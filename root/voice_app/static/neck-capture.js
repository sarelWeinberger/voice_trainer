/* Bounded neck-only acquisition: camera sampling never waits for HTTP responses. */
(function(root) {
    'use strict';
    class NeckCapture {
        constructor({video, canvas, sessionId, startedAt, headers, onResult, motionSeed = null}) {
            Object.assign(this, {video, canvas, sessionId, startedAt, headers, onResult, motionSeed});
            this.queue = []; this.active = false; this.encoding = false; this.pending = null;
            this.controller = null; this.timer = null;
            // Covers every frame of a supported 25-second recording in RAM.
            this.maxQueue = 150; this.maxQueueBytes = 16*1024*1024; this.queueBytes = 0;
            this.requestTimeoutMs = 8000; this.drainTimeoutMs = 12000; this.retryDelays = [250,750];
            this.sequence = 0; this.closed = false; this.cancelled = false; this.fatal = false;
            this.stats = {captured_frames:0,acknowledged_frames:0,encoding_failed_frames:0,
                overflow_frames:0,retry_count:0,failed_batches:0};
        }
        start() {
            this.active = true;
            if (root.NeckMotionCapture) {
                this.motion = new root.NeckMotionCapture(this.video, this.startedAt, this.motionSeed);
                this.motion.start();
            }
            this.timer = setInterval(() => this.capture(), 200);
            this.capture();
        }
        async capture() {
            if (!this.active || this.encoding || !this.video.videoWidth || !this.video.videoHeight) return;
            this.encoding = true;
            const timestamp = (Date.now() - this.startedAt) / 1000;
            try {
                const scale = Math.min(1, 480 / Math.max(this.video.videoWidth, this.video.videoHeight));
                this.canvas.width = Math.max(1, Math.round(this.video.videoWidth * scale));
                this.canvas.height = Math.max(1, Math.round(this.video.videoHeight * scale));
                this.canvas.getContext('2d').drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
                this.encodingTask = new Promise(resolve => this.canvas.toBlob(resolve, 'image/jpeg', .95));
                const blob = await this.encodingTask;
                if (this.closed || this.cancelled) return;
                if (!blob) {this.stats.encoding_failed_frames++;return;}
                this.stats.captured_frames++;
                if (this.queue.length >= this.maxQueue || this.queueBytes+blob.size>this.maxQueueBytes) {
                    this.stats.overflow_frames++;return;
                }
                this.queue.push({blob, timestamp});
                this.queueBytes+=blob.size;
                if (this.active && this.queue.length >= 5) this.pump();
            } catch (_) {this.stats.encoding_failed_frames++;}
            finally { this.encoding = false; this.encodingTask = null; }
        }
        pump(force = false) {
            if (this.pending || this.closed || this.fatal || !this.queue.length || (!force && this.queue.length < 5)) return this.pending;
            const batch = this.queue.splice(0, 10);
            this.queueBytes-=batch.reduce((sum,item)=>sum+item.blob.size,0);
            const batchId=String(++this.sequence);
            this.pending = (async () => {
                for(let attempt=0;attempt<=this.retryDelays.length;attempt++){
                    if(this.closed||this.cancelled)return;
                    if(attempt){
                        this.stats.retry_count++;
                        await new Promise(resolve=>setTimeout(resolve,this.retryDelays[attempt-1]));
                        if(this.closed||this.cancelled)return;
                    }
                    const controller=new AbortController();this.controller=controller;
                    const timeout=setTimeout(()=>controller.abort(),this.requestTimeoutMs);
                    const form=new FormData();
                    form.append('session_id',this.sessionId);form.append('batch_id',batchId);
                    form.append('timestamps',JSON.stringify(batch.map(item=>item.timestamp)));
                    for(const item of batch)form.append('frames',item.blob,'frame.jpg');
                    let retryable=true;
                    try{
                        const response=await fetch('/api/neck-effort/frames',{
                            method:'POST',body:form,headers:this.headers(),signal:controller.signal});
                        if(!response.ok){
                            retryable=[408,425,429].includes(response.status)||response.status>=500;
                            throw Error('neck_capture_http');
                        }
                        const result=await response.json();
                        if(result.batch_id!==batchId||result.accepted_frames!==batch.length)throw Error('neck_capture_receipt');
                        if(this.closed||this.cancelled)return;
                        this.stats.acknowledged_frames+=batch.length;
                        if(this.active){try{this.onResult(result);}catch(_){/* Receipt remains acknowledged. */}}
                        return;
                    }catch(_){
                        if(this.closed||this.cancelled)return;
                        if(!retryable){this.fatal=true;break;}
                    }finally{clearTimeout(timeout);}
                }
                this.stats.failed_batches++;
            })().finally(() => {
                this.pending = null;
                if (this.active) this.pump();
            });
            return this.pending;
        }
        requestMotionReference() { return this.motion?.requestReference() || false; }
        updateMotionReference(seed) { this.motion?.updateReference(seed); }
        motionTrace() { return this.motion?.export() || null; }
        localMotionReport() {
            if (!this.localReportPromise) {
                const trace=this.motionTrace();
                this.localReportPromise=trace && root.NeckMotionAnalysis
                    ? root.NeckMotionAnalysis.analyzeAsync(trace).catch(()=>null) : Promise.resolve(null);
            }
            return this.localReportPromise;
        }
        transportDiagnostics() {
            return {version:'neck-capture-transport-v1',...this.stats,
                unconfirmed_frames:Math.max(0,this.stats.captured_frames-this.stats.acknowledged_frames)};
        }
        async stop() {
            this.motion?.stop();
            this.active = false; clearInterval(this.timer); this.timer = null;
            const deadline = performance.now() + this.drainTimeoutMs;
            const withinDeadline=async task=>{
                let timeout;
                await Promise.race([task,new Promise(resolve=>{timeout=setTimeout(resolve,Math.max(1,deadline-performance.now()));})]);
                clearTimeout(timeout);
            };
            // A JPEG whose capture already began belongs to this recording.
            if(this.encodingTask)await withinDeadline(this.encodingTask);
            while ((this.pending || this.queue.length) && performance.now() < deadline) {
                const task = this.pending || this.pump(true);
                if (!task) break;
                await withinDeadline(task);
            }
            this.closed=true;
            if (this.controller && this.pending) this.controller.abort();
            // Aborted upload may already have been accepted. Session checks prevent cross-session writes.
            this.queue = [];this.queueBytes=0;
            return this.transportDiagnostics();
        }
        cancel() {
            this.motion?.cancel();
            this.closed=true;this.cancelled=true;
            this.active = false; clearInterval(this.timer); this.timer = null;
            if (this.controller) this.controller.abort();
            this.queue = [];this.queueBytes=0;
        }
    }
    root.NeckCapture = NeckCapture;
    if (typeof module !== 'undefined') module.exports = {NeckCapture};
})(typeof window !== 'undefined' ? window : globalThis);
