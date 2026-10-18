'use strict';
const assert=require('node:assert/strict');
const {NeckCapture}=require('../root/voice_app/static/neck-capture.js');
(async()=>{
    global.FormData=class { constructor(){this.fields=new Map();} append(key,value){this.fields.set(key,value);} get(key){return this.fields.get(key);} };
    const savedNow=Date.now;let clock=0;Date.now=()=>clock;
    const video={videoWidth:640,videoHeight:480};
    const canvas={getContext:()=>({drawImage(){}}),toBlob:cb=>cb(new Blob(['synthetic-image']))};
    let pendingRequests=[],accepted=0,inflight=0,maxInflight=0;
    global.fetch=(_,opts)=>new Promise(resolve=>{
        inflight++;maxInflight=Math.max(inflight,maxInflight);
        const times=JSON.parse(opts.body.get('timestamps'));
        assert(times.every((t,i)=>i===0||t>times[i-1]));
        assert(times.length<=10);assert.equal(opts.body.get('session_id'),'test');
        pendingRequests.push(()=>{accepted+=times.length;inflight--;resolve({ok:true,json:async()=>({mode:'running',batch_id:opts.body.get('batch_id'),accepted_frames:times.length})});});
    });
    try {
        const sampler=new NeckCapture({video,canvas,sessionId:'test',startedAt:0,headers:()=>({}),onResult(){}});
        sampler.active=true;
        for(let i=0;i<25;i++){clock=i*200;await sampler.capture();}
        // The first HTTP request is still unresolved; sampling nevertheless captured all 25 frames.
        assert.equal(pendingRequests.length,1);assert.equal(sampler.queue.length,20);
        const stopped=sampler.stop();
        for(let i=0;i<10 && (sampler.pending||sampler.queue.length);i++){
            pendingRequests.shift()?.();
            await new Promise(resolve=>setImmediate(resolve));
        }
        await stopped;
        assert.equal(accepted,25);assert.equal(maxInflight,1);assert.equal(sampler.queue.length,0);
        assert.equal(sampler.transportDiagnostics().unconfirmed_frames,0);
        const long=new NeckCapture({video,canvas,sessionId:'test',startedAt:0,headers:()=>({}),onResult(){}});
        long.active=true;
        for(let i=0;i<125;i++){clock=i*200;await long.capture();}
        assert.equal(long.queue.length,120);assert.equal(long.stats.overflow_frames,0);
        const longStopped=long.stop();
        for(let i=0;i<20 && (long.pending||long.queue.length);i++){
            pendingRequests.shift()?.();await new Promise(resolve=>setImmediate(resolve));
        }
        await longStopped;assert.equal(long.stats.acknowledged_frames,125);
        const bounded=new NeckCapture({video,canvas,sessionId:'test',startedAt:0,headers:()=>({}),onResult(){}});
        bounded.active=true;
        for(let i=0;i<200;i++){clock=i*200;await bounded.capture();}
        assert.equal(bounded.queue.length,bounded.maxQueue);
        assert.equal(bounded.stats.overflow_frames,45);
        bounded.cancel();pendingRequests.shift()?.();await bounded.pending;
        assert.equal(bounded.queue.length,0);
        console.log('25/125 frames retained while HTTP is pending; ordered receipts; bounded overflow reported; cancellation PASS');
        const seen=[],stored=new Set();let actualAccepted=0,calls=0;
        global.fetch=async(_,opts)=>{
            const id=opts.body.get('batch_id'),times=JSON.parse(opts.body.get('timestamps'));seen.push(id);calls++;
            if(!stored.has(id)){stored.add(id);actualAccepted+=times.length;}
            if(calls===1)throw new TypeError('Response lost after server acceptance');
            return {ok:true,json:async()=>({batch_id:id,accepted_frames:times.length})};
        };
        const retry=new NeckCapture({video,canvas,sessionId:'test',startedAt:0,headers:()=>({}),onResult(){}});
        retry.retryDelays=[0,0];retry.active=true;
        for(let i=0;i<15;i++){clock=i*200;await retry.capture();}
        await retry.stop();assert.equal(actualAccepted,15);assert.equal(retry.stats.acknowledged_frames,15);
        assert.equal(seen[0],seen[1]);assert.equal(retry.stats.retry_count,1);
        assert.equal(retry.transportDiagnostics().unconfirmed_frames,0);
        let attempts=0;global.fetch=async()=>{attempts++;return {ok:false,status:401};};
        const unauthorized=new NeckCapture({video,canvas,sessionId:'test',startedAt:0,headers:()=>({}),onResult(){}});
        unauthorized.active=true;
        for(let i=0;i<5;i++){clock=i*200;await unauthorized.capture();}
        await unauthorized.stop();assert.equal(attempts,1);assert.equal(unauthorized.stats.failed_batches,1);
        assert.equal(unauthorized.transportDiagnostics().unconfirmed_frames,5);
        const limited=new NeckCapture({video,canvas,sessionId:'test',startedAt:0,headers:()=>({}),onResult(){}});
        limited.maxQueueBytes=1;limited.active=true;
        for(let i=0;i<5;i++){clock=i*200;await limited.capture();}
        await limited.stop();assert.equal(limited.stats.overflow_frames,5);assert.equal(limited.queueBytes,0);
        let finishEncoding;global.fetch=async(_,opts)=>({ok:true,json:async()=>({batch_id:opts.body.get('batch_id'),accepted_frames:1})});
        const delayed=new NeckCapture({video,canvas:{getContext:canvas.getContext,toBlob:cb=>{finishEncoding=cb;}},sessionId:'test',startedAt:0,headers:()=>({}),onResult(){}});
        delayed.active=true;const capturing=delayed.capture(),stopping=delayed.stop();finishEncoding(new Blob(['last-image']));
        await capturing;await stopping;assert.equal(delayed.stats.acknowledged_frames,1);
        global.fetch=async(_,opts)=>new Promise((resolve,reject)=>opts.signal.addEventListener('abort',()=>reject(Error('aborted'))));
        const stalled=new NeckCapture({video,canvas,sessionId:'test',startedAt:0,headers:()=>({}),onResult(){}});
        stalled.active=true;stalled.drainTimeoutMs=20;
        for(let i=0;i<5;i++){clock=i*200;await stalled.capture();}
        await stalled.stop();assert.equal(stalled.transportDiagnostics().unconfirmed_frames,5);
        await new Promise(resolve=>setImmediate(resolve));assert.equal(stalled.pending,null);
        console.log('Lost acknowledgement retried without duplicate receipt; permanent errors, in-flight encoding and drain deadline PASS');
    } finally {Date.now=savedNow;}
})().catch(error=>{console.error(error);process.exitCode=1;});
