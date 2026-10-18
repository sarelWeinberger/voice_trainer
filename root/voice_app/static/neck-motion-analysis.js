/* Pure local motion analysis. Shared by the browser, its worker and Node checks. */
(function(root){
 'use strict';
 const sum=a=>a.reduce((s,x)=>s+x,0),mean=a=>sum(a)/a.length;
 const round=(n,d=3)=>Math.round(n*10**d)/10**d;
 const median=a=>{const b=a.slice().sort((x,y)=>x-y),m=Math.floor(b.length/2);return b.length%2?b[m]:(b[m-1]+b[m])/2;};
 const dot=(a,b)=>sum(a.map((x,i)=>x*b[i]));
 function detrend(t,y,duration=null){
  const vectors=[t.map(()=>1),t.map(x=>x-mean(t))],orthogonal=[];let clean=y.map(row=>row.slice());
  if(duration!==null)for(const k of [1,2])for(const wave of [Math.sin,Math.cos])vectors.push(t.map(x=>wave(2*Math.PI*k*x/duration)));
  for(let vector of vectors){
   for(const q of orthogonal){const projection=dot(q,vector);vector=vector.map((v,i)=>v-q[i]*projection);}
   const norm=Math.sqrt(dot(vector,vector));if(norm<1e-9)continue;
   const q=vector.map(v=>v/norm);orthogonal.push(q);
   const projections=clean[0].map((_,k)=>dot(q,clean.map(r=>r[k])));
   clean=clean.map((r,i)=>r.map((v,k)=>v-q[i]*projections[k]));
  }
  return clean;
 }
 function fit(t,y,f,duration=null){
  const b=detrend(t,t.map(x=>[Math.sin(2*Math.PI*f*x),Math.cos(2*Math.PI*f*x)]),duration);
  const s=b.map(r=>r[0]),c=b.map(r=>r[1]),aa=dot(s,s),ab=dot(s,c),bb=dot(c,c),det=aa*bb-ab*ab;
  if(det<1e-12)return null;
  const coefficients=[[],[]];let gain=0;
  for(let k=0;k<4;k++){
   const sy=dot(s,y.map(r=>r[k])),cy=dot(c,y.map(r=>r[k]));
   const a=(bb*sy-ab*cy)/det,d=(aa*cy-ab*sy)/det;
   coefficients[0].push(a);coefficients[1].push(d);if(k<2)gain+=a*sy+d*cy;
  }
  return {gain,coefficients};
 }
 function analyze(payload){
  const rows=payload?.samples;
  if(!Array.isArray(rows)||rows.length>900)throw Error('Invalid motion samples');
  let previous=-1,segment=0;
  const cols=payload.version==='head-motion-trace-v2'?7:6;
  if(!['head-motion-trace-v1','head-motion-trace-v2'].includes(payload.version)||payload.source!=='browser_patch_tracking')throw Error('Invalid motion trace');
  for(const row of rows){
   if(!Array.isArray(row)||row.length!==cols||row.some(v=>!Number.isFinite(v))||row[0]<=previous||row[0]<0||row[0]>25||row.slice(1,5).some(v=>Math.abs(v)>2)||row[5]<0||row[5]>1||cols===7&&(!Number.isInteger(row[6])||row[6]<segment||row[6]>900))throw Error('Invalid motion sample');
   previous=row[0];if(cols===7)segment=row[6];
  }
  const result={version:'head-movement-stability-v3',status:'insufficient_data',source:'browser_local',unit:'percent_image_width',
   amplitude_kind:'supported_sinusoid_rms',effort_score:null,windows:[],sample_count:rows.length};
  if(['video_frame_callback_unavailable','face_reference_unavailable','image_tracking_failed'].includes(payload.capture_reason))result.capture_reason=payload.capture_reason;
  if(payload.tracking_diagnostics)result.tracking_diagnostics=payload.tracking_diagnostics;
  if(rows.length<24)return {...result,reason:'too_few_motion_samples'};
  const fps=1/median(rows.slice(1).map((r,i)=>r[0]-rows[i][0]));result.effective_fps=round(fps);
  if(fps>60)return {...result,reason:'motion_sampling_invalid'};
  if(fps<12)return {...result,reason:'motion_sampling_too_slow'};
  const start=rows[0][0],end=rows.at(-1)[0];
  for(const duration of [2,6,10]){
   if(end-start<duration-.1)continue;
   for(let lower=start;lower<end-duration+.1-1e-9;lower+=duration===2?1:2){
    const part=rows.filter(r=>r[0]>=lower&&r[0]<lower+duration),usable=part.filter(r=>r[5]>=.5);
    const coverage=usable.length/Math.max(1,part.length),span=usable.length>1?usable.at(-1)[0]-usable[0][0]:0;
    const maxGap=usable.length>1?Math.max(...usable.slice(1).map((r,i)=>r[0]-usable[i][0])):null;
    const temporal=new Set(usable.map(r=>Math.floor((r[0]-lower)/.1))).size/Math.round(duration*10);
    const w={start:round(lower),end:round(lower+duration),status:'insufficient_data',amplitude_percent_image_width:null,dominant_frequency_hz:null,
     background_amplitude_percent_image_width:null,received_samples:part.length,usable_samples:usable.length,tracking_coverage:round(coverage),
     tracked_span_seconds:round(span),max_gap_seconds:maxGap===null?null:round(maxGap),temporal_coverage:round(temporal)};
    result.windows.push(w);
    if(cols===7&&new Set(usable.map(r=>r[6])).size>1){w.reason='tracking_reinitialized';continue;}
    if(usable.length<20){w.reason='too_few_usable_samples';continue;}
    if(coverage<.8){w.reason='tracking_coverage_low';continue;}
    if(span<duration*.85){w.reason='tracked_span_too_short';continue;}
    if(temporal<.8){w.reason='temporal_coverage_low';continue;}
    if(maxGap>.4){w.reason='sampling_gap_too_large';continue;}
    const density=(usable.length-1)/span;
    if(density<12){w.reason='motion_sampling_too_slow';continue;}
    const t=usable.map(r=>r[0]-usable[0][0]),motion=usable.map(r=>r.slice(1,5)),clean=detrend(t,motion,duration);
    const energy=sum(clean.map(r=>r[0]**2+r[1]**2)),low=3/duration,high=Math.min(duration===2?12:duration===6?1.5:.5,.4*Math.min(fps,density));let best=null;
    const searchLow=2/duration,searchHigh=Math.min(high+1/duration,.48*Math.min(fps,density));
    const searchFirst=Math.ceil((searchLow-1e-9)/.05)*.05;
    for(let step=0;searchFirst+step*.05<=searchHigh+.001;step++){
     const frequency=searchFirst+step*.05,candidate=fit(t,clean,frequency,duration);
     if(candidate&&(!best||candidate.gain>best.gain))best={...candidate,frequency};
    }
    if(!best){w.reason='singular_motion_fit';continue;}
    const fraction=Math.max(0,Math.min(1,best.gain/Math.max(energy,1e-20))),halves=[];
    for(const side of [false,true]){
     const indices=t.map((v,i)=>i).filter(i=>(t[i]>=span/2)===side),ht=indices.map(i=>t[i]),hy=indices.map(i=>motion[i]);
     const half=fit(ht,detrend(ht,hy,duration),best.frequency,duration);
     halves.push(half?[...half.coefficients[0].slice(0,2),...half.coefficients[1].slice(0,2)]:[0,0,0,0]);
    }
    const norms=halves.map(v=>Math.sqrt(dot(v,v))),coherence=dot(halves[0],halves[1])/Math.max(norms[0]*norms[1],1e-20),ratio=Math.min(...norms)/Math.max(...norms,1e-20);
    const amplitude=columns=>100*Math.sqrt(sum(best.coefficients.flatMap(row=>columns.map(k=>row[k]**2)))/2);
    const candidate=best.frequency>=low-1e-7&&best.frequency<=high+1e-7&&energy>1e-14&&amplitude([0,1])>=.0003&&fraction>=.55&&coherence>=.8&&ratio>=.5;
    const cameraConfounded=candidate&&amplitude([2,3])>amplitude([0,1])*.5;
    const supported=candidate&&!cameraConfounded,stationary=energy<=1e-14;
    Object.assign(w,{status:'measured',method:'observed_timestamp_periodicity_check',periodicity_status:supported?'supported':cameraConfounded?'camera_confounded':stationary?'stationary':'not_established',
     amplitude_percent_image_width:supported?round(amplitude([0,1]),5):stationary?0:null,
     background_amplitude_percent_image_width:candidate?round(amplitude([2,3]),5):null,dominant_frequency_hz:supported?round(best.frequency,2):null,
     motion_rms_percent_image_width:round(100*Math.sqrt(energy/t.length),5),periodic_fraction:stationary?0:round(fraction,4),half_coherence:stationary?0:round(coherence,4),
     half_amplitude_ratio:stationary?0:round(ratio,4),band_hz:[round(low,2),round(high,2)],effective_sample_density:round(density)});
   }
  }
  if(result.windows.some(w=>w.status==='measured')){result.status='measured';result.reason=null;}else result.reason='no_usable_motion_windows';
  return result;
 }
 function analyzeAsync(payload){
  if(typeof Worker==='undefined'||typeof document==='undefined')return Promise.resolve().then(()=>analyze(payload));
  return new Promise((resolve,reject)=>{
   let worker;
   try{worker=new Worker(new URL('neck-motion-analysis.js?v=2',document.baseURI));}
   catch(_){try{resolve(analyze(payload));}catch(error){reject(error);}return;}
   const timeout=setTimeout(()=>{worker.terminate();try{resolve(analyze(payload));}catch(error){reject(error);}},3000);
   worker.onmessage=event=>{clearTimeout(timeout);worker.terminate();event.data.error?reject(Error(event.data.error)):resolve(event.data.report);};
   worker.onerror=()=>{clearTimeout(timeout);worker.terminate();try{resolve(analyze(payload));}catch(error){reject(error);}};
   worker.postMessage(payload);
  });
 }
 root.NeckMotionAnalysis={analyze,analyzeAsync};
 if(typeof module!=='undefined')module.exports={analyze,analyzeAsync};
 if(typeof WorkerGlobalScope!=='undefined'&&self instanceof WorkerGlobalScope)self.onmessage=event=>{
  try{self.postMessage({report:analyze(event.data)});}catch(error){self.postMessage({error:error.message});}
 };
})(typeof window!=='undefined'?window:globalThis);
