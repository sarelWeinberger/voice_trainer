'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const {MotionTracker,NeckMotionCapture}=require('../root/voice_app/static/neck-motion.js');
const w=256,h=144,anchors=[[.45,.25],[.55,.25],[.5,.35],[.43,.4],[.57,.4],[.5,.5]];
function frame(mode='normal'){
 const gray=new Uint8Array(w*h);
 for(let y=0;y<h;y++)for(let x=0;x<w;x++){
  const background=x<w*.3||x>w*.7;
  gray[y*w+x]=mode==='blank'||mode==='background_blank'&&background?120:
   Math.round(120+40*Math.sin(x*.31+y*.17)+25*Math.cos(x*.57-y*.43)+(mode==='bright'?35:0));
 }
 return gray;
}
const reports={};
const tracker=new MotionTracker(w,h,anchors);
for(let i=0;i<180;i++){
 tracker.push(frame(i===15?'blank':'normal'),i/30,i===17?anchors:null);
 if(i===16)assert(tracker.needsReference());
}
assert(tracker.rows.slice(18).every(r=>r[5]>=.5));
assert(tracker.rows.slice(18).every(r=>r.slice(1,5).every(v=>v===0)));
assert.equal(tracker.reinitializations,1);assert.equal(tracker.segment,1);
assert(tracker.rows.slice(0,15).every(r=>r[6]===0));assert(tracker.rows.slice(17).every(r=>r[6]===1));
reports.head_recovery={version:'head-motion-trace-v2',source:'browser_patch_tracking',samples:tracker.rows};
// A head-free image must never recover just because a new reference was supplied.
const blank=new MotionTracker(w,h,anchors);
for(let i=0;i<60;i++)blank.push(frame('blank'),i/30,i%15===0?anchors:null);
assert(blank.rows.every(r=>r[5]===0));assert(blank.needsReference());
const background=new MotionTracker(w,h,anchors);
for(let i=0;i<180;i++)background.push(frame(i>=15&&i<=20?'background_blank':'normal'),i/30);
assert(background.rows.slice(23).every(r=>r[5]>=.5));
assert(background.rows.slice(23).every(r=>r.slice(1,5).every(v=>v===0)));
assert.equal(background.reinitializations,1);
reports.background_recovery={version:'head-motion-trace-v2',source:'browser_patch_tracking',samples:background.rows};
const brightness=new MotionTracker(w,h,anchors);
for(let i=0;i<120;i++)brightness.push(frame(i>=15&&i<=20?'bright':'normal'),i/30);
assert(brightness.rows.slice(1).every(r=>r[5]>=.5));
assert(brightness.rows.every(r=>r.slice(1,5).every(v=>v===0)));
const capture=new NeckMotionCapture({},0,null);capture.active=true;capture.tracker=blank;
assert(capture.requestReference(1000));assert(!capture.requestReference(1100));assert(capture.requestReference(1500));
capture.tracker=brightness;assert(!capture.requestReference(2000));assert(capture.requestReference(2500));
capture.active=false;assert(!capture.requestReference(4000));
const dir=process.env.NECK_RESEARCH_OUTPUT || require('node:path').join(require('node:os').tmpdir(),'neck-motion-tests');fs.mkdirSync(dir,{recursive:true});
fs.writeFileSync(dir+'/recovery_traces.json',JSON.stringify(reports));
// Run actual desktop/mobile camera callbacks: recognition resumes only on request.
(async()=>{
 for(const filename of ['app.js','mobile.html']){
  const source=fs.readFileSync('root/voice_app/static/'+filename,'utf8');
  const start=source.indexOf('function initFaceMesh()'),end=source.indexOf('function onFaceResults',start);
  let calls=0,requested=false;const context={isRecording:true,video:{},mVideo:{},onFaceResults(){},console,
   neckSampler:{requestMotionReference(){return requested;}},
   FaceMesh:class{setOptions(){}onResults(){}async send(){calls++;}},
   Camera:class{constructor(video,options){this.options=options;}start(){}}};
  vm.createContext(context);vm.runInContext(source.slice(start,end),context);context.initFaceMesh();
  const camera=context.camera||context.mCamera;
  await camera.options.onFrame();assert.equal(calls,0);
  requested=true;await camera.options.onFrame();assert.equal(calls,1);
  requested=false;context.isRecording=false;await camera.options.onFrame();assert.equal(calls,2);
 }
 console.log('Head re-acquisition, background recovery, illumination, no false recovery, segment boundaries and live recognition requests PASS');
})().catch(error=>{console.error(error);process.exitCode=1;});
