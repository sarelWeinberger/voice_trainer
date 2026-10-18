'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const {MotionTracker,NeckMotionCapture}=require('../static/neck-motion.js');
// Exercise real scheduling against camera media clocks, without HTTP or canvas cost.
for(const fps of [25,30,31.25,60]){
 const capture=new NeckMotionCapture({},Date.now(),null),times=[];
 capture.active=true;capture.canvas={width:1,height:1};
 capture.ctx={drawImage(){},getImageData(){return {data:new Uint8Array(4)};}};
 capture.tracker={push(gray,t){times.push(t);}};capture.schedule=()=>{};
 for(let i=0;i<Math.round(fps*4);i++)capture.capture({mediaTime:i/fps});
 assert(times.length>=Math.min(fps,30)*4-2,`${fps}fps captured only ${times.length}`);
 assert(times.length<=122);assert(times.every((t,i)=>!i||t>times[i-1]));
}
console.log('25/30/31.25/60fps sampling grids retain target rate without halving PASS');
const w=256,h=144,anchors=[[.45,.25],[.55,.25],[.5,.35],[.43,.4],[.57,.4],[.5,.5]];
function image(t,mode){
 const result=new Uint8Array(w*h),camera=mode==='camera'?1.8*Math.sin(2*Math.PI*4.4*t):0;
 const head=mode==='shake'?1.8*Math.sin(2*Math.PI*4.4*t):mode==='slow'?1.8*Math.sin(2*Math.PI*.5*t):0;
 for(let y=0;y<h;y++)for(let x=0;x<w;x++){
  const shifted=x-camera-(x>w*.3&&x<w*.7&&y>h*.1&&y<h*.8?head:0);
  result[y*w+x]=Math.round(120+40*Math.sin(shifted*.31+y*.17)+25*Math.cos(shifted*.57-y*.43));
 }
 return result;
}
const reports={};
for(const mode of ['still','shake','slow','camera']){
 const tracker=new MotionTracker(w,h,anchors);
 for(let i=0;i<180;i++)tracker.push(image(i/30,mode),i/30);
 assert.equal(tracker.rows.length,180);assert(tracker.rows.filter(r=>r[5]>=.5).length>170);
 reports[mode]={version:'head-motion-trace-v2',source:'browser_patch_tracking',samples:tracker.rows};
 if(mode==='still')assert(tracker.rows.every(r=>r.slice(1,5).every(v=>Math.abs(v)<1e-6)));
}
assert(Math.max(...reports.shake.samples.map(r=>Math.abs(r[1])))>.003);
const dir=process.env.NECK_RESEARCH_OUTPUT || require('node:path').join(require('node:os').tmpdir(),'neck-motion-tests');fs.mkdirSync(dir,{recursive:true});
fs.writeFileSync(dir+'/synthetic_traces.json',JSON.stringify(reports));
console.log('Real tracker: still, 4.4 Hz head movement, slow movement, and whole-frame camera movement PASS');
