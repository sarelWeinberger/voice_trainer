'use strict';
const assert=require('node:assert/strict');
const {MotionTracker,NeckMotionCapture,backgroundModel}=require('../root/voice_app/static/neck-motion.js');
const w=256,h=144,anchors=[[.45,.25],[.55,.25],[.5,.35],[.43,.4],[.57,.4],[.5,.5]];
const gray=new Uint8Array(w*h);
for(let y=0;y<h;y++)for(let x=0;x<w;x++)gray[y*w+x]=Math.round(120+40*Math.sin(x*.31+y*.17)+25*Math.cos(x*.57-y*.43));
const tracker=new MotionTracker(w,h,anchors);tracker.push(gray,0);tracker.push(gray,1/30);
assert(tracker.face.length>=3);
// Deliberately displace apparently healthy points to scenery; face references
// must invalidate them even though patch count has not fallen below three.
tracker.face=[[20,20],[25,30],[30,40],[25,90]];
tracker.push(gray,2/30,anchors);
assert.equal(tracker.referenceCorrections,1);assert.equal(tracker.segment,1);assert.equal(tracker.rows.at(-1)[5],0);
tracker.push(gray,3/30);assert(tracker.rows.at(-1)[5]>=.5);
const points=[];
for(const x of [20,36,220,236])for(const y of [20,70,120])points.push({origin:[x,y],point:[x+1,y-.5],delta:[1-.003*(y-h/2),-.5+.003*(x-w/2)]});
const model=backgroundModel(points,w,h,[145,60]);assert(model);
assert(Math.abs(model.delta[0]-(1-.003*(60-h/2)))<1e-6);
assert(Math.abs(model.delta[1]-(-.5+.003*(145-w/2)))<1e-6);
assert(!backgroundModel(points.filter(p=>p.origin[0]<w/2),w,h,[145,60]));
const inconsistent=points.map((p,i)=>({...p,delta:[i%2?4:-4,i%3?3:-3]}));
assert(!backgroundModel(inconsistent,w,h,[145,60]));
// Fixed sample pixels can miss textured scenery completely. Corners between
// them must be acquired on the same image pair without resetting head motion.
const scenery=new Uint8Array(w*h).fill(120);
for(let y=0;y<h;y++)for(let x=0;x<w;x++){
 const island=(x>=44&&x<=64||x>=192&&x<=212)&&(y>=20&&y<=50||y>=90&&y<=120);
 const head=x>=100&&x<=156&&y>=25&&y<=85;
 if(island||head)scenery[y*w+x]=Math.round(120+40*Math.sin(x*.31+y*.17)+25*Math.cos(x*.57-y*.43));
}
const replenished=new MotionTracker(w,h,anchors);replenished.push(scenery,0);
assert.equal(replenished.background.length,0);
for(let i=1;i<90;i++)replenished.push(scenery,i/30);
assert.equal(replenished.rows.filter(r=>r[5]>=.5).length,89);
assert.equal(replenished.reinitializations,0);
assert(replenished.rows.every(r=>r.slice(1,5).every(v=>v===0)));
console.log('Scenery corners missed by fixed pixels are recovered without coordinate resets PASS');
const oldNow=Date.now;let now=5000;Date.now=()=>now;
try{
 const rgba=new Uint8Array(w*h*4);gray.forEach((v,i)=>{rgba[i*4]=v;rgba[i*4+1]=v;rgba[i*4+2]=v;rgba[i*4+3]=255;});
 const c=new NeckMotionCapture({},0,null);c.active=true;c.tracker=tracker;c.lastVerifiedAt=1000;
 c.canvas={width:w,height:h};c.ctx={drawImage(){},getImageData(){return {data:rgba};}};c.schedule=()=>{};
 c.capture({mediaTime:1});assert.equal(c.tracker.rows.at(-1)[5],0);
 c.updateReference({at:now,points:anchors});c.capture({mediaTime:1.04});c.capture({mediaTime:1.08});
 assert(c.tracker.rows.at(-1)[5]>=.5);
 c.updateReference(null);c.capture({mediaTime:1.12});assert.equal(c.tracker.rows.at(-1)[5],0);
}finally{Date.now=oldNow;}
console.log('Healthy-count drift, reference expiry/absence, distributed background consensus and camera rotation control PASS');
const {analyzeAsync}=require('../root/voice_app/static/neck-motion-analysis.js');
global.Worker=class{constructor(){throw Error('Worker blocked');}};
global.document={baseURI:'http://localhost/'};
analyzeAsync({version:'head-motion-trace-v2',source:'browser_patch_tracking',samples:[]}).then(report=>{
 assert.equal(report.reason,'too_few_motion_samples');delete global.Worker;delete global.document;
 console.log('Local analysis remains available when worker creation is blocked PASS');
}).catch(error=>{console.error(error);process.exitCode=1;});
