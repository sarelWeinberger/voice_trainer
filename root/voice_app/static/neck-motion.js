/* Dense image motion stays local; only bounded derived coordinates leave the browser. */
(function(root){
 'use strict';
 const median=a=>{const s=a.slice().sort((x,y)=>x-y);return s.length?s[Math.floor(s.length/2)]:0;};
 function texture(gray,w,x,y){
  x=Math.round(x);y=Math.round(y);let sum=0,sq=0,n=0;
  for(let dy=-4;dy<=4;dy++)for(let dx=-4;dx<=4;dx++){const v=gray[(y+dy)*w+x+dx];if(v===undefined)return 0;sum+=v;sq+=v*v;n++;}
  return sq/n-(sum/n)**2;
 }
 function match(previous,current,w,h,point,checkBack=true){
  const x=Math.round(point[0]),y=Math.round(point[1]),r=6,side=2*r+1;
  const variance=texture(previous,w,x,y);
  if(x<r+5||y<r+5||x>=w-r-5||y>=h-r-5||variance<9)return null;
  const costs=[];let best=Infinity,bx=0,by=0;
  function costAt(dx,dy){
   const index=(dy+r)*side+dx+r;if(costs[index]!==undefined)return costs[index];
   let cost=0,sum=0;for(let py=-4;py<=4;py++)for(let px=-4;px<=4;px++){
    const diff=previous[(y+py)*w+x+px]-current[(y+py+dy)*w+x+px+dx];cost+=diff*diff;sum+=diff;
   }
   cost=Math.max(0,cost/81-(sum/81)**2);
   costs[index]=cost;return cost;
  }
  function consider(dx,dy){const cost=costAt(dx,dy);if(cost<best){best=cost;bx=dx;by=dy;}}
  for(let dy=-r;dy<=r;dy+=2)for(let dx=-r;dx<=r;dx+=2)consider(dx,dy);
  const coarseX=bx,coarseY=by;
  for(let dy=Math.max(-r,coarseY-1);dy<=Math.min(r,coarseY+1);dy++)for(let dx=Math.max(-r,coarseX-1);dx<=Math.min(r,coarseX+1);dx++)consider(dx,dy);
  if(best>Math.min(900,Math.max(16,variance*.5))||Math.abs(bx)===r||Math.abs(by)===r||texture(current,w,x+bx,y+by)<9)return null;
  const center=costs[(by+r)*side+bx+r];
  const refine=(minus,plus)=>{if(center<1e-8)return 0;const den=minus-2*center+plus;return den>1e-6?Math.max(-.5,Math.min(.5,.5*(minus-plus)/den)):0;};
  const dx=bx+refine(costAt(bx-1,by),costAt(bx+1,by));
  const dy=by+refine(costAt(bx,by-1),costAt(bx,by+1));
  const target=[point[0]+dx,point[1]+dy];
  if(checkBack){const back=match(current,previous,w,h,target,false);
   if(!back||Math.hypot(back.point[0]-point[0],back.point[1]-point[1])>1)return null;}
  return {point:target,delta:[dx,dy],origin:point};
 }
 const zone=(p,w,h)=>(p[0]<w/2?0:1)+(p[1]<h/2?0:2);
 function corner(gray,w,x,y){
  x=Math.round(x);y=Math.round(y);let xx=0,yy=0,xy=0;
  for(let dy=-3;dy<=3;dy++)for(let dx=-3;dx<=3;dx++){
   const k=(y+dy)*w+x+dx,gx=gray[k+1]-gray[k-1],gy=gray[k+w]-gray[k-w];
   xx+=gx*gx;yy+=gy*gy;xy+=gx*gy;
  }
  // The smaller structure-tensor eigenvalue rejects an edge that cannot
  // constrain movement along itself, even when its pixel variance is high.
  return (xx+yy-Math.hypot(xx-yy,2*xy))/98;
 }
 function backgroundModel(matches,w,h,at){
  if(matches.length<6)return null;
  function fit(points){
   const xs=points.map(p=>p.origin[0]-w/2),ys=points.map(p=>p.origin[1]-h/2),mx=median(xs),my=median(ys);
   const dx=median(points.map(p=>p.delta[0])),dy=median(points.map(p=>p.delta[1]));
   let numerator=0,denominator=0;
   points.forEach((p,i)=>{const x=xs[i]-mx,y=ys[i]-my;numerator+=x*(p.delta[1]-dy)-y*(p.delta[0]-dx);denominator+=x*x+y*y;});
   const rotation=numerator/Math.max(denominator,1e-9);
   const tx=median(points.map((p,i)=>p.delta[0]+rotation*ys[i])),ty=median(points.map((p,i)=>p.delta[1]-rotation*xs[i]));
   return {rotation,predict:p=>[tx-rotation*(p[1]-h/2),ty+rotation*(p[0]-w/2)]};
  }
  let model=fit(matches);
  const inliers=matches.filter(p=>{const d=model.predict(p.origin);return Math.hypot(d[0]-p.delta[0],d[1]-p.delta[1])<=1;});
  if(inliers.length<6||inliers.length/matches.length<.7||new Set(inliers.map(p=>zone(p.origin,w,h))).size<3)return null;
  model=fit(inliers);return {...model,points:inliers,delta:model.predict(at)};
 }
 class MotionTracker{
  constructor(w,h,anchors){this.w=w;this.h=h;this.anchors=anchors;this.face=[];this.background=[];this.previous=null;this.rows=[];this.head=[0,0];this.bg=[0,0];this.headMissing=0;this.backgroundMissing=0;this.segment=0;this.reinitializations=0;this.referenceChecks=0;this.referenceCorrections=0;this.backgroundRejected=0;this.bounds=null;this.lastBackgroundReset=-Infinity;}
  setBounds(anchors){const xs=anchors.map(p=>p[0]*this.w),ys=anchors.map(p=>p[1]*this.h);
   this.bounds={left:Math.min(...xs),right:Math.max(...xs),top:Math.min(...ys),bottom:Math.max(...ys)};}
  backgroundPoints(gray,adaptive=false){
   const candidates=[];
   const bounds=this.bounds,width=bounds?bounds.right-bounds.left:0,height=bounds?bounds.bottom-bounds.top:0;
   if(!adaptive){
    for(const x of [.08,.14,.86,.92])for(const y of [.12,.25,.38,.51,.64,.77,.9]){
     const p=[x*this.w,y*this.h];
     if(!bounds||p[0]<bounds.left-width*.75||p[0]>bounds.right+width*.75||p[1]<bounds.top-height*.5){
      const score=texture(gray,this.w,...p);if(score>=9)candidates.push({p,score});
     }
    }
   }else{
   for(let y=12;y<this.h-12;y+=6)for(let x=12;x<this.w-12;x+=6){
    if(x>this.w*.25&&x<this.w*.75)continue;
    if(bounds&&x>=bounds.left-width*.75&&x<=bounds.right+width*.75&&y>=bounds.top-height*.5)continue;
    const score=corner(gray,this.w,x,y);
    if(score>=3&&texture(gray,this.w,x,y)>=9)candidates.push({p:[x,y],score});
   }
   }
   candidates.sort((a,b)=>b.score-a.score);
   const selected=[];
   for(let q=0;q<4;q++){
    let count=0;
    for(const c of candidates)if(zone(c.p,this.w,this.h)===q&&(!adaptive||selected.every(p=>Math.hypot(p[0]-c.p[0],p[1]-c.p[1])>=10))){
     selected.push(c.p);if(++count===3)break;
    }
   }
   return selected;
  }
  headPoints(gray,anchors){
   const points=anchors.map(p=>[p[0]*this.w,p[1]*this.h]);
   const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);
   const left=Math.min(...xs),right=Math.max(...xs),top=Math.min(...ys),bottom=Math.max(...ys);
   // Seed additional textured patches inside the upper face rather than depend
   // on six exact landmark pixels, several of which can lie on smooth skin.
   for(const x of [.1,.3,.5,.7,.9])for(const y of [.15,.35,.55,.75])points.push([left+x*(right-left),top+y*(bottom-top)]);
   const ranked=points.filter(p=>p[0]>=11&&p[1]>=11&&p[0]<this.w-11&&p[1]<this.h-11)
    .map(p=>({p,texture:texture(gray,this.w,...p)})).filter(c=>c.texture>=9).sort((a,b)=>b.texture-a.texture);
   const selected=[];
   for(const candidate of ranked)if(selected.every(p=>Math.hypot(p[0]-candidate.p[0],p[1]-candidate.p[1])>=5)){
    selected.push(candidate.p);if(selected.length===16)break;
   }
   return selected;
  }
  needsReference(){return !this.previous||this.face.length<3;}
  initialize(gray,anchors,t){
   if(this.previous){this.segment++;this.reinitializations++;}
   this.setBounds(anchors);this.lastBackgroundReset=t;
   this.face=this.headPoints(gray,anchors);this.background=this.backgroundPoints(gray);
   this.head=[0,0];this.bg=[0,0];this.previous=gray;
   this.rows.push([t,0,0,0,0,0,this.segment]);
  }
  push(gray,t,freshAnchors=null){
   if(this.rows.length>=900||!Number.isFinite(t)||t<0||t>25||this.rows.length&&t<=this.rows.at(-1)[0])return;
   if(!this.previous||freshAnchors&&this.needsReference()){this.initialize(gray,freshAnchors||this.anchors,t);return;}
   const track=points=>points.map(p=>match(this.previous,gray,this.w,this.h,p)).filter(Boolean);
   let face=track(this.face),allBackground=track(this.background);
   if(freshAnchors){
    this.referenceChecks++;this.setBounds(freshAnchors);
    const b=this.bounds,pad=(b.right-b.left)*.2;
    const consistent=face.filter(p=>p.point[0]>=b.left-pad&&p.point[0]<=b.right+pad&&p.point[1]>=b.top-pad&&p.point[1]<=b.bottom+pad);
    if(consistent.length<3||consistent.length/Math.max(1,face.length)<.6){this.referenceCorrections++;this.initialize(gray,freshAnchors,t);return;}
    face=consistent;
   }
   const at=[median(face.map(p=>p.point[0])),median(face.map(p=>p.point[1]))];
   let model=backgroundModel(allBackground,this.w,this.h,at);
   // Re-detect scenery corners in the PREVIOUS image and retry the same
   // observed image pair. This recovers point depletion without resetting
   // coordinates, bridging an unobserved gap or discarding healthy head tracks.
   if(!model){
    for(const adaptive of [true,false]){
     allBackground=track(this.backgroundPoints(this.previous,adaptive));model=backgroundModel(allBackground,this.w,this.h,at);
     if(model)break;
    }
   }
   const bg=model?.points||[];
   if(face.length<3)this.headMissing++;
   if(!model){this.backgroundMissing++;this.backgroundRejected++;}
   const good=face.length>=3&&!!model;
   if(!good&&face.length>=3&&t-this.lastBackgroundReset>=.5&&this.backgroundPoints(gray).length>=6){
    this.initialize(gray,face.map(p=>[p.point[0]/this.w,p.point[1]/this.h]),t);return;
   }
   if(good){
    const fd=[0,1].map(k=>median(face.map(p=>p.delta[k]))),bd=model.delta;
    for(let k=0;k<2;k++){this.head[k]+=(fd[k]-bd[k])/this.w;this.bg[k]+=bd[k]/this.w;}
   }
   this.face=freshAnchors?this.headPoints(gray,freshAnchors):face.map(p=>p.point);
   this.background=(model?bg:allBackground).map(p=>p.point);this.previous=gray;
   this.rows.push([t,...this.head,...this.bg,good?Math.min(1,face.length/6):0,this.segment].map(v=>Math.round(v*1e6)/1e6));
  }
 }
 class NeckMotionCapture{
  constructor(video,startedAt,seed){this.video=video;this.startedAt=startedAt;this.seed=seed;this.active=false;this.tracker=null;this.handle=null;this.nextMedia=null;this.reason=null;this.pendingSeed=null;this.lastReferenceRequest=-Infinity;this.lastVerifiedAt=0;this.faceAbsent=false;}
  start(){
   this.active=true;
   if(!this.video.requestVideoFrameCallback){this.reason='video_frame_callback_unavailable';return;}
   this.canvas=document.createElement('canvas');this.canvas.width=256;
   this.canvas.height=Math.round(256*this.video.videoHeight/this.video.videoWidth);
   this.ctx=this.canvas.getContext('2d',{willReadFrequently:true});
   if(this.seed&&Date.now()-this.seed.at<=1000){this.pendingSeed=this.seed;this.lastVerifiedAt=this.seed.at;}
   else this.reason='face_reference_unavailable';
   this.schedule();
  }
  requestReference(now=Date.now()){
   const interval=this.tracker&&!this.tracker.needsReference()?1000:500;
   if(!this.active||now-this.lastReferenceRequest<interval)return false;
   this.lastReferenceRequest=now;return true;
  }
  updateReference(seed){
   if(!seed){this.faceAbsent=true;return;}
   if(this.active&&seed&&Date.now()-seed.at<=1000&&Array.isArray(seed.points)&&seed.points.length===6&&
    seed.points.every(p=>Array.isArray(p)&&p.length===2&&p.every(v=>Number.isFinite(v)&&v>=0&&v<=1))){this.pendingSeed=seed;this.faceAbsent=false;this.lastVerifiedAt=seed.at;}
  }
  schedule(){if(this.active)this.handle=this.video.requestVideoFrameCallback((_,meta)=>this.capture(meta));}
  capture(meta){
   if(!this.active)return;
   try{
    // Advance a fixed sampling grid rather than restarting a minimum interval
    // at every accepted frame. The latter halves 31.25fps cameras to 15.625fps.
    const period=1/30;
    if(this.nextMedia!==null&&meta.mediaTime+1e-6<this.nextMedia){this.schedule();return;}
    if(this.nextMedia===null)this.nextMedia=meta.mediaTime;
    this.nextMedia+=(Math.floor(Math.max(0,meta.mediaTime-this.nextMedia)/period)+1)*period;
    if(this.baseMedia===undefined){this.baseMedia=meta.mediaTime;this.baseTime=(Date.now()-this.startedAt)/1000;}
    this.ctx.drawImage(this.video,0,0,this.canvas.width,this.canvas.height);
    const rgba=this.ctx.getImageData(0,0,this.canvas.width,this.canvas.height).data,gray=new Uint8Array(rgba.length/4);
    for(let i=0;i<gray.length;i++)gray[i]=Math.round(.299*rgba[i*4]+.587*rgba[i*4+1]+.114*rgba[i*4+2]);
    const seed=this.pendingSeed&&Date.now()-this.pendingSeed.at<=1000?this.pendingSeed:null;
    this.pendingSeed=null;
    if(!this.tracker&&seed)this.tracker=new MotionTracker(this.canvas.width,this.canvas.height,seed.points);
    if(this.tracker){this.reason=null;
     if(this.faceAbsent||Date.now()-this.lastVerifiedAt>3000)this.tracker.face=[];
     this.tracker.push(gray,this.baseTime+meta.mediaTime-this.baseMedia,seed?.points);}
   }catch(_){this.reason='image_tracking_failed';this.stop();return;}
   this.schedule();
  }
  stop(){this.active=false;if(this.handle!==null)this.video.cancelVideoFrameCallback?.(this.handle);this.handle=null;}
  export(){return {version:'head-motion-trace-v2',source:'browser_patch_tracking',width:256,samples:this.tracker?.rows||[],capture_reason:this.reason,
   tracking_diagnostics:this.tracker?{head_missing_samples:this.tracker.headMissing,background_missing_samples:this.tracker.backgroundMissing,
    remaining_head_points:this.tracker.face.length,remaining_background_points:this.tracker.background.length,reinitializations:this.tracker.reinitializations,
    reference_checks:this.tracker.referenceChecks,reference_corrections:this.tracker.referenceCorrections,background_rejected_samples:this.tracker.backgroundRejected}:null};}
  cancel(){this.stop();this.tracker=null;}
 }
 root.NeckMotionCapture=NeckMotionCapture;
 if(typeof module!=='undefined')module.exports={MotionTracker,match,NeckMotionCapture,backgroundModel};
})(typeof window!=='undefined'?window:globalThis);
