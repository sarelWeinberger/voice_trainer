'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const {neckGroupResults,compareNeckResults,neckObservationText}=require('../root/voice_app/static/neck-analysis-v1.js');
const keys=['posture_compensation','shoulder_recruitment','scm_lateral_recruitment','anterior_submental_activity','jaw_compensation'];
const packet=(score=2)=>({neckAnalysis:{status:'complete',versions:{prompt:'v1.6',model:'same-model'},
 dimensions:Object.fromEntries(keys.map(key=>[key,{score,confidence:.8,evidence:['נראית רכינה קדימה בפריימים 1, 2, 3.']}]))}});
const older=packet(3),current=packet(1);
assert.deepEqual(neckGroupResults(current).map(g=>g.score),[1,1,1]);
assert.equal(compareNeckResults(current,older).length,3);
assert(neckObservationText(neckGroupResults(current)[0]).includes('רכינה קדימה'));
assert(!neckObservationText(neckGroupResults(current)[0]).includes('פריימים'));
assert(!neckObservationText(neckGroupResults(packet(3))[0]).includes('שרירי'));
for(const mutate of [
 p=>p.neckAnalysis.status='error',
 p=>p.neckAnalysis.versions.prompt='other-rubric',
 p=>p.neckAnalysis.versions={},
 p=>p.neckAnalysis.diagnostics={capture_transport:{missing_frames:2}},
 p=>Object.values(p.neckAnalysis.dimensions).forEach(d=>d.confidence=.1),
]){
 const unsuitable=packet(0);mutate(unsuitable);
 assert.equal(compareNeckResults(unsuitable,older).length,0);
}
const partial=packet(1);partial.neckAnalysis.dimensions.shoulder_recruitment.score=null;
assert.equal(neckGroupResults(partial)[0].partial,true);
assert(!compareNeckResults(partial,older).some(g=>g.key==='posture'));
assert.equal(compareNeckResults(older,current).length,0);
assert.equal(compareNeckResults(current,current).length,0);

// Exercise each real recording coordinator with audio-first and neck-first completion.
(async()=>{
 for(const mobile of [false,true])for(const audioFirst of [false,true])for(const voiceFailed of [false,true]){
  const source=fs.readFileSync(mobile?'root/voice_app/static/mobile.html':'root/voice_app/static/app.js','utf8');
  const start=source.indexOf('async function stopRecording()');
  const end=source.indexOf('\nfunction pcmToWav(',start);
  let audioDone,neckDone;const audio=new Promise(r=>audioDone=r),neck=new Promise(r=>neckDone=r);
  const saved=[];const dummy={disconnect(){},close(){}};
  const button={classList:{remove(){}},_audioContext:dummy,_ctx:dummy,_proc:dummy,_src:dummy,_pcm:[],_processor:dummy,_source:dummy,_pcmChunks:[]};
  const env={clearInterval(){},timerInterval:1,recordBtn:button,recordLabel:{},timerEl:{},audioContext:dummy,
   neckEffortInterval:null,neckEffortSessionId:"joined-session",pcmToWav:()=>({}),setStatus(){},stopNeckEffortMonitoring(){},
   localStorage:{getItem:()=>JSON.stringify([{neck_effort:older}])},historyKey:()=> 'user-history',_historyKey:()=> 'user-history',
   sendAudio:()=>audio,stopNeckEffortRecordingSession:()=>neck,displayResults(){},saveToHistory:data=>saved.push(data)};
  vm.createContext(env);vm.runInContext(source.slice(start,end),env);
  const running=env.stopRecording();
  if(audioFirst)audioDone(voiceFailed?undefined:{scores:{G:1},session_id:'joined-session'});else neckDone(current);
  await Promise.resolve();await Promise.resolve();assert.equal(saved.length,0);
  if(audioFirst)neckDone(current);else audioDone(voiceFailed?undefined:{scores:{G:1},session_id:'joined-session'});
  await running;
  assert.equal(saved.length,1);assert.equal(saved[0].neck_effort,current);
  if (!voiceFailed) assert.equal(saved[0].neck_previous.neckAnalysis.dimensions.jaw_compensation.score,3);
  else assert.equal(Object.keys(saved[0].scores).length,0);
  assert.equal(saved[0].session_id,'joined-session');
 }
 console.log('Group scores, grounded explanations, guarded positive feedback and joined-history persistence PASS');
})().catch(error=>{console.error(error);process.exitCode=1;});

// An unassessed under-chin region must not become a reassuring group zero.
const incompleteZero=packet(0);
incompleteZero.neckAnalysis.dimensions.anterior_submental_activity.score=null;
const incompleteNeck=neckGroupResults(incompleteZero).find(g=>g.key==='neck');
assert.equal(incompleteNeck.score,null);
assert.equal(incompleteNeck.partial,true);
const appearance=packet(0);
appearance.neckAnalysis.versions.appearance='visible-neck-appearance-v1';
appearance.neckAnalysis.dimensions.anterior_submental_activity.score=2;
assert.equal(neckGroupResults(appearance).find(g=>g.key==='neck').score,2);
assert.equal(compareNeckResults(appearance,older).length,0);
console.log('Visible neck appearance grade preserved, incomplete zero withheld, old construct comparison blocked PASS');
