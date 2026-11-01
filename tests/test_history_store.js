'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');const vm=require('node:vm');
const History=require('../root/voice_app/static/history-store.js');
const rows=Array.from({length:15},(_,i)=>({session_id:'session-'+i,timestamp:new Date(Date.UTC(2026,0,i+1)).toISOString(),scores:{G:i%4}}));
const storage={values:new Map(),getItem(key){return this.values.get(key)||null;},setItem(key,value){this.values.set(key,value);}};
History.save('alice',rows,storage);
assert.equal(History.read('alice',storage).length,10);
assert.equal(History.read('alice',storage)[0].session_id,'session-14');
assert.equal(History.read('bob',storage).length,0);
// A fresh application instance retains saved results and an empty server never wipes them.
const restarted={localStorage:storage,console};vm.createContext(restarted);
vm.runInContext(fs.readFileSync('root/voice_app/static/history-store.js','utf8'),restarted);
const afterRestart=restarted.VoiceHistory.read('alice');
assert.equal(afterRestart.length,10);
assert.equal(restarted.VoiceHistory.merge(afterRestart,[]).length,10);
const final={...rows[14],neck_effort:{neckAnalysis:{status:'complete',dimensions:{jaw_compensation:{score:1}}}}};
const provisional={...rows[14],timestamp:'2026-01-15T00:00:01Z',neck_effort:null};
const merged=History.merge([final],[provisional]);
assert.equal(merged.length,1);assert.equal(merged[0].neck_effort.neckAnalysis.status,'complete');
assert.equal(History.merge([provisional],[final])[0].neck_effort.neckAnalysis.status,'complete');
assert.equal(History.merge([{date:rows[0].timestamp}],[{timestamp:rows[0].timestamp}]).length,1);
assert.equal(History.merge([null,{timestamp:'invalid'}]).length,0);
storage.setItem('broken','bad JSON');assert.deepEqual(History.read('broken',storage),[]);

// Actual desktop/mobile cloud handlers keep local history when the restarted server returns [].
(async()=>{
 for(const mobile of [false,true]){
  const source=fs.readFileSync(mobile?'root/voice_app/static/mobile.html':'root/voice_app/static/app.js','utf8');
  const start=source.indexOf('async function loadCloudHistory()');
  const end=source.indexOf(mobile?'\nfunction renderHistory(':'\n// ---- Quick Stats',start);
  let user='alice',rendered;const env={VoiceHistory:{...History,read:key=>History.read(key,storage),save:(key,entries)=>History.save(key,entries,storage)},
   historyKey:()=>user,_historyKey:()=>user,getAuthHeaders:()=>({Authorization:'Bearer test'}),
   fetch:async()=>({ok:true,json:async()=>({history:[]})}),renderHistory:rows=>rendered=rows,
   renderProgressChart(){},updateQuickStats(){},syncTabEmptyStates(){},console};
  vm.createContext(env);vm.runInContext(source.slice(start,end),env);
  await env.loadCloudHistory();assert.equal(rendered.length,10);assert.equal(History.read('alice',storage).length,10);
  rendered=null;env.fetch=async()=>{user='bob';return{ok:true,json:async()=>({history:rows})};};
  await env.loadCloudHistory();assert.equal(rendered,null);assert.equal(History.read('bob',storage).length,0);
 }
 console.log('Ten per-user results survive application reload, empty cloud history, deduplication and account switches PASS');
})().catch(error=>{console.error(error);process.exitCode=1;});
