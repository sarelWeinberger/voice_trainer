'use strict';
const assert = require('node:assert/strict');
class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.textContent = ''; this.style = {}; this.attributes = {}; }
    setAttribute(name,value) { this.attributes[name]=value; }
    replaceChildren() { this.children = []; this.textContent = ''; this.style = {}; this.attributes = {}; }
    appendChild(child) { this.children.push(child); return child; }
    insertRow() { return this.appendChild(new Element('tr')); }
    insertCell() { return this.appendChild(new Element('td')); }
}
global.document = {createElement: tag => new Element(tag)};
const {renderNeckAnalysis: renderRaw} = require('../static/neck-analysis-v1.js');
const renderNeckAnalysis = (container,packet) => renderRaw(container,packet,{viewer:{role:1}});
const nodes = node => [node,...node.children.flatMap(nodes)];
const dimensions = ['posture_compensation','scm_lateral_recruitment','anterior_submental_activity','shoulder_recruitment','jaw_compensation'];
const container = new Element('div');
const groupedTable = () => container.children.filter(child=>child.tag==='table').at(-1);
const allText = node => node.textContent + node.children.map(allText).join(' ');
renderNeckAnalysis(container,{neckAnalysis:{status:'complete',dimensions:{scm_lateral_recruitment:{
    score:null,continuous_score:null,evidence:['Local activity not distinguishable from jaw/head mechanics']}},
    overall:{},quality:{},diagnostics:{dimensions:{scm_lateral_recruitment:{observation:{
        status:'ambiguous_local_change',evidence:['שינוי בקונטור <script>לא להריץ</script>'],frame_indices:[0,1]}}}}
}});
const ambiguousCell=groupedTable().children[2].children[1];
assert(allText(ambiguousCell).includes('מקור השינוי אינו ניתן להפרדה ואין ציון עצמאי'));
assert(allText(ambiguousCell).includes('<script>לא להריץ</script>'));
assert(ambiguousCell.children.every(child=>child.children.length===0));
assert(!allText(container).includes('פירוט הערכות ההתנהגות לפי אזור'));
renderNeckAnalysis(container,{neckAnalysis:{status:'complete',dimensions:{},overall:{},quality:{},diagnostics:{
    captured_frames:25,effective_fps:5,observed_span_seconds:4.8,target_fps:5,
    dimensions:{scm_lateral_recruitment:{tracking_quality:.5,adjusted_confidence:.1493,required_confidence:.15,
        rejection_reason:'Insufficient evidence confidence',model_evidence:[]}}
}}});
assert(nodes(container).some(child=>child.textContent.includes('14.93% (סף 15%)')));
renderNeckAnalysis(container,{neckAnalysis:{status:'complete',dimensions:{},overall:{},quality:{},diagnostics:{
    captured_frames:25,effective_fps:5,observed_span_seconds:4.8,target_fps:5,
    dimensions:{scm_lateral_recruitment:{tracking_quality:.5,adjusted_confidence:.15,required_confidence:.15,
        rejection_reason:'Insufficient evidence confidence',model_evidence:[]}}
}}});
assert(nodes(container).some(child=>child.textContent.includes('פחות מ־15% (סף 15%)')));
const packet = {neckAnalysis: {status:'complete', dimensions: Object.fromEntries(dimensions.map((name,index)=>[name,
    {score:index===3?null:1,confidence:.8,evidence:['ראיה <script>alert(1)</script>']} ])),
    overall:{continuous_score:1.25,category:1,confidence:.7},quality:{warnings:[]},summary:'סיכום'}};
renderNeckAnalysis(container,packet);
const table = groupedTable();
assert.equal(table.children.length,4);
assert(allText(table.children[1].children[1]).includes('הערכת האזור אינה זמינה'));
assert(!allText(container).includes('1.25 / 3.0'));
assert.equal(container.children.find(child=>child.tag==='table').children.length,4);
assert(allText(container).includes('1 / 3'));
assert(allText(container).includes('מה אפשר ללמוד'));
assert(allText(container).includes('הערכת האזור אינה זמינה'));
assert(allText(container).includes('לא זוהתה התנהגות חריגה'));
assert.equal(table.children[2].children[2].textContent.includes('לתעד את הממצא המשולב'),true);
assert(!allText(container).includes('70%'));
assert(allText(container).includes('80%'));
assert(allText(table.children[1].children[1]).includes('ראיה <script>alert(1)</script>'));
renderNeckAnalysis(container,{neckAnalysis:{status:'error',dimensions:{},overall:{},quality:{}}});
assert.equal(groupedTable().children.length,4);
assert(!allText(container).includes('0 / 3'));
assert(allText(container).includes('כשל שירות'));
assert(!allText(groupedTable()).includes('אין די ראיות'));
console.log('Three explained groups, no five-score option, null and safe text PASS');
renderNeckAnalysis(container,{neckAnalysis:{status:'error',dimensions:{},overall:{},quality:{},observable_change:{
    version:'neck-observable-change-v1',reference:{start:0,end:5,measurements:{jaw_opening:{value:.1,iqr:.02}}},
    windows:[{start:5,end:7,measurements:{jaw_opening:{value:.3,delta_from_reference:.2,valid_frames:10,coverage:1},
        lateral_motion:{value:null,delta_from_reference:null,reason:'reference_not_assessable'}}}]
}}});
const changeTables=[...container.children.find(child=>child.tag==='details').children.filter(child=>child.tag==='table'),groupedTable()];
assert.equal(changeTables.length,3);
assert.equal(changeTables[0].children[3].children[1].textContent,'0.1');
assert.equal(changeTables[1].children[3].children[3].textContent,'+0.2');
assert.equal(changeTables[1].children[6].children[3].textContent,'לא ניתן להשוות');
assert(nodes(container).some(child=>child.textContent.includes('ההשוואה המקומית זמינה')));
assert(!allText(container).includes('0 / 3'));
assert(allText(container).includes('כשל שירות'));
assert(!allText(groupedTable()).includes('אין די ראיות'));
console.log('Reference values, later deltas, missing comparisons and local results on provider failure PASS');
for (const [reason, expected] of [
    ['OpenAI HTTP 429 [credit_balance_exhausted]', 'יתרת הקרדיטים'],
    ['OpenAI HTTP 429 [rate_limit_exceeded]', 'קצב הבקשות'],
    ['OpenAI HTTP 429 [project_spend_limit_exceeded]', 'הפרויקט הגיע'],
]) {
    renderNeckAnalysis(container,{reasons:[reason],neckAnalysis:{status:'error',dimensions:{},overall:{},quality:{}}});
    assert(nodes(container).some(child=>child.textContent.includes(expected)));
}
const fs = require('node:fs');
const vm = require('node:vm');
const source=fs.readFileSync(require.resolve('../static/app.js'),'utf8');
const start=source.indexOf('function updateNeckEffortUI(data) {');
const end=source.indexOf('\n}',start)+2;
const ui={};
for (const name of ['neckEffortMode','neckEffortScore','neckEffortLabel','neckEffortConfidence','neckEffortReasons','neckEffortFeedback']) ui[name]=new Element('div');
vm.createContext(ui);
vm.runInContext(source.slice(start,end),ui);
ui.updateNeckEffortUI({model_version:'neck-analysis-v1',mode:'running',confidence:.97});
assert(ui.neckEffortConfidence.textContent.includes('97%'));
assert(ui.neckEffortConfidence.textContent.includes('לא התקדמות הניתוח'));
ui.updateNeckEffortUI({model_version:'neck-analysis-v1',mode:'error',neckAnalysis:{status:'error'}});
assert.equal(ui.neckEffortMode.textContent,'נכשל');
assert(!ui.neckEffortConfidence.textContent.includes('97%'));
console.log('OpenAI quota/rate errors and tracking-versus-progress display PASS');
renderNeckAnalysis(container,{neckAnalysis:{status:'insufficient_quality',dimensions:{},overall:{},quality:{},diagnostics:{captured_frames:12,observed_span_seconds:18.8,effective_fps:.595,target_fps:5,dimensions:{jaw_compensation:{tracking_quality:.75,adjusted_confidence:.07,required_confidence:.15,rejection_reason:'Insufficient evidence confidence',model_evidence:['עדות <img src=x>']}}}}});
assert(nodes(container).some(child=>child.textContent.includes('12 פריימים')));
assert(nodes(container).some(child=>child.textContent.includes('קצב הפריימים נמוך')));
assert(nodes(container).some(child=>child.textContent.includes('ביטחון משוקלל 7%')));
console.log('Sampling counts and per-dimension abstention diagnostics PASS');
const shouldersOnly={neckAnalysis:{status:'complete',dimensions:Object.fromEntries(dimensions.map(name=>[name,
    {score:name==='shoulder_recruitment'?1:null,confidence:name==='shoulder_recruitment'?.57:0,evidence:[]} ])),
    overall:{continuous_score:.9,category:1,confidence:.057},quality:{warnings:[]}}};
renderNeckAnalysis(container,shouldersOnly);
assert(!allText(container).includes('0.9 / 3.0'));
assert(allText(container).includes('תנוחה ותנועה — ראש, צוואר וכתפיים'));
assert(allText(container).includes('תנועות קשורות אינן מחוברות'));
for(const name of dimensions) shouldersOnly.neckAnalysis.dimensions[name].score=0;
renderNeckAnalysis(container,shouldersOnly);
assert(!allText(container).includes('מאמץ צוואר נצפה כולל:'));
assert(!allText(container).includes('קטגוריה כוללת:'));
console.log('Old and new reports avoid summing potentially coupled observations PASS');
const confounded={neckAnalysis:{dimensions:Object.fromEntries(dimensions.map(name=>[name,
    {score:name==='posture_compensation'?1:name==='jaw_compensation'?0:null,confidence:.52,
        evidence:name==='scm_lateral_recruitment'?['Local activity not distinguishable from jaw/head mechanics']:[]} ])),
    overall:{continuous_score:.4714,category:0,confidence:.1905},quality:{},diagnostics:{
        captured_frames:25,observed_span_seconds:4.808,effective_fps:4.992,target_fps:5,
        dimensions:{scm_lateral_recruitment:{rejection_reason:'Local activity not distinguishable from jaw/head mechanics',tracking_quality:.81}}}}};
renderNeckAnalysis(container,confounded);
assert(!allText(container).includes('0.4714 / 3.0'));
assert(allText(container).includes('מקור השינוי אינו ניתן להפרדה ואין ציון עצמאי'));
assert(!nodes(container).some(child=>child.textContent.includes('0.5 / 3.0')));
const confoundedTable=groupedTable();
assert(allText(confoundedTable.children[2].children[1]).includes('מקור השינוי אינו ניתן להפרדה'));
assert(nodes(container).some(child=>child.textContent.includes('איכות מעקב 81%')&&child.textContent.includes('לא ניתן להפריד')));
console.log('Category-boundary precision and Hebrew confound explanation in table/diagnostics PASS');
renderNeckAnalysis(container,{neckAnalysis:{dimensions:{},overall:{},quality:{},diagnostics:{
    captured_frames:25,observed_span_seconds:4.8,effective_fps:5,target_fps:5,
    audio_timing:'browser_estimated_offset',speech_status:'active',usable_speech_duration:3.8}}});
assert(nodes(container).some(child=>child.textContent.includes('זוהתה פעילות דיבור באודיו בכ־3.8')));
assert(nodes(container).some(child=>child.textContent.includes('התזמון משוער')));
(async()=>{
    for(const file of ['../static/app.js','../static/mobile.html']) {
        const code=fs.readFileSync(require.resolve(file),'utf8');
        const begin=code.indexOf('async function stopNeckEffortRecordingSession(');
        const finish=code.indexOf('\n}',begin)+2;
        let request,requestedTimeout;
        const nodes={};
        class Form {constructor(){this.values={};} append(key,value){this.values[key]=value;}}
        const transport={version:'neck-capture-transport-v1',captured_frames:25};
        const env={neckSampler:{stop:async()=>{},motionTrace:()=>({version:'head-motion-trace-v1',samples:[]}),transportDiagnostics:()=>transport},neckEffortSessionId:'matching-session',
            getAuthHeaders:()=>({Authorization:'Bearer test'}),FormData:Form,
            document:{getElementById:id=>(nodes[id]||=new Element('div')),querySelectorAll:()=>[]},
            AbortSignal:{timeout:ms=>{requestedTimeout=ms;return null;}},renderNeckAnalysis:()=>{},updateNeckEffortUI:()=>{},
            updateNeckUI:()=>{},activateTab:()=>{},fetch:async(url,options)=>{
                request={url,...options};return {ok:true,json:async()=>({neck_effort:{neckAnalysis:{status:'complete'}}})};}};
        vm.createContext(env);vm.runInContext(code.slice(begin,finish),env);
        const blob={recordedAudio:true};await env.stopNeckEffortRecordingSession(blob,.04);
        assert.equal(request.url,'/api/neck-effort/recording/stop');
        assert.equal(requestedTimeout,55000);
        assert.equal(request.body.values.audio,blob);
        assert.equal(request.body.values.session_id,'matching-session');
        assert.equal(JSON.parse(request.body.values.capture_transport).captured_frames,25);
        assert.equal(request.body.values.audio_offset_seconds,'0.04');
        assert.equal(JSON.parse(request.body.values.motion_trace).version,'head-motion-trace-v1');
        assert(!request.headers['Content-Type']);
        const reports=[];
        env.neckSampler.localMotionReport=async()=>({version:'head-movement-stability-v3',status:'measured',source:'browser_local',windows:[]});
        env.renderNeckAnalysis=(container,packet)=>reports.push(packet);
        env.fetch=async()=>{throw new TypeError('offline');};
        const offline=await env.stopNeckEffortRecordingSession(blob,.04);
        assert.equal(reports[0].neckAnalysis.status,'pending');
        assert.equal(offline.neckAnalysis.movement_stability.source,'browser_local');
        assert.equal(offline.neckAnalysis.status,'error');
        env.fetch=async()=>{const error=new Error('timeout');error.name='TimeoutError';throw error;};
        const timedOut=await env.stopNeckEffortRecordingSession(blob,.04);
        assert.equal(timedOut.neckAnalysis.movement_stability.version,'head-movement-stability-v3');
        assert.equal(timedOut.reasons[0],'analysis_timeout');
    }
    console.log('Desktop/mobile send same WAV, session and estimated offset; provisional timing display PASS');
    console.log('Desktop/mobile local motion survives offline and timeout responses and appears before server reply PASS');
})().catch(error=>{console.error(error);process.exitCode=1;});
// Historical V1 packets retain numeric aggregates in JSON, but no longer contribute
// to the visual progress graph or combined neck/voice severity.
const progressStart=source.indexOf('function _progressDimValue(');
const progressEnd=source.indexOf('\n}',progressStart)+2;
const progressEnv={};vm.createContext(progressEnv);
vm.runInContext(source.slice(progressStart,progressEnd),progressEnv);
assert.equal(progressEnv._progressDimValue({neck_effort:{model_version:'neck-analysis-v1',score:2}},'N'),null);
assert.equal(progressEnv._progressDimValue({neck_effort:{neckAnalysis:{},score:2}},'N'),null);
assert.equal(progressEnv._progressDimValue({neck_effort:{score:2}},'N'),2);
assert.equal(progressEnv._progressDimValue({scores:{G:1}},'G'),1);
console.log('Historical V1 aggregate excluded from progress graph; voice values remain available PASS');

// Group grade uses the largest accepted regional grade, not a sum/average.
const groupedPacket={neckAnalysis:{status:'complete',dimensions:{
 posture_compensation:{score:2,confidence:.4,evidence:[]},
 shoulder_recruitment:{score:1,confidence:.3,evidence:[]},
 anterior_submental_activity:{score:0,confidence:.35,evidence:[]},
 scm_lateral_recruitment:{score:null,evidence:['Local activity not distinguishable from jaw/head mechanics']},
 jaw_compensation:{score:1,confidence:.5,evidence:[]}},quality:{}}};
renderNeckAnalysis(container,groupedPacket);
assert(allText(groupedTable().children[1].children[3]).startsWith('2 / 3'));
assert(!allText(groupedTable().children[1].children[3]).includes('דירוג חלקי'));
assert(allText(groupedTable().children[2].children[3]).includes('דירוג חלקי'));
assert(allText(groupedTable().children[3].children[3]).startsWith('1 / 3'));
groupedPacket.neckAnalysis.dimensions.anterior_submental_activity.score=null;
renderNeckAnalysis(container,groupedPacket);
assert(allText(groupedTable().children[2].children[3]).startsWith('לא ניתן להעריך'));
groupedPacket.neckAnalysis.status='error';
renderNeckAnalysis(container,groupedPacket);
assert(groupedTable().children.slice(1).every(row=>row.children[3].textContent.includes('כשל שירות')));
assert(!allText(groupedTable()).includes('אין די ראיות'));
assert(!allText(groupedTable()).includes(' / 3'));
console.log('Max group grade, partial coverage, unavailable group, and service failure distinguished PASS');

for (const viewer of [null,{role:3,is_admin:false},{role:2},{role:1},{is_admin:true}]) {
 groupedPacket.neckAnalysis.status='complete';
 groupedPacket.neckAnalysis.observable_change={reference:{start:0,end:5,measurements:{}},windows:[]};
 renderRaw(container,groupedPacket,{viewer});
 const admin=viewer?.is_admin===true||viewer?.role===1||viewer?.role===2;
 assert.equal(container.children.some(child=>child.tag==='details'),admin);
 assert.equal(allText(container).includes('ערך ייחוס חציוני'),admin);
 assert.equal(allText(container).includes('הדירוג הוא הגבוה'),admin);
 assert(allText(groupedTable()).includes('2 / 3'));
 if (!admin) assert(!allText(container).includes('פירוש התיאורים:'));
}
renderRaw(container,groupedPacket);
assert(!container.children.some(child=>child.tag==='details'));
console.log('Regular/default viewers get grades only; admin roles get collapsed diagnostics PASS');

groupedPacket.neckAnalysis.movement_stability={version:'head-movement-stability-v1',status:'measured',sample_count:150,effective_fps:30,
 windows:[{start:0,end:2,status:'measured',amplitude_percent_image_width:.25,dominant_frequency_hz:4.5,background_amplitude_percent_image_width:.01,band_hz:[2,8]},
 {start:1,end:3,status:'insufficient_data',amplitude_percent_image_width:null,dominant_frequency_hz:null}]};
groupedPacket.neckAnalysis.status='error';
renderRaw(container,groupedPacket,{viewer:{role:3}});
assert(allText(container).includes('יציבות ותנודתיות בתנועת הראש'));
assert(allText(container).includes('0.250'));
assert(allText(container).includes('4.50'));
assert(allText(container).includes('לא ניתן למדוד'));
assert(!allText(container).includes('רקע 0–2:'));
assert(!container.children.some(child=>child.tag==='details'));
console.log('Movement measurements survive provider failure, missing windows stay null, admin diagnostics hidden PASS');
groupedPacket.neckAnalysis.movement_stability.version='head-movement-stability-v2';
groupedPacket.neckAnalysis.movement_stability.windows[0].temporal_coverage=.9;
groupedPacket.neckAnalysis.movement_stability.windows[1].reason='sampling_gap_too_large';
renderRaw(container,groupedPacket,{viewer:{role:3}});
assert(allText(container).includes('כיסוי זמן 90%'));
assert(allText(container).includes('פער בין תמונות'));
assert(allText(container).includes('המדידה חלקית: 1 מתוך 2'));
console.log('V2 irregular-time estimate, coverage and explicit missing-window reason PASS');
const movement=groupedPacket.neckAnalysis.movement_stability;
movement.status='insufficient_data';movement.reason='no_usable_motion_windows';
movement.windows=[{usable_samples:12,tracked_span_seconds:.4,reason:'too_few_usable_samples'},
 {usable_samples:0,tracked_span_seconds:0,reason:'too_few_usable_samples'}];
renderRaw(container,groupedPacket,{viewer:{role:3}});
assert(allText(container).includes('0.40 שניות'));
assert(allText(container).includes('אינו מבדיל בין אובדן מעקב ראש'));
assert(!allText(container).includes('המתינו שהפנים יזוהו'));
movement.capture_reason='face_reference_unavailable';movement.windows=[];
renderRaw(container,groupedPacket,{viewer:{role:3}});
assert(allText(container).includes('לא היה זיהוי פנים עדכני'));
delete movement.capture_reason;
movement.windows=[{usable_samples:0}];movement.tracking_diagnostics={head_missing_samples:0,background_missing_samples:100};
renderRaw(container,groupedPacket,{viewer:{role:3}});
assert(allText(container).includes('אבד מעקב של הרקע'));
assert(allText(container).includes('משקוף או מדף'));
assert(!allText(container).includes('אבד מעקב של הראש.'));
movement.tracking_diagnostics={head_missing_samples:1,background_missing_samples:33};
renderRaw(container,groupedPacket,{viewer:{role:3}});
assert(allText(container).includes('הגורם העיקרי לחסר'));
assert(!allText(container).includes('הן של הראש והן של הרקע'));
assert(!allText(container).includes('תאורה אחידה על הפנים'));
console.log('Tracking loss, face initialization and background failure yield distinct actionable messages PASS');
groupedPacket.neckAnalysis.status='complete';
movement.version='head-movement-stability-v3';movement.status='measured';movement.source='browser_local';
movement.windows=[{start:0,end:2,status:'measured',periodicity_status:'not_established',amplitude_percent_image_width:null,dominant_frequency_hz:null,temporal_coverage:1}];
renderRaw(container,groupedPacket,{viewer:{role:3}});
assert(allText(container).includes('לא נמצאה מחזוריות עקבית'));
assert(!allText(container).includes('ביטחון בראיות באזור:'));
assert(allText(container).includes('הדיוק שלהם טרם אומת קלינית'));
movement.windows[0].motion_rms_percent_image_width=.125;
renderRaw(container,groupedPacket,{viewer:{role:3}});
assert(allText(container).includes('פיזור התנועה (% מרוחב התמונה)'));
assert(allText(container).includes('0.125'));
assert(allText(container).includes('אין תדירות בולטת למדידה'));
assert(!allText(container).includes('לא ניתן למדוד'));
groupedPacket.neckAnalysis.diagnostics ||= {};
groupedPacket.neckAnalysis.diagnostics.capture_transport={captured_frames:30,acknowledged_frames:25,unconfirmed_frames:5,encoding_failed_frames:0,retry_count:2};
renderRaw(container,groupedPacket,{viewer:{role:3}});
assert(allText(container).includes('הניתוח חלקי: חלק מהתמונות'));
assert(!allText(container).includes('ניסיונות חוזרים'));
groupedPacket.neckAnalysis.diagnostics.capture_transport.server_received_frames=30;
groupedPacket.neckAnalysis.diagnostics.capture_transport.missing_frames=0;
renderRaw(container,groupedPacket,{viewer:{role:3}});
assert(!allText(container).includes('הניתוח חלקי: חלק מהתמונות'));
