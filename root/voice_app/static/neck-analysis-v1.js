/* Shared renderer used only by the existing desktop/mobile neck sections. */
(function (root) {
    'use strict';
    const names = {
        posture_compensation: 'תנוחת ראש וצוואר',
        scm_lateral_recruitment: 'מראה צדי הצוואר (אזור SCM)',
        anterior_submental_activity: 'מראה קדמת הצוואר ומתחת לסנטר',
        shoulder_recruitment: 'תנועת כתפיים',
        jaw_compensation: 'תנועת לסת'
    };
    const reasonsHe = {
        'Analysis failed': 'הניתוח לא הושלם',
        'Region not sufficiently observable': 'אין די ראיות נצפות; הדוח הישן אינו מבחין בין סיבות הפסילה',
        'Tracking quality below threshold': 'איכות זיהוי האזור נמוכה מהסף',
        'Model found insufficient observable evidence': 'המודל לא מצא די ראיות נצפות באזור',
        'Insufficient evidence confidence': 'הביטחון לאחר שקלול ראיות, איכות צילום ותזמון דיבור נמוך מהסף',
        'Insufficient evidence relating observed behavior to speech': 'לא נמצא קשר מספיק בין התנועה לדיבור',
        'Local activity not distinguishable from jaw/head mechanics': 'לא ניתן להפריד פעילות מקומית בצוואר מתנועות הראש והלסת',
        'Activity explained by a confound': 'התנועה מוסברת בגורם אחר, כגון תנועת מצלמה, בליעה או תנועות דיבור רגילות',
        'Insufficient temporal sampling confidence': 'איכות וקצב הפריימים מגבילים את הביטחון מתחת לסף',
        'Insufficient visibility': 'האזור אינו נראה מספיק בבירור',
        'No usable speech/phonation evidence in aligned audio': 'לא זוהתה פעילות דיבור שימושית באודיו החופף לצילום'
    };
    function renderNeckAnalysis(container, packet, options = {}) {
        if (!container) return;
        container.replaceChildren();
        const viewer = options.viewer || {};
        const showDebug = viewer.is_admin === true || viewer.role === 1 || viewer.role === 2;
        let target = container;
        const analysis = packet && (packet.neckAnalysis || (packet.version === 'neck-analysis-v1' ? packet : null));
        function paragraph(text) {
            const p = document.createElement('p'); p.textContent = text; target.appendChild(p);
        }
        if (!analysis) {
            paragraph('ניתוח הצוואר יופיע בסיום ההקלטה.');
            return;
        }
        const debug = showDebug ? document.createElement('details') : null;
        if (debug) {
            debug.className = 'neck-admin-debug';
            const summary = document.createElement('summary');
            summary.textContent = 'נתוני בדיקה למנהלים';
            debug.appendChild(summary);
            target = debug;
        }
        if (showDebug) paragraph('ניתוח צוואר V1 · התנהגות חיצונית נצפית; אינו מדידה ישירה של כוח או מתח שרירי ואינו אבחנה.');
        const change = analysis.observable_change;
        if (showDebug && change?.reference && Array.isArray(change.windows)) {
            paragraph('שינוי נצפה לאורך זמן ביחס לתחילת ההקלטה');
            paragraph(`קטע הייחוס: ${change.reference.start}–${change.reference.end} שניות. אין הנחה שבקטע הזה היית רגוע. השינויים אינם ציוני מאמץ.`);
            const signalNames = {
                head_pitch_deg:'זווית ראש משוערת (°)', head_roll_deg:'הטיית ראש בתמונה (°)',
                jaw_opening:'פתיחת לסת (יחס למרחק העיניים)',
                shoulder_height:'מרחק כתפיים מהסנטר (יחס למרחק העיניים)',
                anterior_motion:'תנועה בקדמת הצוואר (מרחקי עיניים/שנייה)',
                lateral_motion:'תנועה בצדי הצוואר (מרחקי עיניים/שנייה)'
            };
            const number = value => Number.isFinite(value) ? String(Math.round(value*10000)/10000) : 'לא ניתן למדוד';
            const referenceTable = document.createElement('table');
            referenceTable.className='neck-change-table';
            const referenceHead=referenceTable.insertRow();
            ['מדידה מקומית','ערך ייחוס חציוני','פיזור בייחוס (IQR)'].forEach(text=>{
                const th=document.createElement('th');th.textContent=text;referenceHead.appendChild(th);
            });
            Object.entries(signalNames).forEach(([key,title])=>{
                const measure=change.reference.measurements?.[key] || {};
                const row=referenceTable.insertRow();
                [title,number(measure.value),number(measure.iqr)].forEach(text=>{row.insertCell().textContent=text;});
            });
            debug.appendChild(referenceTable);
            const changesTable=document.createElement('table');
            changesTable.className='neck-change-table';
            const changesHead=changesTable.insertRow();
            ['זמן (שניות)','מדידה','ערך חציוני','הפרש מהייחוס','איכות הנתונים'].forEach(text=>{
                const th=document.createElement('th');th.textContent=text;changesHead.appendChild(th);
            });
            change.windows.forEach(window=>{
                Object.entries(signalNames).forEach(([key,title])=>{
                    const measure=window.measurements?.[key] || {};
                    const row=changesTable.insertRow();
                    const delta=measure.delta_from_reference;
                    const status=measure.reason === 'reference_not_assessable' ? 'קטע הייחוס אינו ניתן למדידה' :
                        Number.isFinite(delta) ? `${measure.valid_frames} תמונות שימושיות; כיסוי ${Math.round((measure.coverage || 0)*100)}%` : 'אין די תמונות שימושיות';
                    [`${window.start}–${window.end}`,title,number(measure.value),
                        Number.isFinite(delta) ? (delta>0?'+':'')+number(delta) : 'לא ניתן להשוות',status]
                        .forEach(text=>{row.insertCell().textContent=text;});
                });
            });
            debug.appendChild(changesTable);
            paragraph('עלייה או ירידה במדידה אינה מוכיחה עלייה או ירידה במאמץ. תנועות ראש ולסת, תאורה ומצלמה עשויות להשפיע. תנועת הרקמה בתמונה אינה מדידה של כוח או מתח שרירי; קווי מתאר של שרירים אינם מבודדים כאן.');
            if (analysis.status === 'error') paragraph('ההשוואה המקומית זמינה גם כששירות ציוני המאמץ לא השלים את הניתוח.');
            paragraph('ממצאים נצפים — סיכום ההקלטה');
        }
        target = container;
        paragraph('התנהגות חיצונית נצפית · גרסת מחקר');
        if (analysis.status === 'error') paragraph('שירות הניתוח לא השלים את ההערכה; הציונים אינם זמינים.');
        const transport=analysis.diagnostics?.capture_transport;
        if(transport && ((transport.missing_frames ?? transport.unconfirmed_frames)>0 || transport.encoding_failed_frames>0))
            paragraph('הניתוח חלקי: חלק מהתמונות לא התקבלו או לא הוכנו לשליחה. הממצאים מתייחסים לתמונות הזמינות בלבד.');
        target = debug || container;
        if(showDebug && transport)paragraph(`העברת תמונות: הוכנו ${transport.captured_frames}; אושרו בדפדפן ${transport.acknowledged_frames}; התקבלו בשרת ${transport.server_received_frames ?? 'לא ידוע'}; ניסיונות חוזרים ${transport.retry_count}; לא אושרו ${transport.unconfirmed_frames}.`);
        if (showDebug) paragraph('דירוג לכל קבוצה: 0 — מזערי / לא נצפה, 1 — קל, 2 — בינוני, 3 — גבוה. הדירוג הוא הגבוה מבין הערכות האזורים שניתן להעריך בקבוצה; תנועות קשורות אינן מחוברות. זהו כלל מחקר לתיאור התנהגות חיצונית, ולא מדידה מאומתת של מתח שרירי.');
        target = container;
        const movement = analysis.movement_stability;
        if (['head-movement-stability-v1','head-movement-stability-v2','head-movement-stability-v3'].includes(movement?.version)) {
            paragraph('יציבות ותנודתיות בתנועת הראש');
            if (movement.status === 'measured') {
                const hasMotionRms=(movement.windows || []).some(w=>w.status==='measured'&&Number.isFinite(w.motion_rms_percent_image_width));
                paragraph('גודל ותדירות של תנועת הראש ביחס לרקע, להשוואת קטעים באותה הקלטה. זהו מדד תנועה בתמונה.');
                if(hasMotionRms)paragraph('פיזור התנועה מציג גם תנועה שאינה מחזורית, לאחר הסרת מגמה איטית. אפשר להשוות חלונות באותו טווח תדרים; זהו גודל תנועה ולא ציון מאמץ.');
                if(movement.version==='head-movement-stability-v3')
                    paragraph('תדירות מוצגת רק כשנמצאה מחזוריות עקבית. הקלטה של 10–20 שניות מאפשרת גם בדיקת תנודות איטיות; הטווח העליון מוגבל לפי התמונות שהתקבלו. היעדר תדירות אינו ציון אפס במאמץ.');
                if(movement.source==='browser_local')paragraph('מדידת התנועה חושבה בדפדפן; היא זמינה גם כשהשרת אינו משיב.');
                const measuredCount=(movement.windows || []).filter(w=>w.status==='measured').length;
                if (measuredCount < (movement.windows || []).length)
                    paragraph(`המדידה חלקית: ${measuredCount} מתוך ${movement.windows.length} חלונות ניתנים למדידה. בחלונות החסרים היו מגבלות מעקב או רציפות צילום.`);
                const motionTable = document.createElement('table');
                motionTable.className = 'neck-change-table neck-motion-table';
                const header = motionTable.insertRow();
                const columns=['זמן (שניות)',hasMotionRms?'פיזור התנועה (% מרוחב התמונה)':'גודל תנודה (% מרוחב התמונה)','תדירות בולטת בטווח שנבדק (תנודות/שנייה)','איכות המדידה'];
                if(movement.version==='head-movement-stability-v3')columns.splice(1,0,'טווח שנבדק (Hz)');
                columns.forEach(text=>{
                    const th=document.createElement('th');th.textContent=text;header.appendChild(th);
                });
                (movement.windows || []).forEach(window=>{
                    const row=motionTable.insertRow();
                    const measured=window.status === 'measured';
                    const reasons={sampling_gap_too_large:'פער בין תמונות',tracking_coverage_low:'מעקב ראש או רקע לא רציף',
                        too_few_usable_samples:'מעט מדי דגימות שימושיות',tracked_span_too_short:'רצף המעקב קצר מדי',
                        temporal_coverage_low:'חסרים קטעים בחלון',motion_sampling_too_slow:'קצב דגימה נמוך בחלון',
                        tracking_reinitialized:'המעקב התחבר מחדש בחלון הזה'};
                    const cells=[`${window.start}–${window.end}`,
                     measured && hasMotionRms && Number.isFinite(window.motion_rms_percent_image_width) ? window.motion_rms_percent_image_width.toFixed(3) : measured && Number.isFinite(window.amplitude_percent_image_width) ? window.amplitude_percent_image_width.toFixed(3) : measured && window.periodicity_status==='not_established' ? 'לא נמצאה מחזוריות עקבית' : measured && window.periodicity_status==='camera_confounded' ? 'תנועת הרקע מגבילה את ההפרדה' : 'לא ניתן למדוד',
                     measured && Number.isFinite(window.dominant_frequency_hz) ? movement.version==='head-movement-stability-v3' ? `כ־${window.dominant_frequency_hz.toFixed(1)}` : window.dominant_frequency_hz.toFixed(2) : measured ? 'אין תדירות בולטת למדידה' : 'לא ניתן למדוד',
                     measured ? ['head-movement-stability-v2','head-movement-stability-v3'].includes(movement.version) ? `כיסוי זמן ${Math.round(window.temporal_coverage*100)}%` : 'רצף שימושי' : reasons[window.reason] || 'מגבלת מעקב או תזמון']
                    if(movement.version==='head-movement-stability-v3')cells.splice(1,0,window.band_hz?.join('–') || 'לא זמין');
                    cells.forEach(text=>{row.insertCell().textContent=text;});
                });
                container.appendChild(motionTable);
            } else {
                paragraph('לא התקבלה מדידת תנודתיות בהקלטה הזאת.');
                const captureMessages={
                    face_reference_unavailable:'המעקב לא התחיל: לא היה זיהוי פנים עדכני בתחילת ההקלטה. המתינו שהפנים יזוהו בתצוגה המקדימה ואז התחילו הקלטה.',
                    video_frame_callback_unavailable:'המעקב לא התחיל: הדפדפן אינו תומך בשיטת קבלת תמונות הווידאו הדרושה. נסו דפדפן Chrome או Edge מעודכן.',
                    image_tracking_failed:'קריאת תמונות הווידאו נכשלה במהלך ההקלטה. הפעילו מחדש את המצלמה באפליקציה; אם הכשל חוזר, נדרשת בדיקת הדפדפן.'};
                const windows=movement.windows || [],diagnostics=movement.tracking_diagnostics || {};
                if (captureMessages[movement.capture_reason]) paragraph(captureMessages[movement.capture_reason]);
                else if (movement.reason==='motion_sampling_too_slow')
                    paragraph(`קצב הדגימה היה ${movement.effective_fps ?? 'לא ידוע'} תמונות בשנייה; החישוב דורש לפחות 12. סגרו תוכנות שמשתמשות במצלמה והקלטו שוב.`);
                else if (movement.reason==='motion_sampling_invalid')
                    paragraph('זמני דגימות המצלמה אינם תקינים לחישוב. זו בעיה בקבלת נתוני הווידאו; נדרשת בדיקת תזמון הדפדפן.');
                else if (movement.reason==='too_few_motion_samples')
                    paragraph(`התקבלו רק ${movement.sample_count ?? 0} דגימות תנועה. נדרשת הקלטה של לפחות 3 שניות עם מצלמה פעילה. בדוח הזה אין מידע מספיק לקבוע למה התקבלו מעט דגימות.`);
                else if (windows.length && windows.every(w=>(w.usable_samples ?? 0)===0))
                    paragraph('התקבלו תמונות, אבל לא נמצא בהן מעקב שימושי של הראש ביחס לרקע.');
                else if (windows.length && windows.every(w=>w.reason==='too_few_usable_samples'||w.reason==='tracking_coverage_low')) {
                    const longest=Math.max(...windows.map(w=>w.tracked_span_seconds || 0));
                    paragraph(`התמונות התקבלו, אבל המעקב השימושי בתוך חלון כיסה לכל היותר ${longest.toFixed(2)} שניות. החישוב דורש לפחות 1.7 שניות בחלון של שתי שניות.`);
                } else if (windows.length)
                    paragraph('החלונות נפסלו בגלל רציפות צילום או כיסוי מעקב. פירוט הסיבות מופיע בהמשך.');
                else paragraph('הנתונים אינם מכסים חלון מדידה של שתי שניות. הקלטו לפחות 3 שניות.');
                if (diagnostics.head_missing_samples>0 || diagnostics.background_missing_samples>0) {
                    const head=diagnostics.head_missing_samples>0,background=diagnostics.background_missing_samples>0;
                    const mostlyBackground=background&&diagnostics.background_missing_samples>Math.max(3,diagnostics.head_missing_samples*3);
                    paragraph(mostlyBackground?'אבד מעקב של הרקע שמאפשר להפריד תנועת מצלמה מתנועת ראש; זהו הגורם העיקרי לחסר במדידה.':head&&background?'אבד מעקב הן של הראש והן של הרקע.':head?'אבד מעקב של הראש.':'אבד מעקב של הרקע שמאפשר להפריד תנועת מצלמה מתנועת ראש.');
                    paragraph(head&&!mostlyBackground?'נסו תאורה אחידה על הפנים והשאירו אותן בתוך התמונה. אין צורך להפסיק את הרעידות כדי לבצע את הבדיקה.':'כללו בתמונה רקע נייח עם פרטים, למשל משקוף או מדף; קיר חלק עלול לא לספק נקודות מעקב.');
                } else if (windows.some(w=>(w.usable_samples ?? 0)<20))
                    paragraph('הדוח הזה אינו מבדיל בין אובדן מעקב ראש לאובדן מעקב רקע. ההודעה אינה מעידה שלא היו רעידות; היא מציינת שהמערכת לא הצליחה למדוד אותן.');
                const failures=[...new Set(windows.map(w=>w.reason).filter(Boolean))];
                const failureMessages={sampling_gap_too_large:'נמצאו פערי צילום ארוכים מדי בין דגימות שימושיות.',temporal_coverage_low:'חלקים גדולים מחלונות הזמן אינם מכוסים בתמונות שימושיות.',
                    motion_sampling_too_slow:'צפיפות התמונות השימושיות נמוכה מדי בחלונות המדידה.',tracked_span_too_short:'המעקב בכל חלון היה קצר מדי.',
                    tracking_reinitialized:'המעקב התחבר מחדש. חלון הכולל חיבור בין רצפים אינו משמש לחישוב תנודות.'};
                failures.forEach(reason=>{if(failureMessages[reason])paragraph(failureMessages[reason]);});
            }
            if (showDebug) {
                target=debug;
                paragraph(`מעקב תנועה צפוף: ${movement.sample_count} דגימות; קצב חציוני ${movement.effective_fps ?? 'לא זמין'} דגימות/שנייה. טווח התדרים הוא כלל מחקר, ללא סף אבחנתי או ציון 0–3.`);
                if (movement.tracking_diagnostics?.reinitializations!==undefined)
                    paragraph(`חיבורים מחדש של מעקב התנועה: ${movement.tracking_diagnostics.reinitializations}.`);
                if (movement.tracking_diagnostics?.head_missing_samples!==undefined)
                    paragraph(`דגימות ללא מעקב ראש: ${movement.tracking_diagnostics.head_missing_samples}; ללא מעקב רקע: ${movement.tracking_diagnostics.background_missing_samples ?? 'לא ידוע'}; סך דגימות: ${movement.sample_count}.`);
                (movement.windows || []).filter(w=>w.status==='measured').forEach(w=>paragraph(`רקע ${w.start}–${w.end}: גודל תנודה ${w.background_amplitude_percent_image_width}% מרוחב התמונה; טווח נבדק ${w.band_hz?.join('–')} Hz.`));
                (movement.windows || []).filter(w=>w.received_samples!==undefined).forEach(w=>paragraph(`חלון ${w.start}–${w.end}: ${w.usable_samples}/${w.received_samples} דגימות שימושיות; כיסוי מעקב ${Math.round(w.tracking_coverage*100)}%; פער מרבי ${w.max_gap_seconds ?? 'לא זמין'} שניות; ${w.reason || 'נמדד'}.`));
                target=container;
            }
        }
        if(analysis.status==='pending'){
            paragraph('מדידת התנועה מוכנה. ניתוח התצפיות עדיין מתבצע; ציוני הקבוצות יוצגו כאשר יתקבלו.');
            if(debug)container.appendChild(debug);
            return;
        }
        paragraph('דירוגי הקבוצות הם דירוגי תצפית ניסיוניים; הדיוק שלהם טרם אומת קלינית.');
        const groupTable = document.createElement('table');
        groupTable.className = 'neck-change-table';
        const groupHead = groupTable.insertRow();
        ['קבוצת תצפיות', 'ממצאים והקשר', 'מה אפשר ללמוד', 'דירוג התנהגות נצפית'].forEach(text => {
            const th = document.createElement('th'); th.textContent = text; groupHead.appendChild(th);
        });
        const groups = [
            ['תנוחה ותנועה — ראש, צוואר וכתפיים', ['posture_compensation', 'shoulder_recruitment'], 'אפשר לזהות שינויים במנח הראש והצוואר ובמיקום הכתפיים, ולהשוות את תחילת ההקלטה להמשך. התיאור מאפשר לראות אילו אזורים נעים יחד; טבלת ההשוואה מציגה מתי ובאיזה כיוון השתנו המדידות ביחס לקטע הייחוס.'],
            ['מראה הצוואר — צדדים וקדמת הצוואר', ['scm_lateral_recruitment', 'anterior_submental_activity'], 'אפשר לתאר שינויים חיצוניים בקווי המתאר ובתנועה בצדי הצוואר ובקדמתו, ולהשוות אותם למראה בתחילת ההקלטה. כשהשינוי מופיע יחד עם תנועת ראש או לסת, אפשר לתעד את הממצא המשולב ולבחון באילו קטעים הוא מופיע.'],
            ['תנועת לסת', ['jaw_compensation'], 'אפשר לתאר פתיחה ותנועות לסת נראות ולבדוק איך הן משתנות במהלך הדיבור. ההשוואה המקומית מציגה את פתיחת הלסת ביחס לקטע הייחוס, והתיאור מספק הקשר לפירוש שינויים סמוכים מתחת לסנטר.']
        ];
        groups.forEach(([title, keys, explanation]) => {
            const row = groupTable.insertRow();
            row.insertCell().textContent = title;
            const cell = row.insertCell();
            keys.forEach(key => {
                const item = analysis.dimensions?.[key] || {};
                const diagnostic = analysis.diagnostics?.dimensions?.[key] || {};
                const observation = diagnostic.observation;
                const evidence = (item.evidence || []).map(text => reasonsHe[text] || text).join(' ');
                const coupled = !Number.isInteger(item.score) && (
                    observation?.status === 'ambiguous_local_change' ||
                    diagnostic.rejection_reason === 'Local activity not distinguishable from jaw/head mechanics' ||
                    (item.evidence || []).includes('Local activity not distinguishable from jaw/head mechanics'));
                const serviceFailed = analysis.status === 'error';
                const p = document.createElement('p');
                p.textContent = names[key] + ': ' + (serviceFailed
                    ? 'שירות הניתוח לא השלים את ההערכה; אין מסקנה על איכות הראיות באזור.'
                    : coupled
                    ? 'שינוי במראה הצוואר יחד עם תנועות ראש או לסת; מקור השינוי אינו ניתן להפרדה ואין ציון עצמאי. ' +
                      (observation?.evidence || diagnostic.model_evidence || []).join(' ')
                    : Number.isInteger(item.score)
                    ? (item.score === 0 ? key==='posture_compensation'&&movement?.status==='measured' ? 'בתמונות שנבחרו לא דורג פיצוי בתנוחה. ' : 'לא זוהתה התנהגות חריגה באזור הנצפה. ' : 'תואר שינוי נראה באזור. ') + evidence
                    : 'הערכת האזור אינה זמינה: ' + (evidence || 'לא התקבלה הערכה שימושית.'));
                cell.appendChild(p);
                if (showDebug && !serviceFailed && Number.isInteger(item.score)) {
                    const confidence = document.createElement('p');
                    confidence.textContent = `ביטחון בראיות באזור: ${Math.round((item.confidence || 0)*100)}% · אינו הסתברות לדיוק רפואי.`;
                    cell.appendChild(confidence);
                }
                if (diagnostic.model_confounds?.length) {
                    const confounds = document.createElement('p');
                    confounds.textContent = 'גורמים המשפיעים על הפרשנות: ' + diagnostic.model_confounds.join(' ');
                    cell.appendChild(confounds);
                }
            });
            if(keys.includes('posture_compensation')&&movement?.status==='measured'&&analysis.dimensions?.posture_compensation?.score===0){
                const p=document.createElement('p');p.textContent='מדידת תנועת הראש זמינה בטבלה, גם כשהדירוג בתמונות הוא 0. הדירוג ומדידת התנועה מתארים היבטים שונים של ההקלטה.';cell.appendChild(p);
            }
            if (keys.includes('scm_lateral_recruitment')) {
                const p = document.createElement('p');
                p.textContent = 'התיאור מציין היכן נראה השינוי ובאילו תמונות הוא הופיע, כשמידע זה זמין. אפשר להיעזר בו להשוואת קטעים ולסקירה של הסרטון.';
                cell.appendChild(p);
            }
            row.insertCell().textContent = explanation;
            const rated = keys.filter(key => Number.isInteger(analysis.dimensions?.[key]?.score));
            const scoreCell = row.insertCell();
            if (analysis.status === 'error') {
                scoreCell.textContent = 'הניתוח לא הושלם — כשל שירות';
            } else if (!rated.length) {
                scoreCell.textContent = 'לא ניתן להעריך — הסיבה מפורטת בממצאים';
            } else {
                const winner = rated.reduce((best,key) => analysis.dimensions[key].score > analysis.dimensions[best].score ? key : best);
                const item = analysis.dimensions[winner];
                const labels = ['מזערי / לא נצפה', 'קל', 'בינוני', 'גבוה'];
                const partial = rated.length < keys.length;
                const scoreRow = document.createElement('div');
                scoreRow.className = 'score-row neck-group-score';
                const value = document.createElement('span');
                value.className = `score-value grade-${item.score}`;
                value.textContent = `${item.score} / 3`;
                scoreRow.appendChild(value);
                const label = document.createElement('span');
                label.className = 'score-label';
                label.textContent = labels[item.score];
                scoreRow.appendChild(label);
                const background = document.createElement('div');
                background.className = 'score-bar-bg';
                background.setAttribute('role','meter');
                background.setAttribute('aria-label',`דירוג ${title}`);
                background.setAttribute('aria-valuemin','0');
                background.setAttribute('aria-valuemax','3');
                background.setAttribute('aria-valuenow',String(item.score));
                const fill = document.createElement('div');
                fill.className = 'score-bar';
                fill.style.width = `${item.score / 3 * 100}%`;
                fill.style.background = `var(--neck-grade-${item.score})`;
                value.style.color = `var(--neck-grade-${item.score})`;
                background.appendChild(fill);
                scoreRow.appendChild(background);
                scoreCell.appendChild(scoreRow);
                if (partial) {
                    const badge = document.createElement('p');
                    badge.className = 'neck-group-score-note';
                    badge.textContent = 'דירוג חלקי';
                    scoreCell.appendChild(badge);
                }
                const note = document.createElement('p');
                note.className = 'neck-group-score-note';
                note.textContent = (showDebug ? `הדירוג נלקח מ: ${names[winner]}. ` : '') +
                    (partial ? `מבוסס רק על ${rated.map(key=>names[key]).join(', ')}; האזורים החסרים אינם נכללים בדירוג. ` : '') +
                    (showDebug ? `מדד ראיות לא מכויל: ${Math.round((item.confidence || 0)*100)}%.` : '');
                scoreCell.appendChild(note);
            }
        });
        container.appendChild(groupTable);
        target = debug || container;
        if (showDebug) paragraph('פירוש התיאורים: ״נראה שינוי״ מציין שינוי חיצוני שתואר, ולא מאמץ שרירי מוכח. ״לא זוהתה התנהגות חריגה״ מתייחס למה שנראה בלבד ואינו מוכיח שאין מאמץ. ״מקור השינוי אינו ניתן להפרדה״ מציין שינוי שאפשר לראות אך לא לייחס בנפרד לאזור או לשריר. ״אין די ראיות״ מציין מגבלת הערכה ולא ציון אפס.');
        if (showDebug) paragraph('אחוזי הביטחון מתייחסים לראיות בתיאור האזור ומשקללים איכות צילום ותזמון. הם כללי הנדסה שלא כוילו לדיוק רפואי; אין אחוז ביטחון מאוחד לקבוצה.');
        if (showDebug && (analysis.quality || {}).warnings?.some(warning => warning !== 'analysis_failed')) paragraph('קיימות מגבלות איכות צילום או תזמון; הן נלקחו בחשבון בביטחון.');
        const diag = analysis.diagnostics || {};
        const warnings = (analysis.quality || {}).warnings || [];
        if (showDebug && warnings.includes('head_pose_unreliable')) paragraph('בחלק מהתמונות אומדן זווית הראש לא היה יציב ונפסל; תמונות אלו אינן מספקות מדידת זווית.');
        if (showDebug && warnings.includes('shoulder_motion_not_torso_referenced')) paragraph('האגן אינו זמין למעקב: תנועת הכתפיים נמדדה ביחס לראש בלבד, ואינה מפרידה באופן מלא בין תנועת גוף לתנועת כתפיים.');
        if (showDebug && Number.isInteger(diag.captured_frames) && Number.isFinite(diag.effective_fps)) {
            paragraph(`התקבלו ${diag.captured_frames} פריימים לאורך ${diag.observed_span_seconds} שניות; קצב בפועל ${diag.effective_fps} פריימים/שנייה (יעד: ${diag.target_fps}).`);
            if (Number.isFinite(diag.mean_cv_processing_seconds)) paragraph(`זמן עיבוד ממוצע לפריים בשרת: ${diag.mean_cv_processing_seconds.toFixed(3)} שניות.`);
            if (diag.audio_timing === 'browser_estimated_offset') {
                paragraph(diag.speech_status === 'active'
                    ? `זוהתה פעילות דיבור באודיו בכ־${diag.usable_speech_duration} שניות בחלונות החופפים לצילום; התזמון משוער ואינו מוכיח קשר סיבתי לתנועה.`
                    : diag.speech_status === 'silent'
                    ? 'לא זוהתה פעילות דיבור שימושית באודיו החופף לצילום; אין ציוני התנהגות בזמן דיבור.'
                    : 'האודיו התקבל, אך לא ניתן לקבוע ממנו זמני דיבור שימושיים; הביטחון נשאר מוגבל.');
                if (diag.speech_status === 'active' && diag.usable_speech_duration < 3) paragraph('זוהו פחות מ־3 שניות בחלונות דיבור חופפים; כלל השקלול הנוכחי מפחית בשל כך את הביטחון. נסו לדבר ברצף לאורך ההקלטה, עם פנים, צוואר וכתפיים בתמונה. הזיהוי והתזמון משוערים.');
            } else if (diag.audio_timing === 'unverified') {
                paragraph('אין זמני דיבור מהאודיו בניתוח הצוואר הזה; הקשר לדיבור נבחן מהתמונות בלבד.');
            }
            if (diag.effective_fps < 2.5) paragraph('קצב הפריימים נמוך ומגביל את היכולת להעריך שינויים בזמן דיבור. הקלטה ארוכה יותר לבדה אינה מתקנת קצב נמוך.');
            Object.entries(diag.dimensions || {}).forEach(([key, item]) => {
                if (!item.rejection_reason) return;
                const percent = value => (value * 100).toFixed(2).replace(/\.?0+$/, '');
                // Rejected historical results may have been rounded to the threshold in storage.
                const below = item.rejection_reason === 'Insufficient evidence confidence' &&
                    Math.abs(item.adjusted_confidence - item.required_confidence) < .00005;
                paragraph(`${names[key] || key}: ${reasonsHe[item.rejection_reason] || item.rejection_reason}. ` +
                    `איכות מעקב ${Math.round(item.tracking_quality * 100)}%; ` +
                    (Number.isFinite(item.maximum_possible_confidence) ? `תקרת ביטחון אפשרית ${Math.round(item.maximum_possible_confidence * 100)}% (סף ${Math.round(item.required_confidence * 100)}%). ` : '') +
                    (Number.isFinite(item.adjusted_confidence) ? `ביטחון משוקלל ${below ? 'פחות מ־' + percent(item.required_confidence) : percent(item.adjusted_confidence)}% (סף ${percent(item.required_confidence)}%). ` : '') +
                    (item.model_evidence || []).join(' '));
            });
        }
        if (debug) container.appendChild(debug);
        target = container;
        if (analysis.status === 'error' && packet.reasons?.length) {
            const reasons = packet.reasons.join(' ');
            paragraph(/OPENAI_API_KEY/.test(reasons) ? 'שירות הניתוח טרם הוגדר בשרת.'
                : /timeout|timed out/i.test(reasons) ? 'שירות הניתוח עבר את זמן ההמתנה. נסו שוב.'
                : /401|403/.test(reasons) ? 'OpenAI דחה את האימות או ההרשאה של השרת; הניתוח לא הושלם. יש לבדוק את מפתח OPENAI_API_KEY הפעיל בשרת. זהו כשל שירות, ולא מחסור במידע בתמונות.'
                : /credit_balance_exhausted/.test(reasons) ? 'יתרת הקרדיטים ב־OpenAI נגמרה. הוסיפו קרדיטים בחשבון שאליו שייך המפתח.'
                : /project_spend_limit_exceeded/.test(reasons) ? 'הפרויקט הגיע למגבלת ההוצאה ב־OpenAI. בדקו את הגדרות המגבלה.'
                : /organization_spend_limit_exceeded/.test(reasons) ? 'הארגון הגיע למגבלת ההוצאה ב־OpenAI. בדקו את הגדרות המגבלה.'
                : /organization_usage_limit_exceeded/.test(reasons) ? 'החשבון הגיע למכסת השימוש של OpenAI.'
                : /insufficient_quota/.test(reasons) ? 'אין מכסה זמינה ב־OpenAI. בדקו קרדיטים ומגבלות בחשבון שאליו שייך המפתח.'
                : /rate_limit_exceeded|slow_down/.test(reasons) ? 'קצב הבקשות ל־OpenAI גבוה מדי. המתינו לפני ניסיון נוסף.'
                : /429/.test(reasons) ? 'OpenAI החזיר שגיאת 429 ללא סיבה מפורטת זמינה. בדקו יתרה ומגבלות שימוש בחשבון.'
                : 'לא ניתן להשלים את הניתוח כרגע.');
        }
    }
    root.renderNeckAnalysis = renderNeckAnalysis;
    if (typeof module !== 'undefined') module.exports = {renderNeckAnalysis};
})(typeof window !== 'undefined' ? window : globalThis);
