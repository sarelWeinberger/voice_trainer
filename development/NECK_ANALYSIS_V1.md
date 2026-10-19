# Neck Analysis V1

אמינות ההעברה ובדיקת ציוני האפס — 16 באוקטובר 2026: נוספו אישורי קבלה לאצוות, שליחה חוזרת והגנה מפני עיבוד כפול, לרבות חידוש אצווה שעובדה חלקית. התור שומר עד 150 תמונות ועד 16MiB של JPEGs ממתינים; כשל שנותר מופיע כניתוח חלקי. בבדיקת דפדפן עם עיכוב של 6.2 שניות ואובדן אישור קבלה התקבלו 37/37 תמונות במחשב ובתצוגת נייד, ללא כפילות. נתוני תנועה צפופים מחושבים בשרת ונשלחים כעת למודל לפני הדירוג. השוואת OpenAI אמיתית עם אותן תמונות, עם ובלי הנתונים, השאירה את כל ציוני האזורים על 0; המודל הכיר בתנועה הלא סדירה אך לא דירג אותה כפיצוי תנוחתי. זיהוי המאמץ המדווח עדיין לא אומת ולא נפתר באמצעות שינוי ספים. 68 בדיקות Python ובדיקות העברה/תצוגה ב־JavaScript עברו. [היישום, ההשוואה המבוקרת והתוצאה השלילית](research/recording_failure_2026_10_15/RELIABILITY_2026_10_16.md).

תיקון הכשל בסרטון החדש — 15 באוקטובר 2026: שוחזרו אתחולים חוזרים עקב אובדן נקודות רקע, גם כשמעקב הראש נשאר תקין. נוסף חיפוש פינות רקע וניסיון התאמה מחדש בין אותן שתי תמונות, תוך שמירת הספים הקיימים וללא מילוי תנועה חסרה. בסרטון החדש התקבלו 7/7 חלונות שימושיים, גם ב־Chromium למחשב ובתצוגת נייד עם FaceMesh אמיתי. מוצג גם פיזור התנועה שאינה מחזורית; תדירות עדיין מחייבת עמידה בתנאי המחזוריות. בקשת OpenAI אמיתית עם המפתח המעודכן ארכה 22.36 שניות, ולכן נחתכה במגבלה הישנה. מגבלת הסוקט עודכנה ל־45 שניות והמתנת הדפדפן לניתוח הצוואר ל־55 שניות; אין הבטחת זמן תגובה. בדיקה נוספת דרך מסלולי האפליקציה עם OpenAI אמיתי ובסיס נתונים זמני הושלמה בכ־10.12 שניות. ציוני התצפית נשארו אפס בסרטון הזה, ולכן טרם הוכחה יכולת לזהות את המתח שהמשתמש מתאר. [הסיבות, התיקונים, הבדיקות והמגבלות](research/recording_failure_2026_10_15/REPORT.md). העדכון הזה מחליף את מגבלות ההמתנה שבתיעוד הקודם להלן.

הקשחת מדידת תנועה — 15 באוקטובר 2026: נוספו בדיקות למחזוריות, עקביות בין חצאי חלון, רעש ותלות בתנועת הרקע; החישוב מסומן `head-movement-stability-v3`. חלונות של 2/6/10 שניות בודקים תחומי תדרים שונים, עד לטווח שהצילום תומך בו. המעקב נבדק מול הפנים גם כשהנקודות טרם אבדו, ומבוצעת בדיקת התאמה קדימה־אחורה. נקודות רקע נבחרות מחוץ לאזור אדם משוער, נדרשת הסכמה מרחבית ומבוקרת הזזה וסיבוב קטן; עדיין אין הוכחה שהרקע נייח. חישוב מקומי בדפדפן מאפשר הצגת המדידה גם בזמן המתנה, ניתוק או timeout, בלי בקשת OpenAI נוספת. אחוזי ביטחון לא מכוילים מופיעים רק למנהלים; ה־API מסמן `validation_status=not_clinically_validated` ו־`confidence_kind=uncalibrated_evidence_index`, גם בתוצאות כשל. בדיקות עם FaceMesh אמיתי וניגון הסרטון עברו במחשב ובתצוגת נייד, וכן בדפדפן מנותק מרשת. אין אימות במצלמה פיזית או כיול קליני. [פירוט ששת הנושאים, מה נפתר ומה נשאר](research/motion_hardening_2026_10_15/REPORT.md).

התאוששות מעקב — אומת ב־15 באוקטובר 2026: תוקן מצב שבו נקודות שנכשלו נמחקו לצמיתות, ללא ניסיון לזהות שוב את הפנים במהלך ההקלטה. המעקב משתמש כעת בעד 16 נקודות תמונה מפוזרות באזור הפנים העליון, בטווח חיפוש מורחב ובהשוואת טלאים שמפחיתה השפעת שינוי בהירות אחיד. כאשר מעקב הראש אובד, מסלול זיהוי הפנים הקיים מופעל לפי צורך, לכל היותר פעם בחצי שנייה. מעקב רקע יכול להיבנות מחדש כשיש די פרטים ורצף מעקב ראש תקין. כל חיבור מחדש מתחיל מקטע תנועה נפרד; חלונות החוצים מקטעים נפסלים לחישוב כדי שקפיצת החיבור לא תיחשב לרעידה. פורמט קלט `head-motion-trace-v2` כולל מזהה מקטע, עם תאימות לקלט V1. בבדיקת Chromium הסרטון נתן 8/8 חלונות מדידים ללא הפרעה, ולאחר הסתרה יזומה המעקב התאושש והפיק 4/8 חלונות במחשב ו־6/8 בנייד. זו בדיקת ניגון מקומי עם נקודות FaceMesh שחושבו מראש, ללא מצלמה פיזית או בדיקת FaceMesh חי בדפדפן. [ממצאים, בדיקות ומגבלות](research/tracking_recovery_2026_10_12/REPORT.md).

הסברי כשל מדידת תנועה — 11 באוקטובר 2026: ההקלטה החיה האחרונה סיפקה כ־31.19 דגימות בשנייה, אך רק 12 דגימות שימושיות וכ־0.4 שניות מעקב בחלון הראשון, ואפס דגימות שימושיות בהמשך. הודעת ״אין רצף מספיק״ הוחלפה בהסבר המבוסס על נתוני הדוח: כשל אתחול פנים, דפדפן לא נתמך, כשל קריאת תמונה, מחסור בדגימות, פערי זמן או אובדן מעקב. דוחות ישנים אינם מבחינים בין אובדן נקודות ראש ורקע; מעתה הדפדפן שולח מוני אובדן נפרדים שנשמרים לאחר בדיקת טווחים בשרת. הסברים מתודולוגיים ארוכים אינם מופיעים לפני הודעת כשל. בדיקות ההסברים, אימות המטא־נתונים וה־API עברו. השינוי מסביר את הכשל ומוסיף אבחון; אינו מתקן עדיין את אובדן המעקב ואינו משחזר נתונים חסרים.

תיקון מעקב תנועה — 11 באוקטובר 2026: דוח המשתמש הראה 135 דגימות וקצב חציוני 15.62Hz, עם חלון מדיד אחד מתוך תשעה. תוקנה מגבלת קצב שדילגה על כל פריים שני במצלמות 31.25fps. בנוסף, חישוב V2 מתאים תנודה דומיננטית לזמני הדגימה שנצפו בפועל, במקום לפסול חלון שלם עקב פער יחיד של מעל 0.12 שניות. אין השלמת תנועה בתוך הפערים. נשמרים תנאי כיסוי זמן, צפיפות דגימה, משך ואיכות מעקב; סיבת הפסילה מופיעה בדוח. גודל התנודה ב־V2 הוא RMS של הרכיב המחזורי הדומיננטי, ולכן אינו זהה ל־RMS המסונן של V1. בדיקת Chromium חוזרת הפיקה שמונה חלונות מדידים מתוך שמונה בדסקטופ ובנייד. אין עדיין בדיקה חוזרת של הקלטה חדשה מהמצלמה הפיזית של המשתמש. [תיעוד התיקון](research/head_stability_2026_10_11/SAMPLING_FIX.md).

עדכון מעקב תנועה — 11 באוקטובר 2026: נוסף מדד נפרד ״יציבות ותנודתיות בתנועת הראש״. הדפדפן עוקב אחר אזורי תמונה קטנים בראש וברקע בקצב הפריימים הזמין, עד כ־30 דגימות בשנייה, במקביל למסלול התמונות הקיים של 5 פריימים בשנייה. לשרת נשלחות רק קואורדינטות נגזרות ותזמון, ללא תמונות נוספות עבור המדד הזה. מוצגים גודל תנודה באחוז מרוחב התמונה ותדירות בחלונות של שתי שניות; אין המרה לציון מאמץ או לציון רעד 0–3. המדד נשמר בדוח וזמין גם בכשל מובנה של שירות OpenAI. קצב נמוך, פערי תזמון או אובדן מעקב משאירים חלונות ללא מדידה. זיהוי הפנים בתצוגה המקדימה נדרש להתחלת המעקב. מדידה של תנודת הראש אינה מדידה מבודדת של פעילות צוואר או מתח שרירי. בדיקות יחידה, API עם ספק מדומה ו־Chromium עם הסרטון האמיתי עברו; בתצוגת מחשב נמדדו 4 מתוך 8 חלונות ובתצוגת נייד 5 מתוך 8, עקב פערי דגימה. [שיטה, תוצאות ומגבלות](research/head_stability_2026_10_11/REPORT.md).

עדכון תצוגה לפי תפקיד — 10 באוקטובר 2026: משתמשים רגילים רואים את שלוש קבוצות התצפיות, ציוני הקבוצות וההסברים לממצאים. טבלאות מדידה מקומית, קטע ייחוס, כלל חישוב הדירוג ואבחוני איכות/פריימים מוצגים רק בתצוגת מנהלים (`is_admin=true` או תפקיד 1/2), בתוך אזור סגור ״נתוני בדיקה למנהלים״. ברירת המחדל ללא פרטי משתמש היא תצוגת משתמש רגיל. השינוי חל גם על דוחות שמורים בדסקטופ ובנייד. הודעות כשל שירות נשארות גלויות. זהו סינון תצוגה; מבנה נתוני המחקר וה־API נשמר. בדיקות JavaScript ו־Chromium לתפקידים 1, 2 ו־3 בשני גדלי מסך עברו; התוצאות ב־`research/grouped_observations_2026_10_10/browser_admin_view_results.json`.

עדכון ציוני קבוצות — 10 באוקטובר 2026: לבקשת המשתמש, שלוש קבוצות התצפיות מציגות כעת דירוג התנהגות חיצונית 0–3. הדירוג הוא המקסימום מבין דירוגי האזורים הזמינים בקבוצה, ללא סכימה או ממוצע; מצוין האזור שממנו נלקח הדירוג והביטחון בראיותיו. כאשר חסרים אזורים, הדירוג מסומן כחלקי ומפרט את האזורים שנכללו. כאשר כל האזורים חסרים אין דירוג, וכאשר השירות נכשל מוצג כשל שירות במקום מחסור בראיות. אין ציון מאמץ כולל ואין אפשרות לפתוח טבלת חמשת ציוני האזורים. כלל המקסימום הוא כלל הנדסי תיאורי שלא עבר אימות קליני או כיול של ציוני קבוצות.

בדיקת אימות אמיתית קצרה עם המפתח המקומי החזירה HTTP 401; המפתח של שרת פורט 5000 נמצא זהה למפתח בקובץ `.env` הראשי, ללא חשיפת המפתח. תיקון האימות דורש החלפתו במפתח תקף. בדיקות ציוני הקבוצות והבחנת כשל השירות עברו ב־JavaScript וב־Chromium בדסקטופ ובתצוגת נייד, עם תוצאה אמיתית שמורה ובהדמיית שגיאה; ראו `research/grouped_observations_2026_10_10/browser_group_grades_results.json`. לא נשלחו תמונות בבדיקת האימות ולא בוצע ניתוח הקלטה מוצלח נוסף.

עדכון תצוגה — 10 באוקטובר 2026: האפשרות לפתיחת חמשת ציוני האזורים הוסרה לגמרי מהדוח, גם בדוחות שמורים. מוצגות שלוש קבוצות תצפיות עם תיאור ממצאים, מגבלות ועמודת ״מה אפשר ללמוד״. נוסף מדריך לפירוש שינוי נראה, היעדר התנהגות חריגה, שינוי בלתי ניתן להפרדה ומחסור בראיות. אין ציוני אזורים מספריים או ציון מאמץ כולל בתצוגה. נתוני המקור הפנימיים נשמרים לתאימות ולמחקר. בדיקות JavaScript ו־Chromium בדסקטופ ובתצוגת נייד עברו; תוצאות: `research/grouped_observations_2026_10_10/browser_explained_groups_results.json`. לא נדרשה בקשת OpenAI נוספת.

2026-10-09 initial-reference comparison: [implementation, real-provider repeatability check, browser verification and limitations](research/reference_comparison_2026_10_09/REPORT.md). Local observable changes are shown separately from recording-level effort interpretations; the initial reference is not assumed relaxed.

Offline compression/resolution experiment (2026-10-08): [method, measured results and negative findings](research/compression_validation_2026_10_08/REPORT.md). This experiment made no provider calls and did not change live image-quality defaults.

This replaces the previous single-score neck heuristics and the posture-only `ObservableEffortEngine` wrapper. It estimates **observable external behavior during speech**, not diagnosis, actual muscle activation, force or internal laryngeal tension. Weights, thresholds, confidence and rubric are engineering choices, not clinically validated measurements.

## Existing application integration

The recording/upload workflow and authentication are unchanged:

- `/api/neck-effort/reset` and `/recording/start` initialize a bounded session.
- `/api/neck-effort` collects the existing sampled camera images and deterministic CV measurements. No OpenAI request is made per frame.
- `/api/neck-effort/recording/stop` finalizes once and caches the result for the existing audio-analysis response. Existing `NeckRecording.summary` JSON stores all five dimensions, quality and versions. Feature-series storage contains derived measurements only, not images or raw landmark arrays. Existing `Result.raw_scores` retains the richer neck object as before.
- `/api/observable-effort/recording` is the already-existing video/WAV upload route and uses the same analyzer. Audio is optional at engine level; the existing upload form still expects WAV. Its offset is caller-declared, not independently verified.
- Responses extend `neck_effort` with `neckAnalysis`. Legacy `score`, `continuous_score`, `score_label` and `confidence` are derived from V1 overall output for existing callers. `continuous_score` now has the requested **0–3** scale, rather than the older 0–1 scale. Version distinguishes these scales.
- No database migration, unrelated route, authentication, GRBAS/acoustic model, face-analysis or breathing-analysis changes were made. Existing shared product scoring/recommendation code remains intact and receives the legacy overall neck category.

The current desktop/mobile renderer shows three descriptive observation groups with explanatory notes. Five-region numeric grades and overall neck scores are not displayed; see the current update above. Missing scores display “not assessable.” Raw CV/debug measurements are not rendered in the V1 neck UI. Existing separate audio-feature displays remain unchanged.

## Backend configuration

Set server-side `OPENAI_API_KEY`. Never put it in HTML, JavaScript, source code or the repository. Optional `NECK_OPENAI_MODEL` defaults to `gpt-6-luna`. No new SDK dependency: backend calls the documented HTTPS Responses API using Python's standard library.

Model choice is configurable; an alternative must support image input, strict structured output and the request parameters used here. Default uses temperature 0 and `store:false`. The parser rejects incomplete responses, refusals, malformed data, extra fields, invalid ranges and unavailable dimensions carrying scores. It does not parse unconstrained prose. See official [structured output documentation](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses), [image inputs](https://developers.openai.com/api/docs/guides/images-vision) and the default model, `gpt-6-luna`.

Missing key/provider failure yields an explicit error with null scores. No deterministic fallback invents effort severity. The existing recording-stop route retains its response envelope with `neck_effort.ok=false`; the existing upload route reports HTTP 502 on provider failure. GRBAS may still finish independently; its response includes the neck error object.

## Preprocessing and confound controls

OpenCV + the repository's pinned MediaPipe extract face/chin anchors, jaw opening, image-plane roll, approximate perspective pitch, yaw proxy, shoulders and hips when visible. Depth is not interpreted as a physical neck distance. Approximate image-space ROIs cover left/right lateral neck, anterior neck, submental area, jaw and shoulders. Geometric ROI reliability does **not** prove skin visibility; the multimodal interpreter must additionally identify hair, beard, clothing and occlusion and abstain.

A deterministic least-squares similarity transform aligns previous face anchors to current anchors. Previous grayscale pixels are warped before dense Farneback flow is measured. Thus rigid head translation is controlled before measuring local residual motion. Border feature tracks independently estimate camera movement when sufficiently textured background is available; otherwise stability is conservatively reduced and marked unverified. Turning/depth/parallax, lighting changes, compression, tracking noise and nonrigid head/jaw movement remain confounds. Optical flow is not muscle contraction.

Jaw opening is lip separation normalized by inter-eye distance. Anterior/submental residual-motion traces are regressed against signed jaw velocity and its absolute magnitude, with intercept and ridge regularization. The explained-variance fraction and residual are supplied as confound evidence. This is temporal association, not causal mechanical separation; the interpreter must independently see local deformation before attributing positive anterior/lateral recruitment. Shoulder displacement is controlled relative to hips when both are reliable; otherwise only geometry is supplied and interpretation confidence remains limited.

No raw magnitude is converted directly to a clinical/effort score. The model uses ordered pictures, deterministic traces and the fixed rubric together. Model-proposed positive lateral/anterior scores are unavailable if independent visible local activity is not supported. Positive scores based solely on camera motion, normal articulation or swallowing are not accepted. No-speech aligned audio causes all speech-effort dimensions to abstain, rather than score zero.

When supplied, WAV audio is processed by neck-specific acoustic helpers and WebRTC VAD on 30ms 16kHz windows. VAD votes are summarized in one-second windows. VAD, clipping and RMS establish provisional timing/context only; no acoustic diagnosis is added. Noise and sustained vowels can cause VAD errors. Missing VAD or clipped audio makes speech timing uncertain. Live images use browser capture timestamps. Live recording stop now receives the same WAV used by voice analysis, the neck session ID, and an estimated first-audio-buffer offset. Device latency, callback delay, gaps and clock drift are not independently verified; live audio timing remains approximate.

## Evidence selection and timeline

Up to 300 processed frames are kept in bounded session RAM. Live frames are downscaled preserving aspect ratio to a maximum dimension of 480px; the existing live 5fps sampling is retained. Upload replay samples roughly 5fps from a constant-frame-rate timebase, up to 60 seconds. Variable-rate media and device clock drift are not corrected.

Repeated API judgments are not guaranteed bit-for-bit identical even with a pinned model and temperature zero. Session results are cached to prevent duplicate calls.

Frame selection is deterministic: beginning/end, extrema of anterior/lateral activity, jaw opening, head/shoulder movement, and spaced temporal context, up to 8 frames. Temporal spacing limits redundant neighboring frames. This is not a learned semantic frame selector or perceptual duplicate detector. Up to two neck-context crops accompany full frames (at most 10 images total). Crops retain enough chin/lateral/clavicle context to judge jaw and global movement.

An internal one-second timeline reports residual motion evidence. It is an arbitrary normalized observational motion index, not a force/tension scale and not a per-second LLM score. The five interpreted scores summarize the recording.

## Fixed dimensions and rubric

| Dimension | Evidence sought | Major exclusion |
|---|---|---|
| Head/neck posture compensation | Repeated/sustained extension, chin elevation, altered alignment or excess motion associated with phonation | Conversational head movements or static camera roll alone |
| SCM/lateral neck recruitment | Visible contour change, local tightening or prominence that changes with speech | Invisible/deep activation, static anatomy, head rotation alone |
| Anterior/submental activity | Under-chin/anterior local deformation beyond normal speech mechanics | Jaw-explained motion, swallowing |
| Shoulder recruitment | Repeated/sustained visible elevation or bracing with speech | Static asymmetry or shoulders outside frame |
| Jaw compensation | Disproportionate forceful excursion or bracing | Normal articulation/open mouth alone |

All dimensions: 0 = minimal/none observed with adequate visibility; 1 = mild/intermittent; 2 = clear repeated/moderate or moderately sustained; 3 = strong persistent/prominent. Continuous component estimates use these anchors on 0–3. No population-normal “excessive jaw” threshold is claimed. Normal vowel articulation is explicitly excluded in the model rubric; real-world confound robustness remains to be evaluated.

## Deterministic score and confidence

### Gate audit and ambiguous change reporting (2026-10-08)

`neck-scoring-v1.1` fixes a misplaced agreement penalty: `jaw_explained_fraction` fits **anterior motion** against jaw velocity, so its .7 multiplier now applies only to positive anterior/submental proposals. It previously lowered lateral confidence as well, despite no lateral regression. A controlled independent lateral proposal with confidence .17 now remains .17 when the anterior fit exceeds .7; the anterior proposal remains .119 and unavailable at the unchanged .15 confidence cutoff. This is a software evidence-scope correction, not threshold calibration.

Positive lateral/anterior proposals rejected because they cannot be separated from head/jaw mechanics now retain a separate `diagnostics.dimensions[name].observation` with model-described evidence and supplied frame indices. The table explains that the model described a visible change but its source/effort cannot be determined. The rejected effort score remains null and excluded from aggregation. Poorly visible regions are not upgraded to an observed change. Model speech-related and local-separability decisions are retained for review. Historical records without this metadata are displayed as before; images or observations are not reconstructed.

Confidence diagnostics retain the unrounded value used by the decision, avoiding new records that appear equal to a cutoff after storage rounding. The provider prompt, model, independent-local-activity gate and all thresholds are unchanged.

[Controlled threshold audit](research/threshold_audit_2026_10_08.json), generated by [neck_threshold_audit.py](research/neck_threshold_audit.py), compares .15/.10/.05 using explicit simulated model outputs. Supported independent positive changes pass at .15; head/jaw-confounded positives remain null at all three cutoffs. Lower-confidence proposals can pass lower cutoffs, which says nothing about whether those proposals are correct. No new OpenAI requests or clinically labeled recordings were used in this audit. Regression tests cover the scope fix, ambiguous evidence preservation, subcategory positive confounding, supported positives and safe display of model text.

Verification after the gate audit: all 52 neck Python tests and frontend JavaScript tests passed. Chromium on desktop/mobile viewports verified the new main-table observation with a controlled aggregate result, retained null scores, accepted zero shoulders, and partial-summary labels, without JavaScript errors. [Browser results](research/ambiguity_browser_2026_10_08.json). These are controlled software checks, not sensitivity/accuracy validation on positive real recordings. The local app was restarted with the existing environment on port 5000; no Hugging Face deployment occurred.

Weights in `NeckEffortConfig`: posture .15, lateral .25, anterior/submental .30, shoulders .10, jaw .20. Only observable components contribute. Effective weights are divided by the sum of available component weights. With no available dimensions, overall is null, not zero. Categories: `<.5 → 0`, `<1.5 → 1`, `<2.5 → 2`, otherwise 3. Computation uses unrounded values and then rounds the reported continuous value to four decimals. Labels are fixed in code, and summary is constructed from accepted dimensions so rejected model scores cannot dominate it.

Per-component confidence is model evidence confidence × geometric visibility/ROI reliability × square-root(stability × lighting × resolution) × duration/sampling factor, with an additional .7 agreement penalty for positive anterior/submental activity when anterior jaw-explained variance exceeds .7. Duration factor is capped at five seconds; sampling factor at five processed frames/second. Provisional active-audio-window coverage is additionally capped at three seconds and clipped to the actual observed video span. Unknown speech context applies a .8 factor. Live browser-estimated audio alignment also retains the .8 uncertainty factor even when VAD detects speech; it is not applied twice for unknown context. Below .25 ROI quality or .15 component confidence the dimension is unavailable. These cutoffs are engineering configuration.

Overall confidence is the available-weight average of component confidence multiplied by available original weight coverage. Missing regions therefore reduce overall confidence even though severity weights are renormalized. This is an **uncalibrated evidence-confidence index**, not the probability of correctness. The provider's confidence alone never determines final confidence.

### Partial coverage and supported zero observations (2026-10-07)

Prompt `neck-rubric-v1.2` clarifies that `observable` means assessable, rather than abnormal activity present. Adequately visible regions compared during supported speech may receive zero when no visible recruitment/compensation is observed. This does not establish relaxed muscles or absence of internal tension. Missing speech context, poor visibility or inadequate evidence still require abstention. The application never converts model null scores into zero based on a negative evidence sentence or high geometric tracking quality. Positive lateral/anterior scores still require independent local activity.

Prompt `neck-rubric-v1.3` additionally uses supplied provisional audio windows as temporal context, while explicitly distinguishing temporal association from causation. Audio activity alone cannot establish visible recruitment or compensate for inadequate images.

Weight renormalization remains as requested. `diagnostics.aggregation` records assessed dimensions/count, original weight coverage and partial status. Desktop/mobile label aggregates with fewer than five assessed dimensions as summaries of assessable regions and explicitly warn against interpreting them as whole-neck estimates. Coverage is derived from dimension scores in the renderer, so stored older reports receive the same clarification. For shoulders alone with continuous score .9 and confidence .57, the partial summary remains .9 but its evidence index is .057 (.57 × shoulder weight .10); four missing dimensions remain null.

Oct 8 display correction: continuous scores retain their reported precision instead of rounding to one decimal (the observed .4714/category-0 report had displayed .5). Summary activity foci include only dimensions with an accepted category above zero; an accepted zero jaw score no longer appears as an activity focus. Local head/jaw confounding and other-confound rejections are translated in both table and diagnostics. Confounded dimensions remain unavailable; this change does not establish independent neck activation or relax acceptance rules. Regression tests reproduce the .4714 report, zero-jaw exclusion and Hebrew confound text.

Validation: regression tests cover the shoulders-only report, confidence reduction, preservation of missing scores, supported zero observations, speech-context rejection, and partial/full coverage display. These use controlled model outputs; the prompt clarification has not yet been evaluated on new real patient recordings or clinically validated.

## Privacy and development evidence

OpenAI receives selected full-frame JPEGs, approximate neck crops, deterministic measurements/landmark metadata, quality and audio timing summaries. These can be identifiable patient images. Raw video files and raw audio are not sent. Normal users see a neck-specific disclosure in the existing camera section. `store:false` is a request setting, not a promise of zero provider-side retention; deployments must review the provider's applicable data controls and their patient-data authorization.

Frames/previous image buffers are cleared after finalization, failure, reset or engine close. Abandoned sessions expire after 120 seconds. No media is written to ordinary logs or application database. Existing consent/retention architecture is preserved; no recordings are repurposed for research training.

Optional `NECK_DEBUG=1` retains selected frames and CV/ROI/jaw metadata **only in the engine's in-memory `debug_evidence`**, inaccessible through ordinary result serialization. It expires at the same session deadline and is cleared on reset/close. It does not persist images to disk. Do not enable it for routine patient use.

## Verification and limits

Tests use synthetic image motion controls, mocked provider results and neck-only regression fixtures: relaxed speech, jaw/head motion, shoulder elevation, visible lateral evidence, poor lighting, camera movement, neck occlusion, absent shoulders and silence. They verify software behavior, **not clinical accuracy or real-world model judgments**. A live provider request requires the deployment key and was not executed in the implementation environment.

Commands from repository root:

```bash
MPLCONFIGDIR=/tmp/neck-v1-mpl root/voice_app/venv/bin/python -m unittest discover -s root/voice_app/tests -p 'test_neck_effort_engine.py'
MPLCONFIGDIR=/tmp/neck-v1-mpl root/voice_app/venv/bin/python -m unittest discover -s root/voice_app/tests -p 'test_observable_effort.py'
MPLCONFIGDIR=/tmp/neck-v1-mpl timeout 60 root/voice_app/venv/bin/python -m unittest discover -s root/voice_app/tests -p 'test_neck_v1_api.py' -v
MPLCONFIGDIR=/tmp/neck-v1-mpl timeout 60 root/voice_app/venv/bin/python -m unittest discover -s root/voice_app/tests -p 'test_neck_v1_media.py'
node root/voice_app/tests/test_neck_v1_frontend.js
node --check root/voice_app/static/app.js
node --check root/voice_app/static/neck-analysis-v1.js
node --check root/voice_app/static/observable-effort.js
python3 -m py_compile root/voice_app/neck_effort_engine.py root/voice_app/neck_v1_cv.py root/voice_app/neck_v1_openai.py root/voice_app/neck_v1_scoring.py root/voice_app/observable_effort.py root/voice_app/app.py
git diff --check
```

The repository has no configured static type checker, Python linter or frontend build command. Strict Pydantic input/output validation, Python compilation, JavaScript syntax checks and renderer tests are used. API tests invoke the existing route callables and validate response/OpenAPI schemas; they do not claim an HTTP middleware/end-to-end browser test. Threadpool-backed HTTP testing did not complete in this execution environment and was bounded by timeout; callable tests use the same in-memory spooled upload type as normal small multipart uploads.

Professional validation still requires clinician-labeled real recordings, independent blinded assessment, uncertainty calibration and paired physiological recordings if claims about muscle activity are desired. The prior Lombard experiment is historical proxy research; it does not validate these five V1 dimensions.

## Intentional file inventory

Replaced: `neck_effort_engine.py`; adapted neck-only compatibility/audio/upload helpers in `observable_effort.py` and `neck_effort_smoketest.py`. Added `neck_v1_cv.py`, `neck_v1_scoring.py`, `neck_v1_openai.py`. Extended only `NeckEffortOut`, `analyze_synchronized_effort`, and `stop_neck_effort_recording` in `app.py`; AST comparison confirmed all other backend definitions unchanged.

Neck UI changes: existing sections/integration hooks in `static/index.html`, `static/app.js`, `static/mobile.html`, `static/observable-effort.js`; shared neck-only renderer `static/neck-analysis-v1.js`. No stylesheet edits.

Tests: updated obsolete neck-engine and observable compatibility tests; added `test_neck_v1_api.py`, `test_neck_v1_media.py`, `test_neck_v1_frontend.js` and `fixtures/neck_v1_cases.json`. Documentation: this file plus a neck-specific README note. Pre-existing workspace modifications outside this inventory were preserved.

## Responsiveness update (2026-10-06)

Desktop and mobile offer 5/10/20-second recordings, default 5 seconds; no automatic extension for a neck score. Voice and neck requests start independently. Updated October 15: the neck stop request has a 55-second browser deadline; the independent voice request retains its existing deadline. The OpenAI socket timeout is 45 seconds (not a guaranteed whole-request deadline), without automatic retries. A real 22.36-second provider response demonstrated that the former 15-second socket / 20-second browser deadlines could discard valid results. Output budget remains 1600 tokens with concise evidence. The same pinned model, image quality and scoring rubric are retained. Accuracy/latency on real clinician-rated recordings has not been established. This is not a promise of a successful score within 5 or 20 seconds.

Neck results/errors are displayed even if audio fails. CPU audio feature extraction runs in a worker thread so it cannot directly block event-loop delivery of the neck response. Concurrent results are paired by session ID in the UI and saved history. HTTP 401/403/429 and timeout failures are reported without logging keys, patient images or provider response bodies.

## Error identification and live tracking (2026-10-07)

HTTP 429 responses now retain only allowlisted error codes (no raw provider message). The UI distinguishes exhausted credits, project/organization spending caps, usage limits, and rate limits; absent codes remain explicitly unspecified. A live tracking percentage is an image/face tracking quality value, not OpenAI completion progress. Terminal results update the capture panel to finished/failed and remove stale tracking percentages. The synthetic `analysis_failed` warning is not presented as evidence of poor camera quality.

## Recording diagnostics

Results now include a `diagnostics` object: processed-frame count, observed time span, median interval-derived frame rate, temporal multiplier, dimension tracking quality, and mean CV processing time for newly captured recordings. Per-dimension decisions retain model observability, model confidence/evidence/confounds, adjusted confidence when applicable, threshold and rejection reason. These are derived metadata, without images, landmarks, API keys or raw provider bodies. The UI separates tracking rejection from model abstention and shows sparse sampling explicitly. Old records cannot recover discarded model interpretations or processing times; no historical values are invented. Acceptance thresholds remain unchanged.

Before provider inference, the engine now calculates an upper bound on attainable dimension confidence (tracking quality × temporal multiplier, since model confidence and agreement cannot exceed 1). If every bound is below the existing acceptance threshold, it abstains immediately and skips the provider. This avoids spending time and API credits on an inference that the deterministic scorer could never accept. Scores and acceptance thresholds are unchanged. Provider failures retain available acquisition diagnostics and feature timelines; raw images are still cleared. Processing times are available only after restarting the local Python server with the updated code.

## Independent capture transport (2026-10-07)

Live neck capture samples at up to 5fps independently of HTTP completion. Updated October 16: queued JPEGs are bounded by 150 images and 16MiB, sent serially to `/api/neck-effort/frames` in batches of at most ten. Stable batch IDs and content digests allow receipt replay and partial-batch resume without duplicate frames. There are up to two bounded retries for temporary failures; each request has an eight-second timeout, and stop drains for up to twelve seconds. Transfer counters are reconciled with actual server frame counts, and incomplete acquisition is identified in the user result. Persistent failures, memory bounds and drain expiry can still leave missing images; none are fabricated. Session identity and ordered timestamps remain mandatory; the old single-frame endpoint and clients without IDs remain compatible. Effective FPS uses the lower of median interval rate and whole-span coverage. Client capture timing remains unverified for audio synchronization. FaceMesh supplies periodic fresh references during recording; server neck preprocessing remains active.

Chromium desktop and mobile tests with synthetic media and an artificial 1.7-second delay per batch both delivered 25 frames across 4.8 seconds, effective rate 5fps, with no uncaught JavaScript errors. Requests used the actual isolated local API; OpenAI inference was skipped for nonhuman input. This establishes transport behavior under the tested delay, not clinical accuracy or guaranteed real-device performance. Unit tests verify ordered batches, old-session rejection, bounded queue/cancellation and acquisition independent of unresolved HTTP requests.

## Live audio context (2026-10-08)

`/api/neck-effort/recording/stop` optionally accepts multipart WAV `audio`, `session_id`, and `audio_offset_seconds`. Legacy bodyless calls still work with unknown audio context. The live decoder limits the upload to 5MiB, 25 seconds, 8–96kHz and one/two channels; offsets must be finite and within one second. Malformed audio is rejected before finalization; audio requires a matching current neck session. Raw WAV is decoded in memory and never sent to OpenAI or persisted by this neck endpoint. Derived audio windows enter the model context; diagnostics expose speech status, usable window span, VAD availability and provisional timing. Supported silence skips OpenAI and retains null scores; unavailable VAD remains unknown rather than silent.

The frontend estimates the first captured buffer start as callback wall time minus input-buffer duration, relative to the camera acquisition start. This is a heuristic, not verified synchronization; acquisition gaps and device delay are unresolved. Both desktop and mobile pass the same recording WAV used by the independent voice request. This duplicates a small bounded upload, while avoiding dependency on completion of the voice classifier. One-second VAD windows are coarse; the reported active-window span is not an exact voiced duration.

Validation includes controlled provider-context tests, stale-session/invalid-audio rejection, real WAV silence, missing-VAD uncertainty, decoder bounds, and desktop/mobile request construction. Synthetic browser tests check actual multipart parsing and local VAD without OpenAI inference. An Oct 8 stress run with artificial 1.7s batch delay delivered only 15 frames over 3.198s (4.378fps; mean server CV .2292s/frame), although audio context arrived and VAD detected activity. The bounded upload/drain deadlines can lose evidence under load; the earlier successful stress run is not a performance guarantee. See the accompanying live-audio validation report for ordinary-network desktop/mobile results. No patient-level or clinical validation has been performed for this change.
# תצוגת תצפיות מקובצות — 10 באוקטובר 2026

בדוחות חדשים ושמורים, התצוגה הראשית מציגה שלוש קבוצות תיאוריות: תנוחה ותנועה (ראש, צוואר וכתפיים), מראה הצוואר (צדדים וקדמת הצוואר), ותנועת לסת. אין חיבור ציונים בין תצפיות שעשויות לשקף אותה תנועה ואין ציון מאמץ כולל בתצוגה. תיאורים לפי אזור, ביטחון ומגבלות נשמרים. בעדכון הנוכחי פירוט חמשת הדירוגים הוסר; התיאור לפי אזור מופיע בתוך שלוש הקבוצות בלבד.

שינוי מקומי שאינו ניתן להפרדה ממכניקת הראש והלסת מוצג כתיאור משולב ללא ציון עצמאי, גם בדוחות ישנים ללא שדה התצפית החדש. אין הסקת סיבתיות מעצם החפיפה. מדד חסר נשאר חסר ולא אפס. אפס נתמך מתאר היעדר התנהגות חיצונית חריגה שזוהתה, ולא היעדר מתח פנימי.

זהו שינוי בהצגה ובשימוש בציון המסכם: כללי הניקוד, הספים, בקשת OpenAI והנתונים ההיסטוריים נשמרו לצורכי תאימות ומחקר. שדות הסיכום הישנים עדיין קיימים ב־JSON אך אינם מוצגים או משמשים בסיכום המשולב, בתגי ההיסטוריה או בגרף ההתקדמות של דוחות V1. אין משקלים חדשים או ציון קבוצתי מאומת.

אימות: בדיקות JavaScript עברו, כולל דוחות שמורים, טקסט בטוח, תצפית מעורפלת ללא ציון, השוואה מקומית בעת כשל שירות והוצאת סיכום V1 מהגרף. בדיקת Chromium בדסקטופ ובתצוגת נייד עברה עם תוצאת הסרטון האמיתית האחרונה; תוצאות ב־`research/grouped_observations_2026_10_10/browser_results.json`. לא נשלחה בקשת OpenAI נוספת ולא בוצע אימות קליני עבור שינוי תצוגה זה. השינוי מקומי בלבד.
