// Voice Training App - Frontend Logic

// Interface selection: mobile devices go to the phone UI automatically, and a
// button in the top bar switches manually. No blocking modal on first load --
// asking "which interface?" before the user has seen anything is a dead end.
(function () {
    const savedUI = localStorage.getItem("preferred_ui");
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i
        .test(navigator.userAgent);

    if (savedUI === "mobile" || (isMobile && savedUI !== "desktop")) {
        window.location.href = "mobile.html";
        return;
    }
    localStorage.setItem("preferred_ui", "desktop");

    function bindSwitch() {
        const btn = document.getElementById("switch-to-mobile");
        if (!btn) return;
        btn.addEventListener("click", () => {
            localStorage.setItem("preferred_ui", "mobile");
            window.location.href = "mobile.html";
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", bindSwitch);
    } else {
        bindSwitch();
    }
})();

const GRBAS_LABELS = {
    G: "דירוג כללי",
    R: "צרידות",
    B: "נשיפות",
    A: "חולשה",
    S: "מאמץ",
    N: "מאמץ צוואר",
};

// Index = score: 0=normal(green), 1=slight(yellow), 2=moderate(orange), 3=severe(red)
const GRADE_COLORS = ["#22c55e", "#f59e0b", "#f97316", "#ef4444"];

// ---- State ----
let mediaStream = null;
let audioStream = null;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let recordingStart = 0;
let timerInterval = null;
let faceMesh = null;
let camera = null;
let initInProgress = false;
let neckEffortInterval = null;
let neckEffortRequestInFlight = false;
let neckSampler = null;
let neckMotionSeed = null;
let neckCaptureCanvas = null;
// How often a camera frame is sent to the neck-effort engine while recording.
const NECK_FRAME_INTERVAL_MS = 200;

// ---- DOM refs ----
const video = document.getElementById("video");
const faceCanvas = document.getElementById("face-canvas");
const faceCtx = faceCanvas.getContext("2d");
const faceMetrics = document.getElementById("face-metrics");
const neckEffortMode = document.getElementById("neck-effort-mode");
const neckEffortScore = document.getElementById("neck-effort-score");
const neckEffortLabel = document.getElementById("neck-effort-label");
const neckEffortConfidence = document.getElementById("neck-effort-confidence");
const neckEffortFeedback = document.getElementById("neck-effort-feedback");
const neckEffortReasons = document.getElementById("neck-effort-reasons");
const recordBtn = document.getElementById("record-btn");
const recordLabel = recordBtn.querySelector(".record-label");
const timerEl = document.getElementById("recording-timer");
const statusEl = document.getElementById("status");
const resultsPanel = document.getElementById("results-panel");
const combinedSummary = document.getElementById("combined-summary");
const combinedSummaryTitle = document.getElementById("combined-summary-title");
const combinedSummaryBadge = document.getElementById("combined-summary-badge");
const combinedSummaryText = document.getElementById("combined-summary-text");
const combinedSummaryMetrics = document.getElementById("combined-summary-metrics");
const scoresList = document.getElementById("scores-list");
const radarCanvas = document.getElementById("radar-chart");
const modelBadge = document.getElementById("model-badge");
const featuresPanel = document.getElementById("features-panel");
const featuresGrid = document.getElementById("features-grid");
const historyList = document.getElementById("history-list");
const clearHistoryBtn = document.getElementById("clear-history-btn");
const recommendationsPanel = document.getElementById("recommendations-panel");
const recommendationsList = document.getElementById("recommendations-list");
const progressPanel = document.getElementById("progress-panel");
const progressChart = document.getElementById("progress-chart");
const progressSummary = document.getElementById("progress-summary");

// ---- Initialize Camera + Mic ----
async function init() {
    if (initInProgress || mediaStream || audioStream) {
        return;
    }

    initInProgress = true;
    try {
        // Request camera
        mediaStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
            audio: false,
        });
        video.srcObject = mediaStream;

        // Request mic separately
        audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });

        initFaceMesh();
        await resetNeckEffortSession();
        setStatus("מצלמה ומיקרופון מוכנים.");
    } catch (err) {
        setStatus("שגיאה: " + err.message, true);
    } finally {
        initInProgress = false;
    }

    loadHistory();
}

// ---- MediaPipe FaceMesh ----
function initFaceMesh() {
    try {
        faceMesh = new FaceMesh({
            locateFile: (file) =>
                `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
        });

        faceMesh.setOptions({
            maxNumFaces: 1,
            refineLandmarks: false,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5,
        });

        faceMesh.onResults(onFaceResults);

        camera = new Camera(video, {
            onFrame: async () => {
                if (!isRecording || neckSampler?.requestMotionReference?.()) await faceMesh.send({ image: video });
            },
            width: 640,
            height: 480,
        });
        camera.start();
    } catch (e) {
        console.warn("FaceMesh init failed:", e);
    }
}

function onFaceResults(results) {
    faceCanvas.width = video.videoWidth || 640;
    faceCanvas.height = video.videoHeight || 480;
    faceCtx.clearRect(0, 0, faceCanvas.width, faceCanvas.height);

    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        const landmarks = results.multiFaceLandmarks[0];
        neckMotionSeed = {at: Date.now(), points: [10,151,168,33,263,1].map(i=>[landmarks[i].x,landmarks[i].y])};
        if (isRecording) neckSampler?.updateMotionReference?.(neckMotionSeed);

        // Draw face mesh
        drawConnectors(faceCtx, landmarks, FACEMESH_TESSELATION, {
            color: "rgba(79,140,255,0.15)",
            lineWidth: 0.5,
        });
        drawConnectors(faceCtx, landmarks, FACEMESH_FACE_OVAL, {
            color: "rgba(79,140,255,0.5)",
            lineWidth: 1,
        });
        drawConnectors(faceCtx, landmarks, FACEMESH_LIPS, {
            color: "rgba(255,79,94,0.6)",
            lineWidth: 1,
        });

        // Calculate facial metrics
        const metrics = computeFaceMetrics(landmarks);
        faceMetrics.style.display = "block";
        faceMetrics.textContent =
            `לסת: ${metrics.jawOpenness.toFixed(1)}% | ` +
            `סימטריית שפתיים: ${metrics.lipSymmetry.toFixed(1)}% | ` +
            `מתח: ${metrics.facialTension.toFixed(1)}%`;
    } else {
        neckMotionSeed = null;
        if (isRecording) neckSampler?.updateMotionReference?.(null);
        faceMetrics.style.display = "none";
    }
}

function computeFaceMetrics(landmarks) {
    // Jaw openness: distance between upper and lower lip relative to face height
    const upperLip = landmarks[13]; // upper lip center
    const lowerLip = landmarks[14]; // lower lip center
    const noseTip = landmarks[1];
    const chin = landmarks[152];

    const faceHeight = Math.abs(chin.y - noseTip.y) || 0.001;
    const lipGap = Math.abs(lowerLip.y - upperLip.y);
    const jawOpenness = (lipGap / faceHeight) * 100;

    // Lip symmetry: compare left and right mouth corners
    const leftCorner = landmarks[61];
    const rightCorner = landmarks[291];
    const mouthCenter = (leftCorner.y + rightCorner.y) / 2;
    const leftDev = Math.abs(leftCorner.y - mouthCenter);
    const rightDev = Math.abs(rightCorner.y - mouthCenter);
    const maxDev = Math.max(leftDev, rightDev) || 0.001;
    const lipSymmetry = 100 - (Math.abs(leftDev - rightDev) / maxDev) * 100;

    // Facial tension: eyebrow raise relative to eye position
    const leftBrow = landmarks[66];
    const leftEye = landmarks[159];
    const rightBrow = landmarks[296];
    const rightEye = landmarks[386];
    const browRaise =
        (Math.abs(leftBrow.y - leftEye.y) + Math.abs(rightBrow.y - rightEye.y)) / 2;
    const facialTension = Math.min(100, (browRaise / faceHeight) * 500);

    return { jawOpenness, lipSymmetry, facialTension };
}

async function resetNeckEffortSession() {
    if (typeof getAuthHeaders !== "function") return;
    const headers = getAuthHeaders();
    if (!headers.Authorization) return;

    try {
        await fetch("/api/neck-effort/reset", {
            method: "POST",
            headers: headers,
        });
        updateNeckEffortUI({
            mode: "waiting_recording",
            calibrated: false,
            calibration_progress: 0,
            score: null,
            score_label: null,
            confidence: 0,
            reasons: ["recording_required"],
        });
    } catch (err) {
        console.warn("Neck effort reset failed:", err);
    }
}

// Links the neck-effort recording to the GRBAS audio analysis of the same
// session, so the two modalities can be paired for model training.
let neckEffortSessionId = null;

async function startNeckEffortRecordingSession() {
    neckEffortSessionId = null;
    if (typeof getAuthHeaders !== "function") return;
    const headers = getAuthHeaders();
    if (!headers.Authorization) return;
    try {
        const resp = await fetch("/api/neck-effort/recording/start", {
            method: "POST",
            headers: headers,
        });
        if (resp.ok) {
            const data = await resp.json();
            neckEffortSessionId = data.session_id || null;
        }
    } catch (err) {
        console.warn("Neck effort recording start failed:", err);
    }
}

async function stopNeckEffortRecordingSession(audioBlob, audioOffsetSeconds = 0) {
    const sampler=neckSampler,sessionId=neckEffortSessionId;
    await sampler?.stop();
    const captureTransport=sampler?.transportDiagnostics?.();
    const localMotion = await sampler?.localMotionReport?.();
    const headers = getAuthHeaders();
    const container = document.getElementById("neck-analysis-v1-results");
    function show(packet) {
        if (sessionId && sessionId!==neckEffortSessionId) return packet;
        if(packet.neckAnalysis){packet.neckAnalysis.validation_status ??= 'not_clinically_validated';packet.neckAnalysis.confidence_kind ??= 'uncalibrated_evidence_index';
            if(captureTransport){packet.neckAnalysis.diagnostics ||= {};packet.neckAnalysis.diagnostics.capture_transport ??= captureTransport;}}
        container.hidden = false;
        updateNeckEffortUI(packet);
        renderNeckAnalysis(container, packet, {viewer: typeof _currentUser !== "undefined" ? _currentUser : null});
        activateTab("results");
        return packet;
    }
    if (localMotion) show({model_version:'neck-analysis-v1',neckAnalysis:{version:'neck-analysis-v1',status:'pending',dimensions:{},movement_stability:localMotion}});
    if (!headers.Authorization) return localMotion ? show({model_version:'neck-analysis-v1',neckAnalysis:{version:'neck-analysis-v1',status:'error',dimensions:{},movement_stability:localMotion}}) : null;
    try {
        const body = new FormData();
        if (sessionId) body.append('session_id', sessionId);
        if (captureTransport) body.append('capture_transport',JSON.stringify(captureTransport));
        const motionTrace = sampler?.motionTrace?.();
        if (motionTrace) body.append('motion_trace', JSON.stringify(motionTrace));
        if (audioBlob) {
            body.append('audio', audioBlob, 'neck-context.wav');
            body.append('audio_offset_seconds', String(audioOffsetSeconds));
        }
        const resp = await fetch("/api/neck-effort/recording/stop", {
            method: "POST", headers, body, signal: AbortSignal.timeout(55000)
        });
        if (!resp.ok) throw new Error("neck_service_http_" + resp.status);
        const result = await resp.json();
        if (!result.neck_effort) throw new Error("neck_result_missing");
        if (localMotion && !result.neck_effort.neckAnalysis?.movement_stability) {
            result.neck_effort.neckAnalysis ||= {version:'neck-analysis-v1',status:'error',dimensions:{}};
            result.neck_effort.neckAnalysis.movement_stability=localMotion;
        }
        return show(result.neck_effort);
    } catch (error) {
        return show({model_version: "neck-analysis-v1", reasons: [error.name === "TimeoutError" ? "analysis_timeout" : "neck_service_unavailable"],
            neckAnalysis: {version: "neck-analysis-v1", status: "error", dimensions: {},
                overall: {continuous_score: null, category: null, confidence: 0},
                movement_stability: localMotion || {},
                summary: "ניתוח הצוואר לא הושלם. אפשר לנסות הקלטה חדשה."}});
    }
}

function startNeckEffortMonitoring() {
    stopNeckEffortMonitoring();
    neckCaptureCanvas = document.createElement("canvas");
    neckCaptureCanvas.width = 480;
    neckCaptureCanvas.height = 360;
    // Collect temporal evidence at up to 5fps; scoring starts after recording.
    neckSampler = new NeckCapture({video,canvas:neckCaptureCanvas,sessionId:neckEffortSessionId,
        startedAt:recordingStart,motionSeed:neckMotionSeed,headers:getAuthHeaders,onResult:updateNeckEffortUI});
    neckSampler.start();
}

function stopNeckEffortMonitoring(cancel = true) {
    if (cancel) neckSampler?.cancel();
    if (neckEffortInterval) {
        window.clearInterval(neckEffortInterval);
        neckEffortInterval = null;
    }
    neckEffortRequestInFlight = false;
}

async function captureNeckEffortFrame() {
    if (!isRecording) {
        return;
    }
    if (neckEffortRequestInFlight || !neckCaptureCanvas || !video.videoWidth || !video.videoHeight) {
        return;
    }
    if (typeof getAuthHeaders !== "function") return;
    const headers = getAuthHeaders();
    if (!headers.Authorization) return;

    neckEffortRequestInFlight = true;
    try {
        resizeNeckCaptureCanvas(video.videoWidth, video.videoHeight);
        const ctx = neckCaptureCanvas.getContext("2d");
        ctx.drawImage(video, 0, 0, neckCaptureCanvas.width, neckCaptureCanvas.height);
        const blob = await new Promise((resolve) => {
            neckCaptureCanvas.toBlob(resolve, "image/jpeg", 0.8);
        });
        if (!blob) {
            return;
        }

        const formData = new FormData();
        formData.append("frame", blob, "frame.jpg");

        const resp = await fetch("/api/neck-effort", {
            method: "POST",
            body: formData,
            headers: headers,
        });

        if (resp.status === 401) {
            if (typeof logout === "function") logout();
            return;
        }

        if (!resp.ok) {
            throw new Error("HTTP " + resp.status);
        }

        const data = await resp.json();
        // The engine is making progress as long as it answers with a real
        // mode; "running" means observable tracking has finished calibration.
        if (data.mode === "calibrating" || data.mode === "running" || data.mode === "no_score") {
            neckEffortProgressing = true;
        }
        if (data.mode === "running") {
            neckEffortScored = true;
        }
        updateNeckEffortUI(data);
    } catch (err) {
        console.warn("Neck effort analysis failed:", err);
        updateNeckEffortUI({
            mode: "error",
            score: null,
            score_label: null,
            confidence: 0,
            reasons: ["neck_effort_unavailable"],
        });
    } finally {
        neckEffortRequestInFlight = false;
    }
}

function resizeNeckCaptureCanvas(sourceWidth, sourceHeight) {
    if (!neckCaptureCanvas || !sourceWidth || !sourceHeight) return;
    const maxLongEdge = 480;
    const scale = Math.min(1, maxLongEdge / Math.max(sourceWidth, sourceHeight));
    const width = Math.max(240, Math.round(sourceWidth * scale));
    const height = Math.max(180, Math.round(sourceHeight * scale));
    if (neckCaptureCanvas.width !== width || neckCaptureCanvas.height !== height) {
        neckCaptureCanvas.width = width;
        neckCaptureCanvas.height = height;
    }
}

function updateNeckEffortUI(data) {
    if (!data) {
        neckEffortMode.textContent = "לא פעיל";
        neckEffortScore.textContent = "--";
        neckEffortScore.className = "neck-effort-score";
        neckEffortLabel.textContent = "המצלמה לא הופעלה";
        neckEffortConfidence.textContent = "איכות תמונה: --";
        neckEffortReasons.textContent = "יישרו את הראש, הצוואר והכתפיים מול המצלמה.";
        return;
    }

    if (data.model_version === "neck-analysis-v1") {
        neckEffortMode.textContent = data.neckAnalysis?.status === 'pending' ? 'ממתין לתצפיות' : data.mode === "error" || data.neckAnalysis?.status === "error" ? "נכשל" : data.mode === "complete" || data.neckAnalysis ? "הסתיים" : "אוסף ראיות";
        neckEffortScore.textContent = "—";
        neckEffortScore.className = "neck-effort-score";
        neckEffortLabel.textContent = "תצפיות התנוחה, הצוואר והלסת יוצגו בסיום ההקלטה";
        neckEffortConfidence.textContent = data.neckAnalysis ? "פרטי התוצאה מופיעים בלשונית תוצאות." : "איכות מעקב בתמונה (לא התקדמות הניתוח): " + Math.round((data.confidence || 0) * 100) + "%";
        neckEffortReasons.textContent = "שמרו את הפנים, הצוואר והכתפיים בתמונה.";
        if (neckEffortFeedback) neckEffortFeedback.replaceChildren();
        return;
    }
    if (data.mode === "calibrating") {
        const progress = Math.round((data.calibration_progress || 0) * 100);
        neckEffortMode.textContent = "כיול";
        neckEffortScore.textContent = progress + "%";
        neckEffortScore.className = "neck-effort-score";
        neckEffortLabel.textContent = "החזיקו תנוחה נינוחה";
        neckEffortConfidence.textContent = "איכות תמונה: " + Math.round((data.confidence || 0) * 100) + "%";
        neckEffortReasons.textContent = "הרפו את הכתפיים ושמרו על הצוואר במרכז עד סיום הכיול.";
        return;
    }

    if (data.mode === "waiting_recording") {
        neckEffortMode.textContent = "מוכן";
        neckEffortScore.textContent = "--";
        neckEffortScore.className = "neck-effort-score";
        neckEffortLabel.textContent = "הכיול יתחיל בזמן ההקלטה";
        neckEffortConfidence.textContent = "איכות תמונה: --";
        neckEffortReasons.textContent = "שבו מול המצלמה ולחצו על כפתור ההקלטה כשאתם מוכנים.";
        return;
    }

    if (data.mode === "running" && Number.isInteger(data.score)) {
        neckEffortMode.textContent = "פעיל";
        neckEffortScore.textContent = String(data.score);
        neckEffortScore.className = "neck-effort-score grade-" + data.score;
        neckEffortLabel.textContent = formatNeckEffortScoreLabel(data.score, data.score_label);
        neckEffortConfidence.textContent = "איכות תמונה: " + Math.round((data.confidence || 0) * 100) + "%";
        updateNeckEffortFeedback(data.score, data.reasons || []);
        neckEffortReasons.textContent = formatNeckEffortReasons(data.reasons || []);
        return;
    }

    if (data.mode === "no_score") {
        neckEffortMode.textContent = "התאמה";
        neckEffortScore.textContent = data.last_valid_score ?? "--";
        neckEffortScore.className = Number.isInteger(data.last_valid_score)
            ? "neck-effort-score grade-" + data.last_valid_score
            : "neck-effort-score";
        neckEffortLabel.textContent = "נדרש מיסגור טוב יותר";
        neckEffortConfidence.textContent = "איכות תמונה: " + Math.round((data.confidence || 0) * 100) + "%";
        neckEffortReasons.textContent = formatNeckEffortReasons(data.reasons || []);
        return;
    }

    neckEffortMode.textContent = "שגיאה";
    neckEffortScore.textContent = "--";
    neckEffortScore.className = "neck-effort-score";
    neckEffortLabel.textContent = "הניתוח לא זמין";
    neckEffortConfidence.textContent = "ביטחון: 0%";
    neckEffortReasons.textContent = formatNeckEffortReasons(data.reasons || []);
}

function formatNeckEffortScoreLabel(score, rawLabel) {
    const byScore = {
        0: "עומס נצפה מזערי",
        1: "עומס נצפה קל",
        2: "עומס נצפה בינוני",
        3: "עומס נצפה גבוה",
    };
    const byRawLabel = {
        appears_relaxed: byScore[0],
        possible_mild_effort: byScore[1],
        possible_moderate_effort: byScore[2],
        possible_significant_effort: byScore[3],
    };
    if (Number.isInteger(score) && byScore[score]) {
        return byScore[score];
    }
    return byRawLabel[rawLabel] || "לא ידוע";
}

function formatNeckEffortReasons(reasons) {
    const labels = {
        head_forward_posture: "ייתכן שהראש ממוקם קדימה ביחס לכתפיים.",
        neck_tilt: "ייתכן שקיימת הטיה צידית של הראש או הצוואר.",
        shoulder_imbalance: "ייתכן שקיים חוסר איזון בקו הכתפיים.",
        raised_shoulders: "ייתכן שהכתפיים מורמות.",
        head_instability: "ייתכן שמנח הראש אינו יציב.",
        excess_mouth_opening: "פתיחת הפה גדולה מהצפוי.",
        head_rotation: "הפנים מופנות הצידה מהמצלמה.",
        minimal_visible_effort: "לא נראים סימני מאמץ בולטים בתמונה.",
        no_face_or_pose: "ודאו שהפנים והכתפיים נראים למצלמה.",
        low_confidence_view: "התאימו את המיסגור והפנו ישירות למצלמה.",
        recording_required: "ניתוח הצוואר פעיל רק בזמן הקלטה.",
        neck_effort_unavailable: "ניתוח מאמץ הצוואר אינו זמין זמנית.",
    };
    if (!reasons || reasons.length === 0) {
        return "יישרו את הראש, הצוואר והכתפיים מול המצלמה.";
    }
    return reasons.map((reason) => labels[reason] || reason.replaceAll("_", " ")).join(" ");
}

// ---- Neck Effort Positive Reinforcement ----
let _lastNeckScore = null;
let _neckGoodStreak = 0;

function updateNeckEffortFeedback(score, reasons) {
    if (!neckEffortFeedback) return;

    // Track streak of good scores
    if (score <= 1) {
        _neckGoodStreak++;
    } else {
        _neckGoodStreak = 0;
    }

    const improved = _lastNeckScore !== null && score < _lastNeckScore;
    _lastNeckScore = score;

    let icon = "";
    let msg = "";
    let cls = "";

    if (score === 0) {
        icon = "\u2728"; // ✨
        if (_neckGoodStreak >= 10) {
            msg = "מצוין! תנוחה רפויה בעקביות \u2014 המשיכו!";
        } else if (improved) {
            msg = "שיפור מעולה! הצוואר משוחרר יפה.";
        } else {
            msg = "נראה מצוין \u2014 תנוחה רפויה ומיושרת.";
        }
        cls = "neck-feedback-great";
    } else if (score === 1) {
        icon = "\u2705"; // ✅
        if (improved) {
            msg = "הולך והשתפר! מאמץ קל \u2014 התאמות קטנות עוזרות.";
        } else if (_neckGoodStreak >= 5) {
            msg = "עקביות טובה! שמרו על הכתפיים רפויות.";
        } else {
            msg = "זוהה מאמץ קל \u2014 נסו להרפות את הכתפיים קצת.";
        }
        cls = "neck-feedback-good";
    } else if (score === 2) {
        icon = "\u26A0\uFE0F"; // ⚠️
        if (improved) {
            msg = "יותר טוב! המשיכו לשחרר מתח מאזור הצוואר.";
        } else {
            msg = "מתח מורגש \u2014 הורידו את הכתפיים ונשמו.";
        }
        cls = "neck-feedback-caution";
    } else {
        icon = "\uD83D\uDCA1"; // 💡
        msg = "מאמץ גבוה זוהה \u2014 עצרו, נשמו החוצה והורידו את הכתפיים.";
        cls = "neck-feedback-alert";
    }

    neckEffortFeedback.className = "neck-effort-feedback " + cls;
    neckEffortFeedback.innerHTML = `<span class="neck-feedback-icon">${icon}</span> <span class="neck-feedback-msg">${msg}</span>`;
}

// ---- Audio Recording ----
// Recording duration is independent of analysis. Never extend for a neck score.
const MAX_RECORDING_MS = 20000;
let recordingLimitMs = 5000;
let neckEffortScored = false;
let neckEffortProgressing = false;
let recordingExtended = false;

recordBtn.addEventListener("click", toggleRecording);

function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

async function startRecording() {
    if (!audioStream) {
        setStatus("המיקרופון לא זמין.", true);
        return;
    }

    audioChunks = [];
    document.getElementById("neck-analysis-v1-results").hidden = true;

    // Use AudioContext to capture WAV-compatible audio
    const audioContext = new AudioContext({ sampleRate: 44100 });
    const source = audioContext.createMediaStreamSource(audioStream);
    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    const pcmChunks = [];
    recordBtn._neckAudioOffset = 0;

    processor.onaudioprocess = (e) => {
        if (isRecording) {
            if (!pcmChunks.length) {
                // Approximate the first input buffer start; device/callback latency remains unverified.
                recordBtn._neckAudioOffset = (Date.now() - recordingStart) / 1000 - e.inputBuffer.duration;
            }
            const channelData = e.inputBuffer.getChannelData(0);
            pcmChunks.push(new Float32Array(channelData));
        }
    };

    source.connect(processor);
    processor.connect(audioContext.destination);

    await resetNeckEffortSession();
    await startNeckEffortRecordingSession();
    isRecording = true;
    neckEffortScored = false;
    neckEffortProgressing = false;
    recordingExtended = false;
    recordingLimitMs = Math.min(MAX_RECORDING_MS, Math.max(5000, Number(document.getElementById("recording-duration").value) * 1000));
    recordingStart = Date.now();
    startNeckEffortMonitoring();
    recordBtn.classList.add("recording");
    recordLabel.textContent = "עצור הקלטה";
    timerEl.hidden = false;
    setStatus("מקליט... ההקלטה תיעצר אחרי " + recordingLimitMs / 1000 + " שניות.");

    timerInterval = setInterval(() => {
        const elapsedMs = Date.now() - recordingStart;
        timerEl.textContent = elapsedMs < recordingLimitMs
            ? (elapsedMs / 1000).toFixed(1) + "s / " + (recordingLimitMs / 1000) + "s"
            : (elapsedMs / 1000).toFixed(1) + "s";

        if (!isRecording || elapsedMs < recordingLimitMs) return;

        stopRecording();
    }, 100);

    // Store references for stop
    recordBtn._audioContext = audioContext;
    recordBtn._processor = processor;
    recordBtn._source = source;
    recordBtn._pcmChunks = pcmChunks;
}

async function stopRecording() {
    isRecording = false;
    clearInterval(timerInterval);
    recordBtn.classList.remove("recording");
    recordLabel.textContent = "התחל הקלטה";

    const audioContext = recordBtn._audioContext;
    const processor = recordBtn._processor;
    const source = recordBtn._source;
    const pcmChunks = recordBtn._pcmChunks;

    processor.disconnect();
    source.disconnect();

    // Convert PCM chunks to WAV blob
    const wavBlob = pcmToWav(pcmChunks, audioContext.sampleRate);
    audioContext.close();

    setStatus("מנתח...");
    stopNeckEffortMonitoring(false);
    recordBtn.disabled = true;
    try {
        const [neck, voice] = await Promise.all([stopNeckEffortRecordingSession(wavBlob, recordBtn._neckAudioOffset), sendAudio(wavBlob)]);
        if (voice && neck) { voice.neck_effort = neck; displayResults(voice); }
        setStatus(neck?.neckAnalysis?.status === "error"
            ? "ניתוח הצוואר לא הושלם; פרטים מופיעים בתוצאות."
            : voice ? "הניתוח הושלם." : "תוצאת הצוואר מוצגת; ניתוח הקול לא הושלם.", !voice || neck?.neckAnalysis?.status === "error");
    } finally { recordBtn.disabled = false; }
}

function pcmToWav(pcmChunks, sampleRate) {
    // Merge all chunks
    let totalLength = 0;
    for (const chunk of pcmChunks) totalLength += chunk.length;
    const merged = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of pcmChunks) {
        merged.set(chunk, offset);
        offset += chunk.length;
    }

    // Convert to 16-bit PCM
    const buffer = new ArrayBuffer(44 + merged.length * 2);
    const view = new DataView(buffer);

    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++)
            view.setUint8(offset + i, string.charCodeAt(i));
    };

    writeString(0, "RIFF");
    view.setUint32(4, 36 + merged.length * 2, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, "data");
    view.setUint32(40, merged.length * 2, true);

    // PCM data
    for (let i = 0; i < merged.length; i++) {
        const s = Math.max(-1, Math.min(1, merged[i]));
        view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }

    return new Blob([buffer], { type: "audio/wav" });
}

// ---- Send to Backend ----
async function sendAudio(blob) {
    const formData = new FormData();
    formData.append("audio", blob, "recording.wav");
    if (neckEffortSessionId) {
        formData.append("session_id", neckEffortSessionId);
    }

    try {
        const headers = typeof getAuthHeaders === "function" ? getAuthHeaders() : {};
        const resp = await fetch("/api/analyze", {
            signal: AbortSignal.timeout(20000),
            method: "POST",
            body: formData,
            headers: headers,
        });
        const data = await resp.json();

        if (!resp.ok) {
            setStatus("Error: " + (data.detail || data.error || "Analysis failed"), true);
            return;
        }

        showImprovementBanner(data);
        displayResults(data);
        saveToHistory(data);
        loadCloudHistory();
        // Take the user straight to what they just earned, and flag the tabs
        // that quietly gained content while they were on the results tab.
        activateTab("results");
        markTabUpdated("tips");
        markTabUpdated("progress");
        setStatus("Analysis complete.");
        return data;
    } catch (err) {
        setStatus(err.name === "TimeoutError" ? "ניתוח הקול עבר את זמן ההמתנה. נסו שוב." : "Error: " + err.message, true);
    }
}

// ---- Display Results ----
function displayResults(data) {
    resultsPanel.hidden = false;
    featuresPanel.hidden = false;
    renderCombinedSummary(data);

    // Scores list
    scoresList.innerHTML = "";
    for (const dim of "GRBAS") {
        const score = data.scores[dim];
        scoresList.appendChild(createScoreRow(dim, score));
    }

    const effortPanel = document.getElementById("observable-effort-results");
    if (effortPanel) {
        effortPanel.replaceChildren();
        const report = data.observable_effort;
        if (report) {
            const intro = document.createElement("p");
            intro.textContent = "מדדים נצפים לאורך זמן — גרסת מחקר. אין ציון מאמץ מאומת; עוצמת ההקלטה מושפעת גם מהמיקרופון והמרחק. הסנכרון בהקלטה החיה אינו מאומת.";
            effortPanel.appendChild(intro);
            const table = document.createElement("table");
            const heading = table.insertRow();
            ["זמן (שניות)", "עוצמה (dBFS)", "יחס אנרגיה מעל 1kHz", "איכות"].forEach(label => { const cell = document.createElement("th"); cell.textContent = label; heading.appendChild(cell); });
            (report.audio_timeline || []).forEach(row => {
                const tr = table.insertRow();
                [row.t.toFixed(1), row.rms_dbfs, row.high_frequency_energy_ratio === null ? "—" : row.high_frequency_energy_ratio.toFixed(3), row.usable ? "ניתן למדידה" : "שקט / רוויה"].forEach(value => { tr.insertCell().textContent = value; });
            });
            effortPanel.appendChild(table);
        }
    }
    const neckSection = document.getElementById("neck-analysis-v1-results");
    if (neckSection) {
        if (data.neck_effort) neckSection.hidden = false;
        if (data.neck_effort) renderNeckAnalysis(neckSection, data.neck_effort, {viewer: typeof _currentUser !== "undefined" ? _currentUser : null});
    }
    const neckScore = data.neck_effort && !data.neck_effort.neckAnalysis && data.neck_effort.model_version !== "neck-analysis-v1" && Number.isInteger(data.neck_effort.score)
        ? data.neck_effort.score
        : null;
    if (neckScore !== null) {
        scoresList.appendChild(createScoreRow("N", neckScore));
    }

    modelBadge.textContent = `מודל: ${data.model_type}`;

    // Clinician referral: if any GRBAS dimension scores 2-3 (moderate/severe),
    // surface a clear "see a clinician" notice. This is both clinically right
    // (high scores warrant professional review) and a legal safety net — it
    // prevents users treating high scores as a green light to self-manage.
    const grbasMax = Math.max(...Array.from("GRBAS").map(d => data.scores[d] ?? 0));
    let warningBox = document.getElementById("grbas-clinician-warning");
    if (grbasMax >= 2) {
        if (!warningBox) {
            warningBox = document.createElement("div");
            warningBox.id = "grbas-clinician-warning";
            warningBox.className = "grbas-clinician-warning";
            scoresList.parentElement.insertBefore(warningBox, scoresList.nextSibling);
        }
        warningBox.innerHTML = `
            <strong>שימו לב:</strong> אחד או יותר מציוני GRBAS שלך גבוה (2 או 3).
            ציון זה מאפיין קולות שלעיתים יש להם אתגרים שדורשים בדיקה מקצועית.
            <strong>מומלץ לפנות לקלינאי תקשורת מוסמך או לרופא א.א.ג</strong> לבדיקה מעמיקה.
            <br>זוהי המלצה, לא אבחנה — הכלי הזה הוא חינוכי בלבד.
        `;
        warningBox.style.display = "";
    } else if (warningBox) {
        warningBox.style.display = "none";
    }

    // Radar chart
    drawRadarChart(data.scores);

    // Features
    featuresGrid.innerHTML = "";
    const featureLabels = {
        f0_mean: "תדר יסוד F0 (Hz)",
        f0_std: "סטיית תקן F0",
        f0_range: "טווח F0 (Hz)",
        jitter_percent: "רעידה (%)",
        shimmer_percent: "ריצוד (%)",
        hnr: "HNR (dB)",
        spectral_centroid: "מרכז ספקטרלי",
        spectral_tilt: "נטיית ספקטרלית",
        hf_energy_ratio: "יחס אנרגיה גבוהה",
        intensity_mean: "עוצמה (dB)",
        f1: "F1 (Hz)",
        f2: "F2 (Hz)",
        f3: "F3 (Hz)",
    };

    for (const [key, label] of Object.entries(featureLabels)) {
        const val = data.features[key];
        if (val === undefined) continue;
        const item = document.createElement("div");
        item.className = "feature-item";
        item.innerHTML = `<span class="feature-name">${label}</span><span class="feature-value">${val}</span>`;
        featuresGrid.appendChild(item);
    }

    // Recommendations
    displayRecommendations(data.recommendations);
}

function createScoreRow(dim, score) {
    const row = document.createElement("div");
    row.className = "score-row";
    row.innerHTML = `
        <span class="score-label">${dim}</span>
        <div class="score-bar-bg">
            <div class="score-bar" style="width:${(score / 3) * 100}%;background:${GRADE_COLORS[score]}"></div>
        </div>
        <span class="score-value grade-${score}">${score}</span>
        <span class="score-desc">${GRBAS_LABELS[dim] || dim}</span>
    `;
    return row;
}

function renderCombinedSummary(data) {
    if (!combinedSummary) {
        return;
    }

    const voiceGrade = Number(data.scores?.G ?? 0);
    const strain = Number(data.scores?.S ?? 0);
    const neckEffort = data.neck_effort && !data.neck_effort.neckAnalysis && data.neck_effort.model_version !== "neck-analysis-v1" && Number.isInteger(data.neck_effort.score)
        ? data.neck_effort.score
        : null;
    const overallLevel = Math.max(voiceGrade, strain, neckEffort ?? 0);

    const summaryByLevel = {
        0: {
            title: "עומס נמוך על הקול והצוואר",
            badge: "יציב",
            text: "ההקלטה נראית מסודרת. איכות הקול קרובה לתקינה ואין סימנים למאמץ צוואר מוגבר.",
        },
        1: {
            title: "עומס קל זוהה",
            badge: "מעקב",
            text: "יש סימנים מוקדמים למאמץ נוסף. המשיכו לעקוב אחר התנוחה, זרימת האוויר והקלות בתרגול.",
        },
        2: {
            title: "עומס משולב בינוני",
            badge: "תשומת לב",
            text: "המדדים שנצפו עשויים להתאים למאמץ קולי או צווארי מורגש. התמקדו בשחרור תנוחה, הרפיית כתפיים והפקה יעילה.",
        },
        3: {
            title: "עומס משולב גבוה",
            badge: "גבוה",
            text: "המדדים שנצפו עשויים להתאים למאמץ קולי ו/או צווארי גבוה. מומלץ לתעדף הפחתת מתח לפני הגברת העומס הקולי.",
        },
    };

    const summary = summaryByLevel[overallLevel] || summaryByLevel[0];
    const neckText = neckEffort === null
        ? ""
        : `מאמץ צוואר ${neckEffort}/3${data.neck_effort ? ` (${formatNeckEffortScoreLabel(data.neck_effort.score, data.neck_effort.score_label)})` : ""}.`;

    combinedSummary.hidden = false;
    combinedSummaryTitle.textContent = summary.title;
    combinedSummaryBadge.textContent = summary.badge;
    combinedSummaryBadge.className = "combined-summary-badge level-" + overallLevel;
    combinedSummaryText.textContent = [
        summary.text,
        `דירוג ${voiceGrade}/3, מאמץ ${strain}/3.`,
        neckText,
    ].filter(Boolean).join(" ");

    combinedSummaryMetrics.innerHTML = "";
    const chips = [
        { label: "דירוג קולי", value: `${voiceGrade}/3`, score: voiceGrade },
        { label: "מאמץ", value: `${strain}/3`, score: strain },
        {
            label: "מודל",
            value: data.model_type || "לא ידוע",
            score: null,
        },
    ];

    if (neckEffort !== null) {
        chips.splice(2, 0, {
            label: "מאמץ צוואר",
            value: `${neckEffort}/3`,
            score: neckEffort,
        });
    }

    for (const chip of chips) {
        const element = document.createElement("div");
        element.className = "summary-chip";
        const valueClass = Number.isInteger(chip.score) ? `grade-${chip.score}` : "";
        element.innerHTML = `
            <span class="summary-chip-label">${chip.label}</span>
            <span class="summary-chip-value ${valueClass}">${chip.value}</span>
        `;
        combinedSummaryMetrics.appendChild(element);
    }
}

// ---- Display Recommendations ----
const MEASURE_NAMES = { G: "דירוג כללי", R: "צרידות", B: "נשיפות", A: "חולשה", S: "מאמץ", N: "מאמץ צוואר" };

function displayRecommendations(recommendations) {
    if (!recommendations || recommendations.length === 0) {
        recommendationsPanel.hidden = true;
        return;
    }

    recommendationsPanel.hidden = false;
    recommendationsList.innerHTML = "";

    // Show appropriate subtitle
    const recSubtitle = recommendationsPanel.querySelector(".rec-subtitle");
    const maxScore = Math.max(...recommendations.map((r) => r.score));
    if (recSubtitle) {
        if (maxScore === 0) {
            recSubtitle.textContent = "הקול נשמע תקין! הנה תרגילים לשמירה על בריאות הקול.";
        } else if (maxScore === 1) {
            recSubtitle.textContent = "זוהו מממצאים קלים — תרגילים אלו יעזרו לשמור על בריאות הקול.";
        } else {
            recSubtitle.textContent = "תרגילי YouTube מותאמים לתוצאות GRBAS שלך";
        }
    }

    for (const rec of recommendations) {
        const group = document.createElement("div");
        group.className = "rec-group";

        const header = document.createElement("div");
        header.className = "rec-header";
        header.innerHTML = `
            <span class="rec-measure grade-${rec.score}">${rec.measure} – ${MEASURE_NAMES[rec.measure] || rec.measure}</span>
            <span class="rec-score">ציון: ${rec.score}/3</span>
            <span class="rec-priority">עדיפות #${rec.priority_rank}</span>
        `;
        group.appendChild(header);

        const list = document.createElement("div");
        list.className = "rec-resources";

        for (const res of rec.resources) {
            const card = document.createElement("a");
            card.className = "rec-card";
            card.href = res.url;
            card.target = "_blank";
            card.rel = "noopener noreferrer";
            card.innerHTML = `
                <span class="rec-card-course">${res.course}</span>
                <span class="rec-card-section">${res.section}</span>
                <span class="rec-card-why">${res.why}</span>
            `;
            list.appendChild(card);
        }

        group.appendChild(list);
        recommendationsList.appendChild(group);
    }
}

// ---- Radar Chart ----
function drawRadarChart(scores) {
    const ctx = radarCanvas.getContext("2d");
    const w = radarCanvas.width;
    const h = radarCanvas.height;
    const cx = w / 2;
    const cy = h / 2;
    const maxR = Math.min(cx, cy) - 30;
    const dims = ["G", "R", "B", "A", "S"];
    const n = dims.length;
    const angleStep = (2 * Math.PI) / n;
    const startAngle = -Math.PI / 2;

    ctx.clearRect(0, 0, w, h);

    // Draw grid rings (levels 1, 2, 3)
    for (let level = 1; level <= 3; level++) {
        const r = (level / 3) * maxR;
        ctx.beginPath();
        for (let i = 0; i <= n; i++) {
            const angle = startAngle + i * angleStep;
            const x = cx + r * Math.cos(angle);
            const y = cy + r * Math.sin(angle);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = "rgba(0,0,0,0.08)";
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    // Draw axes
    for (let i = 0; i < n; i++) {
        const angle = startAngle + i * angleStep;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + maxR * Math.cos(angle), cy + maxR * Math.sin(angle));
        ctx.strokeStyle = "rgba(0,0,0,0.1)";
        ctx.stroke();
    }

    // Draw data polygon
    ctx.beginPath();
    for (let i = 0; i <= n; i++) {
        const idx = i % n;
        const angle = startAngle + idx * angleStep;
        const val = scores[dims[idx]] || 0;
        const r = (val / 3) * maxR;
        const x = cx + r * Math.cos(angle);
        const y = cy + r * Math.sin(angle);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.fillStyle = "rgba(74,124,255,0.15)";
    ctx.fill();
    ctx.strokeStyle = "#4a7cff";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw data points and labels
    for (let i = 0; i < n; i++) {
        const angle = startAngle + i * angleStep;
        const val = scores[dims[i]] || 0;
        const r = (val / 3) * maxR;
        const px = cx + r * Math.cos(angle);
        const py = cy + r * Math.sin(angle);

        // Point
        ctx.beginPath();
        ctx.arc(px, py, 4, 0, Math.PI * 2);
        ctx.fillStyle = GRADE_COLORS[val];
        ctx.fill();

        // Label
        const lx = cx + (maxR + 18) * Math.cos(angle);
        const ly = cy + (maxR + 18) * Math.sin(angle);
        ctx.fillStyle = "#1a2332";
        ctx.font = "bold 14px 'Rubik', sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(dims[i], lx, ly);
    }
}

// ---- Improvement Banner ----
const improvementBanner = document.getElementById("improvement-banner");

const ENCOURAGEMENT_MESSAGES = [
    { title: "כל הכבוד! יש שיפור!", msg: "התרגול משתלם – הקול שלך נשמע טוב יותר מהפעם הקודמת!" },
    { title: "מעולה! התקדמות יפה!", msg: "העבודה הקשה משתלמת. המשיכו/י להתאמן!" },
    { title: "איזה קפיצה קדימה!", msg: "הנתונים מראים שיפור אמיתי באיכות הקול. כל הכבוד!" },
    { title: "ראית שיפור!", msg: "המאמץ שלך נותן פירות – הקול נשמע חזק ויציב יותר." },
    { title: "ואו מדהים!", msg: "הקול שלך השתפר. זו בדיוק התוצאה של עבודה קשה!" },
];

const DIM_NAMES_HE = { G: "דירוג כללי", R: "צרידות", B: "נשיפות", A: "חולשה", S: "מאמץ", N: "מאמץ צוואר" };

function showImprovementBanner(data) {
    if (!improvementBanner) return;
    improvementBanner.hidden = true;

    // Get previous history to compare
    const key = _historyKey();
    const history = JSON.parse(localStorage.getItem(key) || "[]");
    if (history.length === 0) return;

    const prev = history[0]; // most recent previous entry
    const improved = [];

    for (const dim of "GRBAS") {
        const prevScore = prev.scores[dim] ?? 0;
        const curScore = data.scores[dim] ?? 0;
        if (curScore < prevScore) {
            improved.push({ dim, from: prevScore, to: curScore });
        }
    }

    // Check neck effort improvement
    const prevNeck = prev.neck_effort && !prev.neck_effort.neckAnalysis && prev.neck_effort.model_version !== "neck-analysis-v1" && Number.isInteger(prev.neck_effort.score) ? prev.neck_effort.score : null;
    const curNeck = data.neck_effort && !data.neck_effort.neckAnalysis && data.neck_effort.model_version !== "neck-analysis-v1" && Number.isInteger(data.neck_effort.score) ? data.neck_effort.score : null;
    if (prevNeck !== null && curNeck !== null && curNeck < prevNeck) {
        improved.push({ dim: "N", from: prevNeck, to: curNeck });
    }

    if (improved.length === 0) return;

    const pick = ENCOURAGEMENT_MESSAGES[Math.floor(Math.random() * ENCOURAGEMENT_MESSAGES.length)];

    let chipsHtml = "";
    for (const imp of improved) {
        chipsHtml += `<span class="banner-chip">${DIM_NAMES_HE[imp.dim]}: ${imp.from} \u2192 ${imp.to} \u2193</span>`;
    }

    improvementBanner.innerHTML = `
        <div class="banner-title"><span class="banner-icon">\u2728</span>${pick.title}</div>
        <div class="banner-message">${pick.msg}</div>
        <div class="banner-details">${chipsHtml}</div>
    `;
    improvementBanner.hidden = false;
}

// ---- Session History ----

/** Return a per-user localStorage key so each account has its own history. */
function _historyKey() {
    if (typeof _currentUser !== "undefined" && _currentUser && _currentUser.username) {
        return "grbas_history_" + _currentUser.username;
    }
    return "grbas_history";
}

function saveToHistory(data) {
    const key = _historyKey();
    const history = JSON.parse(localStorage.getItem(key) || "[]");
    history.unshift({
        timestamp: new Date().toISOString(),
        scores: data.scores,
        features: data.features,
        neck_effort: data.neck_effort || null,
        observable_effort: data.observable_effort || null,
    });
    // Keep last 50 entries
    if (history.length > 50) history.length = 50;
    localStorage.setItem(key, JSON.stringify(history));
    loadHistory();
}

function loadHistory() {
    const key = _historyKey();
    const history = JSON.parse(localStorage.getItem(key) || "[]");

    if (history.length === 0) {
        historyList.innerHTML =
            '<p class="empty-state">אין עדיין הקלטות. התחילו בהקלטת הקול למעלה.</p>';
        clearHistoryBtn.hidden = true;
        return;
    }

    clearHistoryBtn.hidden = false;
    historyList.innerHTML = "";

    for (const entry of history) {
        const item = document.createElement("div");
        item.className = "history-item";

        const time = new Date(entry.timestamp);
        const timeStr = time.toLocaleDateString() + " " + time.toLocaleTimeString();

        let scoresHtml = "";
        for (const dim of "GRBAS") {
            const s = entry.scores[dim];
            scoresHtml += `<span class="history-score grade-${s}">${dim}:${s}</span>`;
        }
        if (entry.neck_effort && !entry.neck_effort.neckAnalysis && entry.neck_effort.model_version !== "neck-analysis-v1" && Number.isInteger(entry.neck_effort.score)) {
            scoresHtml += `<span class="history-score grade-${entry.neck_effort.score}">N:${entry.neck_effort.score}</span>`;
        }

        item.innerHTML = `
            <span class="history-time">${timeStr}</span>
            <div class="history-scores">${scoresHtml}</div>
            <span class="history-view-btn" title="צפה בהמלצות">&#x25B6;</span>
        `;
        item.className = "history-item history-item-clickable";
        item.addEventListener("click", () => loadHistoryEntry(entry));
        historyList.appendChild(item);
    }
}

clearHistoryBtn.addEventListener("click", () => {
    localStorage.removeItem(_historyKey());
    loadHistory();
});

// ---- Helpers ----
function setStatus(msg, isError = false) {
    statusEl.textContent = msg;
    statusEl.className = "status" + (isError ? " error" : "");
}

// ---- Cloud History (when authenticated) ----
async function loadCloudHistory() {
    if (typeof getAuthHeaders !== "function") return;
    const headers = getAuthHeaders();
    if (!headers.Authorization) return;

    try {
        const resp = await fetch("/api/history", { headers });
        if (!resp.ok) return;
        const data = await resp.json();

        if (data.history && data.history.length > 0) {
            renderHistory(data.history);
            renderProgressChart(data.history);
            updateQuickStats(data.history);
            syncTabEmptyStates();
        }
    } catch (err) {
        console.warn("Cloud history load failed:", err);
    }
}

// ---- Quick Stats ----
function updateQuickStats(entries) {
    const qsTotal = document.getElementById("qs-total");
    const qsLastDate = document.getElementById("qs-last-date");
    const qsBestG = document.getElementById("qs-best-g");
    if (!qsTotal) return;

    qsTotal.textContent = entries.length;

    if (entries.length > 0) {
        const sorted = entries.slice().sort((a, b) =>
            new Date(b.timestamp) - new Date(a.timestamp)
        );
        const lastDate = new Date(sorted[0].timestamp);
        qsLastDate.textContent = lastDate.toLocaleDateString("he-IL");

        let bestG = 3;
        for (const e of entries) {
            if (e.scores && typeof e.scores.G === "number" && e.scores.G < bestG) {
                bestG = e.scores.G;
            }
        }
        qsBestG.textContent = bestG;
    }
}

function renderHistory(entries) {
    if (!entries || entries.length === 0) {
        historyList.innerHTML =
            '<p class="empty-state">אין עדיין הקלטות. התחילו בהקלטת הקול למעלה.</p>';
        clearHistoryBtn.hidden = true;
        return;
    }

    clearHistoryBtn.hidden = false;
    historyList.innerHTML = "";

    for (const entry of entries) {
        const item = document.createElement("div");
        item.className = "history-item history-item-clickable";

        const time = new Date(entry.timestamp);
        const timeStr = time.toLocaleDateString() + " " + time.toLocaleTimeString();

        let scoresHtml = "";
        for (const dim of "GRBAS") {
            const s = entry.scores[dim];
            scoresHtml += `<span class="history-score grade-${s}">${dim}:${s}</span>`;
        }
        if (entry.neck_effort && !entry.neck_effort.neckAnalysis && entry.neck_effort.model_version !== "neck-analysis-v1" && Number.isInteger(entry.neck_effort.score)) {
            scoresHtml += `<span class="history-score grade-${entry.neck_effort.score}">N:${entry.neck_effort.score}</span>`;
        }

        item.innerHTML = `
            <span class="history-time">${timeStr}</span>
            <div class="history-scores">${scoresHtml}</div>
            <span class="history-view-btn" title="צפה בהמלצות">&#x25B6;</span>
        `;
        item.className = "history-item history-item-clickable";
        item.addEventListener("click", () => loadHistoryEntry(entry));
        historyList.appendChild(item);
    }
}

// ---- Load History Entry into Results ----

// ---- Progress Chart ----
// Dimensions the user has toggled off by clicking the legend, plus the state
// needed to redraw the chart without re-fetching history.
const progressHiddenDims = new Set();
let progressChartEntries = null;
let progressLegendHits = [];
let progressLegendBound = false;

function renderProgressChart(entries) {
    if (!progressPanel || !progressChart || !entries || entries.length < 2) {
        if (progressPanel) progressPanel.hidden = true;
        return;
    }

    progressPanel.hidden = false;
    progressChartEntries = entries;
    _bindProgressLegend();

    // Defer drawing to next frame so the browser can layout the now-visible panel
    requestAnimationFrame(() => _drawProgressChart(entries));
}

/** Clicking a legend entry hides or re-shows that line. */
function _bindProgressLegend() {
    if (progressLegendBound || !progressChart) return;
    progressLegendBound = true;

    progressChart.addEventListener("click", (ev) => {
        const rect = progressChart.getBoundingClientRect();
        const x = ev.clientX - rect.left;
        const y = ev.clientY - rect.top;
        const hit = progressLegendHits.find(
            (r) => x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h
        );
        if (!hit || !progressChartEntries) return;

        if (progressHiddenDims.has(hit.key)) progressHiddenDims.delete(hit.key);
        else progressHiddenDims.add(hit.key);
        _drawProgressChart(progressChartEntries);
    });

    progressChart.addEventListener("mousemove", (ev) => {
        const rect = progressChart.getBoundingClientRect();
        const x = ev.clientX - rect.left;
        const y = ev.clientY - rect.top;
        const over = progressLegendHits.some(
            (r) => x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h
        );
        progressChart.style.cursor = over ? "pointer" : "default";
    });

    window.addEventListener("resize", () => {
        if (progressChartEntries && progressPanel && !progressPanel.hidden) {
            _drawProgressChart(progressChartEntries);
        }
    });
}

/**
 * Pull one 0-3 score out of a history entry.
 * "N" is the neck-effort score, which lives outside `scores` but shares the
 * same 0-3 scale, so it can be plotted on the same axis as GRBAS.
 * Returns null when the session has no value for that dimension (e.g. an
 * audio-only session has no neck score) so it can be skipped, not read as 0.
 */
function _progressDimValue(entry, key) {
    if (key === "N") {
        const score = entry.neck_effort && !entry.neck_effort.neckAnalysis && entry.neck_effort.model_version !== "neck-analysis-v1" ? entry.neck_effort.score : null;
        return Number.isFinite(score) ? score : null;
    }
    const score = entry.scores ? entry.scores[key] : null;
    return Number.isFinite(score) ? score : null;
}

/**
 * Centered weighted moving average over a series (nulls skipped).
 * One unusually good or bad session should nudge the trend, not flip it, so
 * every point is averaged with its neighbours using triangular weights. The
 * window widens as history grows: +-1 session early on, up to +-3 later.
 */
function _smoothSeries(raw) {
    const n = raw.length;
    const half = n < 5 ? 1 : (n < 10 ? 2 : 3);
    const out = [];
    for (let i = 0; i < n; i++) {
        let sum = 0;
        let weight = 0;
        for (let j = i - half; j <= i + half; j++) {
            if (j < 0 || j >= n || raw[j] === null) continue;
            const w = (half + 1) - Math.abs(j - i);
            sum += raw[j] * w;
            weight += w;
        }
        out.push(weight > 0 ? sum / weight : null);
    }
    return out;
}

/** Stroke a soft Catmull-Rom curve through the given points. */
function _strokeSmoothPath(ctx, pts) {
    if (pts.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 0; i < pts.length - 1; i++) {
        const p0 = pts[i - 1] || pts[i];
        const p1 = pts[i];
        const p2 = pts[i + 1];
        const p3 = pts[i + 2] || p2;
        ctx.bezierCurveTo(
            p1.x + (p2.x - p0.x) / 6, p1.y + (p2.y - p0.y) / 6,
            p2.x - (p3.x - p1.x) / 6, p2.y - (p3.y - p1.y) / 6,
            p2.x, p2.y,
        );
    }
    ctx.stroke();
}

function _drawProgressChart(entries) {
    // Sort oldest first
    const sorted = [...entries].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
    const n = sorted.length;

    // GRBAS + neck effort, all on the same 0-3 scale
    const dims = [
        { key: "G", name: "דירוג", color: "#4a7cff" },
        { key: "R", name: "צרידות", color: "#f97316" },
        { key: "B", name: "נשיפות", color: "#a855f7" },
        { key: "A", name: "חולשה", color: "#f59e0b" },
        { key: "S", name: "מאמץ", color: "#ef4444" },
        { key: "N", name: "מאמץ צוואר", color: "#14b8a6" },
    ];

    for (const dim of dims) {
        dim.raw = sorted.map((entry) => _progressDimValue(entry, dim.key));
        dim.smooth = _smoothSeries(dim.raw);
    }
    // A dimension with fewer than two measurements has no trend to show
    const visible = dims.filter((d) => d.raw.filter((v) => v !== null).length >= 2);
    if (visible.length === 0) {
        if (progressPanel) progressPanel.hidden = true;
        return;
    }
    const shown = visible.filter((d) => !progressHiddenDims.has(d.key));

    const canvas = progressChart;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const container = canvas.parentElement;
    const cssW = (container && container.clientWidth > 0) ? container.clientWidth : 760;
    const cssH = 336;
    canvas.style.width = cssW + "px";
    canvas.style.height = cssH + "px";
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const pad = { top: 26, right: 18, bottom: 78, left: 36 };
    const w = cssW - pad.left - pad.right;
    const h = cssH - pad.top - pad.bottom;
    const xAt = (i) => pad.left + (i / Math.max(1, n - 1)) * w;
    const yAt = (score) => pad.top + h - (score / 3) * h;

    // Dimensions frequently land on the same grade (three 1's hide two lines).
    // Fan the drawn lines apart by a couple of pixels each so none disappears
    // behind another; the shift is far below one grade step.
    const laneGap = shown.length > 1 ? Math.min(2.4, (h / 3) * 0.06) : 0;
    shown.forEach((dim, i) => {
        dim.lane = (i - (shown.length - 1) / 2) * laneGap;
    });

    ctx.clearRect(0, 0, cssW, cssH);

    // Background grid
    ctx.lineWidth = 1;
    for (let level = 0; level <= 3; level++) {
        const y = yAt(level);
        ctx.strokeStyle = level === 0 ? "rgba(0,0,0,0.12)" : "rgba(0,0,0,0.06)";
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(pad.left + w, y);
        ctx.stroke();

        ctx.fillStyle = "rgba(0,0,0,0.4)";
        ctx.font = "12px 'Rubik', sans-serif";
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillText(String(level), pad.left - 8, y);
    }

    // X-axis date labels
    ctx.fillStyle = "rgba(0,0,0,0.4)";
    ctx.font = "11px 'Rubik', sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const maxLabels = Math.min(n, 8);
    const step = Math.max(1, Math.ceil(n / maxLabels));
    for (let i = 0; i < n; i += step) {
        const date = new Date(sorted[i].timestamp);
        const label = date.toLocaleDateString("he-IL", { month: "numeric", day: "numeric" });
        ctx.fillText(label, xAt(i), pad.top + h + 8);
    }

    // Each dimension: faint raw sessions + a smoothed trend line on top
    for (const dim of shown) {
        ctx.save();
        ctx.globalAlpha = 0.22;
        ctx.fillStyle = dim.color;
        for (let i = 0; i < n; i++) {
            if (dim.raw[i] === null) continue;
            ctx.beginPath();
            ctx.arc(xAt(i), yAt(dim.raw[i]) + dim.lane, 2.4, 0, Math.PI * 2);
            ctx.fill();
        }
        ctx.restore();

        // A white casing under each line keeps overlapping lines readable
        const runs = [];
        let run = [];
        for (let i = 0; i <= n; i++) {
            const value = i < n ? dim.smooth[i] : null;
            if (value === null) {
                if (run.length) runs.push(run);
                run = [];
                continue;
            }
            run.push({ x: xAt(i), y: yAt(value) + dim.lane });
        }
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.strokeStyle = "rgba(255,255,255,0.85)";
        ctx.lineWidth = 5.5;
        for (const r of runs) _strokeSmoothPath(ctx, r);
        ctx.strokeStyle = dim.color;
        ctx.lineWidth = 2.6;
        for (const r of runs) _strokeSmoothPath(ctx, r);

        // Mark where the trend currently stands
        const lastIdx = dim.smooth.reduce((acc, v, i) => (v === null ? acc : i), -1);
        if (lastIdx >= 0) {
            ctx.beginPath();
            ctx.arc(xAt(lastIdx), yAt(dim.smooth[lastIdx]) + dim.lane, 4, 0, Math.PI * 2);
            ctx.fillStyle = dim.color;
            ctx.fill();
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    }

    // Legend — clickable, falls back to letters if the names do not fit
    const legendY = pad.top + h + 40;
    ctx.font = "12px 'Rubik', sans-serif";
    ctx.textBaseline = "middle";
    const gap = 16;
    const dotW = 14;
    const labels = visible.map((d) => d.name);
    const rowWidth = () => visible.reduce(
        (sum, d, i) => sum + dotW + ctx.measureText(labels[i]).width + gap, -gap);
    if (rowWidth() > w) visible.forEach((d, i) => { labels[i] = d.key; });

    let legendX = pad.left + Math.max(0, (w - rowWidth()) / 2);
    progressLegendHits = [];
    visible.forEach((dim, i) => {
        const off = progressHiddenDims.has(dim.key);
        const textW = ctx.measureText(labels[i]).width;

        ctx.save();
        ctx.globalAlpha = off ? 0.35 : 1;
        ctx.fillStyle = dim.color;
        ctx.beginPath();
        ctx.arc(legendX + 5, legendY, 5, 0, Math.PI * 2);
        if (off) { ctx.lineWidth = 1.5; ctx.strokeStyle = dim.color; ctx.stroke(); }
        else ctx.fill();
        ctx.fillStyle = "#5a6a7e";
        ctx.textAlign = "left";
        ctx.fillText(labels[i], legendX + dotW, legendY);
        ctx.restore();

        progressLegendHits.push({
            key: dim.key,
            x: legendX - 4, y: legendY - 11,
            w: dotW + textW + 8, h: 22,
        });
        legendX += dotW + textW + gap;
    });

    // Hint that the legend is interactive
    ctx.fillStyle = "rgba(0,0,0,0.35)";
    ctx.font = "11px 'Rubik', sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("לחצו על שם במקרא כדי להסתיר או להציג קו", pad.left + w / 2, legendY + 20);

    // Progress summary
    renderProgressSummary(sorted, visible);
}

function renderProgressSummary(sorted, dims) {
    if (!progressSummary || sorted.length < 2) return;

    let html = "";
    for (const dim of dims) {
        // Compare smoothed endpoints so a single outlier session cannot flip the arrow
        const trend = dim.smooth.filter((v) => v !== null);
        if (trend.length < 2) continue;
        const start = trend[0];
        const end = trend[trend.length - 1];
        const diff = end - start;

        let arrow, cls;
        if (diff <= -0.15) {
            arrow = "↓"; // ↓ improved (lower is better)
            cls = "progress-improved";
        } else if (diff >= 0.15) {
            arrow = "↑"; // ↑ worsened
            cls = "progress-worsened";
        } else {
            arrow = "→"; // → stable
            cls = "progress-stable";
        }

        const dimmed = progressHiddenDims.has(dim.key) ? " progress-chip-off" : "";
        html += `<div class="progress-chip ${cls}${dimmed}">
            <span class="progress-chip-dim" style="color:${dim.color}">${dim.key}</span>
            <span class="progress-chip-label">${dim.name}</span>
            <span class="progress-chip-change">${start.toFixed(1)} ${arrow} ${end.toFixed(1)}</span>
        </div>`;
    }

    progressSummary.innerHTML = html;
}

// ---- Load History Entry into Results ----
/**
 * Load a past analysis entry into the results panels.
 * If the entry already has recommendations show them immediately,
 * otherwise fetch from the server.
 */
async function loadHistoryEntry(entry) {
    const fakeData = {
        scores: entry.scores,
        features: entry.features || {},
        model_type: entry.model_type || "unknown",
        neck_effort: entry.neck_effort || null,
        observable_effort: entry.observable_effort || null,
        recommendations: entry.recommendations || null,
    };

    displayResults(fakeData);

    // If no embedded recommendations, try fetching from the server
    if (!fakeData.recommendations && entry.id) {
        try {
            const headers = typeof getAuthHeaders === "function" ? getAuthHeaders() : {};
            const resp = await fetch(`/api/results/${encodeURIComponent(entry.id)}/recommendations`, { headers });
            if (resp.ok) {
                const data = await resp.json();
                if (data.recommendations) {
                    displayRecommendations(data.recommendations);
                }
            }
        } catch (err) {
            console.warn("Failed to load recommendations for history entry:", err);
        }
    }

    // If still no recommendations, generate client-side from scores
    if (!fakeData.recommendations && !entry.id) {
        try {
            const headers = typeof getAuthHeaders === "function" ? getAuthHeaders() : {};
            headers["Content-Type"] = "application/json";
            const resp = await fetch("/api/recommendations", {
                method: "POST",
                headers,
                body: JSON.stringify({
                    scores: entry.scores,
                    neck_effort: entry.neck_effort || null,
        observable_effort: entry.observable_effort || null,
                }),
            });
            if (resp.ok) {
                const data = await resp.json();
                if (data.recommendations) {
                    displayRecommendations(data.recommendations);
                }
            }
        } catch (err) {
            console.warn("Failed to fetch recommendations:", err);
        }
    }

    resultsPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---- Show / Hide app content based on auth ----
const appContent = document.getElementById("app-content");
const loginRequired = document.getElementById("login-required");

function showApp() {
    if (appContent) appContent.style.display = "";
    if (loginRequired) loginRequired.style.display = "none";
}

function hideApp() {
    if (appContent) appContent.style.display = "none";
    if (loginRequired) loginRequired.style.display = "";
}

// Auth hooks called by auth.js
function onAuthReady(user) {
    showApp();
    init();
    loadCloudHistory();
    syncTabEmptyStates();
    const avatar = document.getElementById("user-avatar");
    if (avatar && user && user.username) {
        avatar.textContent = user.username.trim().charAt(0).toUpperCase();
    }
}

function onAuthSignedOut() {
    hideApp();
    stopNeckEffortMonitoring();
    updateNeckEffortUI(null);
    if (camera) { camera.stop(); camera = null; }
    if (mediaStream) { mediaStream.getTracks().forEach((track) => track.stop()); mediaStream = null; }
    if (audioStream) { audioStream.getTracks().forEach((track) => track.stop()); audioStream = null; }
    faceMesh = null;
    if (combinedSummary) combinedSummary.hidden = true;
    resultsPanel.hidden = true;
    featuresPanel.hidden = true;
    recommendationsPanel.hidden = true;
    if (progressPanel) progressPanel.hidden = true;
}

// ---- Start ----
// Don't auto-init; wait for auth. If already logged in, _restoreSession in
// auth.js will call onAuthReady which triggers init().

// ---- Tab Navigation ----
// The app used to be one long scroll with a floating dot navigator. Each stage
// of the flow is now its own tab, so nothing below the fold gets lost.
function activateTab(name) {
    const tabs = document.querySelectorAll("#app-tabs .tab");
    const panels = document.querySelectorAll(".tab-panel");
    if (!tabs.length) return;
    tabs.forEach((tab) => tab.classList.toggle("is-active", tab.dataset.tab === name));
    panels.forEach((panel) => panel.classList.toggle("is-active", panel.dataset.tab === name));
    syncTabEmptyStates();
    window.scrollTo({ top: 0, behavior: "smooth" });
}

/** Show a placeholder in any tab whose real content has not been produced yet. */
function syncTabEmptyStates() {
    const pairs = [
        ["results-empty", "results-panel"],
        ["tips-empty", "recommendations-panel"],
        ["progress-empty", "progress-panel"],
    ];
    for (const [emptyId, panelId] of pairs) {
        const empty = document.getElementById(emptyId);
        const panel = document.getElementById(panelId);
        if (empty && panel) empty.hidden = !panel.hidden;
    }
}

/** Mark a tab as holding something new until the user opens it. */
function markTabUpdated(name) {
    const tab = document.querySelector(`#app-tabs .tab[data-tab="${name}"]`);
    if (tab && !tab.classList.contains("is-active")) tab.classList.add("has-update");
}

(function initTabs() {
    const tabBar = document.getElementById("app-tabs");
    if (tabBar) {
        tabBar.addEventListener("click", (e) => {
            const tab = e.target.closest(".tab");
            if (!tab) return;
            tab.classList.remove("has-update");
            activateTab(tab.dataset.tab);
        });
    }

    // "Go to recording" buttons inside the empty states
    document.addEventListener("click", (e) => {
        const btn = e.target.closest("[data-go-tab]");
        if (btn) activateTab(btn.dataset.goTab);
    });

    const scrollTopBtn = document.getElementById("scroll-top-btn");
    if (scrollTopBtn) {
        scrollTopBtn.addEventListener("click", () => {
            window.scrollTo({ top: 0, behavior: "smooth" });
        });
        const toggleScrollBtn = () => {
            scrollTopBtn.classList.toggle("is-visible", window.scrollY > 400);
        };
        window.addEventListener("scroll", toggleScrollBtn, { passive: true });
        toggleScrollBtn();
    }

    syncTabEmptyStates();
})();

// ---- Auth form modes ----
// One card, two modes: the class-code field and the emphasised button follow
// the chosen mode instead of showing every field at once.
(function initAuthModes() {
    const tabs = document.getElementById("auth-mode-tabs");
    if (!tabs) return;
    const classField = document.getElementById("class-code-field");
    const loginButton = document.getElementById("login-btn");
    const registerButton = document.getElementById("register-btn");

    function setMode(mode) {
        const registering = mode === "register";
        tabs.querySelectorAll(".tab").forEach((tab) => {
            tab.classList.toggle("is-active", tab.dataset.mode === mode);
        });
        if (classField) classField.hidden = !registering;
        if (loginButton) {
            loginButton.hidden = registering;
            loginButton.className = "btn btn-primary btn-block";
        }
        if (registerButton) {
            registerButton.hidden = !registering;
            registerButton.className = registering
                ? "btn btn-primary btn-block"
                : "btn btn-secondary btn-block";
        }
        const heading = document.querySelector(".auth-heading");
        if (heading) heading.textContent = registering ? "יצירת חשבון" : "התחברות";
    }

    tabs.addEventListener("click", (e) => {
        const tab = e.target.closest(".tab");
        if (tab) setMode(tab.dataset.mode);
    });
    setMode("login");
})();
