// Voice Training App - Frontend Logic

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
let neckCaptureCanvas = null;

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
        startNeckEffortMonitoring();
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
            refineLandmarks: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5,
        });

        faceMesh.onResults(onFaceResults);

        camera = new Camera(video, {
            onFrame: async () => {
                await faceMesh.send({ image: video });
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
            mode: "calibrating",
            calibrated: false,
            calibration_progress: 0,
            score: null,
            score_label: null,
            confidence: 0,
            reasons: ["החזיקו תנוחה נינוחה לכיול"],
        });
    } catch (err) {
        console.warn("Neck effort reset failed:", err);
    }
}

function startNeckEffortMonitoring() {
    stopNeckEffortMonitoring();
    neckCaptureCanvas = document.createElement("canvas");
    neckCaptureCanvas.width = 320;
    neckCaptureCanvas.height = 240;
    neckEffortInterval = window.setInterval(captureNeckEffortFrame, 900);
}

function stopNeckEffortMonitoring() {
    if (neckEffortInterval) {
        window.clearInterval(neckEffortInterval);
        neckEffortInterval = null;
    }
    neckEffortRequestInFlight = false;
}

async function captureNeckEffortFrame() {
    if (neckEffortRequestInFlight || !neckCaptureCanvas || !video.videoWidth || !video.videoHeight) {
        return;
    }
    if (typeof getAuthHeaders !== "function") return;
    const headers = getAuthHeaders();
    if (!headers.Authorization) return;

    neckEffortRequestInFlight = true;
    try {
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

function updateNeckEffortUI(data) {
    if (!data) {
        neckEffortMode.textContent = "לא פעיל";
        neckEffortScore.textContent = "--";
        neckEffortScore.className = "neck-effort-score";
        neckEffortLabel.textContent = "המצלמה לא הופעלה";
        neckEffortConfidence.textContent = "ביטחון: --";
        neckEffortReasons.textContent = "יישרו את הראש, הצוואר והכתפיים מול המצלמה.";
        return;
    }

    if (data.mode === "calibrating") {
        const progress = Math.round((data.calibration_progress || 0) * 100);
        neckEffortMode.textContent = "כיול";
        neckEffortScore.textContent = progress + "%";
        neckEffortScore.className = "neck-effort-score";
        neckEffortLabel.textContent = "החזיקו תנוחה נינוחה";
        neckEffortConfidence.textContent = "ביטחון: " + Math.round((data.confidence || 0) * 100) + "%";
        neckEffortReasons.textContent = "הרפו את הכתפיים ושמרו על הצוואר במרכז עד סיום הכיול.";
        return;
    }

    if (data.mode === "running" && Number.isInteger(data.score)) {
        neckEffortMode.textContent = "פעיל";
        neckEffortScore.textContent = String(data.score);
        neckEffortScore.className = "neck-effort-score grade-" + data.score;
        neckEffortLabel.textContent = (data.score_label || "לא ידוע").replace(/^./, (char) => char.toUpperCase());
        neckEffortConfidence.textContent = "ביטחון: " + Math.round((data.confidence || 0) * 100) + "%";
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
        neckEffortConfidence.textContent = "ביטחון: " + Math.round((data.confidence || 0) * 100) + "%";
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

function formatNeckEffortReasons(reasons) {
    const labels = {
        head_forward_posture: "הראש נראה קדימה ביחס לכתפיים.",
        neck_tilt: "הטיית הצוואר מוגבהת.",
        shoulder_imbalance: "הכתפיים נראות לא מאוזנות.",
        raised_shoulders: "הכתפיים נראות מורמות.",
        head_instability: "מצב הראש לא יציב.",
        excess_mouth_opening: "פתיחת הפה גדולה מהצפוי.",
        head_rotation: "הפנים מופנות הצידה מהמצלמה.",
        minimal_visible_effort: "מאמץ צוואר מינימלי.",
        no_face_or_pose: "ודאו שהפנים והכתפיים נראים למצלמה.",
        low_confidence_view: "התאימו את המיסגור והפנו ישירות למצלמה.",
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
recordBtn.addEventListener("click", toggleRecording);

function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

function startRecording() {
    if (!audioStream) {
        setStatus("המיקרופון לא זמין.", true);
        return;
    }

    audioChunks = [];

    // Use AudioContext to capture WAV-compatible audio
    const audioContext = new AudioContext({ sampleRate: 44100 });
    const source = audioContext.createMediaStreamSource(audioStream);
    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    const pcmChunks = [];

    processor.onaudioprocess = (e) => {
        if (isRecording) {
            const channelData = e.inputBuffer.getChannelData(0);
            pcmChunks.push(new Float32Array(channelData));
        }
    };

    source.connect(processor);
    processor.connect(audioContext.destination);

    isRecording = true;
    recordingStart = Date.now();
    recordBtn.classList.add("recording");
    recordLabel.textContent = "עצור הקלטה";
    timerEl.hidden = false;
    setStatus("מקליט...");

    timerInterval = setInterval(() => {
        const elapsed = ((Date.now() - recordingStart) / 1000).toFixed(1);
        timerEl.textContent = elapsed + "s";
    }, 100);

    // Store references for stop
    recordBtn._audioContext = audioContext;
    recordBtn._processor = processor;
    recordBtn._source = source;
    recordBtn._pcmChunks = pcmChunks;
}

function stopRecording() {
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
    sendAudio(wavBlob);
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

    try {
        const headers = typeof getAuthHeaders === "function" ? getAuthHeaders() : {};
        const resp = await fetch("/api/analyze", {
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
        setStatus("Analysis complete.");
    } catch (err) {
        setStatus("Error: " + err.message, true);
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

    const neckScore = data.neck_effort && Number.isInteger(data.neck_effort.score)
        ? data.neck_effort.score
        : null;
    if (neckScore !== null) {
        scoresList.appendChild(createScoreRow("N", neckScore));
    }

    modelBadge.textContent = `מודל: ${data.model_type}`;

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
    const neckEffort = data.neck_effort && Number.isInteger(data.neck_effort.score)
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
            text: "הניתוח מראה מאמץ קולי או צווארי מורגש. התמקדו בשחרור תנוחה, הרפיית כתפיים והפקה יעילה.",
        },
        3: {
            title: "עומס משולב גבוה",
            badge: "גבוה",
            text: "הדגימה מראה מאמץ קולי ו/או צווארי חזק. מומלץ לתעדף הפחתת מתח לפני הגברת העומס הקולי.",
        },
    };

    const summary = summaryByLevel[overallLevel] || summaryByLevel[0];
    const neckText = neckEffort === null
        ? ""
        : `מאמץ צוואר ${neckEffort}/3${data.neck_effort?.score_label ? ` (${data.neck_effort.score_label})` : ""}.`;

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
    const prevNeck = prev.neck_effort && Number.isInteger(prev.neck_effort.score) ? prev.neck_effort.score : null;
    const curNeck = data.neck_effort && Number.isInteger(data.neck_effort.score) ? data.neck_effort.score : null;
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
        if (entry.neck_effort && Number.isInteger(entry.neck_effort.score)) {
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
        if (entry.neck_effort && Number.isInteger(entry.neck_effort.score)) {
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
function renderProgressChart(entries) {
    if (!progressPanel || !progressChart || !entries || entries.length < 2) {
        if (progressPanel) progressPanel.hidden = true;
        return;
    }

    progressPanel.hidden = false;

    // Defer drawing to next frame so the browser can layout the now-visible panel
    requestAnimationFrame(() => _drawProgressChart(entries));
}

function _drawProgressChart(entries) {
    // Sort oldest first
    const sorted = [...entries].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
    const dims = ["G", "R", "B", "A", "S"];
    const dimColors = {
        G: "#4a7cff",
        R: "#f97316",
        B: "#a855f7",
        A: "#f59e0b",
        S: "#ef4444",
    };

    const canvas = progressChart;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const container = canvas.parentElement;
    const cssW = (container && container.clientWidth > 0) ? container.clientWidth : 760;
    const cssH = 300;
    canvas.style.width = cssW + "px";
    canvas.style.height = cssH + "px";
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    ctx.scale(dpr, dpr);

    const pad = { top: 30, right: 20, bottom: 50, left: 40 };
    const w = cssW - pad.left - pad.right;
    const h = cssH - pad.top - pad.bottom;
    const n = sorted.length;

    ctx.clearRect(0, 0, cssW, cssH);

    // Background grid
    ctx.strokeStyle = "rgba(0,0,0,0.06)";
    ctx.lineWidth = 1;
    for (let level = 0; level <= 3; level++) {
        const y = pad.top + h - (level / 3) * h;
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
    const maxLabels = Math.min(n, 10);
    const step = Math.max(1, Math.floor(n / maxLabels));
    for (let i = 0; i < n; i += step) {
        const x = pad.left + (i / Math.max(1, n - 1)) * w;
        const date = new Date(sorted[i].timestamp);
        const label = date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
        ctx.fillText(label, x, pad.top + h + 8);
    }

    // Draw lines for each dimension
    for (const dim of dims) {
        ctx.beginPath();
        ctx.strokeStyle = dimColors[dim];
        ctx.lineWidth = 2.5;
        ctx.lineJoin = "round";

        for (let i = 0; i < n; i++) {
            const x = pad.left + (i / Math.max(1, n - 1)) * w;
            const score = sorted[i].scores[dim] ?? 0;
            const y = pad.top + h - (score / 3) * h;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Draw dots
        for (let i = 0; i < n; i++) {
            const x = pad.left + (i / Math.max(1, n - 1)) * w;
            const score = sorted[i].scores[dim] ?? 0;
            const y = pad.top + h - (score / 3) * h;
            ctx.beginPath();
            ctx.arc(x, y, 3.5, 0, Math.PI * 2);
            ctx.fillStyle = dimColors[dim];
            ctx.fill();
        }
    }

    // Legend
    const legendY = pad.top + h + 28;
    let legendX = pad.left;
    ctx.font = "bold 12px sans-serif";
    for (const dim of dims) {
        ctx.fillStyle = dimColors[dim];
        ctx.beginPath();
        ctx.arc(legendX + 6, legendY, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#e8eaf0";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(dim, legendX + 15, legendY);
        legendX += 50;
    }

    // Progress summary
    renderProgressSummary(sorted, dims);
}

function renderProgressSummary(sorted, dims) {
    if (!progressSummary || sorted.length < 2) return;

    const first = sorted[0];
    const last = sorted[sorted.length - 1];
    const dimNames = { G: "דירוג", R: "צרידות", B: "נשיפות", A: "חולשה", S: "מאמץ" };

    let html = "";
    for (const dim of dims) {
        const start = first.scores[dim] ?? 0;
        const end = last.scores[dim] ?? 0;
        const diff = end - start;

        let arrow, cls;
        if (diff < 0) {
            arrow = "\u2193"; // ↓ improved (lower is better)
            cls = "progress-improved";
        } else if (diff > 0) {
            arrow = "\u2191"; // ↑ worsened
            cls = "progress-worsened";
        } else {
            arrow = "\u2192"; // → stable
            cls = "progress-stable";
        }

        html += `<div class="progress-chip ${cls}">
            <span class="progress-chip-dim">${dim}</span>
            <span class="progress-chip-label">${dimNames[dim]}</span>
            <span class="progress-chip-change">${start} ${arrow} ${end}</span>
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

// ---- Scroll Navigation ----
(function initScrollNav() {
    const scrollTopBtn = document.getElementById("scroll-top-btn");
    const sectionNav = document.getElementById("section-nav");

    if (scrollTopBtn) {
        scrollTopBtn.addEventListener("click", () => {
            window.scrollTo({ top: 0, behavior: "smooth" });
        });
    }

    // Section navigator dot clicks
    if (sectionNav) {
        sectionNav.addEventListener("click", (e) => {
            const btn = e.target.closest(".snav-dot");
            if (!btn) return;
            const targetId = "section-" + btn.dataset.target;
            const el = document.getElementById(targetId);
            if (el) {
                const navHeight = document.querySelector(".top-nav")?.offsetHeight || 0;
                const top = el.getBoundingClientRect().top + window.scrollY - navHeight - 16;
                window.scrollTo({ top, behavior: "smooth" });
            }
        });
    }
})();
