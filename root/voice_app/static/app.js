// Voice Training App - Frontend Logic

const GRBAS_LABELS = {
    G: "Grade (overall)",
    R: "Roughness",
    B: "Breathiness",
    A: "Asthenia",
    S: "Strain",
    N: "Neck effort",
};

// Index = score: 0=normal(green), 1=slight(yellow), 2=moderate(orange), 3=severe(red)
const GRADE_COLORS = ["#3ddc84", "#ffb74d", "#ff8a50", "#ff4f5e"];

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
        setStatus("Camera and microphone ready.");
    } catch (err) {
        setStatus("Error: " + err.message, true);
        updateNeckEffortUI(null);
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
            `Jaw: ${metrics.jawOpenness.toFixed(1)}% | ` +
            `Lip Sym: ${metrics.lipSymmetry.toFixed(1)}% | ` +
            `Tension: ${metrics.facialTension.toFixed(1)}%`;
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
            reasons: ["Hold a relaxed neutral posture for calibration"],
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
        neckEffortMode.textContent = "Idle";
        neckEffortScore.textContent = "--";
        neckEffortScore.className = "neck-effort-score";
        neckEffortLabel.textContent = "Camera not started";
        neckEffortConfidence.textContent = "Confidence: --";
        neckEffortReasons.textContent = "Align your head, neck, and shoulders in frame.";
        return;
    }

    if (data.mode === "calibrating") {
        const progress = Math.round((data.calibration_progress || 0) * 100);
        neckEffortMode.textContent = "Calibrating";
        neckEffortScore.textContent = progress + "%";
        neckEffortScore.className = "neck-effort-score";
        neckEffortLabel.textContent = "Hold a neutral posture";
        neckEffortConfidence.textContent = "Confidence: " + Math.round((data.confidence || 0) * 100) + "%";
        neckEffortReasons.textContent = "Relax your shoulders and keep your neck centered until calibration finishes.";
        return;
    }

    if (data.mode === "running" && Number.isInteger(data.score)) {
        neckEffortMode.textContent = "Live";
        neckEffortScore.textContent = String(data.score);
        neckEffortScore.className = "neck-effort-score grade-" + data.score;
        neckEffortLabel.textContent = (data.score_label || "unknown").replace(/^./, (char) => char.toUpperCase());
        neckEffortConfidence.textContent = "Confidence: " + Math.round((data.confidence || 0) * 100) + "%";
        neckEffortReasons.textContent = formatNeckEffortReasons(data.reasons || []);
        return;
    }

    if (data.mode === "no_score") {
        neckEffortMode.textContent = "Adjust";
        neckEffortScore.textContent = data.last_valid_score ?? "--";
        neckEffortScore.className = Number.isInteger(data.last_valid_score)
            ? "neck-effort-score grade-" + data.last_valid_score
            : "neck-effort-score";
        neckEffortLabel.textContent = "Need better framing";
        neckEffortConfidence.textContent = "Confidence: " + Math.round((data.confidence || 0) * 100) + "%";
        neckEffortReasons.textContent = formatNeckEffortReasons(data.reasons || []);
        return;
    }

    neckEffortMode.textContent = "Error";
    neckEffortScore.textContent = "--";
    neckEffortScore.className = "neck-effort-score";
    neckEffortLabel.textContent = "Analysis unavailable";
    neckEffortConfidence.textContent = "Confidence: 0%";
    neckEffortReasons.textContent = formatNeckEffortReasons(data.reasons || []);
}

function formatNeckEffortReasons(reasons) {
    const labels = {
        head_forward_posture: "Head appears forward relative to shoulders.",
        neck_tilt: "Neck tilt is elevated.",
        shoulder_imbalance: "Shoulders look uneven.",
        raised_shoulders: "Shoulders appear raised.",
        head_instability: "Head position is unstable.",
        excess_mouth_opening: "Mouth opening is larger than expected.",
        head_rotation: "Face is turned away from frontal view.",
        minimal_visible_effort: "Minimal visible neck effort.",
        no_face_or_pose: "Keep your face and shoulders visible to the camera.",
        low_confidence_view: "Adjust framing and face the camera more directly.",
        neck_effort_unavailable: "Neck effort analysis is temporarily unavailable.",
    };
    if (!reasons || reasons.length === 0) {
        return "Align your head, neck, and shoulders in frame.";
    }
    return reasons.map((reason) => labels[reason] || reason.replaceAll("_", " ")).join(" ");
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
        setStatus("Microphone not available.", true);
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
    recordLabel.textContent = "Stop Recording";
    timerEl.hidden = false;
    setStatus("Recording...");

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
    recordLabel.textContent = "Start Recording";

    const audioContext = recordBtn._audioContext;
    const processor = recordBtn._processor;
    const source = recordBtn._source;
    const pcmChunks = recordBtn._pcmChunks;

    processor.disconnect();
    source.disconnect();

    // Convert PCM chunks to WAV blob
    const wavBlob = pcmToWav(pcmChunks, audioContext.sampleRate);
    audioContext.close();

    setStatus("Analyzing...");
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

        if (data.error) {
            setStatus("Error: " + data.error, true);
            return;
        }

        displayResults(data);
        saveToHistory(data);
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

    modelBadge.textContent = `Model: ${data.model_type}`;

    // Radar chart
    drawRadarChart(data.scores);

    // Features
    featuresGrid.innerHTML = "";
    const featureLabels = {
        f0_mean: "F0 Mean (Hz)",
        f0_std: "F0 Std Dev",
        f0_range: "F0 Range (Hz)",
        jitter_percent: "Jitter (%)",
        shimmer_percent: "Shimmer (%)",
        hnr: "HNR (dB)",
        spectral_centroid: "Spectral Centroid",
        spectral_tilt: "Spectral Tilt",
        hf_energy_ratio: "HF Energy Ratio",
        intensity_mean: "Intensity (dB)",
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
            title: "Low visible load across voice and neck",
            badge: "Stable",
            text: "The recording looks organized. Voice quality is close to normal and there is no clear sign of elevated neck effort.",
        },
        1: {
            title: "Mild load detected",
            badge: "Watch",
            text: "There are early signs of extra effort. Keep monitoring posture, airflow, and ease during practice.",
        },
        2: {
            title: "Moderate combined load",
            badge: "Attention",
            text: "The analysis suggests noticeable vocal or cervical effort. Focus on posture release, shoulder relaxation, and efficient phonation.",
        },
        3: {
            title: "High combined load",
            badge: "High",
            text: "The sample shows strong vocal and/or neck effort. Practice should prioritize unloading tension before increasing vocal demand.",
        },
    };

    const summary = summaryByLevel[overallLevel] || summaryByLevel[0];
    const neckText = neckEffort === null
        ? ""
        : `Neck effort scored ${neckEffort}/3${data.neck_effort?.score_label ? ` (${data.neck_effort.score_label})` : ""}.`;

    combinedSummary.hidden = false;
    combinedSummaryTitle.textContent = summary.title;
    combinedSummaryBadge.textContent = summary.badge;
    combinedSummaryBadge.className = "combined-summary-badge level-" + overallLevel;
    combinedSummaryText.textContent = [
        summary.text,
        `Grade is ${voiceGrade}/3, strain is ${strain}/3.`,
        neckText,
    ].filter(Boolean).join(" ");

    combinedSummaryMetrics.innerHTML = "";
    const chips = [
        { label: "Voice Grade", value: `${voiceGrade}/3`, score: voiceGrade },
        { label: "Strain", value: `${strain}/3`, score: strain },
        {
            label: "Model",
            value: data.model_type || "unknown",
            score: null,
        },
    ];

    if (neckEffort !== null) {
        chips.splice(2, 0, {
            label: "Neck Effort",
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
const MEASURE_NAMES = { G: "Grade", R: "Roughness", B: "Breathiness", A: "Asthenia", S: "Strain", N: "Neck Effort" };

function displayRecommendations(recommendations) {
    if (!recommendations || recommendations.length === 0) {
        recommendationsPanel.hidden = true;
        return;
    }

    recommendationsPanel.hidden = false;
    recommendationsList.innerHTML = "";

    for (const rec of recommendations) {
        const group = document.createElement("div");
        group.className = "rec-group";

        const header = document.createElement("div");
        header.className = "rec-header";
        header.innerHTML = `
            <span class="rec-measure grade-${rec.score}">${rec.measure} – ${MEASURE_NAMES[rec.measure] || rec.measure}</span>
            <span class="rec-score">Score: ${rec.score}/3</span>
            <span class="rec-priority">Priority #${rec.priority_rank}</span>
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
        ctx.strokeStyle = "rgba(255,255,255,0.1)";
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    // Draw axes
    for (let i = 0; i < n; i++) {
        const angle = startAngle + i * angleStep;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + maxR * Math.cos(angle), cy + maxR * Math.sin(angle));
        ctx.strokeStyle = "rgba(255,255,255,0.15)";
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
    ctx.fillStyle = "rgba(79,140,255,0.25)";
    ctx.fill();
    ctx.strokeStyle = "#4f8cff";
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
        ctx.fillStyle = "#e8eaf0";
        ctx.font = "bold 13px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(dims[i], lx, ly);
    }
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
            '<p class="empty-state">No recordings yet. Start by recording your voice above.</p>';
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
        `;
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
        }
    } catch (err) {
        console.warn("Cloud history load failed:", err);
    }
}

function renderHistory(entries) {
    if (!entries || entries.length === 0) {
        historyList.innerHTML =
            '<p class="empty-state">No recordings yet. Start by recording your voice above.</p>';
        clearHistoryBtn.hidden = true;
        return;
    }

    clearHistoryBtn.hidden = false;
    historyList.innerHTML = "";

    for (const entry of entries) {
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
        `;
        historyList.appendChild(item);
    }
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
}

// ---- Start ----
// Don't auto-init; wait for auth. If already logged in, _restoreSession in
// auth.js will call onAuthReady which triggers init().
