// Voice Training App - Frontend Logic

const GRBAS_LABELS = {
    G: "Grade (overall)",
    R: "Roughness",
    B: "Breathiness",
    A: "Asthenia",
    S: "Strain",
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

// ---- DOM refs ----
const video = document.getElementById("video");
const faceCanvas = document.getElementById("face-canvas");
const faceCtx = faceCanvas.getContext("2d");
const faceMetrics = document.getElementById("face-metrics");
const recordBtn = document.getElementById("record-btn");
const recordLabel = recordBtn.querySelector(".record-label");
const timerEl = document.getElementById("recording-timer");
const statusEl = document.getElementById("status");
const resultsPanel = document.getElementById("results-panel");
const scoresList = document.getElementById("scores-list");
const radarCanvas = document.getElementById("radar-chart");
const modelBadge = document.getElementById("model-badge");
const featuresPanel = document.getElementById("features-panel");
const featuresGrid = document.getElementById("features-grid");
const historyList = document.getElementById("history-list");
const clearHistoryBtn = document.getElementById("clear-history-btn");

// ---- Initialize Camera + Mic ----
async function init() {
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
        setStatus("Camera and microphone ready.");
    } catch (err) {
        setStatus("Error: " + err.message, true);
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

    // Scores list
    scoresList.innerHTML = "";
    for (const dim of "GRBAS") {
        const score = data.scores[dim];
        const row = document.createElement("div");
        row.className = "score-row";
        row.innerHTML = `
            <span class="score-label">${dim}</span>
            <div class="score-bar-bg">
                <div class="score-bar" style="width:${(score / 3) * 100}%;background:${GRADE_COLORS[score]}"></div>
            </div>
            <span class="score-value grade-${score}">${score}</span>
            <span class="score-desc">${GRBAS_LABELS[dim]}</span>
        `;
        scoresList.appendChild(row);
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
function saveToHistory(data) {
    const history = JSON.parse(localStorage.getItem("grbas_history") || "[]");
    history.unshift({
        timestamp: new Date().toISOString(),
        scores: data.scores,
        features: data.features,
    });
    // Keep last 50 entries
    if (history.length > 50) history.length = 50;
    localStorage.setItem("grbas_history", JSON.stringify(history));
    loadHistory();
}

function loadHistory() {
    const history = JSON.parse(localStorage.getItem("grbas_history") || "[]");

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

        item.innerHTML = `
            <span class="history-time">${timeStr}</span>
            <div class="history-scores">${scoresHtml}</div>
        `;
        historyList.appendChild(item);
    }
}

clearHistoryBtn.addEventListener("click", () => {
    localStorage.removeItem("grbas_history");
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

        item.innerHTML = `
            <span class="history-time">${timeStr}</span>
            <div class="history-scores">${scoresHtml}</div>
        `;
        historyList.appendChild(item);
    }
}

// Auth hooks called by auth.js
function onAuthReady(user) {
    loadCloudHistory();
}

function onAuthSignedOut() {
    loadHistory();
}

// ---- Start ----
init();
