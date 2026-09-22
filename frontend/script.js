/**
 * Domain-Adaptive Speech Emotion Recognition Frontend Logic
 * Vanilla JavaScript client communicating with FastAPI backend
 * Includes Client-Side Audio Recording & 16kHz PCM WAV Encoding
 */

const API_BASE = "http://127.0.0.1:8000";

// Emotion Metadata: Emojis and Color Accents
const EMOTION_CONFIG = {
    happy: { emoji: "😄", color: "#f59e0b" },
    sad: { emoji: "😢", color: "#38bdf8" },
    angry: { emoji: "😠", color: "#f43f5e" },
    fearful: { emoji: "😨", color: "#a855f7" },
    disgust: { emoji: "🤢", color: "#10b981" },
    neutral: { emoji: "😐", color: "#94a3b8" },
};

// DOM Elements: Header & Tabs
const apiStatusBadge = document.getElementById("apiStatusBadge");
const apiStatusText = document.getElementById("apiStatusText");
const tabUpload = document.getElementById("tabUpload");
const tabRecord = document.getElementById("tabRecord");
const panelUpload = document.getElementById("panelUpload");
const panelRecord = document.getElementById("panelRecord");

// DOM Elements: File Upload Dropzone
const dropzone = document.getElementById("dropzone");
const audioFileInput = document.getElementById("audioFileInput");
const dropzoneContent = document.getElementById("dropzoneContent");
const fileInfoBox = document.getElementById("fileInfoBox");
const selectedFileName = document.getElementById("selectedFileName");
const selectedFileSize = document.getElementById("selectedFileSize");
const audioPreview = document.getElementById("audioPreview");
const btnClearFile = document.getElementById("btnClearFile");

// DOM Elements: Microphone Recorder
const recordStandby = document.getElementById("recordStandby");
const recordActive = document.getElementById("recordActive");
const recordPreview = document.getElementById("recordPreview");
const btnStartRecord = document.getElementById("btnStartRecord");
const btnStopRecord = document.getElementById("btnStopRecord");
const btnRerecord = document.getElementById("btnRerecord");
const recordTimer = document.getElementById("recordTimer");
const recordedFileName = document.getElementById("recordedFileName");
const recordedFileSize = document.getElementById("recordedFileSize");
const recordedAudioPreview = document.getElementById("recordedAudioPreview");

// DOM Elements: Controls & Results
const coralToggle = document.getElementById("coralToggle");
const btnAnalyze = document.getElementById("btnAnalyze");
const btnSpinner = document.getElementById("btnSpinner");
const errorAlert = document.getElementById("errorAlert");
const errorMessage = document.getElementById("errorMessage");

const emptyState = document.getElementById("emptyState");
const loadingState = document.getElementById("loadingState");
const resultContent = document.getElementById("resultContent");

const resEmotionEmoji = document.getElementById("resEmotionEmoji");
const resEmotionTitle = document.getElementById("resEmotionTitle");
const resConfidence = document.getElementById("resConfidence");
const resCoralBanner = document.getElementById("resCoralBanner");
const resCoralBadge = document.getElementById("resCoralBadge");
const resCoralDesc = document.getElementById("resCoralDesc");
const probabilityBars = document.getElementById("probabilityBars");
const resFileName = document.getElementById("resFileName");

// Application State
let currentSelectedFile = null;
let uploadedBlobUrl = null;
let recordedBlobUrl = null;

// MediaRecorder State
let mediaRecorder = null;
let audioStream = null;
let audioChunks = [];
let recordingStartTime = 0;
let recordingTimerInterval = null;
const MAX_RECORDING_SECONDS = 10;

// Initialize
document.addEventListener("DOMContentLoaded", () => {
    checkBackendHealth();
    setupEventListeners();
});

/**
 * Check backend health status on load
 */
async function checkBackendHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`, { method: "GET" });
        if (response.ok) {
            const data = await response.json();
            if (data.status === "healthy") {
                apiStatusBadge.className = "api-status-badge online";
                apiStatusText.textContent = "Backend Online (Model Ready)";
                return;
            }
        }
        setBackendOffline();
    } catch (err) {
        setBackendOffline();
    }
}

function setBackendOffline() {
    apiStatusBadge.className = "api-status-badge offline";
    apiStatusText.textContent = "Backend Offline";
}

/**
 * Setup Event Listeners
 */
function setupEventListeners() {
    // Mode Switcher Tabs
    tabUpload.addEventListener("click", () => switchInputMode("upload"));
    tabRecord.addEventListener("click", () => switchInputMode("record"));

    // File Upload Input
    audioFileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    // Drag & Drop
    ["dragenter", "dragover"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("dragover");
        });
    });

    dropzone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        if (dt && dt.files && dt.files.length > 0) {
            handleFileUpload(dt.files[0]);
        }
    });

    // Clear Uploaded File
    btnClearFile.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        clearUploadedFile();
    });

    // Microphone Recording Controls
    btnStartRecord.addEventListener("click", startMicrophoneRecording);
    btnStopRecord.addEventListener("click", stopMicrophoneRecording);
    btnRerecord.addEventListener("click", resetMicrophoneRecorder);

    // Analyze Action
    btnAnalyze.addEventListener("click", handleAnalyzeEmotion);
}

/**
 * Switch between "upload" and "record" input modes
 */
function switchInputMode(mode) {
    hideError();
    if (mode === "upload") {
        tabUpload.classList.add("active");
        tabRecord.classList.remove("active");
        panelUpload.classList.remove("hidden");
        panelRecord.classList.add("hidden");

        // If an active recording is in progress, stop it
        if (mediaRecorder && mediaRecorder.state === "recording") {
            stopMicrophoneRecording();
        }
    } else {
        tabRecord.classList.add("active");
        tabUpload.classList.remove("active");
        panelRecord.classList.remove("hidden");
        panelUpload.classList.add("hidden");
    }
}

/**
 * Handles uploaded WAV file
 */
function handleFileUpload(file) {
    hideError();

    if (!file.name.toLowerCase().endsWith(".wav")) {
        showError("Invalid file type. Please select a valid WAV (.wav) audio file.");
        clearUploadedFile();
        return;
    }

    currentSelectedFile = file;

    const sizeKB = (file.size / 1024).toFixed(1);
    const sizeFormatted = sizeKB > 1024 ? `${(sizeKB / 1024).toFixed(2)} MB` : `${sizeKB} KB`;

    selectedFileName.textContent = file.name;
    selectedFileSize.textContent = sizeFormatted;

    if (uploadedBlobUrl) {
        URL.revokeObjectURL(uploadedBlobUrl);
    }
    uploadedBlobUrl = URL.createObjectURL(file);
    audioPreview.src = uploadedBlobUrl;

    dropzoneContent.classList.add("hidden");
    fileInfoBox.classList.remove("hidden");
    btnAnalyze.disabled = false;
}

function clearUploadedFile() {
    if (currentSelectedFile && currentSelectedFile.name !== "recorded_voice.wav") {
        currentSelectedFile = null;
    }
    audioFileInput.value = "";

    if (uploadedBlobUrl) {
        URL.revokeObjectURL(uploadedBlobUrl);
        uploadedBlobUrl = null;
    }
    audioPreview.src = "";

    dropzoneContent.classList.remove("hidden");
    fileInfoBox.classList.add("hidden");
    
    if (!currentSelectedFile) {
        btnAnalyze.disabled = true;
    }
    hideError();
}

/**
 * Microphone Recording: Start
 */
async function startMicrophoneRecording() {
    hideError();

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showError("Microphone recording is not supported in this browser.");
        return;
    }

    try {
        audioStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true,
            },
        });

        audioChunks = [];
        
        // Find supported MIME type
        const mimeTypes = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg", "audio/mp4"];
        let selectedMime = "";
        for (const mime of mimeTypes) {
            if (MediaRecorder.isTypeSupported(mime)) {
                selectedMime = mime;
                break;
            }
        }

        const options = selectedMime ? { mimeType: selectedMime } : {};
        mediaRecorder = new MediaRecorder(audioStream, options);

        mediaRecorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = processRecordedAudio;

        mediaRecorder.start(100);
        recordingStartTime = Date.now();

        // UI Updates for Active Recording
        recordStandby.classList.add("hidden");
        recordPreview.classList.add("hidden");
        recordActive.classList.remove("hidden");
        btnAnalyze.disabled = true;

        updateTimerDisplay();
        recordingTimerInterval = setInterval(() => {
            const elapsed = (Date.now() - recordingStartTime) / 1000;
            updateTimerDisplay();

            if (elapsed >= MAX_RECORDING_SECONDS) {
                stopMicrophoneRecording();
                showError("Maximum recording limit (10s) reached.");
            }
        }, 200);

    } catch (err) {
        console.error("Microphone access error:", err);
        if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
            showError("Microphone permission denied. Please allow microphone access in your browser to record speech.");
        } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
            showError("No microphone detected. Please connect an audio input device.");
        } else {
            showError(`Microphone recording error: ${err.message || err.name}`);
        }
        resetMicrophoneRecorder();
    }
}

function updateTimerDisplay() {
    const elapsedSec = Math.floor((Date.now() - recordingStartTime) / 1000);
    const clamped = Math.min(elapsedSec, MAX_RECORDING_SECONDS);
    const format = (s) => (s < 10 ? `0${s}` : `${s}`);
    recordTimer.textContent = `00:${format(clamped)} / 00:${format(MAX_RECORDING_SECONDS)}`;
}

/**
 * Microphone Recording: Stop
 */
function stopMicrophoneRecording() {
    if (recordingTimerInterval) {
        clearInterval(recordingTimerInterval);
        recordingTimerInterval = null;
    }

    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
    }

    if (audioStream) {
        audioStream.getTracks().forEach((track) => track.stop());
        audioStream = null;
    }
}

/**
 * Convert Browser Recorded Blob to Standard 16 kHz 16-bit Mono PCM WAV
 */
async function processRecordedAudio() {
    const elapsedSec = (Date.now() - recordingStartTime) / 1000;

    if (elapsedSec < 0.5) {
        showError("Recording too short (< 0.5s). Please hold and speak for at least 1-2 seconds.");
        resetMicrophoneRecorder();
        return;
    }

    try {
        const rawBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || "audio/webm" });
        const arrayBuffer = await rawBlob.arrayBuffer();

        // Decode using Web Audio API AudioContext
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        const audioCtx = new AudioContextClass();
        const decodedBuffer = await audioCtx.decodeAudioData(arrayBuffer);

        // Convert / downsample to 16 kHz Mono
        const targetSampleRate = 16000;
        const { samples, sampleRate } = resampleAndDownmixToMono(decodedBuffer, targetSampleRate);

        // Encode authentic 16-bit PCM RIFF/WAVE file
        const wavBlob = encodePCM16Wav(samples, sampleRate);
        const wavFile = new File([wavBlob], "recorded_voice.wav", {
            type: "audio/wav",
            lastModified: Date.now(),
        });

        currentSelectedFile = wavFile;

        // Update preview player
        if (recordedBlobUrl) {
            URL.revokeObjectURL(recordedBlobUrl);
        }
        recordedBlobUrl = URL.createObjectURL(wavBlob);
        recordedAudioPreview.src = recordedBlobUrl;

        const sizeKB = (wavBlob.size / 1024).toFixed(1);
        recordedFileName.textContent = "recorded_voice.wav";
        recordedFileSize.textContent = `${elapsedSec.toFixed(1)}s • ${sizeKB} KB (16kHz PCM WAV)`;

        recordActive.classList.add("hidden");
        recordStandby.classList.add("hidden");
        recordPreview.classList.remove("hidden");
        btnAnalyze.disabled = false;

        audioCtx.close();
    } catch (err) {
        console.error("WAV conversion error:", err);
        showError(`Failed to convert recorded audio: ${err.message || err}`);
        resetMicrophoneRecorder();
    }
}

function resetMicrophoneRecorder() {
    stopMicrophoneRecording();

    recordActive.classList.add("hidden");
    recordPreview.classList.add("hidden");
    recordStandby.classList.remove("hidden");

    if (currentSelectedFile && currentSelectedFile.name === "recorded_voice.wav") {
        currentSelectedFile = null;
        btnAnalyze.disabled = true;
    }

    if (recordedBlobUrl) {
        URL.revokeObjectURL(recordedBlobUrl);
        recordedBlobUrl = null;
    }
    recordedAudioPreview.src = "";
}

/**
 * Resamples and mixes multi-channel AudioBuffer to single-channel (mono) Float32Array
 */
function resampleAndDownmixToMono(audioBuffer, targetSampleRate = 16000) {
    const numChannels = audioBuffer.numberOfChannels;
    const originalLength = audioBuffer.length;
    const originalSampleRate = audioBuffer.sampleRate;

    // 1. Downmix to mono
    const monoData = new Float32Array(originalLength);
    if (numChannels === 1) {
        monoData.set(audioBuffer.getChannelData(0));
    } else {
        for (let ch = 0; ch < numChannels; ch++) {
            const chData = audioBuffer.getChannelData(ch);
            for (let i = 0; i < originalLength; i++) {
                monoData[i] += chData[i] / numChannels;
            }
        }
    }

    if (originalSampleRate === targetSampleRate) {
        return { samples: monoData, sampleRate: targetSampleRate };
    }

    // 2. Resample via linear interpolation
    const ratio = originalSampleRate / targetSampleRate;
    const newLength = Math.round(originalLength / ratio);
    const resampledData = new Float32Array(newLength);

    for (let i = 0; i < newLength; i++) {
        const originalPos = i * ratio;
        const index = Math.floor(originalPos);
        const decimal = originalPos - index;
        const s1 = monoData[index] || 0;
        const s2 = monoData[index + 1] !== undefined ? monoData[index + 1] : s1;
        resampledData[i] = s1 + decimal * (s2 - s1);
    }

    return { samples: resampledData, sampleRate: targetSampleRate };
}

/**
 * Encodes Float32Array PCM samples into a 16-bit Mono RIFF/WAVE Blob
 */
function encodePCM16Wav(samples, sampleRate = 16000) {
    const numChannels = 1;
    const bytesPerSample = 2; // 16-bit
    const blockAlign = numChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = samples.length * bytesPerSample;
    const bufferSize = 44 + dataSize;

    const buffer = new ArrayBuffer(bufferSize);
    const view = new DataView(buffer);

    // RIFF chunk descriptor
    writeAsciiString(view, 0, "RIFF");
    view.setUint32(4, 36 + dataSize, true);
    writeAsciiString(view, 8, "WAVE");

    // fmt sub-chunk
    writeAsciiString(view, 12, "fmt ");
    view.setUint32(16, 16, true);          // Subchunk1Size (16 for PCM)
    view.setUint16(20, 1, true);           // AudioFormat (1 = PCM)
    view.setUint16(22, numChannels, true); // NumChannels (1)
    view.setUint32(24, sampleRate, true);  // SampleRate (16000)
    view.setUint32(28, byteRate, true);    // ByteRate (32000)
    view.setUint16(32, blockAlign, true);  // BlockAlign (2)
    view.setUint16(34, 16, true);          // BitsPerSample (16)

    // data sub-chunk
    writeAsciiString(view, 36, "data");
    view.setUint32(40, dataSize, true);

    // Write 16-bit signed PCM samples with clipping [-1.0, 1.0] -> [-32768, 32767]
    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
        let s = Math.max(-1, Math.min(1, samples[i]));
        let val = s < 0 ? s * 0x8000 : s * 0x7FFF;
        view.setInt16(offset, val, true);
        offset += 2;
    }

    return new Blob([view], { type: "audio/wav" });
}

function writeAsciiString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

/**
 * Executes Emotion Analysis via FastAPI POST /predict
 */
async function handleAnalyzeEmotion() {
    if (!currentSelectedFile) {
        showError("Please select a WAV file or record microphone audio first.");
        return;
    }

    hideError();
    setLoading(true);

    const useCoral = coralToggle.checked;
    const formData = new FormData();
    formData.append("file", currentSelectedFile, currentSelectedFile.name || "audio.wav");

    try {
        const url = `${API_BASE}/predict?use_coral=${useCoral}`;
        const response = await fetch(url, {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            const errorMsg = data.detail || `Server error (${response.status})`;
            showError(errorMsg);
            setLoading(false);
            return;
        }

        renderPredictionResults(data);
    } catch (err) {
        console.error("Inference request failed:", err);
        showError(
            "Could not connect to FastAPI backend at http://127.0.0.1:8000. Please ensure the backend server is running."
        );
    } finally {
        setLoading(false);
    }
}

/**
 * Renders prediction payload and updates UI cards
 */
function renderPredictionResults(data) {
    const emotion = (data.predicted_emotion || "").toLowerCase();
    const config = EMOTION_CONFIG[emotion] || { emoji: "🎭", color: "#6366f1" };

    // Header
    resEmotionEmoji.textContent = config.emoji;
    resEmotionTitle.textContent = emotion;
    resEmotionTitle.style.color = config.color;

    const confPct = ((data.confidence || 0) * 100).toFixed(1);
    resConfidence.textContent = `${confPct}%`;
    resFileName.textContent = data.filename || currentSelectedFile.name;

    // CORAL Banner
    if (data.coral_applied) {
        resCoralBanner.className = "coral-status-banner";
        resCoralBadge.textContent = "CORAL APPLIED";
        resCoralDesc.textContent = "Target speech features aligned using pre-fitted covariance transform.";
    } else {
        resCoralBanner.className = "coral-status-banner no-coral";
        resCoralBadge.textContent = "BASELINE (NO CORAL)";
        resCoralDesc.textContent = "Raw unadapted MFCC features evaluated directly on source classifier.";
    }

    // Probability Bars
    renderProbabilityBars(data.probabilities || {}, emotion);

    // Switch views
    emptyState.classList.add("hidden");
    loadingState.classList.add("hidden");
    resultContent.classList.remove("hidden");
}

/**
 * Dynamically builds animated probability bars
 */
function renderProbabilityBars(probabilities, winnerEmotion) {
    probabilityBars.innerHTML = "";

    const sortedEntries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

    sortedEntries.forEach(([emoKey, probValue]) => {
        const emo = emoKey.toLowerCase();
        const config = EMOTION_CONFIG[emo] || { emoji: "•", color: "#6366f1" };
        const percent = (probValue * 100).toFixed(1);
        const isWinner = emo === winnerEmotion;

        const row = document.createElement("div");
        row.className = `prob-row ${isWinner ? "winner" : ""}`;

        row.innerHTML = `
            <div class="prob-row-header">
                <span class="prob-name">
                    <span>${config.emoji}</span>
                    <span>${emo}</span>
                </span>
                <span class="prob-percent">${percent}%</span>
            </div>
            <div class="prob-track">
                <div class="prob-fill" style="width: 0%; background-color: ${config.color};"></div>
            </div>
        `;

        probabilityBars.appendChild(row);

        setTimeout(() => {
            const fill = row.querySelector(".prob-fill");
            if (fill) {
                fill.style.width = `${percent}%`;
            }
        }, 30);
    });
}

/**
 * UI State Helpers
 */
function setLoading(isLoading) {
    if (isLoading) {
        btnAnalyze.disabled = true;
        btnSpinner.classList.remove("hidden");
        emptyState.classList.add("hidden");
        resultContent.classList.add("hidden");
        loadingState.classList.remove("hidden");
    } else {
        btnAnalyze.disabled = currentSelectedFile === null;
        btnSpinner.classList.add("hidden");
        loadingState.classList.add("hidden");
    }
}

function showError(msg) {
    errorMessage.textContent = msg;
    errorAlert.classList.remove("hidden");
}

function hideError() {
    errorAlert.classList.add("hidden");
}
