import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12";

// DOM Elements
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const statusOverlay = document.getElementById("status-overlay");
const statusIcon = document.getElementById("status-icon");
const statusText = document.getElementById("status-text");
const statusSubtext = document.getElementById("status-subtext");
const alertOverlay = document.getElementById("alert-overlay");
const alarmSound = document.getElementById("alarm-sound");
const liveIndicator = document.getElementById("live-indicator");
const videoWrapper = document.querySelector(".video-wrapper");

// Dashboard & UI Elements
const dashboardPanel = document.getElementById("dashboard-panel");
const summaryModal = document.getElementById("summary-modal");
const closeModalBtn = document.getElementById("close-modal-btn");
const earThresholdInput = document.getElementById("ear-threshold");
const earValDisplay = document.getElementById("ear-val");
const marThresholdInput = document.getElementById("mar-threshold");
const marValDisplay = document.getElementById("mar-val");
const toggleYawn = document.getElementById("toggle-yawn");
const toggleDistraction = document.getElementById("toggleDistraction");

// State & Settings
let faceLandmarker;
let runningMode = "VIDEO";
let webcamRunning = false;
let lastVideoTime = -1;
let animationFrameId = null;
let isAlarmPlaying = false;
let vibrateInterval = null;

let EAR_THRESHOLD = parseFloat(earThresholdInput.value);
let MAR_THRESHOLD = parseFloat(marThresholdInput.value);
const CLOSED_EYES_DURATION_MS = 1000;
const DISTRACTION_DURATION_MS = 3000;
const DISTRACTION_ANGLE_THRESHOLD = 25; // degrees

// Session State Architecture
const sessionStats = {
    startTime: null,
    drowsyEvents: 0,
    yawnEvents: 0,
    distractionEvents: 0
};

const currentStatus = {
    state: "ACTIVE", // ACTIVE, DROWSY, DISTRACTED, YAWNING
    yawning: false,
    eyesClosedSince: null,
    distractedSince: null,
    lastChartUpdate: 0
};

// MediaPipe Indices
const LEFT_EYE = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE = [362, 385, 387, 263, 373, 380];
const INNER_LIP = [13, 14, 78, 308]; // Top, Bottom, Left, Right

// Chart.js Setup
let earChart;
const chartDataBuffer = [];
const CHART_MAX_POINTS = 60; // 60 seconds rolling window roughly if updated 1/sec
const CHART_UPDATE_INTERVAL_MS = 500;

function initChart() {
    const ctx = document.getElementById('earChart').getContext('2d');
    earChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array(CHART_MAX_POINTS).fill(''),
            datasets: [{
                label: 'Eye Aspect Ratio (EAR)',
                data: Array(CHART_MAX_POINTS).fill(null),
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 2,
                pointRadius: 0,
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                y: { min: 0.1, max: 0.4, grid: { color: 'rgba(255,255,255,0.05)' } },
                x: { grid: { display: false } }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

function updateChart(earValue) {
    const now = performance.now();
    if (now - currentStatus.lastChartUpdate > CHART_UPDATE_INTERVAL_MS) {
        currentStatus.lastChartUpdate = now;
        
        const data = earChart.data.datasets[0].data;
        data.push(earValue);
        if (data.length > CHART_MAX_POINTS) {
            data.shift();
        }
        earChart.update();
    }
}

// Settings Listeners
earThresholdInput.addEventListener("input", (e) => {
    EAR_THRESHOLD = parseFloat(e.target.value);
    earValDisplay.textContent = EAR_THRESHOLD.toFixed(2);
});

marThresholdInput.addEventListener("input", (e) => {
    MAR_THRESHOLD = parseFloat(e.target.value);
    marValDisplay.textContent = MAR_THRESHOLD.toFixed(2);
});

closeModalBtn.addEventListener("click", () => {
    summaryModal.classList.add("hidden");
});

// Initialize MediaPipe
async function setupFaceLandmarker() {
    updateStatus("⏳", "Loading AI Model...", "Please wait while we download the vision tasks.");
    try {
        const filesetResolver = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/wasm"
        );
        faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
                modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                delegate: "GPU"
            },
            outputFaceBlendshapes: false,
            outputFacialTransformationMatrixes: true,
            runningMode: runningMode,
            numFaces: 1
        });
        updateStatus("📷", "Camera Off", "Click Start Detection to begin");
        startBtn.disabled = false;
        initChart();
    } catch (error) {
        console.error(error);
        updateStatus("❌", "Model Failed to Load", "Refresh the page or check your connection.");
    }
}

// Math Utils
function calculateDistance(a, b, width, height) {
    return Math.sqrt(
        Math.pow((a.x - b.x) * width, 2) + 
        Math.pow((a.y - b.y) * height, 2)
    );
}

function calculateEAR(landmarks, eyeIndices, width, height) {
    const p1 = landmarks[eyeIndices[0]];
    const p2 = landmarks[eyeIndices[1]];
    const p3 = landmarks[eyeIndices[2]];
    const p4 = landmarks[eyeIndices[3]];
    const p5 = landmarks[eyeIndices[4]];
    const p6 = landmarks[eyeIndices[5]];

    const vertical1 = calculateDistance(p2, p6, width, height);
    const vertical2 = calculateDistance(p3, p5, width, height);
    const horizontal = calculateDistance(p1, p4, width, height);

    if (horizontal === 0) return 0;
    return (vertical1 + vertical2) / (2.0 * horizontal);
}

function calculateMAR(landmarks, width, height) {
    // 13: Top inner lip, 14: Bottom inner lip
    // 78: Left inner corner, 308: Right inner corner
    const top = landmarks[13];
    const bottom = landmarks[14];
    const left = landmarks[78];
    const right = landmarks[308];

    const vertical = calculateDistance(top, bottom, width, height);
    const horizontal = calculateDistance(left, right, width, height);
    
    if (horizontal === 0) return 0;
    return vertical / horizontal;
}

// Euler angles from transformation matrix
// Returns yaw, pitch, roll in degrees
function getEulerAngles(matrix) {
    const m00 = matrix[0], m01 = matrix[1], m02 = matrix[2];
    const m10 = matrix[4], m11 = matrix[5], m12 = matrix[6];
    const m20 = matrix[8], m21 = matrix[9], m22 = matrix[10];

    // Decomposition
    let yaw, pitch, roll;
    
    const sy = Math.sqrt(m00 * m00 + m10 * m10);
    const singular = sy < 1e-6;

    if (!singular) {
        pitch = Math.atan2(m21, m22);
        yaw = Math.atan2(-m20, sy);
        roll = Math.atan2(m10, m00);
    } else {
        pitch = Math.atan2(-m12, m11);
        yaw = Math.atan2(-m20, sy);
        roll = 0;
    }

    // Convert to degrees
    const toDeg = 180 / Math.PI;
    return {
        pitch: pitch * toDeg,
        yaw: yaw * toDeg,
        roll: roll * toDeg
    };
}

// Start Camera & Processing
async function startDetection() {
    if (!faceLandmarker) return;

    alarmSound.play().then(() => alarmSound.pause()).catch(e => console.warn("Audio priming failed", e));

    startBtn.disabled = true;
    updateStatus("🎥", "Requesting Camera...", "Please allow camera access.");

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 1280, height: 720, facingMode: "user" } 
        });
        
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
        webcamRunning = true;
        stopBtn.disabled = false;
        
        // Reset Session Stats
        sessionStats.startTime = Date.now();
        sessionStats.drowsyEvents = 0;
        sessionStats.yawnEvents = 0;
        sessionStats.distractionEvents = 0;
        
        currentStatus.state = "ACTIVE";
        currentStatus.yawning = false;
        currentStatus.eyesClosedSince = null;
        currentStatus.distractedSince = null;

        // Reset chart data
        if(earChart) {
             earChart.data.datasets[0].data = Array(CHART_MAX_POINTS).fill(null);
             earChart.update();
        }

        // Show dashboard, hide overlay
        dashboardPanel.style.display = "flex";
        statusOverlay.classList.add("hidden");
        liveIndicator.classList.remove("offline");
        liveIndicator.classList.add("live");
        liveIndicator.querySelector(".text").textContent = "MONITORING";
        videoWrapper.classList.add("detecting");

    } catch (err) {
        console.error("Error accessing webcam:", err);
        startBtn.disabled = false;
        if (err.name === 'NotAllowedError') {
            updateStatus("🔒", "Camera Blocked", "Please grant camera permissions to use this app.");
        } else {
            updateStatus("❌", "Camera Error", "Could not access your webcam. Is it in use?");
        }
    }
}

// Stop Camera & Show Summary
function stopDetection() {
    webcamRunning = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
        video.srcObject = null;
    }
    
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
    }

    clearCanvas();
    hideAlert();
    statusOverlay.classList.remove("hidden");
    updateStatus("📷", "Camera Off", "Click Start Detection to begin");
    
    liveIndicator.classList.remove("live");
    liveIndicator.classList.add("offline");
    liveIndicator.querySelector(".text").textContent = "OFFLINE";
    videoWrapper.classList.remove("detecting");
    videoWrapper.classList.remove("drowsy");
    
    dashboardPanel.style.display = "none";
    
    showSummaryModal();
}

function showSummaryModal() {
    const elapsedSeconds = Math.floor((Date.now() - sessionStats.startTime) / 1000);
    const mins = Math.floor(elapsedSeconds / 60).toString().padStart(2, '0');
    const secs = (elapsedSeconds % 60).toString().padStart(2, '0');
    
    // Safety Score Formula: 100 - penalties
    const score = Math.max(0, 100 - (sessionStats.drowsyEvents * 10 + sessionStats.yawnEvents * 3 + sessionStats.distractionEvents * 5));

    document.getElementById("stat-duration").textContent = `${mins}:${secs}`;
    document.getElementById("stat-score").textContent = score;
    document.getElementById("stat-drowsy").textContent = sessionStats.drowsyEvents;
    document.getElementById("stat-yawn").textContent = sessionStats.yawnEvents;
    document.getElementById("stat-distracted").textContent = sessionStats.distractionEvents;
    
    summaryModal.classList.remove("hidden");
}

// Main processing loop
async function predictWebcam() {
    if (!webcamRunning) return;

    if (canvasElement.width !== video.videoWidth) {
        canvasElement.width = video.videoWidth;
        canvasElement.height = video.videoHeight;
    }

    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        
        const results = faceLandmarker.detectForVideo(video, startTimeMs);
        clearCanvas();

        if (results.faceLandmarks && results.faceLandmarks.length > 0) {
            const landmarks = results.faceLandmarks[0];
            const matrix = results.facialTransformationMatrixes ? results.facialTransformationMatrixes[0] : null;
            
            drawEyeLandmarks(landmarks);
            drawMouthLandmarks(landmarks);

            // 1. Drowsiness (EAR)
            const leftEAR = calculateEAR(landmarks, LEFT_EYE, video.videoWidth, video.videoHeight);
            const rightEAR = calculateEAR(landmarks, RIGHT_EYE, video.videoWidth, video.videoHeight);
            const avgEAR = (leftEAR + rightEAR) / 2.0;
            
            updateChart(avgEAR);

            if (avgEAR < EAR_THRESHOLD) {
                if (!currentStatus.eyesClosedSince) {
                    currentStatus.eyesClosedSince = performance.now();
                } else {
                    const elapsed = performance.now() - currentStatus.eyesClosedSince;
                    if (elapsed > CLOSED_EYES_DURATION_MS) {
                        if (currentStatus.state !== "DROWSY") {
                            sessionStats.drowsyEvents++;
                            triggerAlert("⚠️ DROWSY ALERT ⚠️", "WAKE UP!");
                            currentStatus.state = "DROWSY";
                        }
                    }
                }
            } else {
                currentStatus.eyesClosedSince = null;
                if (currentStatus.state === "DROWSY") {
                    hideAlert();
                    currentStatus.state = "ACTIVE";
                }
            }

            // Only run other checks if not already actively drowsy alerting
            if (currentStatus.state !== "DROWSY") {
                
                // 2. Yawn Detection (MAR)
                const isYawnEnabled = document.getElementById("toggle-yawn").checked;
                if (isYawnEnabled) {
                    const mar = calculateMAR(landmarks, video.videoWidth, video.videoHeight);
                    if (mar > MAR_THRESHOLD) {
                        if (!currentStatus.yawning) {
                            sessionStats.yawnEvents++;
                            currentStatus.yawning = true;
                        }
                    } else {
                        currentStatus.yawning = false;
                    }
                }

                // 3. Distraction Detection (Head Pose)
                const isDistractEnabled = document.getElementById("toggle-distraction").checked;
                if (isDistractEnabled && matrix) {
                    const euler = getEulerAngles(matrix);
                    
                    // Check if yaw or pitch exceeds threshold
                    if (Math.abs(euler.yaw) > DISTRACTION_ANGLE_THRESHOLD || Math.abs(euler.pitch) > DISTRACTION_ANGLE_THRESHOLD) {
                        if (!currentStatus.distractedSince) {
                            currentStatus.distractedSince = performance.now();
                        } else {
                            const elapsed = performance.now() - currentStatus.distractedSince;
                            if (elapsed > DISTRACTION_DURATION_MS) {
                                if (currentStatus.state !== "DISTRACTED") {
                                    sessionStats.distractionEvents++;
                                    triggerAlert("⚠️ DISTRACTED ⚠️", "KEEP EYES ON ROAD!");
                                    currentStatus.state = "DISTRACTED";
                                }
                            }
                        }
                    } else {
                        currentStatus.distractedSince = null;
                        if (currentStatus.state === "DISTRACTED") {
                            hideAlert();
                            currentStatus.state = "ACTIVE";
                        }
                    }
                }
            }
            
        } else {
            // No face found
            currentStatus.eyesClosedSince = null;
            currentStatus.distractedSince = null;
            hideAlert();
            currentStatus.state = "ACTIVE";
        }
    }

    animationFrameId = requestAnimationFrame(predictWebcam);
}

// Helpers
function clearCanvas() {
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
}

function drawEyeLandmarks(landmarks) {
    canvasCtx.fillStyle = "#3b82f6"; // Blueish for eyes
    const drawPoint = (index) => {
        const point = landmarks[index];
        canvasCtx.beginPath();
        canvasCtx.arc(point.x * canvasElement.width, point.y * canvasElement.height, 2.5, 0, 2 * Math.PI);
        canvasCtx.fill();
    };
    LEFT_EYE.forEach(drawPoint);
    RIGHT_EYE.forEach(drawPoint);
}

function drawMouthLandmarks(landmarks) {
    const isYawnEnabled = document.getElementById("toggle-yawn").checked;
    if (!isYawnEnabled) return;
    canvasCtx.fillStyle = "#8b5cf6"; // Purple for mouth
    INNER_LIP.forEach(index => {
        const point = landmarks[index];
        canvasCtx.beginPath();
        canvasCtx.arc(point.x * canvasElement.width, point.y * canvasElement.height, 2.5, 0, 2 * Math.PI);
        canvasCtx.fill();
    });
}

function updateStatus(icon, text, subtext) {
    statusIcon.textContent = icon;
    statusText.textContent = text;
    statusSubtext.textContent = subtext;
}

// Audio and Alert Logic
function triggerAlert(title = "⚠️ DROWSY ALERT ⚠️", msg = "WAKE UP!") {
    const alertBox = alertOverlay.querySelector('.alert-box');
    alertBox.querySelector('h2').textContent = title;
    alertBox.querySelector('p').textContent = msg;
    
    alertOverlay.classList.remove("hidden");
    videoWrapper.classList.add("drowsy");
    
    if (!isAlarmPlaying) {
        alarmSound.currentTime = 0;
        alarmSound.loop = true;
        alarmSound.play().catch(e => console.error("Audio play blocked", e));
        
        if ("vibrate" in navigator) {
            navigator.vibrate([500, 200, 500]);
            vibrateInterval = setInterval(() => {
                navigator.vibrate([500, 200, 500]);
            }, 1500);
        }
        isAlarmPlaying = true;
    }
}

function hideAlert() {
    alertOverlay.classList.add("hidden");
    videoWrapper.classList.remove("drowsy");
    if (isAlarmPlaying) {
        alarmSound.pause();
        if ("vibrate" in navigator) {
            clearInterval(vibrateInterval);
            navigator.vibrate(0);
        }
        isAlarmPlaying = false;
    }
}

// Event Listeners
startBtn.addEventListener("click", startDetection);
stopBtn.addEventListener("click", stopDetection);

// Initialization
startBtn.disabled = true;
setupFaceLandmarker();
