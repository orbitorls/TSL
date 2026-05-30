const video = document.getElementById("video");
const landmarkCanvas = document.getElementById("landmarkCanvas");
const videoCard = document.getElementById("videoCard");
const startOverlay = document.getElementById("startOverlay");
const btnStart = document.getElementById("btnStart");
const btnStop = document.getElementById("btnStop");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const handsBadge = document.getElementById("handsBadge");
const fpsBadge = document.getElementById("fpsBadge");
const bufferFill = document.getElementById("bufferFill");
const bufferPct = document.getElementById("bufferPct");
const loadingScreen = document.getElementById("loadingScreen");
const loadingStep = document.getElementById("loadingStep");
const cameraHint = document.getElementById("cameraHint");
const top1Label = document.getElementById("top1Label");
const top1Conf = document.getElementById("top1Conf");
const confidenceFill = document.getElementById("confidenceFill");
const confidenceState = document.getElementById("confidenceState");
const historyList = document.getElementById("historyList");
const btnClearHistory = document.getElementById("btnClearHistory");
const stepReady = document.getElementById("stepReady");
const stepHands = document.getElementById("stepHands");
const stepResult = document.getElementById("stepResult");

const predLabels = [
    document.querySelector("#pred1 .pred-label"),
    document.querySelector("#pred2 .pred-label"),
    document.querySelector("#pred3 .pred-label"),
];
const predBars = [
    document.getElementById("bar1"),
    document.getElementById("bar2"),
    document.getElementById("bar3"),
];
const predConfs = [
    document.getElementById("conf1"),
    document.getElementById("conf2"),
    document.getElementById("conf3"),
];

let tfLib = null;
let handPoseDetectionLib = null;
let holisticLandmarker = null;
let handDetector = null;
let taskHandDetector = null;
let tasksVision = null;
let handLandmarkerClass = null;
let mediaPipeHandsDetector = null;
let mediaPipeHandsResultWait = null;
let mediaPipeHandsRequestId = 0;
let stream = null;
let running = false;
let animationId = null;
let sendTimerId = null;
let processingFrame = false;
let isSendingBatch = false;
let frameSequence = 0;
let predictAbortController = null;
let lastTfjsHandState = null;
let lastTfjsHandStateFrame = -1;
let lastStableHands = null;
let lastStableHandsFrame = -1;
let frameTimestamps = [];
let lastFpsUpdate = 0;
let lastHistoryLabel = "";
let missingHandFrames = 0;
let handDetectTimestampHint = null;
let mediaPipeHandsErrorCount = 0;
let landmarkCtx = null;
let landmarkCanvasRatio = 0;
let forceSearchEveryFrame = false;
const TFJS_DETECT_EVERY_N_FRAMES = 3;
const TASK_LANDMARK_DETECT_EVERY_N_FRAMES = 2;
const HANDS_PIPELINE_DETECT_EVERY_N_FRAMES = 2;
const TFJS_STALE_STATE_WINDOW = 6;

function resolveHandPoseDetectionModule(moduleHandle) {
    if (!moduleHandle || typeof moduleHandle !== "object") {
        return null;
    }

    const candidates = [
        moduleHandle,
        moduleHandle.default,
        moduleHandle.handPoseDetection,
        moduleHandle.HandPoseDetection,
        window.handPoseDetection,
        window.HandPoseDetection,
    ];

    const direct = candidates.find((candidate) => candidate?.createDetector);
    return direct && typeof direct.createDetector === "function"
        ? direct
        : null;
}

function setHandPoseDetectionLibrary(moduleHandle = null) {
    handPoseDetectionLib = resolveHandPoseDetectionModule(moduleHandle ?? window);
    return handPoseDetectionLib;
}

const VISION_BUNDLE_URLS = [
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/vision_bundle.mjs",
    "https://unpkg.com/@mediapipe/tasks-vision@0.10.34/vision_bundle.mjs",
];
const VISION_WASM_URLS = [
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/wasm",
    "https://unpkg.com/@mediapipe/tasks-vision@0.10.34/wasm",
];
const TFJS_SCRIPT_URLS = [
    "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js",
    "https://unpkg.com/@tensorflow/tfjs@4.22.0/dist/tf.min.js",
];
const TFJS_WASM_SCRIPT_URLS = [
    "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/tf-backend-wasm.min.js",
    "https://unpkg.com/@tensorflow/tfjs-backend-wasm@4.22.0/dist/tf-backend-wasm.min.js",
];
const HAND_POSE_MODULE_URLS = [
    "https://cdn.jsdelivr.net/npm/@tensorflow-models/hand-pose-detection@2.0.1/dist/hand-pose-detection.mjs",
    "https://unpkg.com/@tensorflow-models/hand-pose-detection@2.0.1/dist/hand-pose-detection.mjs",
    "https://cdn.jsdelivr.net/npm/@tensorflow-models/hand-pose-detection@2.1.0/dist/hand-pose-detection.mjs",
];
const HAND_POSE_SCRIPT_URLS = [
    "https://cdn.jsdelivr.net/npm/@tensorflow-models/hand-pose-detection@2.0.1/dist/hand-pose-detection.min.js",
    "https://cdn.jsdelivr.net/npm/@tensorflow-models/hand-pose-detection@2.1.0/dist/hand-pose-detection.min.js",
    "https://unpkg.com/@tensorflow-models/hand-pose-detection@2.0.1/dist/hand-pose-detection.min.js",
    "https://unpkg.com/@tensorflow-models/hand-pose-detection@2.1.0/dist/hand-pose-detection.min.js",
];
const MEDIAPIPE_HANDS_SCRIPT_URLS = [
    "https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/hands.min.js",
    "https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/hands.js",
    "https://unpkg.com/@mediapipe/hands@0.4.1675469240/hands.min.js",
    "https://unpkg.com/@mediapipe/hands@0.4.1675469240/hands.js",
];
const TFJS_HAND_POSE_LOAD_ERROR_TEXT = "โหลดตัวจับมือ TFJS ไม่สำเร็จ";
const HAND_LANDMARK_TASK_URLS = [
    "https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe/third_party/mediapipe/tasks/hand_landmarker/hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task",
    "/model/hand_landmarker.task",
];
const HOLISTIC_LANDMARK_TASK_URLS = [
    "/model/holistic_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker.task",
    "https://storage.googleapis.com/mediapipe/third_party/mediapipe/models/holistic_landmarker.task",
];
const _scriptPromises = new Map();
const _modulePromises = new Map();
const SCRIPT_TIMEOUT_MS = 6000;
const SCRIPT_POLL_MS = 80;
const MEDIA_PIPE_HANDS_DETECT_TIMEOUT_MS = 260;
const MEDIA_PIPE_HANDS_RETRY_THRESHOLD = 2;

const landmarkBuffer = [];
const historyItems = [];
const SEND_INTERVAL_MS = 500;
const MISSING_HAND_RESET_FRAMES = 12;
const STALE_HAND_KEEP_FRAMES = 5;
const HISTORY_THRESHOLD = 0.7;
const HIGH_CONFIDENCE = 0.72;
const LOW_CONFIDENCE = 0.45;
const HISTORY_MAX = 20;
const HAND_SCORE_THRESHOLD = 0.15;
const HAND_SCORE_THRESHOLD_RELAXED = 0.06;
const HAND_SCORE_THRESHOLD_FAST = 0.12;
const MAX_CLIENT_BATCH = 120;
const PREDICT_REQUEST_TIMEOUT_MS = 9000;
const POSE_INDICES = [11, 12, 13, 14, 15, 16];
const FACE_INDICES = [70, 107, 336, 300, 61, 291];
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
    [5, 9], [9, 13], [13, 17],
];
const POSE_CONNECTIONS = [[11, 13], [13, 15], [12, 14], [14, 16], [11, 12]];
const FACE_CONNECTIONS = [[70, 107], [107, 336], [336, 300], [61, 291]];

async function loadFirstModule(sources, label) {
    const errors = [];
    for (const src of sources) {
        try {
            return await import(src);
        } catch (err) {
            errors.push(err.message);
            console.warn(`[TSL] ${label} module source failed: ${src}`, err);
        }
    }
    throw new Error(`${label} module is not loaded (${errors.join(" | ")})`);
}

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

function isTfJsReady() {
    return typeof window.tf === "object" && typeof window.tf.tensor === "function";
}

function isHandPoseLibReady() {
    const globalHandPose = window.handPoseDetection || window.HandPoseDetection;
    return (
        handPoseDetectionLib
        && typeof handPoseDetectionLib.createDetector === "function"
    ) || (
        typeof globalHandPose === "object"
        && typeof globalHandPose.createDetector === "function"
    );
}

function isHandLandmarkerReady() {
    return Boolean(taskHandDetector && typeof taskHandDetector.detectForVideo === "function");
}

function isMediaPipeHandsReady() {
    return (
        mediaPipeHandsDetector !== null
        && typeof mediaPipeHandsDetector.send === "function"
    );
}

function closeMediaPipeHandsDetector() {
    if (!mediaPipeHandsDetector) return;

    const detector = mediaPipeHandsDetector;
    mediaPipeHandsDetector = null;
    try {
        if (typeof detector.close === "function") {
            detector.close();
            return;
        }
        if (typeof detector.dispose === "function") {
            detector.dispose();
        }
    } catch (err) {
        console.warn("[TSL] MediaPipe Hands cleanup failed:", err);
    }
}

async function waitForPredicate(predicate, timeoutMs, label) {
    const start = performance.now();
    while (true) {
        if (predicate()) return;
        if (performance.now() - start > timeoutMs) {
            throw new Error(`${label} is not available after ${timeoutMs}ms`);
        }
        await sleep(SCRIPT_POLL_MS);
    }
}

async function ensureHandPoseDetectionLib() {
    if (isHandPoseLibReady()) return;

    const errors = [];

    for (const src of HAND_POSE_SCRIPT_URLS) {
        try {
            await loadScriptIfNeeded(src);
            await waitForPredicate(isHandPoseLibReady, SCRIPT_TIMEOUT_MS, "Hand pose library");
            if (setHandPoseDetectionLibrary()) return;
        } catch (err) {
            errors.push(err.message);
            console.warn(`[TSL] Hand pose script source failed: ${src}`, err);
        }
    }

    for (const src of HAND_POSE_MODULE_URLS) {
        try {
            const moduleHandle = await loadModuleIfNeeded(src);
            const resolved = resolveHandPoseDetectionModule(moduleHandle);
            if (resolved) {
                handPoseDetectionLib = resolved;
                return;
            }
            throw new Error("Module loaded but hand-pose createDetector is not available");
        } catch (err) {
            errors.push(err.message);
            console.warn(`[TSL] Hand pose module source failed: ${src}`, err);
        }
    }

    throw new Error(`Hand pose scripts are not loaded (${errors.join(" | ")})`);
}

async function loadModuleIfNeeded(src) {
    if (_modulePromises.has(src)) {
        const existing = _modulePromises.get(src);
        return existing;
    }

    const promise = import(src).catch((err) => {
        _modulePromises.delete(src);
        throw err;
    });
    _modulePromises.set(src, promise);
    return promise;
}

function getExistingScriptTag(src) {
    const requestedUrl = new URL(src, document.baseURI).href;
    return Array.from(document.scripts).find(
        (script) => script.src === src || script.src === requestedUrl,
    );
}

async function loadScriptIfNeeded(src) {
    if (_scriptPromises.has(src)) return _scriptPromises.get(src);

    const promise = new Promise((resolve, reject) => {
        const existing = getExistingScriptTag(src);
        const attachScriptWaiters = (script) => {
            script.dataset.tslState = script.dataset.tslState || "loading";
            if (script.dataset.tslState === "loaded") {
                resolve();
                return;
            }

            if (script.dataset.tslState === "error") {
                script.remove();
                reject(new Error(`Previous load failed: ${src}`));
                return;
            }

            const onLoad = () => {
                clearTimeout(timeoutId);
                script.dataset.tslState = "loaded";
                cleanup();
                resolve();
            };
            const onError = (err) => {
                clearTimeout(timeoutId);
                script.dataset.tslState = "error";
                cleanup();
                _scriptPromises.delete(src);
                reject(err instanceof Error ? err : new Error(`Failed script: ${src}`));
            };

            const cleanup = () => {
                script.removeEventListener("load", onLoad);
                script.removeEventListener("error", onError);
                clearTimeout(timeoutId);
            };

            const onTimeout = () => {
                script.dataset.tslState = "error";
                cleanup();
                _scriptPromises.delete(src);
                reject(new Error(`Script timeout: ${src}`));
            };
            const timeoutId = window.setTimeout(onTimeout, SCRIPT_TIMEOUT_MS);

            script.addEventListener("load", onLoad, { once: true });
            script.addEventListener("error", onError, { once: true });
            if (script.readyState === "complete" || script.readyState === "loaded") {
                onLoad();
            }
        };

        if (existing) {
            attachScriptWaiters(existing);
            return;
        }

        const script = document.createElement("script");
        script.src = src;
        script.crossOrigin = "anonymous";
        script.async = true;
        script.dataset.tslState = "loading";
        attachScriptWaiters(script);
        document.body.appendChild(script);
    });

    _scriptPromises.set(src, promise);
    return promise;
}

async function ensureLibrary(sources, isReady, label) {
    if (isReady()) return;

    const errors = [];
    for (const src of sources) {
        try {
            await loadScriptIfNeeded(src);
            await waitForPredicate(isReady, SCRIPT_TIMEOUT_MS, label);
            return;
        } catch (err) {
            errors.push(err.message);
            console.warn(`[TSL] ${label} source failed: ${src}`, err);
        }
    }

    throw new Error(`${label} scripts are not loaded (${errors.join(" | ")})`);
}

function setStatus(kind, text) {
    statusText.textContent = text;
    statusDot.className = "status-dot";
    confidenceState.className = "";

    if (kind === "on") statusDot.classList.add("on");
    if (kind === "warn") statusDot.classList.add("warn");
    if (kind === "error") statusDot.classList.add("error");
}

function setSteps(activeStep) {
    const steps = [stepReady, stepHands, stepResult];
    for (const step of steps) {
        step.classList.remove("active", "done");
    }

    if (activeStep === "ready") {
        stepReady.classList.add("active");
        return;
    }

    stepReady.classList.add("done");
    if (activeStep === "hands") {
        stepHands.classList.add("active");
        return;
    }

    stepHands.classList.add("done");
    stepResult.classList.add("active");
}

function setConfidenceState(kind, text) {
    confidenceState.textContent = text;
    confidenceState.className = kind;
}

async function initMediaPipe() {
    loadingStep.textContent = "กำลังโหลด MediaPipe WASM...";
    let visionLoaded = false;
    let visionError = null;
    let visionModule;
    let vision;
    let HolisticLandmarker;
    let FilesetResolver;
    let HandLandmarker;

    try {
        visionModule = await loadFirstModule(
            VISION_BUNDLE_URLS,
            "MediaPipe Tasks Vision",
        );
        ({ HolisticLandmarker, FilesetResolver, HandLandmarker } = visionModule);
        handLandmarkerClass = HandLandmarker;
        vision = null;
        for (const wasmUrl of VISION_WASM_URLS) {
            try {
                vision = await FilesetResolver.forVisionTasks(wasmUrl);
                visionLoaded = true;
                break;
            } catch (err) {
                console.warn(`[TSL] MediaPipe WASM source failed: ${wasmUrl}`, err);
            }
        }

        if (!visionLoaded) {
            throw new Error("MediaPipe WASM is not loaded");
        }

        tasksVision = vision;
    } catch (err) {
        visionError = err;
        console.warn("[TSL] Vision bundle initialization failed, continuing with TFJS fallback.", err);
    }

    const createHolisticLandmarker = async () => {
        if (!visionLoaded || !tasksVision) return false;

        for (const modelPath of HOLISTIC_LANDMARK_TASK_URLS) {
            try {
                holisticLandmarker = await HolisticLandmarker.createFromOptions(vision, {
                    baseOptions: { modelAssetPath: modelPath },
                    outputHandLandmarks: true,
                    outputPoseLandmarks: true,
                    outputFaceLandmarks: true,
                    outputFaceBlendshapes: false,
                    runningMode: "VIDEO",
                });
                return true;
            } catch (err) {
                console.warn("[TSL] Holistic model load failed:", modelPath, err);
            }
        }

        holisticLandmarker = null;
        return false;
    };

    if (visionLoaded && tasksVision) {
        try {
            loadingStep.textContent = "กำลังโหลด Holistic Model...";
            const loadedHolistic = await createHolisticLandmarker();
            if (!loadedHolistic) {
                throw new Error("Holistic model load failed");
            }
        } catch (err) {
            console.warn("[TSL] Holistic model load failed, will continue with hand-only mode:", err);
        }
    }

    try {
        await initHandDetector();

        if (!holisticLandmarker && !(handDetector || isHandLandmarkerReady() || isMediaPipeHandsReady())) {
            loadingStep.textContent = "ไม่สามารถโหลดตัวจับมือได้: รอสัญญาณระบบ";
            loadingStep.style.color = "#f59e0b";
            setStatus("warn", "รอโมดูลตรวจจับมือ");
            if (visionError) {
                loadingStep.textContent = `ไม่สามารถโหลด MediaPipe/TFJS ครบทุกส่วน: ${visionError.message}`;
            }
            return;
        }

        loadingScreen.classList.add("hidden");
        setStatus("idle", "พร้อมเริ่ม");
    } catch (err) {
        loadingStep.textContent = `โหลดโมเดลตรวจจับท่าทางล้มเหลว: ${err.message}`;
        loadingStep.style.color = "#c24135";
        setStatus("error", "โหลดระบบไม่สำเร็จ");
        console.error("[TSL] Init error:", err);
    }
}

async function initTaskHandDetector() {
    if (!tasksVision || !handLandmarkerClass) {
        return false;
    }

    for (const taskPath of HAND_LANDMARK_TASK_URLS) {
        try {
            taskHandDetector = await handLandmarkerClass.createFromOptions(tasksVision, {
                baseOptions: { modelAssetPath: taskPath },
                numHands: 2,
                runningMode: "VIDEO",
            });
            return true;
        } catch (err) {
            console.warn("[TSL] HandLandmarker path failed:", taskPath, err);
        }
    }
    taskHandDetector = null;
    return false;
}

async function initHandDetector() {
    loadingStep.textContent = "กำลังโหลดตัวจับมือ...";
    handDetector = null;
    taskHandDetector = null;
    tfLib = null;
    handPoseDetectionLib = null;
    closeMediaPipeHandsDetector();
    mediaPipeHandsResultWait = null;

    const taskReady = await initTaskHandDetector();
    if (taskReady) {
        loadingStep.style.color = "#16a34a";
        loadingStep.textContent = "พร้อมจับมือจาก Task Landmarker";
        setStatus("warn", "รอโครงมือจาก Task Vision");
        return;
    }

    const fallbackReady = await initMediaPipeHandsFallback();
    if (fallbackReady) {
        loadingStep.style.color = "#16a34a";
        loadingStep.textContent = "พร้อมจับมือจาก MediaPipe Hands";
        setStatus("warn", "รอโครงมือจาก MediaPipe Hands");
        return;
    }

    if (!tasksVision || !handLandmarkerClass) {
        loadingStep.textContent = `${TFJS_HAND_POSE_LOAD_ERROR_TEXT}: ใช้สำรองแทน`;
        console.warn("[TSL] Task Landmarker ไม่พร้อม ใช้ TFJS แทน");
    }

    try {
        await ensureLibrary(TFJS_SCRIPT_URLS, isTfJsReady, "TFJS");
        await ensureHandPoseDetectionLib();

        const tf = window.tf;
        const handPoseDetection = setHandPoseDetectionLibrary();
        tfLib = tf;
        handPoseDetectionLib = handPoseDetection;
        const supportedModels = handPoseDetection.SupportedModels || {};
        const mediaPipeHandsModel = supportedModels.MediaPipeHands
            || supportedModels.MediaPipeHandsModel
            || supportedModels.HandPose;
        const selectedModel = mediaPipeHandsModel
            || supportedModels.HandPoseDetection
            || supportedModels?.HandPose;
        if (!selectedModel) {
            throw new Error("Hand pose model symbol not found");
        }

        if (typeof tf.setWasmPaths === "function") {
            tf.setWasmPaths("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/");
        }

        try {
            await tf.setBackend("webgl");
            await tf.ready();
        } catch (webglErr) {
            console.warn("[TSL] TFJS WebGL backend unavailable, fallback:", webglErr);
            try {
                await ensureLibrary(
                    TFJS_WASM_SCRIPT_URLS,
                    () => typeof window.tf?.findBackend === "function" && Boolean(window.tf.findBackend("wasm")),
                    "TFJS WASM backend",
                );
                await tf.setBackend("wasm");
                await tf.ready();
            } catch (wasmErr) {
                console.warn("[TSL] TFJS WASM backend unavailable, fallback:", wasmErr);
                await tf.setBackend("cpu");
                await tf.ready();
            }
        }

        handDetector = await handPoseDetection.createDetector(
            selectedModel,
            {
                runtime: "tfjs",
                modelType: "lite",
                maxHands: 2,
            },
        );
        taskHandDetector = null;
        loadingStep.style.color = "#16a34a";
        loadingStep.textContent = "พร้อมจับมือ TFJS";
        setStatus("warn", "กำลังรอโครงมือจาก TFJS");
        return;
    } catch (err) {
        loadingStep.textContent = `${TFJS_HAND_POSE_LOAD_ERROR_TEXT}: ใช้สำรองแทน`;
        console.warn("[TSL] TFJS Hand Pose not available:", err);
    }

    console.warn("[TSL] TFJS, Task Landmarker และ MediaPipe Hands โหลดไม่สำเร็จ: ใช้ Holistic ชั่วคราว");
    loadingStep.textContent = "ไม่สามารถใช้งานตัวจับมือได้: ใช้ Holistic ชั่วคราว";
    loadingStep.style.color = "#f59e0b";
    setStatus("warn", "รอมือผ่านโครง Holistic");
    handDetector = null;
    taskHandDetector = null;
    mediaPipeHandsDetector = null;
    tfLib = null;
    handPoseDetectionLib = null;
}

function normalizeTaskLandmarks(handLandmarks, frameWidth, frameHeight) {
    return Array.isArray(handLandmarks)
        ? (() => {
            const normalized = [];
            for (let i = 0; i < 21; i++) {
                const point = handLandmarks[i];
                normalized.push(point ? normalizeHandPoint(point, frameWidth, frameHeight) : ZERO_LANDMARK);
            }
            return normalized;
        })()
        : null;
}

const ZERO_LANDMARK = { x: 0, y: 0, z: 0 };

function parseTaskHandResult(landmarks, handednesses) {
    if (!Array.isArray(landmarks)) return [];

    const normalizedHandedness = Array.isArray(handednesses)
        ? handednesses
        : null;

    return landmarks.map((handLandmarks, index) => {
        const handednessRaw = normalizedHandedness?.[index]?.[0]
            || normalizedHandedness?.[index]
            || null;
        const label = typeof handednessRaw === "string"
            ? handednessRaw
            : (handednessRaw?.categoryName || handednessRaw?.label || handednessRaw?.category || "");

        const score = Number(handednessRaw?.score ?? 1);
        return {
            hand: { landmarks: handLandmarks },
            handedness: (label || "").toLowerCase(),
            score: Number.isFinite(score) ? score : 1,
            points2d: handLandmarks,
        };
    });
}

function parseTfjsHandResult(tfjsHand) {
    const handednessRaw = tfjsHand.handedness?.[0]
        || tfjsHand.handednesses?.[0]
        || tfjsHand.handedness;
    const label = typeof handednessRaw === "string"
        ? handednessRaw
        : (handednessRaw?.categoryName || handednessRaw?.label || handednessRaw?.category || "");

    const score = Number(tfjsHand.score ?? handednessRaw?.score ?? 1);
    const points2d = Array.isArray(tfjsHand)
        ? tfjsHand
        : (tfjsHand.keypoints
            || tfjsHand.keypoints2D
            || tfjsHand.landmarks
            || tfjsHand.handLandmarks
            || []);
    const normalizedHand = Array.isArray(tfjsHand)
        ? { keypoints: tfjsHand }
        : tfjsHand;
    return {
        hand: normalizedHand,
        handedness: (label || "").toLowerCase(),
        score: Number.isFinite(score) ? score : 1,
        points2d,
    };
}

function parseTaskHandResultFromEntry(hand) {
    if (!hand || typeof hand !== "object") return null;

    const handednessRaw = hand.handedness?.[0]
        || hand.handednesses?.[0]
        || hand.handedness;
    const landmarks = Array.isArray(hand.landmarks)
        ? hand.landmarks
        : Array.isArray(hand.keypoints)
            ? hand.keypoints
            : Array.isArray(hand.handLandmarks)
                ? hand.handLandmarks
                : [];
    const score = Number(hand.score ?? hand.handedness?.[0]?.score ?? hand.handednesses?.[0]?.score ?? 1);

    return {
        hand: { landmarks },
        handedness: (typeof handednessRaw === "string"
            ? handednessRaw
            : (handednessRaw?.categoryName || handednessRaw?.label || handednessRaw?.category || "")).toLowerCase(),
        score: Number.isFinite(score) ? score : 1,
        points2d: landmarks,
    };
}

function getMediaPipeHandsBaseUrl(scriptUrl) {
    return scriptUrl
        .replace(/\/hands(?:\.min)?\.js(?:\?.*)?$/, "")
        .replace(/\/$/, "");
}

function parseMediaPipeHandsResult(rawResult) {
    if (!rawResult) {
        return [];
    }

    const handLandmarks = Array.isArray(rawResult.multiHandLandmarks)
        ? rawResult.multiHandLandmarks
        : [];

    const handednesses = Array.isArray(rawResult.multiHandedness)
        ? rawResult.multiHandedness
        : [];

    return handLandmarks
        .map((handLandmarks, index) => {
            const handednessRaw = handednesses[index];
            const label = typeof handednessRaw === "string"
                ? handednessRaw
                : (handednessRaw?.label || handednessRaw?.categoryName || handednessRaw?.category || "");
            const score = Number(handednessRaw?.score ?? 1);
            return {
                hand: { landmarks: handLandmarks },
                handedness: (label || "").toLowerCase(),
                score: Number.isFinite(score) ? score : 1,
                points2d: handLandmarks,
            };
        })
        .filter((candidate) => Array.isArray(candidate.points2d));
}

function onMediaPipeHandsResult(rawResult) {
    if (!mediaPipeHandsResultWait) return;

    const waiter = mediaPipeHandsResultWait;
    mediaPipeHandsResultWait = null;
    clearTimeout(waiter.timeoutId);

    const hands = parseMediaPipeHandsResult(rawResult)
        .map((hand) => ({ source: "MediaPipeHands", ...hand }));
    waiter.resolve(hands);
}

async function initMediaPipeHandsFallback() {
    if (mediaPipeHandsDetector) return true;

    for (const scriptUrl of MEDIAPIPE_HANDS_SCRIPT_URLS) {
        try {
            await loadScriptIfNeeded(scriptUrl);
            await waitForPredicate(
                () => typeof window.Hands === "function",
                SCRIPT_TIMEOUT_MS,
                "MediaPipe Hands",
            );

            const baseUrl = getMediaPipeHandsBaseUrl(scriptUrl);
            const detector = new window.Hands({
                locateFile: (fileName) => `${baseUrl}/${fileName}`,
            });
            detector.setOptions({
                maxNumHands: 2,
                modelComplexity: 1,
                minDetectionConfidence: 0.22,
                minTrackingConfidence: 0.28,
            });
            detector.onResults(onMediaPipeHandsResult);

            if (typeof detector.initialize === "function") {
                await detector.initialize();
            }

            mediaPipeHandsDetector = detector;
            return true;
        } catch (err) {
            console.warn("[TSL] MediaPipe Hands fallback failed:", scriptUrl, err);
        }
    }

    mediaPipeHandsDetector = null;
    return false;
}

function mapDetectedHands(candidates, frameWidth, frameHeight) {
    return candidates
        .map((candidate) => {
            const landmarks = candidate.source === "TaskLandmarker"
                ? normalizeTaskLandmarks(candidate.points2d, frameWidth, frameHeight)
                : normalizeTfjsHand(candidate.hand, frameWidth, frameHeight);
            return {
                handedness: candidate.handedness,
                landmarks,
                score: candidate.score,
            };
        })
        .filter((hand) => hand.landmarks);
}

function isLandmarkPoint(point) {
    return (
        point
        && typeof point.x === "number"
        && typeof point.y === "number"
        && Number.isFinite(point.x)
        && Number.isFinite(point.y)
        && (typeof point.z === "undefined" || Number.isFinite(point.z))
    );
}

function getLandmarkList(landmarks) {
    const values = Array.isArray(landmarks)
        ? landmarks
        : (landmarks?.landmarks && Array.isArray(landmarks.landmarks) ? landmarks.landmarks : null);

    if (!Array.isArray(values) || values.length === 0) return null;

    const first = values[0];
    if (Array.isArray(first)) {
        if (first.length === 0) return null;
        return isLandmarkPoint(first[0]) ? first : null;
    }

    return isLandmarkPoint(first) ? values : null;
}

function emptyHandState(error = "") {
    return { leftHand: null, rightHand: null, handCount: 0, source: "TFJS", error };
}

function normalizeHandPoint(point, frameWidth, frameHeight) {
    if (!Number.isFinite(point?.x) || !Number.isFinite(point?.y)) {
        return ZERO_LANDMARK;
    }

    const isNormalized = point.x >= 0 && point.x <= 1 && point.y >= 0 && point.y <= 1;
    const z = typeof point.z === "number" ? point.z : 0;

    return {
        x: Math.min(1, Math.max(0, isNormalized ? point.x : point.x / frameWidth)),
        y: Math.min(1, Math.max(0, isNormalized ? point.y : point.y / frameHeight)),
        z: Number.isFinite(z) ? z : 0,
    };
}

function normalizeTfjsHand(hand, frameWidth, frameHeight) {
    const points2d = hand.keypoints || hand.landmarks;
    if (!Array.isArray(points2d)) return null;

    return (() => {
        const normalized = [];
        for (let i = 0; i < 21; i++) {
            const point = points2d[i];
            const depth = hand.keypoints3D?.[i]?.z ?? point?.z ?? 0;
            normalized.push(
                point
                    ? normalizeHandPoint({ x: point.x, y: point.y, z: depth }, frameWidth, frameHeight)
                    : ZERO_LANDMARK,
            );
        }

        return normalized;
    })();
}

async function detectWithMediaPipeHands(videoFrame) {
    if (!isMediaPipeHandsReady()) return [];

    if (mediaPipeHandsResultWait) {
        return mediaPipeHandsResultWait.promise;
    }

    const requestId = ++mediaPipeHandsRequestId;
    const inference = new Promise((resolve, reject) => {
        const timeoutId = setTimeout(() => {
            if (mediaPipeHandsResultWait?.requestId === requestId) {
                mediaPipeHandsResultWait = null;
                clearTimeout(timeoutId);
                resolve([]);
            }
        }, MEDIA_PIPE_HANDS_DETECT_TIMEOUT_MS);

        mediaPipeHandsResultWait = {
            requestId,
            timeoutId,
            resolve,
            reject,
            promise: null,
        };

        try {
            const sendResult = mediaPipeHandsDetector.send({ image: videoFrame });
            if (sendResult && typeof sendResult.catch === "function") {
                sendResult.catch((err) => {
                    if (mediaPipeHandsResultWait?.requestId !== requestId) return;
                    mediaPipeHandsResultWait = null;
                    clearTimeout(timeoutId);
                    reject(err);
                });
            }
        } catch (err) {
            if (mediaPipeHandsResultWait?.requestId !== requestId) return;
            mediaPipeHandsResultWait = null;
            clearTimeout(timeoutId);
            reject(err);
        }
    });

    mediaPipeHandsResultWait.promise = inference;
    return inference;
}

async function detectHands(videoFrame) {
    const explicitSearch = arguments.length > 1 ? arguments[1] === true : false;
    const shouldSearchEveryFrame = explicitSearch || forceSearchEveryFrame;
    const detectTimestamp = handDetectTimestampHint || Math.max(1, Math.floor(performance.now()));
    handDetectTimestampHint = null;
    return detectHandsWithTimestamp(videoFrame, detectTimestamp, shouldSearchEveryFrame);
}

async function detectHandsWithTimestamp(videoFrame, detectTimestamp, shouldSearchEveryFrame = false) {
    const state = { leftHand: null, rightHand: null, handCount: 0, source: "TFJS", error: "" };

    if (!handDetector && !isHandLandmarkerReady() && !isMediaPipeHandsReady()) {
        state.error = "ยังไม่สามารถโหลดตัวจับมือได้";
        return state;
    }

    try {
        let rawHands = [];
        const frameWidth = videoFrame.videoWidth || videoFrame.clientWidth || 1;
        const frameHeight = videoFrame.videoHeight || videoFrame.clientHeight || 1;

        const source = handDetector ? "TFJS"
            : isHandLandmarkerReady() ? "TaskLandmarker"
            : "MediaPipeHands";
        if (handDetector) {
            const tfjsHands = await handDetector.estimateHands(videoFrame, { flipHorizontal: false });
            rawHands = Array.isArray(tfjsHands)
                ? tfjsHands.map((hand) => ({ source, ...parseTfjsHandResult(hand) }))
                : [];
        } else if (taskHandDetector) {
            const rawTask = await taskHandDetector.detectForVideo(videoFrame, detectTimestamp);
            const rawCandidates = [];
            if (Array.isArray(rawTask?.hands)) {
                rawCandidates.push(
                    ...rawTask.hands
                        .map((entry) => parseTaskHandResultFromEntry(entry))
                        .filter(Boolean)
                        .map((entry) => ({ source, ...entry })),
                );
            } else if (Array.isArray(rawTask?.landmarks)) {
                rawCandidates.push(
                    ...parseTaskHandResult(rawTask.landmarks, rawTask.handednesses).map((item) => ({ source, ...item })),
                );
            } else if (Array.isArray(rawTask)) {
                rawCandidates.push(...rawTask.map((hand) => ({ source, ...parseTfjsHandResult({ ...hand, keypoints: hand } ) })));
            }
            rawHands = rawCandidates;
            if (!Array.isArray(rawHands)) rawHands = [];
            state.source = "TaskLandmarker";
        } else if (isMediaPipeHandsReady()) {
            rawHands = await detectWithMediaPipeHands(videoFrame);
            state.source = "MediaPipeHands";
        }

        const handScoreCutoff = isMediaPipeHandsReady()
            ? HAND_SCORE_THRESHOLD_RELAXED
            : shouldSearchEveryFrame
                ? HAND_SCORE_THRESHOLD_FAST
                : HAND_SCORE_THRESHOLD;

        const detected = Array.isArray(rawHands)
            ? mapDetectedHands(
                rawHands
                    .filter((hand) => Number(hand.score ?? 1) >= handScoreCutoff)
                    .slice(0, 2),
                frameWidth,
                frameHeight,
            )
            : [];

        state.handCount = detected.length;

        for (const hand of detected) {
            if (hand.handedness === "left" && !state.leftHand) state.leftHand = hand.landmarks;
            else if (hand.handedness === "right" && !state.rightHand) state.rightHand = hand.landmarks;
        }

        for (const hand of detected) {
            if (!state.leftHand) state.leftHand = hand.landmarks;
            else if (!state.rightHand && hand.landmarks !== state.leftHand) {
                state.rightHand = hand.landmarks;
            }
        }
    } catch (err) {
        state.error = "ตรวจจับมือสะดุด";
        console.warn("[TSL] Hand detection error:", err);
        if (isMediaPipeHandsReady()) {
            mediaPipeHandsErrorCount += 1;
            if (mediaPipeHandsErrorCount >= MEDIA_PIPE_HANDS_RETRY_THRESHOLD) {
                closeMediaPipeHandsDetector();
                initMediaPipeHandsFallback().catch((recoveryErr) => {
                    console.warn("[TSL] MediaPipe Hands recovery failed:", recoveryErr);
                });
            }
        }
    }

    if (isMediaPipeHandsReady()) {
        mediaPipeHandsErrorCount = 0;
    }

    return state;
}

async function detectHolisticFrame(videoFrame, timestamp) {
    const result = holisticLandmarker.detectForVideo(videoFrame, timestamp);
    if (result && typeof result.then === "function") return await result;
    return result;
}

function getDetectionState(result, handState = null) {
    const tfLeftHand = handState?.leftHand || null;
    const tfRightHand = handState?.rightHand || null;
    const leftHandLandmarks = getLandmarkList(result.leftHandLandmarks);
    const rightHandLandmarks = getLandmarkList(result.rightHandLandmarks);
    const leftHand = leftHandLandmarks || tfLeftHand;
    const rightHand = rightHandLandmarks || tfRightHand;
    const handCount = Number(Boolean(leftHand)) + Number(Boolean(rightHand));
    const source = (leftHandLandmarks || rightHandLandmarks) ? "Holistic" : (handState?.source || "TFJS");

    return {
        leftHand,
        rightHand,
        handCount,
        handSource: source,
        handError: handState?.error || "",
        hasHands: handCount > 0,
        hasPose: Boolean(getLandmarkList(result.poseLandmarks)),
        hasFace: Boolean(getLandmarkList(result.faceLandmarks)),
    };
}

function buildFeatureVector(result, detection) {
    const feats = [];

    const lh = getLandmarkList(result.leftHandLandmarks) || detection?.leftHand;
    if (lh) {
        for (const lm of lh) feats.push(lm.x, lm.y, lm.z);
    } else {
        for (let i = 0; i < 21; i++) feats.push(0, 0, 0);
    }

    const rh = getLandmarkList(result.rightHandLandmarks) || detection?.rightHand;
    if (rh) {
        for (const lm of rh) feats.push(lm.x, lm.y, lm.z);
    } else {
        for (let i = 0; i < 21; i++) feats.push(0, 0, 0);
    }

    const pose = getLandmarkList(result.poseLandmarks);
    if (pose) {
        for (const idx of POSE_INDICES) {
            const lm = pose[idx];
            feats.push(lm?.x || 0, lm?.y || 0, lm?.z || 0);
        }
    } else {
        for (let i = 0; i < 6; i++) feats.push(0, 0, 0);
    }

    const face = getLandmarkList(result.faceLandmarks);
    if (face) {
        for (const idx of FACE_INDICES) {
            const lm = face[idx];
            feats.push(lm?.x || 0, lm?.y || 0, lm?.z || 0);
        }
    } else {
        for (let i = 0; i < 6; i++) feats.push(0, 0, 0);
    }

    return feats;
}

function syncLandmarkCanvas() {
    if (!landmarkCanvas || !videoCard) return null;

    const width = Math.max(1, Math.round(video.clientWidth || videoCard.clientWidth));
    const height = Math.max(1, Math.round(video.clientHeight || videoCard.clientHeight));
    const ratio = window.devicePixelRatio || 1;
    const targetWidth = Math.max(1, Math.round(width * ratio));
    const targetHeight = Math.max(1, Math.round(height * ratio));

    if (landmarkCanvas.width !== targetWidth || landmarkCanvas.height !== targetHeight) {
        landmarkCanvas.width = targetWidth;
        landmarkCanvas.height = targetHeight;
        landmarkCanvasRatio = ratio;
        landmarkCtx = null;
    }

    if (!landmarkCtx) {
        landmarkCtx = landmarkCanvas.getContext("2d");
    }
    const ctx = landmarkCtx;
    if (!ctx) return null;

    if (landmarkCanvasRatio !== ratio) {
        landmarkCanvasRatio = ratio;
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { ctx, width, height };
}

function clearLandmarks() {
    if (!landmarkCanvas) return;

    const ctx = landmarkCtx || landmarkCanvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);
}

function resetLiveBuffer() {
    landmarkBuffer.length = 0;
    updateBuffer(0);
}

function pointToCanvas(point, width, height) {
    if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y)) return null;
    return {
        x: Math.min(width, Math.max(0, (1 - point.x) * width)),
        y: Math.min(height, Math.max(0, point.y * height)),
    };
}

function drawIndexedPoints(ctx, landmarks, indices, width, height, color, radius) {
    if (!landmarks || !indices?.length) return;

    ctx.save();
    ctx.fillStyle = color;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.92)";
    ctx.lineWidth = 2;

    for (const index of indices) {
        const point = landmarks[index];
        const p = pointToCanvas(point, width, height);
        if (!p) continue;

        ctx.beginPath();
        ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
    }
    ctx.restore();
}

function drawConnections(ctx, landmarks, connections, width, height, color, lineWidth) {
    if (!landmarks) return;

    ctx.save();
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = color;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.shadowColor = "rgba(15, 23, 42, 0.45)";
    ctx.shadowBlur = 6;

    for (const [startIndex, endIndex] of connections) {
        const start = landmarks[startIndex];
        const end = landmarks[endIndex];
        if (!start || !end) continue;

        const a = pointToCanvas(start, width, height);
        const b = pointToCanvas(end, width, height);
        if (!a || !b) continue;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
    }
    ctx.restore();
}

function drawPoints(ctx, landmarks, width, height, color, radius) {
    if (!landmarks) return;

    ctx.save();
    ctx.fillStyle = color;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.92)";
    ctx.lineWidth = 2;

    for (const point of landmarks) {
        const p = pointToCanvas(point, width, height);
        if (!p) continue;
        ctx.beginPath();
        ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
    }
    ctx.restore();
}

function drawLandmarks(result, detection = null) {
    const canvasState = syncLandmarkCanvas();
    if (!canvasState) return;

    const { ctx, width, height } = canvasState;
    ctx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);

    const leftHand = detection?.leftHand || getLandmarkList(result.leftHandLandmarks);
    const rightHand = detection?.rightHand || getLandmarkList(result.rightHandLandmarks);
    const pose = result?.poseLandmarks || null;
    const face = result?.faceLandmarks || null;

    drawConnections(ctx, pose, POSE_CONNECTIONS, width, height, "rgba(96, 165, 250, 0.78)", 3);
    drawConnections(ctx, face, FACE_CONNECTIONS, width, height, "rgba(250, 204, 21, 0.7)", 2);
    drawConnections(ctx, leftHand, HAND_CONNECTIONS, width, height, "rgba(20, 184, 166, 0.95)", 4);
    drawConnections(ctx, rightHand, HAND_CONNECTIONS, width, height, "rgba(37, 99, 235, 0.95)", 4);

    drawIndexedPoints(ctx, pose, POSE_INDICES, width, height, "#60a5fa", 4);
    drawIndexedPoints(ctx, face, FACE_INDICES, width, height, "#facc15", 3);
    drawPoints(ctx, leftHand, width, height, "#14b8a6", 4.5);
    drawPoints(ctx, rightHand, width, height, "#2563eb", 4.5);
}

async function startCamera() {
    if (running) return;

    if (!holisticLandmarker) {
        const hasHandsOnlyDetector = Boolean(handDetector) || isHandLandmarkerReady() || isMediaPipeHandsReady();
        if (!hasHandsOnlyDetector) {
            setStatus("warn", "กำลังโหลดระบบ");
            return;
        }
        console.warn("[TSL] Holistic not ready; continuing with hand-only detector.");
    }

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
            audio: false,
        });
        video.srcObject = stream;
        await video.play();

        startOverlay.classList.add("hidden");
        btnStop.hidden = false;
        videoCard.classList.add("active");
        cameraHint.textContent = "วางมือให้อยู่ในกรอบและขยับท่าทางชัดเจน";
        setStatus("on", "กำลังทำงาน");
        setSteps("hands");
        frameSequence = 0;
        lastTfjsHandState = null;
        lastTfjsHandStateFrame = -1;
        lastStableHands = null;
        lastStableHandsFrame = -1;
        handDetectTimestampHint = null;
        running = true;
        missingHandFrames = 0;
        resetLiveBuffer();
        frameTimestamps = [];
        lastFpsUpdate = performance.now();
        animationId = requestAnimationFrame(processFrame);
        sendTimerId = setInterval(sendLandmarks, SEND_INTERVAL_MS);
    } catch (err) {
        setStatus("error", "เปิดกล้องไม่ได้");
        setConfidenceState("error", "ต้องอนุญาตกล้อง");
        top1Label.textContent = "ไม่พบสิทธิ์กล้อง";
        top1Label.classList.remove("active");
        top1Conf.textContent = "ตรวจสอบสิทธิ์กล้องในเบราว์เซอร์แล้วลองอีกครั้ง";
        cameraHint.textContent = "เบราว์เซอร์ไม่อนุญาตให้เข้าถึงกล้อง";
        console.error("[TSL] Camera error:", err);
    }
}

function stopCamera() {
    running = false;
    if (animationId) cancelAnimationFrame(animationId);
    if (sendTimerId) clearInterval(sendTimerId);
    if (stream) stream.getTracks().forEach((track) => track.stop());
    if (predictAbortController) {
        predictAbortController.abort();
        predictAbortController = null;
    }
    if (mediaPipeHandsResultWait) {
        clearTimeout(mediaPipeHandsResultWait.timeoutId);
        mediaPipeHandsResultWait.resolve([]);
        mediaPipeHandsResultWait = null;
    }
    mediaPipeHandsErrorCount = 0;
    closeMediaPipeHandsDetector();

    animationId = null;
    sendTimerId = null;
    processingFrame = false;
    stream = null;
    video.srcObject = null;
    lastTfjsHandState = null;
    lastTfjsHandStateFrame = -1;
    lastStableHands = null;
    lastStableHandsFrame = -1;
    handDetectTimestampHint = null;
    missingHandFrames = 0;
    resetLiveBuffer();
    clearLandmarks();

    startOverlay.classList.remove("hidden");
    btnStop.hidden = true;
    videoCard.classList.remove("active");
    handsBadge.textContent = "ยังไม่พบมือ";
    handsBadge.classList.remove("on");
    fpsBadge.textContent = "-- FPS";
    bufferFill.style.width = "0%";
    bufferPct.textContent = "0%";
    cameraHint.textContent = "กดเปิดกล้องเพื่อเริ่มใช้งาน";
    setStatus("idle", "พร้อมเริ่ม");
    setSteps("ready");
}

async function processFrame(timestamp) {
    if (!running || processingFrame) {
        if (running) animationId = requestAnimationFrame(processFrame);
        return;
    }
    processingFrame = true;
    frameSequence += 1;

    try {
        const now = performance.now();
        frameTimestamps.push(now);
        while (frameTimestamps.length > 0 && frameTimestamps[0] < now - 1000) {
            frameTimestamps.shift();
        }

        if (now - lastFpsUpdate > 500) {
            fpsBadge.textContent = `${frameTimestamps.length} FPS`;
            lastFpsUpdate = now;
        }

        if ((holisticLandmarker || handDetector || taskHandDetector || isMediaPipeHandsReady()) && video.readyState >= 2) {
            const rawResult = holisticLandmarker
                ? await detectHolisticFrame(video, timestamp)
                : {
                    leftHandLandmarks: null,
                    rightHandLandmarks: null,
                    poseLandmarks: null,
                    faceLandmarks: null,
                };

            if (!rawResult) {
                throw new Error("Holistic result is empty");
            }

            const result = {
                leftHandLandmarks: getLandmarkList(rawResult.leftHandLandmarks),
                rightHandLandmarks: getLandmarkList(rawResult.rightHandLandmarks),
                poseLandmarks: getLandmarkList(rawResult.poseLandmarks),
                faceLandmarks: getLandmarkList(rawResult.faceLandmarks),
            };
            const hasHolisticHands = Boolean(holisticLandmarker)
                && (Boolean(result.leftHandLandmarks) || Boolean(result.rightHandLandmarks));
            let handState = emptyHandState();
            const canUseTfjsHands = Boolean(handDetector) && !hasHolisticHands;
            const canUseTaskHands = Boolean(taskHandDetector) && !hasHolisticHands;
            const canUseMediaPipeHands = isMediaPipeHandsReady() && !hasHolisticHands;
            const shouldSearchEveryFrame = missingHandFrames >= 8;
            const detectEveryNFrames = canUseTaskHands
                ? TASK_LANDMARK_DETECT_EVERY_N_FRAMES
                : canUseMediaPipeHands
                    ? (shouldSearchEveryFrame ? 1 : HANDS_PIPELINE_DETECT_EVERY_N_FRAMES)
                    : (shouldSearchEveryFrame ? 1 : TFJS_DETECT_EVERY_N_FRAMES);

            if ((canUseTfjsHands || canUseTaskHands || canUseMediaPipeHands) && !hasHolisticHands) {
                if (frameSequence % detectEveryNFrames === 0) {
                    handDetectTimestampHint = Math.floor(timestamp);
                    forceSearchEveryFrame = shouldSearchEveryFrame;
                    handState = await detectHands(video);
                    lastTfjsHandState = handState;
                    lastTfjsHandStateFrame = frameSequence;
                } else if (
                    lastTfjsHandState
                    && frameSequence - lastTfjsHandStateFrame <= TFJS_STALE_STATE_WINDOW
                ) {
                    handState = lastTfjsHandState;
                }
            }

            const detection = getDetectionState(result, handState);
            const hasAnyRecentHandLandmarks = lastStableHands && frameSequence - lastStableHandsFrame <= STALE_HAND_KEEP_FRAMES;
            const detectionForRender = detection.hasHands || !hasAnyRecentHandLandmarks
                ? detection
                : { ...detection, ...lastStableHands };

            if (detection.hasHands) {
                missingHandFrames = 0;
                lastStableHands = {
                    leftHand: detection.leftHand,
                    rightHand: detection.rightHand,
                };
                lastStableHandsFrame = frameSequence;
                drawLandmarks(result, detection);
                handsBadge.textContent = detection.handCount === 2
                    ? `${detection.handSource}: ตรวจพบมือ 2 ข้าง`
                    : `${detection.handSource}: ตรวจพบมือ`;
                handsBadge.classList.add("on");
                cameraHint.textContent = "กำลังอ่านลำดับท่าทาง";
                setSteps("result");
                const vec = buildFeatureVector(result, detection);
                if (landmarkBuffer.length < MAX_CLIENT_BATCH) {
                    landmarkBuffer.push(vec);
                }
            } else if (detection.hasPose || detection.hasFace) {
                missingHandFrames += 1;
                drawLandmarks(result, detectionForRender);
                handsBadge.textContent = detection.handError || "เห็นร่างกาย/มือ แต่ยังรอสัญญาณมือเสถียร";
                handsBadge.classList.remove("on");
                cameraHint.textContent = "ยกมือให้อยู่ในกรอบภาพ";
                setSteps("hands");
                if (missingHandFrames >= MISSING_HAND_RESET_FRAMES) resetLiveBuffer();
            } else if (hasAnyRecentHandLandmarks) {
                missingHandFrames += 1;
                drawLandmarks(result, { ...detection, ...lastStableHands });
                handsBadge.textContent = detection.handError || "เห็นร่างกาย แต่ TFJS ยังไม่พบมือ";
                handsBadge.classList.remove("on");
                cameraHint.textContent = "ยกมือให้อยู่ในกรอบภาพ";
                setSteps("hands");
                if (missingHandFrames >= MISSING_HAND_RESET_FRAMES) resetLiveBuffer();
            } else {
                missingHandFrames += 1;
                clearLandmarks();
                handsBadge.textContent = "ยังไม่พบมือ";
                handsBadge.classList.remove("on");
                cameraHint.textContent = "ขยับมือให้อยู่ในกรอบภาพ";
                setSteps("hands");
                if (missingHandFrames >= MISSING_HAND_RESET_FRAMES) resetLiveBuffer();
            }
        }
    } catch (err) {
        setStatus("warn", "ตรวจจับสะดุด");
        console.warn("[TSL] Detection error:", err);
    } finally {
        processingFrame = false;
        if (running) animationId = requestAnimationFrame(processFrame);
    }
}

async function sendLandmarks() {
    if (isSendingBatch || !running) return;
    if (landmarkBuffer.length === 0) return;
    isSendingBatch = true;
    if (predictAbortController) {
        predictAbortController.abort();
    }
    predictAbortController = new AbortController();
    const timeoutHandle = setTimeout(() => predictAbortController.abort(), PREDICT_REQUEST_TIMEOUT_MS);

    const batch = [...landmarkBuffer];
    landmarkBuffer.length = 0;
    try {
        if (batch.length > MAX_CLIENT_BATCH) {
            batch.length = MAX_CLIENT_BATCH;
        }
        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            signal: predictAbortController.signal,
            body: JSON.stringify({ landmarks: batch }),
        });

        if (!res.ok) {
            throw new Error(`Predict failed with HTTP ${res.status}`);
        }

        const result = await res.json();
        updateBuffer(result.buffer_filled || 0);
        updatePredictions(result.top3 || []);
    } catch (err) {
        landmarkBuffer.push(...batch);
        if (landmarkBuffer.length > MAX_CLIENT_BATCH) {
            landmarkBuffer.splice(0, landmarkBuffer.length - MAX_CLIENT_BATCH);
        }
        setStatus("error", "เชื่อมต่อโมเดลไม่ได้");
        setConfidenceState("error", "เกิดข้อผิดพลาด");
        top1Label.textContent = "ลองใหม่อีกครั้ง";
        top1Label.classList.remove("active");
        top1Conf.textContent = "ไม่สามารถส่งข้อมูลไปยังโมเดลได้";
        console.warn("[TSL] Predict error:", err);
    } finally {
        clearTimeout(timeoutHandle);
        predictAbortController = null;
        isSendingBatch = false;
    }
}

function updateBuffer(fill) {
    const pct = Math.min(100, Math.round(fill * 100));
    bufferFill.style.width = `${pct}%`;
    bufferPct.textContent = `${pct}%`;
}

function updatePredictions(top3) {
    if (!top3.length) {
        setConfidenceState("warn", "กำลังสะสมข้อมูล");
        top1Label.textContent = "กำลังอ่านท่าทาง";
        top1Label.classList.remove("active");
        top1Conf.textContent = "รอข้อมูลลำดับภาพให้ครบก่อนแสดงผล";
        confidenceFill.style.width = "0%";
        return;
    }

    const top = top3[0];
    const conf = top.confidence || 0;
    const pct = Math.round(conf * 100);
    const isConfident = conf >= HIGH_CONFIDENCE;
    const isLow = conf < LOW_CONFIDENCE;

    top1Label.textContent = top.label || "-";
    top1Label.classList.toggle("active", !isLow);
    top1Conf.textContent = `ความมั่นใจ ${pct}%`;
    confidenceFill.style.width = `${pct}%`;

    if (isConfident) {
        setStatus("on", "ตรวจพบคำศัพท์");
        setConfidenceState("ok", "มั่นใจ");
    } else if (isLow) {
        setStatus("warn", "ความมั่นใจต่ำ");
        setConfidenceState("warn", "ยังไม่แน่ใจ");
    } else {
        setStatus("on", "กำลังประเมิน");
        setConfidenceState("warn", "ปานกลาง");
    }

    for (let i = 0; i < 3; i++) {
        const item = top3[i] || { label: "-", confidence: 0 };
        predLabels[i].textContent = item.label || "-";
        predBars[i].style.width = `${(item.confidence || 0) * 100}%`;
        predConfs[i].textContent = `${Math.round((item.confidence || 0) * 100)}%`;
    }

    if (conf >= HISTORY_THRESHOLD && top.label && top.label !== lastHistoryLabel) {
        lastHistoryLabel = top.label;
        addHistory(top.label, conf);
    }
}

function addHistory(label, conf) {
    if (historyItems.length >= HISTORY_MAX) {
        historyItems.shift();
        const first = historyList.querySelector(".history-chip");
        if (first) first.remove();
    }
    historyItems.push({ label, conf });

    const empty = document.getElementById("historyEmpty");
    if (empty) empty.remove();

    const chip = document.createElement("div");
    chip.className = "history-chip";

    const nameSpan = document.createElement("span");
    nameSpan.textContent = label;

    const confSpan = document.createElement("span");
    confSpan.className = "chip-conf";
    confSpan.textContent = `${Math.round(conf * 100)}%`;

    chip.appendChild(nameSpan);
    chip.appendChild(confSpan);
    historyList.appendChild(chip);
    historyList.scrollTop = historyList.scrollHeight;
}

function clearHistory() {
    historyItems.length = 0;
    lastHistoryLabel = "";

    while (historyList.firstChild) historyList.removeChild(historyList.firstChild);

    const empty = document.createElement("span");
    empty.className = "history-empty";
    empty.id = "historyEmpty";
    empty.textContent = "ยังไม่มีคำที่มั่นใจพอ";
    historyList.appendChild(empty);
}

btnStart.addEventListener("click", startCamera);
btnStop.addEventListener("click", stopCamera);
btnClearHistory.addEventListener("click", clearHistory);

document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && running) stopCamera();
});

window.addEventListener("beforeunload", () => {
    if (running) stopCamera();
});

if (document.readyState === "complete") initMediaPipe();
else window.addEventListener("load", initMediaPipe);
