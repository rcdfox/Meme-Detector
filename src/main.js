import './style.css';
import {
  FaceLandmarker,
  FilesetResolver,
  GestureRecognizer,
  PoseLandmarker
} from '@mediapipe/tasks-vision';
import {
  MEMES,
  chooseConfidentCandidate,
  chooseStableMatch,
  computeFaceBaseline,
  extractFaceFeatures,
  extractHandFeatures,
  extractPoseFeatures,
  normalizeFaceFeatures,
  rankMemeMatches,
  smoothFeatureGroup
} from './detection.js';
import {
  computeCoverRect,
  computeFaceOverlayTransform,
  smoothOverlayTransform
} from './overlay.js';

const app = document.querySelector('#app');

app.innerHTML = `
  <main class="shell">
    <header class="topbar">
      <div>
        <p class="eyebrow">LOCAL COMPUTER VISION</p>
        <h1>Meme Detector</h1>
        <p class="subtitle">Make a face, pose, or hand gesture and the matching meme will snap onto your face in real time after a quick neutral-face calibration.</p>
      </div>
      <div class="privacy-pill"><span></span> Camera stays local</div>
    </header>

    <section class="workspace">
      <article class="camera-card panel">
        <div class="panel-head">
          <div>
            <p class="label">LIVE MEME CAM</p>
            <p id="status" class="status">Models not loaded</p>
          </div>
          <div class="actions">
            <button id="startBtn" class="primary">Start camera</button>
            <button id="stopBtn" class="ghost" disabled>Stop</button>
          </div>
        </div>
        <div class="video-wrap">
          <video id="webcam" autoplay playsinline muted></video>
          <canvas id="overlay"></canvas>
          <div id="cameraPlaceholder" class="placeholder">
            <div class="placeholder-icon">☺</div>
            <strong>Ready for a meme face</strong>
            <span>Start the camera, then try smiling, staring, pointing, or giving a thumbs-up.</span>
          </div>
          <div id="activeMeme" class="active-meme idle">
            <span class="active-label">ACTIVE MEME</span>
            <strong id="matchTitle">Waiting for a match</strong>
            <small id="matchDetail">Hold a neutral face briefly when the camera starts. Then the meme will lock onto your face.</small>
          </div>
          <div class="tracking-pill">Face-locked square overlay</div>
          <div class="fps" id="fps">0 FPS</div>
        </div>
        <div class="camera-note">
          <span>Tip</span>
          Keep your face visible while posing. The square image follows face position, scale, and head tilt automatically.
        </div>
      </article>
    </section>

    <section class="lower-grid">
      <article class="panel diagnostics">
        <div class="panel-head"><div><p class="label">DETECTION SIGNALS</p><p class="status">Live signals used by the matcher</p></div></div>
        <div id="signals" class="signals"></div>
      </article>
      <article class="panel history-panel">
        <div class="panel-head"><div><p class="label">RECENT MATCHES</p><p class="status">Stabilized detections</p></div></div>
        <div id="history" class="history"><p class="muted">No matches yet.</p></div>
      </article>
    </section>

    <section class="library-section">
      <div class="section-title">
        <div><p class="eyebrow">CAT MEME LIBRARY</p><h2>Try these reactions</h2></div>
        <p>Each meme becomes a face-tracked square overlay once its expression, pose, or gesture rule is detected.</p>
      </div>
      <div id="memeLibrary" class="meme-library"></div>
    </section>

    <footer>
      <span>No account. No uploads. No backend.</span>
      <span>MediaPipe Face + Pose + Gesture Recognition</span>
    </footer>
  </main>
`

const els = {
  video: document.querySelector('#webcam'),
  canvas: document.querySelector('#overlay'),
  placeholder: document.querySelector('#cameraPlaceholder'),
  start: document.querySelector('#startBtn'),
  stop: document.querySelector('#stopBtn'),
  status: document.querySelector('#status'),
  fps: document.querySelector('#fps'),
  activeMeme: document.querySelector('#activeMeme'),
  matchTitle: document.querySelector('#matchTitle'),
  matchDetail: document.querySelector('#matchDetail'),
  signals: document.querySelector('#signals'),
  history: document.querySelector('#history'),
  library: document.querySelector('#memeLibrary')
};

els.library.innerHTML = MEMES.map((m) => `
  <div class="meme-tile">
    <img src="${m.asset}" alt="${m.title}" loading="lazy" />
    <div><strong>${m.title}</strong><span>${m.hint}</span></div>
  </div>
`).join('');

const memeImages = new Map(MEMES.map((meme) => {
  const image = new Image();
  image.decoding = 'async';
  image.src = meme.asset;
  return [meme.id, image];
}));

let faceLandmarker = null;
let poseLandmarker = null;
let gestureRecognizer = null;
let stream = null;
let running = false;
let animationId = null;
let lastVideoTime = -1;
let stableHistory = [];
let recentMatches = [];
let frameTimes = [];
let lastRenderedMatch = null;
let activeMatch = null;
let unmatchedFrames = 0;
let faceOverlayTransform = null;
let lastGestureResult = null;
let lastGestureInferenceAt = -Infinity;
let faceBaseline = null;
let calibrationSamples = [];
let calibrationComplete = false;
let smoothedFace = null;
let smoothedPose = null;
let smoothedHand = null;
const HAND_INFERENCE_INTERVAL_MS = 66;
const MATCH_RELEASE_FRAMES = 8;
const CALIBRATION_SAMPLE_COUNT = 36;

function setStatus(message, error = false) {
  els.status.textContent = message;
  els.status.classList.toggle('error', error);
}

async function loadModels() {
  if (faceLandmarker && poseLandmarker && gestureRecognizer) return;
  setStatus('Loading local MediaPipe models…');
  const vision = await FilesetResolver.forVisionTasks('/wasm');
  const createTasks = (delegate) => Promise.all([
    FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: '/models/face_landmarker.task', delegate },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: true,
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5
    }),
    PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: '/models/pose_landmarker_lite.task', delegate },
      runningMode: 'VIDEO',
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5
    }),
    GestureRecognizer.createFromOptions(vision, {
      baseOptions: { modelAssetPath: '/models/gesture_recognizer.task', delegate },
      runningMode: 'VIDEO',
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5
    })
  ]);
  try {
    [faceLandmarker, poseLandmarker, gestureRecognizer] = await createTasks('GPU');
  } catch (gpuError) {
    console.warn('GPU delegate unavailable; falling back to CPU.', gpuError);
    [faceLandmarker, poseLandmarker, gestureRecognizer] = await createTasks('CPU');
  }
  setStatus('Models loaded. Camera ready.');
}

async function startCamera() {
  els.start.disabled = true;
  try {
    await loadModels();
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false
    });
    els.video.srcObject = stream;
    await els.video.play();
    resizeCanvas();
    running = true;
    faceBaseline = null;
    calibrationSamples = [];
    calibrationComplete = false;
    smoothedFace = null;
    smoothedPose = null;
    smoothedHand = null;
    stableHistory = [];
    activeMatch = null;
    unmatchedFrames = 0;
    els.placeholder.hidden = true;
    els.stop.disabled = false;
    setStatus('Calibration: look at the camera with a neutral face…');
    animationId = requestAnimationFrame(predictLoop);
  } catch (err) {
    console.error(err);
    setStatus(formatCameraError(err), true);
    els.start.disabled = false;
  }
}

function formatCameraError(err) {
  if (err?.name === 'NotAllowedError') return 'Camera permission denied. Allow camera access and try again.';
  if (err?.name === 'NotFoundError') return 'No camera was found.';
  if (String(err?.message).includes('models')) return 'Model files are missing. Run: npm run setup';
  return `Could not start detector: ${err?.message ?? err}`;
}

function stopCamera() {
  running = false;
  if (animationId) cancelAnimationFrame(animationId);
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  els.video.srcObject = null;
  els.placeholder.hidden = false;
  els.start.disabled = false;
  els.stop.disabled = true;
  els.fps.textContent = '0 FPS';
  activeMatch = null;
  faceOverlayTransform = null;
  unmatchedFrames = 0;
  faceBaseline = null;
  calibrationSamples = [];
  calibrationComplete = false;
  smoothedFace = null;
  smoothedPose = null;
  smoothedHand = null;
  clearOverlay();
  updateActiveMemeHud(null, null);
  setStatus('Camera stopped');
}

function resizeCanvas() {
  const width = els.video.videoWidth || 1280;
  const height = els.video.videoHeight || 720;
  if (els.canvas.width !== width) els.canvas.width = width;
  if (els.canvas.height !== height) els.canvas.height = height;
}

function categoriesFromFace(result) {
  return result?.faceBlendshapes?.[0]?.categories ?? [];
}

function clearOverlay() {
  const ctx = els.canvas.getContext('2d');
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
}

function drawMemeFaceMask(faceLandmarks, match) {
  const ctx = els.canvas.getContext('2d');
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);

  if (!faceLandmarks || !match) {
    if (!faceLandmarks) faceOverlayTransform = null;
    return;
  }

  const rawTransform = computeFaceOverlayTransform(faceLandmarks, els.canvas.width, els.canvas.height, {
    widthScale: 1.42,
    heightScale: 1.42,
    verticalShift: -0.03
  });
  if (!rawTransform) return;

  faceOverlayTransform = smoothOverlayTransform(faceOverlayTransform, rawTransform, 0.40);
  const image = memeImages.get(match.id);
  if (!image?.complete || !image.naturalWidth || !image.naturalHeight) return;

  const { x, y, width, height, rotation } = faceOverlayTransform;
  const side = Math.max(width, height);
  const cover = computeCoverRect(image.naturalWidth, image.naturalHeight, side, side);
  if (!cover) return;

  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(rotation);
  ctx.globalAlpha = 1;
  ctx.drawImage(image, cover.x, cover.y, cover.width, cover.height);
  ctx.restore();
}

function updateSignals(face, pose, hand) {
  const values = [
    ['Smile', face.smile], ['Jaw open', face.jawOpen], ['Eyes wide', face.eyeWide],
    ['Squint', face.squint], ['Brows up', face.browUp], ['Brows down', face.browDown],
    ['Frown', face.frown], ['Side eye', face.sideEye], ['Mouth press', face.press],
    ['Thumbs up', hand.thumbUp], ['Pointing', hand.pointingUp], ['Peace / victory', hand.victory],
    ['Open palm', hand.openPalm], ['Closed fist', hand.closedFist], ['Pinch', hand.pinch],
    ['Hand near face', pose.handNearFace], ['One hand up', pose.oneHandUp], ['Both hands up', pose.bothHandsUp]
  ];
  els.signals.innerHTML = values.map(([label, value]) => `
    <div class="signal-row">
      <span>${label}</span>
      <div class="meter"><i style="width:${Math.round((value ?? 0) * 100)}%"></i></div>
      <b>${Math.round((value ?? 0) * 100)}%</b>
    </div>
  `).join('');
}

function updateActiveMemeHud(match, candidate) {
  els.activeMeme.classList.toggle('idle', !match);
  if (!calibrationComplete) {
    els.matchTitle.textContent = 'Calibrating neutral face';
    els.matchDetail.textContent = `${calibrationSamples.length}/${CALIBRATION_SAMPLE_COUNT} samples · Keep a relaxed, neutral expression.`;
    return;
  }
  if (match) {
    els.matchTitle.textContent = match.title;
    els.matchDetail.textContent = `${Math.round(match.score * 100)}% match · ${match.reason}`;
    return;
  }
  els.matchTitle.textContent = candidate ? `Closest: ${candidate.title}` : 'Waiting for a match';
  els.matchDetail.textContent = candidate
    ? `${Math.round(candidate.score * 100)}% · Hold the expression or pose a moment longer.`
    : 'The meme will replace your face when a detection stabilizes.';
}

function renderMatch(match, topCandidate) {
  if (match) {
    unmatchedFrames = 0;
    activeMatch = match;
    if (lastRenderedMatch?.id !== match.id) {
      recentMatches = [{ ...match, at: new Date() }, ...recentMatches.filter((m) => m.id !== match.id)].slice(0, 5);
      renderHistory();
    }
    lastRenderedMatch = match;
  } else if (!topCandidate?.matched) {
    unmatchedFrames += 1;
    if (unmatchedFrames >= MATCH_RELEASE_FRAMES) {
      activeMatch = null;
      faceOverlayTransform = null;
    }
  }
  updateActiveMemeHud(activeMatch, topCandidate);
}

function renderHistory() {
  if (!recentMatches.length) return;
  els.history.innerHTML = recentMatches.map((m) => `
    <div class="history-item">
      <img src="${m.asset}" alt="" />
      <div><strong>${m.title}</strong><span>${m.at.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span></div>
      <b>${Math.round(m.score * 100)}%</b>
    </div>
  `).join('');
}

function updateFps(now) {
  frameTimes.push(now);
  frameTimes = frameTimes.filter((t) => now - t < 1000);
  els.fps.textContent = `${frameTimes.length} FPS`;
}

async function predictLoop(now) {
  if (!running) return;
  if (els.video.readyState >= 2 && els.video.currentTime !== lastVideoTime) {
    lastVideoTime = els.video.currentTime;
    try {
      resizeCanvas();
      const timestamp = performance.now();
      const shouldRunHands = timestamp - lastGestureInferenceAt >= HAND_INFERENCE_INTERVAL_MS;
      const [faceResult, poseResult, handResult] = await Promise.all([
        faceLandmarker.detectForVideo(els.video, timestamp),
        poseLandmarker.detectForVideo(els.video, timestamp),
        shouldRunHands
          ? gestureRecognizer.recognizeForVideo(els.video, timestamp)
          : Promise.resolve(lastGestureResult)
      ]);
      if (shouldRunHands) {
        lastGestureResult = handResult;
        lastGestureInferenceAt = timestamp;
      }
      const faceLandmarks = faceResult?.faceLandmarks?.[0] ?? null;
      const rawFace = extractFaceFeatures(categoriesFromFace(faceResult));
      const rawPose = extractPoseFeatures(poseResult?.landmarks?.[0] ?? []);
      const rawHand = extractHandFeatures(handResult ?? {});

      if (!calibrationComplete) {
        if (rawFace.present) calibrationSamples.push(rawFace);
        if (calibrationSamples.length >= CALIBRATION_SAMPLE_COUNT) {
          faceBaseline = computeFaceBaseline(calibrationSamples);
          calibrationComplete = true;
          setStatus('Detecting calibrated face + pose + hands');
        }
        clearOverlay();
        updateSignals(rawFace, rawPose, rawHand);
        updateActiveMemeHud(null, null);
        updateFps(now);
        animationId = requestAnimationFrame(predictLoop);
        return;
      }

      const normalizedFace = normalizeFaceFeatures(rawFace, faceBaseline ?? {});
      smoothedFace = smoothFeatureGroup(smoothedFace, normalizedFace, 0.48);
      smoothedPose = smoothFeatureGroup(smoothedPose, rawPose, 0.58);
      smoothedHand = smoothFeatureGroup(smoothedHand, rawHand, 0.62);

      const ranked = rankMemeMatches(smoothedFace, smoothedPose, smoothedHand);
      const candidate = chooseConfidentCandidate(ranked, { minMargin: 0.09, strongScore: 0.84 });
      const stable = chooseStableMatch(stableHistory, candidate, { minFrames: 3, maxHistory: 5 });
      stableHistory = stable.history;
      renderMatch(stable.match, candidate ?? ranked[0]);
      drawMemeFaceMask(faceLandmarks, activeMatch);
      updateSignals(smoothedFace, smoothedPose, smoothedHand);
      updateFps(now);
    } catch (err) {
      console.error(err);
      setStatus(`Detection error: ${err.message}`, true);
    }
  }
  animationId = requestAnimationFrame(predictLoop);
}

els.start.addEventListener('click', startCamera);
els.stop.addEventListener('click', stopCamera);
window.addEventListener('resize', resizeCanvas);
window.addEventListener('beforeunload', stopCamera);

if (!navigator.mediaDevices?.getUserMedia) {
  els.start.disabled = true;
  setStatus('This browser does not support webcam access.', true);
} else {
  setStatus('Ready. Start the camera to calibrate detection.');
}
