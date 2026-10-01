import './style.css';
import {
  FaceLandmarker,
  FilesetResolver,
  PoseLandmarker,
  DrawingUtils
} from '@mediapipe/tasks-vision';
import {
  MEMES,
  chooseStableMatch,
  extractFaceFeatures,
  extractPoseFeatures,
  rankMemeMatches
} from './detection.js';

const app = document.querySelector('#app');

app.innerHTML = `
  <main class="shell">
    <header class="topbar">
      <div>
        <p class="eyebrow">LOCAL COMPUTER VISION</p>
        <h1>Meme Detector</h1>
        <p class="subtitle">Match your expression or pose to a meme in real time. Camera frames are processed on your device.</p>
      </div>
      <div class="privacy-pill"><span></span> Local camera processing</div>
    </header>

    <section class="workspace">
      <article class="camera-card panel">
        <div class="panel-head">
          <div>
            <p class="label">LIVE CAMERA</p>
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
            <div class="placeholder-icon">◎</div>
            <strong>Camera is off</strong>
            <span>Press “Start camera” and allow browser access.</span>
          </div>
          <div class="fps" id="fps">0 FPS</div>
        </div>
      </article>

      <article class="match-card panel">
        <div class="panel-head">
          <div>
            <p class="label">BEST MATCH</p>
            <p id="confidence" class="status">Waiting for camera</p>
          </div>
        </div>
        <div id="matchStage" class="match-stage empty">
          <img id="matchImage" alt="Detected meme" />
          <div id="emptyMatch">
            <div class="big-question">?</div>
            <h2>Show me a meme pose</h2>
            <p>Try smiling, looking shocked, raising both arms, shrugging, flexing, crossing your arms, or dabbing.</p>
          </div>
        </div>
        <div class="match-meta">
          <div><span>Detected</span><strong id="matchTitle">Nothing yet</strong></div>
          <div><span>Signal</span><strong id="matchReason">—</strong></div>
        </div>
      </article>
    </section>

    <section class="lower-grid">
      <article class="panel diagnostics">
        <div class="panel-head"><div><p class="label">DETECTION SIGNALS</p><p class="status">Useful for tuning thresholds</p></div></div>
        <div id="signals" class="signals"></div>
      </article>
      <article class="panel history-panel">
        <div class="panel-head"><div><p class="label">RECENT MATCHES</p><p class="status">Stabilized detections only</p></div></div>
        <div id="history" class="history"><p class="muted">No matches yet.</p></div>
      </article>
    </section>

    <section class="library-section">
      <div class="section-title">
        <div><p class="eyebrow">STARTER LIBRARY</p><h2>Try these detections</h2></div>
        <p>These starter graphics are original local SVGs, so the repo can be distributed without bundling third-party meme artwork.</p>
      </div>
      <div id="memeLibrary" class="meme-library"></div>
    </section>

    <footer>
      <span>No account. No uploads. No backend.</span>
      <span>MediaPipe Face + Pose Landmarker</span>
    </footer>
  </main>
`;

const els = {
  video: document.querySelector('#webcam'),
  canvas: document.querySelector('#overlay'),
  placeholder: document.querySelector('#cameraPlaceholder'),
  start: document.querySelector('#startBtn'),
  stop: document.querySelector('#stopBtn'),
  status: document.querySelector('#status'),
  fps: document.querySelector('#fps'),
  matchStage: document.querySelector('#matchStage'),
  matchImage: document.querySelector('#matchImage'),
  emptyMatch: document.querySelector('#emptyMatch'),
  matchTitle: document.querySelector('#matchTitle'),
  matchReason: document.querySelector('#matchReason'),
  confidence: document.querySelector('#confidence'),
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

let faceLandmarker = null;
let poseLandmarker = null;
let stream = null;
let running = false;
let animationId = null;
let lastVideoTime = -1;
let stableHistory = [];
let recentMatches = [];
let frameTimes = [];
let lastRenderedMatch = null;

function setStatus(message, error = false) {
  els.status.textContent = message;
  els.status.classList.toggle('error', error);
}

async function loadModels() {
  if (faceLandmarker && poseLandmarker) return;
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
    })
  ]);
  try {
    [faceLandmarker, poseLandmarker] = await createTasks('GPU');
  } catch (gpuError) {
    console.warn('GPU delegate unavailable; falling back to CPU.', gpuError);
    [faceLandmarker, poseLandmarker] = await createTasks('CPU');
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
    els.placeholder.hidden = true;
    els.stop.disabled = false;
    setStatus('Detecting face + pose');
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
  setStatus('Camera stopped');
}

function resizeCanvas() {
  const width = els.video.videoWidth || 1280;
  const height = els.video.videoHeight || 720;
  els.canvas.width = width;
  els.canvas.height = height;
}

function categoriesFromFace(result) {
  return result?.faceBlendshapes?.[0]?.categories ?? [];
}

function drawLandmarks(faceResult, poseResult) {
  const ctx = els.canvas.getContext('2d');
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  const drawing = new DrawingUtils(ctx);

  const face = faceResult?.faceLandmarks?.[0];
  if (face) {
    drawing.drawConnectors(face, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: '#7dd3fc', lineWidth: 1.5 });
  }
  const pose = poseResult?.landmarks?.[0];
  if (pose) {
    drawing.drawConnectors(pose, PoseLandmarker.POSE_CONNECTIONS, { color: '#a7f3d0', lineWidth: 2 });
    drawing.drawLandmarks(pose, { color: '#ecfeff', radius: 2 });
  }
}

function updateSignals(face, pose) {
  const values = [
    ['Smile', face.smile], ['Jaw open', face.jawOpen], ['Eyes wide', face.eyeWide],
    ['Brows up', face.browUp], ['Side eye', face.sideEye], ['Hand near face', pose.handNearFace],
    ['Both hands up', pose.bothHandsUp], ['Flex', pose.flex], ['Shrug', pose.shrug],
    ['Dab', pose.dab], ['Arms crossed', pose.armsCrossed]
  ];
  els.signals.innerHTML = values.map(([label, value]) => `
    <div class="signal-row">
      <span>${label}</span>
      <div class="meter"><i style="width:${Math.round((value ?? 0) * 100)}%"></i></div>
      <b>${Math.round((value ?? 0) * 100)}%</b>
    </div>
  `).join('');
}

function renderMatch(match, topCandidate) {
  if (match) {
    if (lastRenderedMatch?.id !== match.id) {
      recentMatches = [{ ...match, at: new Date() }, ...recentMatches.filter((m) => m.id !== match.id)].slice(0, 5);
      renderHistory();
    }
    lastRenderedMatch = match;
    els.matchStage.classList.remove('empty');
    els.matchImage.src = match.asset;
    els.matchImage.alt = match.title;
    els.matchImage.hidden = false;
    els.emptyMatch.hidden = true;
    els.matchTitle.textContent = match.title;
    els.matchReason.textContent = match.reason;
    els.confidence.textContent = `${Math.round(match.score * 100)}% rule confidence`;
  } else {
    els.confidence.textContent = topCandidate ? `Closest: ${topCandidate.title} (${Math.round(topCandidate.score * 100)}%)` : 'Looking for a face or pose';
  }
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
      const timestamp = performance.now();
      const [faceResult, poseResult] = await Promise.all([
        faceLandmarker.detectForVideo(els.video, timestamp),
        poseLandmarker.detectForVideo(els.video, timestamp)
      ]);
      const face = extractFaceFeatures(categoriesFromFace(faceResult));
      const pose = extractPoseFeatures(poseResult?.landmarks?.[0] ?? []);
      const ranked = rankMemeMatches(face, pose);
      const top = ranked[0];
      const stable = chooseStableMatch(stableHistory, top, { minFrames: 3, maxHistory: 5 });
      stableHistory = stable.history;
      drawLandmarks(faceResult, poseResult);
      updateSignals(face, pose);
      renderMatch(stable.match, top);
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
  setStatus('Ready. First run requires npm run setup.');
}
