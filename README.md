# Meme Detector

A local-first computer-vision web app that watches a webcam feed, estimates the user's facial expression and body pose, and displays the closest matching meme-style image beside the camera.

The app is designed for a simple **clone -> install -> run** workflow. There is no account system, database, upload endpoint, or application backend. Webcam frames are processed locally in the browser with MediaPipe Face Landmarker and Pose Landmarker.

> Course/project note: the starter meme artwork in this repository is intentionally original SVG artwork rather than copied third-party meme images. This avoids shipping copyrighted image files and makes the initial repository self-contained. Real meme images can be added later by replacing the SVG assets or adding new entries to the meme library.

## What it detects

The initial rule set combines **face blendshapes** and **body landmarks**. It currently includes:

| Detection | Main signal |
|---|---|
| Screaming Cat | open mouth + wide eyes + hands near face |
| Shocked Cat | open mouth + wide eyes |
| Grinning Cat | strong smile |
| Side-Eye Cat | strong sideways eye movement |
| Thinking Cat | hand near face/chin |
| Suspicious Cat | raised eyebrows + wide eyes |
| Absolute Victory | both hands above shoulders |
| Dab Detected | one hand near face + opposite arm extended |
| Built Different | double-arm flex pose |
| It Is What It Is | both bent arms/hands near shoulders |
| Not Impressed | crossed arms |

The matcher is deliberately easy to extend. Adding a new meme is primarily a matter of adding an image and one scoring rule.

---

## Architecture

```text
Webcam
  |
  v
Browser getUserMedia()
  |
  +-----------------------+
  |                       |
  v                       v
MediaPipe             MediaPipe
Face Landmarker       Pose Landmarker
  |                       |
  v                       v
Face blendshapes       33 body landmarks
  |                       |
  +-----------+-----------+
              |
              v
       Feature extraction
              |
              v
        Rule scoring
              |
              v
  Multi-frame stabilization
              |
              v
      Best meme displayed
```

### Why a local browser app?

This architecture gives the project several useful properties:

- webcam permission is handled by the browser;
- no server needs to receive or store video frames;
- the UI is cross-platform on macOS, Windows, and Linux;
- the MediaPipe models and WASM runtime are copied/downloaded locally during setup;
- after setup, the application can use local model/WASM files instead of a model CDN;
- adding visualizations and new meme classes does not require retraining a neural network.

MediaPipe's package documentation states that input processing occurs on-device and that input images/video are not sent to Google servers. It also states that MediaPipe Tasks may send API performance/utilization metrics to Google. This repository itself does not upload, store, or transmit webcam frames.

---

## Prerequisites

Install:

- **Git**
- **Node.js 20.19+ or a newer supported Node release**
- a current Chromium-based browser or Firefox with webcam support
- a webcam

Chrome/Edge are recommended for the most predictable WebAssembly/WebGL behavior.

Check your versions:

```bash
git --version
node --version
npm --version
```

---

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/rcdfox/Meme-Detector.git
cd Meme-Detector
```

### 2. Install JavaScript dependencies

```bash
npm install
```

### 3. Run the app

```bash
npm start
```

`npm start` performs the local asset setup and then starts the Vite development server.

On the first run, setup will:

1. copy MediaPipe's WASM runtime from `node_modules/@mediapipe/tasks-vision/wasm` into `public/wasm`;
2. download the official Face Landmarker model into `public/models`;
3. download the official Pose Landmarker Lite model into `public/models`;
4. start the local web server.

Open the local URL printed in the terminal, normally:

```text
http://localhost:5173
```

Press **Start camera** and allow camera access when the browser asks.

### Later runs

The setup script skips model files that are already present, so you can continue using:

```bash
npm start
```

Or, after setup has already completed:

```bash
npm run dev
```

---

## Commands

| Command | Purpose |
|---|---|
| `npm install` | Install Vite and MediaPipe Tasks Vision |
| `npm run setup` | Copy WASM files and download local model files |
| `npm start` | Run setup, then start the local development server |
| `npm run dev` | Start Vite without re-running setup |
| `npm test` | Run unit tests for feature/matching logic |
| `npm run build` | Produce a production build in `dist/` |
| `npm run preview` | Preview the production build locally |

---

## Project structure

```text
Meme-Detector/
├── index.html
├── package.json
├── README.md
├── LICENSE
├── public/
│   ├── memes/                 # Starter meme-style SVG images
│   ├── models/                # Downloaded MediaPipe .task models
│   └── wasm/                  # Local MediaPipe WASM runtime
├── scripts/
│   └── setup-assets.mjs       # One-time/local asset preparation
├── src/
│   ├── detection.js           # Feature extraction + meme scoring rules
│   ├── main.js                # Webcam, MediaPipe inference, UI updates
│   └── style.css              # App styling
└── test/
    └── detection.test.js      # Unit tests for matcher behavior
```

---

## How detection works

### 1. Face Landmarker

The face model produces facial landmarks and blendshape scores. The app currently reads signals including:

- `mouthSmileLeft` / `mouthSmileRight`
- `jawOpen`
- `eyeWideLeft` / `eyeWideRight`
- `eyeBlinkLeft` / `eyeBlinkRight`
- eyebrow raise blendshapes
- mouth pucker/frown blendshapes
- eye look-in/look-out blendshapes

Those values are converted into normalized features such as `smile`, `jawOpen`, `eyeWide`, `browUp`, and `sideEye`.

### 2. Pose Landmarker

The pose model provides 33 normalized body landmarks. The current rules use the nose, shoulders, elbows, wrists, and hips to estimate:

- hands above shoulders;
- hand near face;
- both hands near face;
- elbow bend;
- extended arms;
- crossed arms;
- flex geometry;
- shrug geometry;
- dab geometry.

### 3. Meme scoring

Each meme has a weighted score. For example, the Shocked Cat rule gives weight to jaw opening and eye widening. The victory rule strongly weights both wrists being above the shoulders.

The output is **not a trained meme classifier**. It is a deterministic rule layer on top of pretrained face/pose landmark models. This makes the prototype easy to understand, tune, and extend.

### 4. Stabilization

A meme does not appear because of one noisy frame. The best candidate must repeat across multiple recent frames before it becomes the displayed match. This reduces flicker and accidental detections.

---

## Adding a new meme

### Step 1: Add an image

Place an image in:

```text
public/memes/
```

Recommended formats:

- `.svg`
- `.png`
- `.jpg`
- `.webp`

Use images you created, own, have permission to redistribute, or that have a license compatible with your project.

### Step 2: Register it

Open `src/detection.js` and add an item to `MEMES`:

```js
{
  id: 'new-meme',
  title: 'New Meme',
  asset: '/memes/new-meme.png',
  hint: 'What the user should do to trigger it.'
}
```

### Step 3: Add a scoring rule

In `rankMemeMatches()`, add a rule:

```js
[
  'new-meme',
  0.7 * f.smile + 0.3 * p.oneHandUp,
  0.55,
  'smile + one raised hand'
]
```

The four fields are:

```text
[meme id, score expression, detection threshold, human-readable reason]
```

### Step 4: Tune using the diagnostics panel

Run the app and watch the live signal percentages below the camera. Those meters make it easier to choose reasonable weights and thresholds for different people and cameras.

---

## Adding actual meme images later

The starter SVGs are placeholders by design. To use recognizable meme artwork:

1. confirm that you can legally redistribute the image;
2. copy it into `public/memes/`;
3. update the matching `asset` path in `src/detection.js`;
4. optionally add an attribution/license section to this README.

The detection logic does not depend on the image format or artwork.

---

## Privacy and local processing

### What this repository does

- asks the browser for webcam permission;
- reads frames directly from the `<video>` element;
- runs MediaPipe inference in the browser;
- renders landmarks and the selected meme locally;
- does **not** define an API server;
- does **not** define a database;
- does **not** save webcam images or recordings;
- does **not** include analytics code of its own.

### Initial setup network access

The first setup needs network access for `npm install` and for downloading the two official MediaPipe model files. After those files are present, model inference uses local files under `public/models` and `public/wasm`.

### MediaPipe telemetry caveat

The official `@mediapipe/tasks-vision` package privacy notice says input data is processed on-device and is not sent to Google servers, but it also says API performance/utilization metrics may be sent to Google. Therefore, this project should be described as **local camera/image processing with no project backend or cloud image storage**, rather than claiming that the browser will never make any outbound network request under any circumstance.

---

## Troubleshooting

### `Model files are missing. Run: npm run setup`

Run:

```bash
npm run setup
```

Then restart:

```bash
npm run dev
```

### Camera permission denied

In your browser's site settings, allow camera permission for `localhost`, reload the page, and press **Start camera** again.

### No camera found

Check that another application is not exclusively using the camera and that your OS has granted camera permission to the browser.

### Detection is slow

Try:

- Chrome or Edge;
- closing other GPU-heavy tabs/apps;
- reducing webcam resolution in `src/main.js`;
- running only one of the two landmarkers while debugging.

The included pose model is the **Lite** version to keep the prototype responsive.

### A pose triggers the wrong meme

Use the signal meters to identify which feature is high, then tune the weights/thresholds in `rankMemeMatches()`.

### A meme flickers on and off

Increase `minFrames` or `maxHistory` in the `chooseStableMatch()` call inside `src/main.js`.

---

## Testing

The matching logic is separated from the webcam/UI code so it can be unit tested without a camera.

Run:

```bash
npm test
```

Current tests verify:

- face blendshape feature extraction;
- smile -> Grinning Cat matching;
- hands-up -> Victory matching;
- multi-frame stabilization behavior.

For final project testing, also manually validate every pose in different lighting conditions and with multiple users.

---

## Possible next upgrades

Good next iterations include:

- add MediaPipe Hand Landmarker for finger gestures;
- add more meme classes and licensed image assets;
- add user-adjustable sensitivity sliders;
- add a three-second "hold pose" progress ring;
- add screenshots that are saved **only when the user explicitly clicks Save**;
- add multi-person detection;
- collect a small labeled pose dataset and replace some heuristics with a lightweight classifier;
- export detection events to a local JSON file for testing;
- package the web app as an Electron/Tauri desktop application.

---

## Dependencies

- [`@mediapipe/tasks-vision`](https://www.npmjs.com/package/@mediapipe/tasks-vision) — face and pose inference
- [Vite](https://vite.dev/) — local development/build tooling

The current package versions are pinned in `package.json` for reproducibility.

## License

Project source code and the original starter SVG artwork are released under the MIT License. See `LICENSE`.
