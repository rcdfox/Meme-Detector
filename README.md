# Meme Detector

A local-first computer-vision web app that uses a webcam to match a user's facial expression, body pose, and hand gesture to a silly cat meme in real time. Once matched, the meme is rendered directly over the user's face as a tracked AR-style mask.

The app runs in the browser with MediaPipe. There is no account system, database, upload endpoint, or application backend. Webcam frames are processed locally on the user's device.

> **Image note:** This version includes a starter set of cat meme images supplied for the project. Before public redistribution, verify that you have permission to redistribute each image. More cat or hamster memes can be added later without retraining the MediaPipe models.

## Current meme library

| Meme | Main trigger |
|---|---|
| Deadpan Cat | Neutral, straight-faced stare |
| Silly Tongue Cat | Open mouth + smile + wide eyes |
| Crying Cat | Frown + worried / raised brows |
| Launch Cat | Both hands raised above shoulders |
| Nerd Cat | Wide eyes + raised brows |
| Happy Cat | Bright smile + open eyes |
| Tired Cat | Squint + open mouth / tired expression |
| Smug Cat | Small smile + squint or sideways glance |
| Buffering Cat | Very wide eyes + mostly closed mouth |
| Judging Cat | Side-eye + squint |
| Thinking Cat | Hand near chin/mouth + pointing or pinch signal |
| Concerned Cat | Lowered brows + frown |
| Crying Thumbs-Up Cat | Sad face + recognized thumbs-up |

The matching layer is rule-based, which makes it straightforward to add and tune memes.

# Getting started

## Prerequisites

Install:

- Git
- Node.js 20.19+ or a newer supported Node.js LTS release
- npm, which is included with Node.js
- a modern browser with webcam support
- a webcam

Check whether Node.js and npm are already installed:

```bash
node --version
npm --version
```

If either command is unavailable, install a current Node.js LTS release from https://nodejs.org/.

On macOS with Homebrew:

```bash
brew install node
```

## Clone and install

```bash
git clone https://github.com/rcdfox/Meme-Detector.git
cd Meme-Detector
npm install
```

## Run

```bash
npm start
```

On the first run, setup will:

1. copy MediaPipe's WASM runtime from `node_modules` into `public/wasm`;
2. download the Face Landmarker model;
3. download the Pose Landmarker Lite model;
4. download the Gesture Recognizer model;
5. start Vite.

Open the local address shown in the terminal, normally:

```text
http://localhost:5173
```

Click **Start camera** and allow webcam access.

Do not open `index.html` directly. Run the project through Vite so browser camera APIs and module loading work correctly.

After initial setup, the normal workflow is:

```bash
cd Meme-Detector
npm start
```

You can start Vite without re-running model setup with:

```bash
npm run dev
```

# Architecture

```text
Webcam
  |
  v
Browser getUserMedia()
  |
  +----------------+----------------+
  |                |                |
  v                v                v
Face Landmarker  Pose Landmarker  Gesture Recognizer
  |                |                |
  v                v                v
blendshapes       body points      hand points + gesture classes
  |                |                |
  +----------------+----------------+
                   |
                   v
             feature extraction
                   |
                   v
              rule scoring
                   |
                   v
        multi-frame stabilization
                   |
                   v
            best meme displayed
```

## Face recognition

The Face Landmarker produces face landmarks and blendshape scores. The app currently derives signals such as:

- smile
- jaw open
- eye widen
- blink
- squint
- brow raise
- brow lower
- mouth pucker
- mouth press
- frown
- side-eye

## Pose recognition

The Pose Landmarker provides 33 normalized body landmarks. The app primarily uses the nose, shoulders, elbows, wrists, and hips to estimate:

- one or both hands raised
- hand near face
- crossed arms
- flex geometry
- shrug geometry
- dab geometry

## Hand and finger recognition

The Gesture Recognizer detects up to two hands and returns 21 landmarks per hand.

The app currently exposes:

- thumbs up
- thumbs down
- pointing up
- victory / peace sign
- open palm
- closed fist
- I-love-you gesture
- pinch distance from thumb/index landmarks
- approximate extended-finger count

Hand recognition runs at a slightly lower cadence than face and pose inference to reduce browser CPU/GPU load. The latest hand result is reused between hand-inference frames.

The **Crying Thumbs-Up Cat** now uses an actual recognized thumbs-up rather than approximating the gesture from wrist position.

## Face-tracked meme mask

The selected meme is no longer displayed in a separate side panel. It is drawn onto the live camera canvas over the detected face.

The overlay uses Face Landmarker geometry to track:

- face center position;
- face width and height;
- head-roll angle from the eye line;
- smoothed movement between frames to reduce jitter.

The meme stays active for a short release window when a rule briefly drops below its threshold, which reduces flicker while the user is moving. The source image is aspect-filled into an elliptical face mask and follows the user's face as they move around the frame.

# Commands

| Command | Purpose |
|---|---|
| `npm install` | Install dependencies |
| `npm run setup` | Copy WASM files and download MediaPipe models |
| `npm start` | Run setup and start the development server |
| `npm run dev` | Start Vite without re-running setup |
| `npm test` | Run unit tests |
| `npm run build` | Build production files into `dist/` |
| `npm run preview` | Preview the production build |

# Project structure

```text
Meme-Detector/
├── index.html
├── package.json
├── package-lock.json
├── README.md
├── LICENSE
├── public/
│   ├── memes/                 # Cat meme WebP images
│   ├── models/                # Downloaded MediaPipe .task models
│   └── wasm/                  # Local MediaPipe WASM runtime
├── scripts/
│   └── setup-assets.mjs
├── src/
│   ├── detection.js
│   ├── main.js
│   └── style.css
└── test/
    └── detection.test.js
```

# How matching works

Each meme has a weighted rule in `rankMemeMatches()`.

Examples:

- **Buffering Cat** emphasizes wide eyes with a mostly closed mouth.
- **Judging Cat** emphasizes side-eye and squinting.
- **Launch Cat** emphasizes both wrists above the shoulders.
- **Thinking Cat** combines hand-near-face pose data with pointing/pinch signals.
- **Crying Thumbs-Up Cat** combines a sad facial expression with Gesture Recognizer's thumbs-up confidence.

The output is not a trained meme classifier. It is a deterministic rule layer on top of pretrained MediaPipe models. This makes the prototype easier to understand, debug, tune, and extend.

A candidate must also remain strong across multiple recent frames before becoming the displayed match. This reduces flicker and brief false detections.

# Adding another cat or hamster meme

1. Put the image in `public/memes/`.
2. Add an entry to the `MEMES` array in `src/detection.js`.
3. Add or tune a scoring rule in `rankMemeMatches()`.
4. Run `npm test`.
5. Run `npm start` and use the live signal meters to tune thresholds.

Example metadata:

```js
{
  id: 'new-meme',
  title: 'New Meme',
  asset: '/memes/new-meme.webp',
  hint: 'What the user should do to trigger it.'
}
```

Example rule:

```js
[
  'new-meme',
  0.6 * h.victory + 0.4 * f.smile,
  0.55,
  'peace sign + smile'
]
```

The face, pose, and hand feature objects are all available to scoring rules.

# Privacy and local processing

This project:

- requests webcam permission through the browser;
- reads frames from the browser's video element;
- runs MediaPipe inference in the browser;
- renders landmarks locally;
- selects and displays memes locally;
- does not define an API server;
- does not define a database;
- does not save webcam images;
- does not record webcam video.

An internet connection is needed initially for `npm install` and to download the official MediaPipe model files. After setup, the models and WASM runtime are loaded from:

```text
public/models/
public/wasm/
```

The repository itself does not upload or store webcam frames.

MediaPipe Tasks may still send API performance or utilization metrics, so the project is best described as **local camera/image processing with no project backend or cloud image storage**, rather than guaranteeing that the browser never makes any outbound network request.

# Troubleshooting

## Node or npm command not found

Install Node.js. npm comes with Node.js.

Then verify:

```bash
node --version
npm --version
```

and run:

```bash
npm install
npm start
```

## Model files are missing

Run:

```bash
npm run setup
npm run dev
```

## Camera permission denied

Open the app through `http://localhost:5173`, allow camera permission for localhost, and reload the page.

## Detection is slow

Try Chrome or Edge, close GPU-heavy applications/tabs, lower webcam resolution in `src/main.js`, or temporarily disable one of the MediaPipe tasks while debugging.

## Wrong meme triggers

Use the live detection signal meters, then adjust weights or thresholds in `rankMemeMatches()`.

## Meme flickers

Increase `minFrames` or `maxHistory` in the `chooseStableMatch()` call in `src/main.js`.

# Testing

Run:

```bash
npm test
```

Current unit tests cover:

- face blendshape extraction
- squint and brow-down extraction
- Buffering Cat matching
- Thinking Cat matching
- hands-up / Launch Cat matching
- Deadpan Cat matching
- multi-frame stabilization
- thumbs-up gesture extraction
- sad face + thumbs-up / Crying Thumbs-Up Cat matching

Manual testing should also cover multiple users, lighting conditions, camera distances, backgrounds, and hand orientations.

# Building

```bash
npm run build
npm run preview
```

# Possible next upgrades

- add more cat and hamster meme classes
- add custom-trained hand gestures beyond MediaPipe's built-in gesture classes
- add user-adjustable sensitivity
- add a hold-pose progress indicator
- add screenshot capture
- support multiple people
- add a meme library browser
- add UI controls for confidence thresholds
- package the app with Electron or Tauri

# License

The project source code is released under the MIT License.

The supplied meme images may have separate copyright or licensing terms and are not automatically relicensed by the MIT License. Verify redistribution rights before publishing them publicly.
