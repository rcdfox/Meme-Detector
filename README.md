# Meme Detector

A local-first computer-vision web app that watches a webcam feed, estimates the user's facial expression and body pose, and displays the closest matching meme-style image beside the camera.

The app is designed for a simple **clone → install → run** workflow. There is no account system, database, upload endpoint, or application backend. Webcam frames are processed locally in the browser using MediaPipe Face Landmarker and Pose Landmarker.

> **Project note:** The starter meme artwork in this repository is intentionally original SVG artwork rather than copied third-party meme images. This keeps the initial repository self-contained and avoids redistributing copyrighted meme images. Recognizable meme images can be added later if their licenses permit redistribution.

---

## What It Detects

The initial rule set combines **face blendshapes** and **body landmarks**.

| Detection | Main Signal |
|---|---|
| Screaming Cat | Open mouth + wide eyes + hands near face |
| Shocked Cat | Open mouth + wide eyes |
| Grinning Cat | Strong smile |
| Side-Eye Cat | Strong sideways eye movement |
| Thinking Cat | Hand near face/chin |
| Suspicious Cat | Raised eyebrows + wide eyes |
| Absolute Victory | Both hands above shoulders |
| Dab Detected | One hand near face + opposite arm extended |
| Built Different | Double-arm flex pose |
| It Is What It Is | Both bent arms/hands near shoulders |
| Not Impressed | Crossed arms |

The matcher is deliberately easy to extend. Adding another meme primarily requires adding an image and defining a scoring rule.

---

# Getting Started

## Prerequisites

Before running Meme Detector, make sure the following are installed:

- **Git**
- **Node.js 20.19+**, or a newer supported Node.js LTS release
- **npm**, which is installed automatically with Node.js
- A modern browser with webcam support
- A webcam

Chrome or Edge are recommended for the most predictable WebAssembly and WebGL behavior.

### 1. Check Whether You Already Have Node.js

Open a terminal and run:

```bash
node --version
npm --version
```

If both commands return version numbers, Node.js and npm are already installed.

Example:

```text
v22.x.x
10.x.x
```

You can continue to the **Clone the Repository** section.

### 2. Install Node.js If Needed

If either command returns something similar to:

```text
command not found: node
```

or:

```text
command not found: npm
```

install Node.js.

**You do not need to install npm separately.** npm is included with Node.js.

The easiest cross-platform option is to download a current Node.js LTS release from:

https://nodejs.org/

### macOS with Homebrew

If you already use Homebrew:

```bash
brew install node
```

### Windows

Install the current Node.js LTS release from:

https://nodejs.org/

The standard installer includes both Node.js and npm.

### Linux

Node.js can be installed through your distribution's package manager or through the official Node.js installation options.

After installation, close and reopen your terminal if necessary and verify:

```bash
node --version
npm --version
```

---

## Clone the Repository

Clone Meme Detector from GitHub:

```bash
git clone https://github.com/rcdfox/Meme-Detector.git
cd Meme-Detector
```

If you already have the repository downloaded, simply navigate into its folder:

```bash
cd path/to/Meme-Detector
```

---

## Install Dependencies

From inside the `Meme-Detector` directory, run:

```bash
npm install
```

This installs the JavaScript dependencies defined in `package.json`, including:

- MediaPipe Tasks Vision
- Vite

You normally only need to run `npm install` once after cloning the repository.

Run it again if:

- `package.json` changes;
- dependencies are updated; or
- you delete the `node_modules` directory.

---

## Run the App

Start Meme Detector with:

```bash
npm start
```

`npm start` prepares the local MediaPipe assets and starts the Vite development server.

On the first run, setup will:

1. copy MediaPipe's WASM runtime from `node_modules/@mediapipe/tasks-vision/wasm` into `public/wasm`;
2. download the official Face Landmarker model into `public/models`;
3. download the official Pose Landmarker Lite model into `public/models`;
4. start the local development server.

You should see output similar to:

```text
Local: http://localhost:5173/
```

Open that address in your browser.

Normally:

```text
http://localhost:5173
```

Click **Start Camera** and allow the browser to access your webcam.

> Do not open `index.html` directly from Finder or File Explorer. Run the application through the local Vite server using `npm start`.

---

## Normal Workflow After Initial Setup

After the first installation and setup, starting the application is normally just:

```bash
cd Meme-Detector
npm start
```

The setup script skips model files that are already present.

You can also start Vite without rerunning asset setup:

```bash
npm run dev
```

---

# Architecture

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

## Why a Local Browser App?

This architecture gives the project several useful properties:

- webcam permission is handled by the browser;
- no project server receives or stores video frames;
- the UI works across macOS, Windows, and Linux;
- MediaPipe models and the WASM runtime are stored locally after setup;
- model inference can use local model and WASM files rather than a model CDN;
- adding visualizations and new meme classes does not require retraining a neural network.

MediaPipe's package documentation states that input processing occurs on-device and input images/video are not sent to Google servers. MediaPipe Tasks may still send API performance or utilization metrics.

This repository itself does not upload, save, or transmit webcam frames.

---

# Commands

| Command | Purpose |
|---|---|
| `npm install` | Install Vite, MediaPipe, and other project dependencies |
| `npm run setup` | Copy WASM files and download local MediaPipe models |
| `npm start` | Run setup and then start the local development server |
| `npm run dev` | Start Vite without rerunning setup |
| `npm test` | Run unit tests for feature and matching logic |
| `npm run build` | Produce a production build in `dist/` |
| `npm run preview` | Preview the production build locally |

---

# Project Structure

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
│   └── setup-assets.mjs       # Local asset preparation
├── src/
│   ├── detection.js           # Feature extraction + meme scoring rules
│   ├── main.js                # Webcam, MediaPipe inference, UI updates
│   └── style.css              # App styling
└── test/
    └── detection.test.js      # Unit tests for matcher behavior
```

---

# How Detection Works

## 1. Face Landmarker

The Face Landmarker produces facial landmarks and blendshape scores.

The application currently reads signals including:

- `mouthSmileLeft`
- `mouthSmileRight`
- `jawOpen`
- `eyeWideLeft`
- `eyeWideRight`
- `eyeBlinkLeft`
- `eyeBlinkRight`
- eyebrow raise blendshapes
- mouth pucker/frown blendshapes
- eye look-in/look-out blendshapes

These values are converted into normalized features such as:

```text
smile
jawOpen
eyeWide
browUp
sideEye
```

---

## 2. Pose Landmarker

The Pose Landmarker provides 33 normalized body landmarks.

The current detection rules primarily use:

- nose;
- shoulders;
- elbows;
- wrists;
- hips.

These landmarks are used to estimate features including:

- hands above shoulders;
- hand near face;
- both hands near face;
- elbow bend;
- extended arms;
- crossed arms;
- flex geometry;
- shrug geometry;
- dab geometry.

---

## 3. Meme Scoring

Each meme has a weighted detection score.

For example, the **Shocked Cat** rule gives significant weight to:

- jaw opening;
- eye widening.

The **Absolute Victory** rule strongly weights both wrists being above the shoulders.

The output is **not a trained meme classifier**.

Instead, the application uses a deterministic rule layer on top of pretrained face and pose landmark models.

This makes the prototype:

- easier to understand;
- easier to debug;
- easier to tune;
- easier to extend.

---

## 4. Multi-Frame Stabilization

A meme is not displayed because of a single noisy frame.

The best candidate must repeatedly score highly across several recent frames before it becomes the displayed match.

This reduces:

- flickering;
- accidental detections;
- brief tracking errors;
- rapid switching between memes.

---

# Adding a New Meme

## Step 1: Add an Image

Place the image inside:

```text
public/memes/
```

Recommended formats include:

- `.svg`
- `.png`
- `.jpg`
- `.webp`

Use images that you:

- created;
- own;
- have permission to redistribute; or
- can redistribute under an appropriate license.

---

## Step 2: Register the Meme

Open:

```text
src/detection.js
```

Add an item to the `MEMES` array:

```js
{
  id: 'new-meme',
  title: 'New Meme',
  asset: '/memes/new-meme.png',
  hint: 'What the user should do to trigger it.'
}
```

---

## Step 3: Add a Scoring Rule

Inside `rankMemeMatches()`, add a rule such as:

```js
[
  'new-meme',
  0.7 * f.smile + 0.3 * p.oneHandUp,
  0.55,
  'smile + one raised hand'
]
```

The fields are:

```text
[
  meme ID,
  score expression,
  detection threshold,
  human-readable reason
]
```

---

## Step 4: Tune the Detector

Run:

```bash
npm start
```

Use the live diagnostic signal percentages displayed by the application.

These values can help determine whether:

- a feature threshold is too high;
- a feature threshold is too low;
- another feature should be included;
- a rule should receive more or less weight.

---

# Adding Recognizable Meme Images

The starter SVGs are placeholders by design.

To use recognizable meme artwork:

1. confirm that you can legally redistribute the image;
2. copy the image into `public/memes/`;
3. update the matching `asset` path in `src/detection.js`;
4. add attribution or licensing information to this README when required.

The detection system does not depend on the artwork or image format.

---

# Privacy and Local Processing

## What the Repository Does

The application:

- asks the browser for webcam permission;
- reads frames directly from the browser's `<video>` element;
- runs MediaPipe inference in the browser;
- renders landmarks locally;
- selects and displays memes locally;
- does **not** define an API server;
- does **not** define a database;
- does **not** save webcam images;
- does **not** record webcam video;
- does **not** include project-specific analytics.

---

## Initial Setup Network Access

An internet connection is required initially for:

```bash
npm install
```

and for downloading the two official MediaPipe model files during setup.

After these files are installed, model inference uses local files from:

```text
public/models/
public/wasm/
```

The application does not require webcam frames to be uploaded to an application backend.

---

## MediaPipe Telemetry Caveat

The official `@mediapipe/tasks-vision` package states that input data is processed on-device and is not sent to Google servers.

MediaPipe Tasks may still send API performance or utilization metrics.

For this reason, this project is best described as:

> **Local camera and image processing with no project backend or cloud image storage.**

It should not be described as guaranteeing that the browser will never make any outbound network request.

---

# Troubleshooting

## `node` or `npm` Command Not Found

If:

```bash
node --version
```

or:

```bash
npm --version
```

returns a command-not-found error, install Node.js.

npm is included automatically with Node.js.

Download a current Node.js LTS release from:

https://nodejs.org/

On macOS with Homebrew:

```bash
brew install node
```

After installing Node.js, close and reopen the terminal if necessary.

Verify:

```bash
node --version
npm --version
```

Then return to the project directory and run:

```bash
npm install
npm start
```

---

## `Model files are missing. Run: npm run setup`

Run:

```bash
npm run setup
```

Then restart the application:

```bash
npm run dev
```

---

## Camera Permission Denied

Make sure you opened the application through:

```text
http://localhost:5173
```

Then check your browser's site permissions and allow camera access for `localhost`.

Reload the page and click **Start Camera** again.

---

## No Camera Found

Check that:

- the computer has a working webcam;
- another application is not exclusively using the camera;
- your operating system has granted camera permission to the browser;
- the browser has permission to access the webcam.

---

## Detection Is Slow

Try:

- Chrome or Edge;
- closing GPU-heavy browser tabs;
- closing other applications using significant GPU resources;
- reducing webcam resolution in `src/main.js`;
- temporarily running only one of the two landmarkers while debugging.

The included Pose Landmarker uses the **Lite** model to keep the prototype responsive.

---

## A Pose Triggers the Wrong Meme

Use the signal meters in the application to identify which feature is producing a high score.

Then adjust the weights or thresholds inside:

```text
rankMemeMatches()
```

in:

```text
src/detection.js
```

---

## A Meme Flickers On and Off

Increase `minFrames` or `maxHistory` in the `chooseStableMatch()` call inside:

```text
src/main.js
```

This requires the detector to see a matching condition for longer before changing the displayed meme.

---

# Testing

The feature extraction and meme-matching logic are separated from the webcam/UI code so they can be unit tested without requiring a camera.

Run:

```bash
npm test
```

The initial tests verify:

- face blendshape feature extraction;
- smile → Grinning Cat matching;
- hands-up → Absolute Victory matching;
- multi-frame stabilization behavior.

For broader validation, manually test each pose using:

- multiple users;
- different lighting conditions;
- different camera positions;
- different distances from the camera;
- different backgrounds.

---

# Building for Production

Create a production build with:

```bash
npm run build
```

Vite will generate the production files inside:

```text
dist/
```

Preview the production build locally with:

```bash
npm run preview
```

---

# Possible Next Upgrades

Possible future improvements include:

- add MediaPipe Hand Landmarker for finger gestures;
- add additional meme classes;
- replace placeholder art with appropriately licensed meme images;
- add user-adjustable detection sensitivity;
- add a three-second "hold pose" progress indicator;
- add an explicit **Save Screenshot** button;
- support multiple people in one frame;
- collect a labeled pose dataset;
- replace selected heuristic rules with a lightweight classifier;
- export detection events to a local JSON file for testing;
- package the application using Electron or Tauri;
- allow users to create custom meme-to-pose mappings;
- add a meme library browser;
- add confidence and threshold configuration through the UI.

---

# Dependencies

Major dependencies include:

- `@mediapipe/tasks-vision` — face and pose inference
- `Vite` — local development and build tooling

Exact dependency versions are pinned in:

```text
package.json
```

for reproducibility.

---

# License

Project source code and the original starter SVG artwork are released under the MIT License.

See:

```text
LICENSE
```

for details.
