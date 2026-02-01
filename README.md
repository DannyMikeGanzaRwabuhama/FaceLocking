# Face Recognition with ArcFace ONNX and 5-Point Alignment

A CPU-friendly face recognition system using ArcFace embeddings, Haar cascade detection, and MediaPipe 5-point landmark alignment.

**NEW**: Now includes **Face Locking** feature for behavioral tracking and action detection!

## Features

- ✅ **CPU-only execution** - No GPU required
- ✅ **5-point facial landmark alignment** - Robust face normalization
- ✅ **ArcFace deep learning embeddings** - State-of-the-art accuracy via ONNX
- ✅ **Open-set recognition** - Automatically rejects unknown faces
- ✅ **Multi-face real-time recognition** - Handles multiple faces simultaneously
- ✅ **Threshold tuning with FAR/FRR analysis** - Data-driven threshold selection
- ✅ **Persistent database storage** - NPZ format for embeddings
- ✅ **Re-enrollment support** - Add more samples to existing identities
- 🆕 **Face Locking** - Lock onto specific identity and track actions over time
- 🆕 **Action Detection** - Detect movement, blinks, and smiles
- 🆕 **Action History Recording** - Persistent timestamped action logs

## Project Structure

```
face-recognition-5pt/
├── data/
│   ├── enroll/          # Aligned face crops (generated)
│   ├── db/              # Recognition database (generated)
│   └── action_history/  # Face locking action logs (generated)
├── models/
│   └── embedder_arcface.onnx  # ArcFace model (download required)
├── src/
│   ├── camera.py        # Camera validation
│   ├── detect.py        # Face detection test
│   ├── landmarks.py     # 5-point landmark test
│   ├── align.py         # Face alignment test
│   ├── embed.py         # Embedding extraction test
│   ├── enroll.py        # Enrollment pipeline
│   ├── evaluate.py      # Threshold evaluation
│   ├── recognize.py     # Real-time recognition
│   ├── face_locking.py  # 🆕 Face locking system (NEW)
│   └── haar_5pt.py      # Core detection module
├── init_project.py      # Project structure generator
├── requirements.txt     # Python dependencies
├── .gitignore
└── README.md
```

## Prerequisites

- Python 3.9 or higher
- Webcam
- Operating System: macOS, Linux, or Windows

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/DannyMikeGanzaRwabuhama/FaceLocking.git
cd FaceLocking
```

### 2. Create Project Structure

```bash
python init_project.py
```

### 3. Set Up Virtual Environment

**Create virtual environment:**
```bash
python -m venv .venv
```

**Activate:**
- **macOS/Linux:** `source .venv/bin/activate`
- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **Windows (CMD):** `.venv\Scripts\activate.bat`

### 4. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Download ArcFace ONNX Model

⚠️ **CRITICAL STEP** - The model file is ~120MB and not included in the repository.

**Download and install:**
```bash
# Download
curl -L -o buffalo_l.zip \
"https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download"

# Extract
unzip -o buffalo_l.zip

# Copy to models directory
cp w600k_r50.onnx models/embedder_arcface.onnx

# Clean up
rm -f buffalo_l.zip w600k_r50.onnx 1k3d68.onnx 2d106det.onnx det_10g.onnx genderage.onnx
```

**Validate installation:**
```bash
python -m src.embed
```

Expected output:
```
embedding dim: 512
norm(before L2): 21.85
cos(prev,this): 0.988
```

### 6. Camera Permissions

**macOS:**
1. Go to System Settings → Privacy & Security → Camera
2. Allow access for Terminal or VS Code
3. Restart terminal

**Windows/Linux:**
- Ensure no other application is using the camera

## Usage

### Step 1: Validate Your Setup

Run these tests in order to verify each component:

```bash
# Test camera access
python -m src.camera

# Test face detection
python -m src.detect

# Test 5-point landmarks
python -m src.landmarks

# Test face alignment
python -m src.align

# Test embedding extraction
python -m src.embed
```

Press `q` to quit each test.

### Step 2: Enroll Identities

```bash
python -m src.enroll
```

**Controls:**
- `SPACE` - Capture one sample
- `a` - Toggle auto-capture mode (captures every 0.25s)
- `s` - Save enrollment (requires 15+ samples)
- `r` - Reset new samples (keeps existing)
- `q` - Quit

**Tips for best results:**
- Use stable lighting
- Capture from different angles
- Include different expressions
- Move slightly left/right during capture
- Enroll at least 2 people for evaluation

### Step 3: Evaluate Threshold

```bash
python -m src.evaluate
```

This analyzes your enrolled data and suggests an optimal recognition threshold based on:
- **Genuine pairs** - Same person comparisons
- **Impostor pairs** - Different person comparisons
- **FAR** (False Accept Rate) - Target: 1%
- **FRR** (False Reject Rate) - Minimized

**Requirements:**
- At least 2 enrolled people
- At least 5 samples per person

### Step 4: Run Recognition

```bash
python -m src.recognize
```

**Controls:**
- `q` - Quit
- `r` - Reload database from disk
- `+` or `=` - Increase threshold (more accepts)
- `-` - Decrease threshold (fewer accepts)
- `d` - Toggle debug overlay

**Display shows:**
- Face bounding boxes with 5-point landmarks
- Identity labels (green = known, red = unknown)
- Distance and similarity scores
- Aligned face thumbnails (right side)
- FPS counter

---

## 🆕 Face Locking Feature (NEW)

### Overview

The **Face Locking** feature extends the recognition system to lock onto a specific enrolled identity and track their behavioral actions over time. Once locked, the system:

1. **Locks onto the target face** - Recognizes and focuses on a specific enrolled identity
2. **Maintains stable tracking** - Continues tracking even during brief recognition failures
3. **Detects actions** - Monitors face movements, eye blinks, and smiles
4. **Records history** - Saves timestamped action logs to persistent files

This implements the assignment requirement: *"recognizing who the person is, then tracking what that person does over time."*

### How Face Locking Works

#### Lock State Machine

The system operates as a state machine with two primary states:

**UNLOCKED State:**
- System scans all detected faces
- Searches for the target identity (configured in code)
- When target identity is detected with high confidence → **transition to LOCKED**

**LOCKED State:**
- System focuses exclusively on the locked identity
- Tracks the locked face across frames
- Detects and records actions continuously
- Tolerates brief recognition failures (up to ~0.67 seconds)
- If face disappears for too long → **transition to UNLOCKED**

#### Lock Stability Mechanism

The system maintains lock stability through:

1. **Confidence Threshold** - Requires similarity ≥ 0.65 to acquire lock
2. **Temporal Tolerance** - Maintains lock for up to 20 frames (~0.67s) even if recognition temporarily fails
3. **Visual Continuity** - Tracks face position and landmarks across frames
4. **Smart Recovery** - Automatically re-acquires lock when target reappears

This ensures the lock remains stable during:
- Brief occlusions (hand passing in front of face)
- Rapid head movements
- Momentary recognition failures
- Changes in lighting or expression

### Actions Detected

The system detects three types of actions while a face is locked:

#### 1. Movement Detection (Left/Right)

**How it works:**
- Tracks the horizontal center position of the face bounding box
- Compares current position with previous frame
- Detects movement when displacement exceeds threshold (25 pixels)

**Detection logic:**
```python
delta_x = current_center_x - previous_center_x

if delta_x > 25:
    → "move_right" action
elif delta_x < -25:
    → "move_left" action
```

**Recorded data:**
- Direction (left or right)
- Displacement magnitude in pixels
- Timestamp

#### 2. Eye Blink Detection

**How it works:**
- Calculates Eye Aspect Ratio (EAR) using 5-point landmarks
- Monitors ratio of vertical to horizontal eye spacing
- Detects when eyes close (ratio drops below 0.25) then open again

**Detection logic:**
```python
EAR = vertical_eye_distance / horizontal_eye_distance

if EAR < 0.25 for 2+ consecutive frames:
    eyes_closed = True

if eyes_closed → eyes_open:
    → "blink" action
```

**Recorded data:**
- Blink event confirmation
- Eye aspect ratio value
- Timestamp

#### 3. Smile/Laugh Detection

**How it works:**
- Calculates mouth width-to-height ratio using landmarks
- Mouth corners and nose form reference triangle
- Detects when mouth widens (ratio > 1.4 for 3+ frames)

**Detection logic:**
```python
mouth_ratio = mouth_width / mouth_height

if mouth_ratio > 1.4 for 3+ consecutive frames:
    → "smile" action
```

**Recorded data:**
- Smile/laugh confirmation
- Mouth ratio value
- Timestamp

**Note on Accuracy:**
These detectors use simplified heuristics based on 5-point landmarks. Perfect accuracy is not required per assignment specifications. The logic is clear, explainable, and works reliably under normal conditions.

### Action History Files

#### File Naming Format

All history files follow the **mandatory naming convention** specified in the assignment:

```
<face>_history_<timestamp>.txt
```

**Example:**
```
gabi_history_20260201143025.txt
fani_history_20260129112099.txt
```

Where:
- `<face>` = enrolled identity name (lowercase)
- `<timestamp>` = YYYYMMDDHHmmss format (e.g., 20260201143025)

#### File Location

All history files are stored in:
```
data/action_history/
```

This directory is automatically created when the face locking system runs.

#### File Format

Each history file contains:

**1. Header Section:**
```
Face Locking Action History
============================================================
Identity: Gabi
Session Start: 2026-02-01 14:30:25
Session End: 2026-02-01 14:32:18
Total Actions: 47
============================================================
```

**2. Action Records Table:**
```
Timestamp                 Action Type     Description                    Value
------------------------- --------------- ------------------------------ ----------
2026-02-01 14:30:25.123   lock_acquired   Locked onto Gabi (confiden...  0.89
2026-02-01 14:30:26.456   move_right      Face moved right by 32.5 pi... 32.50
2026-02-01 14:30:27.789   blink           Eye blink detected (EAR=0.2... 0.21
2026-02-01 14:30:29.012   smile           Smile/laugh detected (ratio... 1.52
2026-02-01 14:30:30.345   move_left       Face moved left by 28.3 pix... 28.30
...
```

**3. Summary Statistics:**
```
============================================================
Action Summary
============================================================
blink: 12
lock_acquired: 1
lock_released: 1
move_left: 8
move_right: 10
smile: 15
```

### Running Face Locking

#### 1. Run the System

```bash
python -m src.face_locking
```

#### 2. Select Target Identity

The system will display all enrolled identities and prompt you to choose:

```
============================================================
Face Locking System - Term 02 Week 04
============================================================

Enrolled identities in database:
  1. Alice
  2. Bob
  3. Gabi

Enter the name of the identity to lock onto: Gabi
```

**Important:** Enter the exact name as shown in the list (case-sensitive).

#### 3. System Startup

The system will:
1. Verify the target identity exists in the database
2. Display configuration (target, thresholds, output directory)
3. Open camera and begin scanning for the target face

**Example startup output:**
```
============================================================
Face Locking System - Term 02 Week 04
============================================================
Target Identity: Gabi
Lock Tolerance: 20 frames (~0.7s at 30fps)
Action Detection: Movement, Blinks, Smiles
History Output: data/action_history/
============================================================

System ready. Press 'q' to quit and save history.
```

#### 4. Lock Acquisition

When the target identity appears and is recognized with high confidence:

```
🔒 LOCKED onto: Gabi
   Confidence: 0.892
   Time: 14:30:25
```

The display will show:
- **Blue bounding box** around the locked face (instead of green)
- **"🔒 LOCKED: Gabi"** banner above the face
- **Lock duration** and **action count** below the face
- **Status overlay** showing lock state and statistics

#### 5. Action Detection

While locked, the system continuously monitors and logs actions:

```
[2026-02-01 14:30:26.456] move_right: Face moved right by 32.5 pixels (value=32.50)
[2026-02-01 14:30:27.789] blink: Eye blink detected (EAR=0.210) (value=0.21)
[2026-02-01 14:30:29.012] smile: Smile/laugh detected (ratio=1.520) (value=1.52)
```

Actions are:
- Displayed in real-time in the terminal
- Recorded to the in-memory history
- Saved to file when lock is released

#### 6. Lock Release

The lock is released when:

**Automatic release:**
- Face disappears for > 20 frames (~0.67 seconds)

**Manual release:**
- Press `q` to quit
- Press `l` to toggle lock off

**On release:**
```
🔓 UNLOCKED: Gabi
   Duration: 113.45s
   Actions recorded: 47

[HISTORY] Saved to: data/action_history/gabi_history_20260201143025.txt
```

### Controls

While the face locking system is running:

| Key | Action |
|-----|--------|
| `q` | Quit and save history (if locked) |
| `s` | Manually save current history (if locked) |
| `l` | Release lock (toggle off) |
| `r` | Reload database from disk |
| `+` or `=` | Increase recognition threshold (more accepts) |
| `-` | Decrease recognition threshold (fewer accepts) |

### Visual Indicators

#### Unlocked State:
- **Green boxes** - Known identities (other than target)
- **Yellow boxes** - Target identity (not yet locked)
- **Red boxes** - Unknown faces
- **Status:** "🔓 UNLOCKED - Searching for: [target]"

#### Locked State:
- **Blue box** - The locked face (thick border)
- **"🔒 LOCKED"** banner - Displayed above locked face
- **Lock stats** - Duration and action count shown below face
- **Other faces** - Still detected but shown as unlocked
- **Status:** "🔒 LOCKED: [target]"

### Configuration Parameters

You can adjust detection sensitivity by editing constants in `src/face_locking.py`:

```python
# Lock stability
MAX_LOST_FRAMES = 20          # Frames to tolerate before release
MIN_CONFIDENCE_TO_LOCK = 0.65 # Minimum similarity to lock

# Action thresholds
MOVEMENT_THRESHOLD = 25        # Pixels for movement detection
BLINK_EAR_THRESHOLD = 0.25     # Eye aspect ratio for blinks
BLINK_CONSECUTIVE_FRAMES = 2   # Frames to confirm blink
SMILE_RATIO_THRESHOLD = 1.4    # Mouth ratio for smiles
SMILE_CONSECUTIVE_FRAMES = 3   # Frames to confirm smile
```

### Example Workflow

**Complete workflow from enrollment to action tracking:**

```bash
# 1. Enroll target identity
python -m src.enroll
# Enter name: Gabi
# Capture 15+ samples
# Press 's' to save

# 2. Run face locking
python -m src.face_locking
# System shows: Enrolled identities in database: 1. Gabi
# Prompt: Enter the name of the identity to lock onto: Gabi

# 3. System locks when Gabi appears
# → Actions are detected and logged

# 4. Press 'q' to quit
# → History saved to: data/action_history/gabi_history_20260201143025.txt
```

### Troubleshooting

**"Target identity not found in database"**
- The entered name doesn't exist in the database
- Solution: 
  - Check the spelling (case-sensitive)
  - Choose from the displayed list
  - Enroll the person first if not listed

**Lock keeps releasing too quickly**
- Face detection is unstable or face is too small
- Solution: Move closer to camera, improve lighting, increase `MAX_LOST_FRAMES`

**No actions detected**
- Thresholds may be too strict
- Solution: Adjust `MOVEMENT_THRESHOLD`, `BLINK_EAR_THRESHOLD`, `SMILE_RATIO_THRESHOLD`

**Actions detected too frequently**
- Thresholds may be too loose
- Solution: Increase threshold values, increase consecutive frame requirements

**Multiple locks/releases**
- Recognition confidence fluctuating around threshold
- Solution: Ensure good lighting, adjust `MIN_CONFIDENCE_TO_LOCK`

---

## How It Works

### Pipeline Architecture

**Enrollment Pipeline:**
```
Camera → Haar Detection → MediaPipe 5pt → Alignment → ArcFace Embedding → Database
```

**Recognition Pipeline:**
```
Camera → Haar Detection → MediaPipe 5pt → Alignment → ArcFace Embedding → Matching → Label
```

**Face Locking Pipeline:** 🆕
```
Recognition Pipeline → Lock Manager → Action Detector → History Logger → File Storage
```

### Key Components

1. **Face Detection** - Haar cascade (CPU-efficient)
2. **5-Point Landmarks** - MediaPipe FaceMesh extracts: left eye, right eye, nose, left mouth, right mouth
3. **Face Alignment** - Affine transform to 112×112 canonical pose
4. **Embedding** - ArcFace ResNet-50 produces 512-dimensional L2-normalized vectors
5. **Matching** - Cosine distance comparison (distance = 1 - similarity)
6. **Threshold** - Accept if distance ≤ threshold
7. **Lock Manager** 🆕 - State machine for lock acquisition and release
8. **Action Detector** 🆕 - Monitors movements, blinks, and smiles
9. **History Logger** 🆕 - Records actions to timestamped files

### Recognition Math

- **Embedding**: 512-dimensional L2-normalized vector
- **Similarity**: `cos(a,b) = dot(a,b)` (since L2-normalized)
- **Distance**: `dist(a,b) = 1 - cos(a,b)`
- **Decision**: Accept if `dist ≤ threshold` (typically ~0.34)

## Data Storage

### Database Files

**`data/db/face_db.npz`** - Binary storage of embeddings
```python
{
  "Alice": [512-dim embedding],
  "Bob": [512-dim embedding],
  ...
}
```

**`data/db/face_db.json`** - Metadata
```json
{
  "updated_at": "2025-01-25 10:30:00",
  "embedding_dim": 512,
  "names": ["Alice", "Bob"],
  "samples_total_used": 30,
  "note": "Embeddings are L2-normalized vectors..."
}
```

### Enrollment Crops

**`data/enroll/<name>/*.jpg`** - Aligned 112×112 face crops
- Saved for inspection and evaluation
- Used for re-enrollment
- Not required at runtime

### Action History Files 🆕

**`data/action_history/<name>_history_<timestamp>.txt`** - Action logs
- Timestamped action records
- Session metadata
- Summary statistics
- Created automatically when lock is released

## Performance

**Typical CPU performance:**
- Enrollment: 10-15 FPS
- Recognition: 10-20 FPS (single face)
- Recognition: 8-15 FPS (2-3 faces)
- Face Locking: 10-18 FPS (with action detection)

**Optimizations:**
- ROI-based detection reduces computation
- Process every N frames (not every frame)
- Temporal smoothing stabilizes predictions
- Action detection uses efficient geometric calculations

## Troubleshooting

### Camera not opening
- Check permissions (macOS: System Settings → Privacy → Camera)
- Try different camera index: `cv2.VideoCapture(1)` instead of `0`
- Close other apps using the camera

### Model not loading
- Verify file exists: `ls -lh models/embedder_arcface.onnx`
- Should be ~120MB
- Re-download if corrupted

### Poor recognition accuracy
- Re-enroll with more samples (20-30 per person)
- Ensure good lighting during enrollment
- Use threshold evaluation to tune threshold
- Check alignment quality: `python -m src.align`

### "FaceMesh returned none"
- Face too small - move closer to camera
- Poor lighting - improve illumination
- Face turned away - look at camera

## Re-enrollment

To add more samples to an existing identity:

```bash
python -m src.enroll
# Enter the same name
# System loads existing samples
# Capture new samples
# Press 's' to merge and save
```

## Project Background

This project is based on the book **"Face Recognition with ArcFace ONNX and 5-Point Alignment"** by Gabriel Baziramwabo (Benax Technologies Ltd · Rwanda Coding Academy).

The **Face Locking** feature was developed as part of **Term-02 Week-04 Assignment** to extend the recognition system with behavioral tracking capabilities.

The system emphasizes:
- Educational transparency over black-box frameworks
- CPU-first architecture for accessibility
- Modular design for debugging and extension
- Production-ready practices

## References

1. Deng et al. (2019) - ArcFace: Additive Angular Margin Loss for Deep Face Recognition
2. InsightFace Project - 2D & 3D Face Analysis
3. ONNX - Open Neural Network Exchange
4. MediaPipe - Framework for Building Perception Pipelines
5. OpenCV - Computer Vision Library

## License

This project is for educational purposes. Please respect the licenses of:
- ArcFace/InsightFace models
- MediaPipe
- OpenCV

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Test thoroughly
4. Submit a pull request