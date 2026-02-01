# src/face_locking.py
"""
Face Locking System - Term 02 Week 04 Assignment

This module extends the face recognition system to:
1. Lock onto a specific enrolled identity
2. Track that identity consistently across frames
3. Detect and record facial actions (movement, blinks, smiles)
4. Maintain action history in timestamped files

Key Features:
- Stable locking with tolerance for brief recognition failures
- Multi-action detection (left/right movement, blinks, smiles)
- Persistent action history logging
- Visual feedback for locked state

Run:
    python -m src.face_locking

Controls:
    q : quit and save history
    s : manually save current history
    r : reload database
    l : toggle lock on/off (manual override)
    +/- : adjust recognition threshold
"""

from __future__ import annotations
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort

try:
    import mediapipe as mp
except Exception as e:
    mp = None
    _MP_IMPORT_ERROR = e

# Reuse existing modules
from .haar_5pt import align_face_5pt

# -------------------------
# CONFIGURATION
# -------------------------
# Lock stability parameters
MAX_LOST_FRAMES = 20  # Frames to tolerate before releasing lock (~0.67s at 30fps)
MIN_CONFIDENCE_TO_LOCK = 0.65  # Minimum similarity to initiate lock

# Action detection thresholds
MOVEMENT_THRESHOLD = 25  # Pixels of horizontal movement to detect
BLINK_EAR_THRESHOLD = 0.25  # Eye aspect ratio threshold for blink
BLINK_CONSECUTIVE_FRAMES = 2  # Frames below threshold to confirm blink
SMILE_RATIO_THRESHOLD = 1.4  # Mouth width/height ratio for smile
SMILE_CONSECUTIVE_FRAMES = 3  # Frames above threshold to confirm smile

# History configuration
HISTORY_DIR = Path("data/action_history")


# -------------------------
# Data Classes
# -------------------------
@dataclass
class FaceDet:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray  # (5,2) float32 in FULL-frame coords


@dataclass
class MatchResult:
    name: Optional[str]
    distance: float
    similarity: float
    accepted: bool


@dataclass
class ActionRecord:
    """Single action event record"""
    timestamp: str
    action_type: str
    description: str
    value: Optional[float] = None


# -------------------------
# Math Helpers
# -------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    return float(np.dot(a, b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_similarity(a, b)


def _clip_xyxy(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(W - 1, round(x1))))
    y1 = int(max(0, min(H - 1, round(y1))))
    x2 = int(max(0, min(W - 1, round(x2))))
    y2 = int(max(0, min(H - 1, round(y2))))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _bbox_from_5pt(
        kps: np.ndarray,
        pad_x: float = 0.55,
        pad_y_top: float = 0.85,
        pad_y_bot: float = 1.15,
) -> np.ndarray:
    k = kps.astype(np.float32)
    x_min = float(np.min(k[:, 0]))
    x_max = float(np.max(k[:, 0]))
    y_min = float(np.min(k[:, 1]))
    y_max = float(np.max(k[:, 1]))
    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)
    x1 = x_min - pad_x * w
    x2 = x_max + pad_x * w
    y1 = y_min - pad_y_top * h
    y2 = y_max + pad_y_bot * h
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _kps_span_ok(kps: np.ndarray, min_eye_dist: float) -> bool:
    k = kps.astype(np.float32)
    le, re, no, lm, rm = k
    eye_dist = float(np.linalg.norm(re - le))
    if eye_dist < float(min_eye_dist):
        return False
    if not (lm[1] > no[1] and rm[1] > no[1]):
        return False
    return True


# -------------------------
# Database Helpers
# -------------------------
def load_db_npz(db_path: Path) -> Dict[str, np.ndarray]:
    if not db_path.exists():
        return {}
    data = np.load(str(db_path), allow_pickle=True)
    out: Dict[str, np.ndarray] = {}
    for k in data.files:
        out[k] = np.asarray(data[k], dtype=np.float32).reshape(-1)
    return out


# -------------------------
# ArcFace Embedder
# -------------------------
class ArcFaceEmbedderONNX:
    def __init__(
            self,
            model_path: str = "models/embedder_arcface.onnx",
            input_size: Tuple[int, int] = (112, 112),
            debug: bool = False,
    ):
        self.model_path = model_path
        self.in_w, self.in_h = int(input_size[0]), int(input_size[1])
        self.debug = bool(debug)
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def _preprocess(self, aligned_bgr_112: np.ndarray) -> np.ndarray:
        img = aligned_bgr_112
        if img.shape[1] != self.in_w or img.shape[0] != self.in_h:
            img = cv2.resize(img, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        x = np.transpose(rgb, (2, 0, 1))[None, ...]
        return x.astype(np.float32)

    @staticmethod
    def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        v = v.astype(np.float32).reshape(-1)
        n = float(np.linalg.norm(v) + eps)
        return (v / n).astype(np.float32)

    def embed(self, aligned_bgr_112: np.ndarray) -> np.ndarray:
        x = self._preprocess(aligned_bgr_112)
        y = self.sess.run([self.out_name], {self.in_name: x})[0]
        emb = np.asarray(y, dtype=np.float32).reshape(-1)
        return self._l2_normalize(emb)


# -------------------------
# Haar + FaceMesh 5pt Detector
# -------------------------
class HaarFaceMesh5pt:
    def __init__(
            self,
            haar_xml: Optional[str] = None,
            min_size: Tuple[int, int] = (70, 70),
            debug: bool = False,
    ):
        self.debug = bool(debug)
        self.min_size = tuple(map(int, min_size))

        if haar_xml is None:
            haar_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(haar_xml)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {haar_xml}")

        if mp is None:
            raise RuntimeError(f"mediapipe import failed: {_MP_IMPORT_ERROR}")

        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.IDX_LEFT_EYE = 33
        self.IDX_RIGHT_EYE = 263
        self.IDX_NOSE_TIP = 1
        self.IDX_MOUTH_LEFT = 61
        self.IDX_MOUTH_RIGHT = 291

    def _haar_faces(self, gray: np.ndarray) -> np.ndarray:
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=self.min_size,
        )
        if faces is None or len(faces) == 0:
            return np.zeros((0, 4), dtype=np.int32)
        return faces.astype(np.int32)

    def _roi_facemesh_5pt(self, roi_bgr: np.ndarray) -> Optional[np.ndarray]:
        H, W = roi_bgr.shape[:2]
        if H < 20 or W < 20:
            return None
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        idxs = [self.IDX_LEFT_EYE, self.IDX_RIGHT_EYE, self.IDX_NOSE_TIP,
                self.IDX_MOUTH_LEFT, self.IDX_MOUTH_RIGHT]
        pts = []
        for i in idxs:
            p = lm[i]
            pts.append([p.x * W, p.y * H])
        kps = np.array(pts, dtype=np.float32)
        if kps[0, 0] > kps[1, 0]:
            kps[[0, 1]] = kps[[1, 0]]
        if kps[3, 0] > kps[4, 0]:
            kps[[3, 4]] = kps[[4, 3]]
        return kps

    def detect(self, frame_bgr: np.ndarray, max_faces: int = 5) -> List[FaceDet]:
        H, W = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._haar_faces(gray)

        if faces.shape[0] == 0:
            return []

        areas = faces[:, 2] * faces[:, 3]
        order = np.argsort(areas)[::-1]
        faces = faces[order][:max_faces]

        out: List[FaceDet] = []
        for (x, y, w, h) in faces:
            mx, my = 0.25 * w, 0.35 * h
            rx1, ry1, rx2, ry2 = _clip_xyxy(x - mx, y - my, x + w + mx, y + h + my, W, H)
            roi = frame_bgr[ry1:ry2, rx1:rx2]
            kps_roi = self._roi_facemesh_5pt(roi)
            if kps_roi is None:
                continue
            kps = kps_roi.copy()
            kps[:, 0] += float(rx1)
            kps[:, 1] += float(ry1)
            if not _kps_span_ok(kps, min_eye_dist=max(10.0, 0.18 * float(w))):
                continue
            bb = _bbox_from_5pt(kps, pad_x=0.55, pad_y_top=0.85, pad_y_bot=1.15)
            x1, y1, x2, y2 = _clip_xyxy(bb[0], bb[1], bb[2], bb[3], W, H)
            out.append(
                FaceDet(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    score=1.0,
                    kps=kps.astype(np.float32),
                )
            )
        return out


# -------------------------
# Face Matcher
# -------------------------
class FaceDBMatcher:
    def __init__(self, db: Dict[str, np.ndarray], dist_thresh: float = 0.34):
        self.db = db
        self.dist_thresh = float(dist_thresh)
        self._names: List[str] = []
        self._mat: Optional[np.ndarray] = None
        self._rebuild()

    def _rebuild(self):
        self._names = sorted(self.db.keys())
        if self._names:
            self._mat = np.stack([self.db[n].reshape(-1).astype(np.float32) for n in self._names], axis=0)
        else:
            self._mat = None

    def reload_from(self, path: Path):
        self.db = load_db_npz(path)
        self._rebuild()

    def match(self, emb: np.ndarray) -> MatchResult:
        if self._mat is None or len(self._names) == 0:
            return MatchResult(name=None, distance=1.0, similarity=0.0, accepted=False)
        e = emb.reshape(1, -1).astype(np.float32)
        sims = (self._mat @ e.T).reshape(-1)
        best_i = int(np.argmax(sims))
        best_sim = float(sims[best_i])
        best_dist = 1.0 - best_sim
        ok = best_dist <= self.dist_thresh
        return MatchResult(
            name=self._names[best_i] if ok else None,
            distance=float(best_dist),
            similarity=float(best_sim),
            accepted=bool(ok),
        )


# -------------------------
# Action History Manager
# -------------------------
class ActionHistory:
    """
    Manages action recording and file persistence.
    File format: <identity>_history_<timestamp>.txt
    """

    def __init__(self, identity_name: str, output_dir: Path = HISTORY_DIR):
        self.identity = identity_name
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records: List[ActionRecord] = []
        self.session_start = datetime.now()

    def log_action(self, action_type: str, description: str, value: Optional[float] = None):
        """Log a single action event"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        record = ActionRecord(
            timestamp=timestamp,
            action_type=action_type,
            description=description,
            value=value
        )
        self.records.append(record)

        # Print to console for real-time feedback
        if value is not None:
            print(f"[{timestamp}] {action_type}: {description} (value={value:.2f})")
        else:
            print(f"[{timestamp}] {action_type}: {description}")

    def save_to_file(self) -> Path:
        """
        Save history to file with mandatory naming format:
        <face>_history_<timestamp>.txt
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{self.identity}_history_{timestamp}.txt"
        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            # Header
            f.write(f"Face Locking Action History\n")
            f.write(f"=" * 60 + "\n")
            f.write(f"Identity: {self.identity}\n")
            f.write(f"Session Start: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Session End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Actions: {len(self.records)}\n")
            f.write(f"=" * 60 + "\n\n")

            # Action records
            f.write(f"{'Timestamp':<25} {'Action Type':<15} {'Description':<30} {'Value'}\n")
            f.write(f"{'-' * 25} {'-' * 15} {'-' * 30} {'-' * 10}\n")

            for record in self.records:
                value_str = f"{record.value:.2f}" if record.value is not None else "N/A"
                f.write(f"{record.timestamp:<25} {record.action_type:<15} {record.description:<30} {value_str}\n")

            # Summary statistics
            f.write(f"\n" + "=" * 60 + "\n")
            f.write(f"Action Summary\n")
            f.write(f"=" * 60 + "\n")

            action_counts = {}
            for record in self.records:
                action_counts[record.action_type] = action_counts.get(record.action_type, 0) + 1

            for action_type, count in sorted(action_counts.items()):
                f.write(f"{action_type}: {count}\n")

        print(f"\n[HISTORY] Saved to: {filepath}")
        return filepath


# -------------------------
# Action Detectors
# -------------------------
class ActionDetector:
    """
    Detects facial actions from landmarks and face position.
    Implements: movement (left/right), blinks, smiles
    """

    def __init__(self):
        # State tracking
        self.prev_center_x: Optional[float] = None
        self.prev_kps: Optional[np.ndarray] = None

        # Blink detection state
        self.eye_open_history: deque = deque(maxlen=5)
        self.was_eyes_closed = False
        self.closed_frame_count = 0

        # Smile detection state
        self.smile_history: deque = deque(maxlen=5)
        self.was_smiling = False
        self.smile_frame_count = 0

    def detect_movement(self, face: FaceDet, history: ActionHistory) -> None:
        """Detect left/right head movement"""
        current_center_x = (face.x1 + face.x2) / 2.0

        if self.prev_center_x is not None:
            delta_x = current_center_x - self.prev_center_x

            if delta_x > MOVEMENT_THRESHOLD:
                history.log_action(
                    "move_right",
                    f"Face moved right by {abs(delta_x):.1f} pixels",
                    abs(delta_x)
                )
            elif delta_x < -MOVEMENT_THRESHOLD:
                history.log_action(
                    "move_left",
                    f"Face moved left by {abs(delta_x):.1f} pixels",
                    abs(delta_x)
                )

        self.prev_center_x = current_center_x

    def _calculate_eye_aspect_ratio(self, kps: np.ndarray) -> float:
        """
        Calculate a simplified eye aspect ratio from 5-point landmarks.
        Lower values indicate closed eyes.

        Since we only have eye centers (not full eye landmarks), we estimate
        eye openness based on the vertical distance between eyes and nose.
        """
        left_eye = kps[0]  # (x, y)
        right_eye = kps[1]
        nose = kps[2]

        # Horizontal distance between eyes
        eye_width = np.linalg.norm(right_eye - left_eye)

        # Vertical distance from eye line to nose (proxy for face height)
        eye_center_y = (left_eye[1] + right_eye[1]) / 2.0
        vertical_dist = abs(nose[1] - eye_center_y)

        # Ratio: when eyes close, vertical distance tends to decrease slightly
        # This is a simplified approximation
        if eye_width > 0:
            ratio = vertical_dist / eye_width
        else:
            ratio = 0.5

        return float(ratio)

    def detect_blink(self, kps: np.ndarray, history: ActionHistory) -> None:
        """
        Detect eye blinks using simplified eye aspect ratio.
        A blink is confirmed when eyes close (ratio drops) then open again.
        """
        ear = self._calculate_eye_aspect_ratio(kps)
        self.eye_open_history.append(ear)

        # Determine if eyes are currently closed
        eyes_closed = ear < BLINK_EAR_THRESHOLD

        if eyes_closed:
            self.closed_frame_count += 1
        else:
            # Eyes just opened after being closed
            if self.was_eyes_closed and self.closed_frame_count >= BLINK_CONSECUTIVE_FRAMES:
                history.log_action(
                    "blink",
                    f"Eye blink detected (EAR={ear:.3f})",
                    ear
                )
            self.closed_frame_count = 0

        self.was_eyes_closed = eyes_closed
        self.prev_kps = kps.copy()

    def _calculate_mouth_ratio(self, kps: np.ndarray) -> float:
        """
        Calculate mouth width/height ratio for smile detection.
        Higher values indicate wider mouth (smile/laugh).
        """
        left_mouth = kps[3]
        right_mouth = kps[4]
        nose = kps[2]

        # Mouth width
        mouth_width = np.linalg.norm(right_mouth - left_mouth)

        # Mouth center to nose (vertical component)
        mouth_center = (left_mouth + right_mouth) / 2.0
        mouth_height = abs(mouth_center[1] - nose[1])

        # Ratio: smile makes mouth wider relative to its height
        if mouth_height > 0:
            ratio = mouth_width / mouth_height
        else:
            ratio = 1.0

        return float(ratio)

    def detect_smile(self, kps: np.ndarray, history: ActionHistory) -> None:
        """
        Detect smiles/laughs using mouth width/height ratio.
        A smile is confirmed when the ratio stays high for consecutive frames.
        """
        mouth_ratio = self._calculate_mouth_ratio(kps)
        self.smile_history.append(mouth_ratio)

        is_smiling = mouth_ratio > SMILE_RATIO_THRESHOLD

        if is_smiling:
            self.smile_frame_count += 1
            # Confirm smile after consecutive frames
            if not self.was_smiling and self.smile_frame_count >= SMILE_CONSECUTIVE_FRAMES:
                history.log_action(
                    "smile",
                    f"Smile/laugh detected (ratio={mouth_ratio:.3f})",
                    mouth_ratio
                )
                self.was_smiling = True
        else:
            self.smile_frame_count = 0
            self.was_smiling = False


# -------------------------
# Face Locking Manager
# -------------------------
class FaceLockManager:
    """
    Manages the face locking state machine.

    States:
    - UNLOCKED: No face is locked, searching for target identity
    - LOCKED: Target identity is locked and being tracked

    Lock transitions:
    - UNLOCKED -> LOCKED: Target identity recognized with high confidence
    - LOCKED -> UNLOCKED: Face lost for too long (>MAX_LOST_FRAMES)
    """

    def __init__(self, target_identity: str):
        self.target_identity = target_identity
        self.is_locked = False
        self.locked_face: Optional[FaceDet] = None
        self.lost_frame_count = 0
        self.lock_start_time: Optional[float] = None
        self.total_lock_duration = 0.0

        # Action detection
        self.action_detector = ActionDetector()
        self.history: Optional[ActionHistory] = None

    def try_lock(self, face: FaceDet, match_result: MatchResult) -> bool:
        """
        Attempt to lock onto a face if it matches the target identity.

        Returns:
            True if lock was acquired, False otherwise
        """
        if self.is_locked:
            return False

        # Check if this is our target identity with sufficient confidence
        if (match_result.accepted and
                match_result.name == self.target_identity and
                match_result.similarity >= MIN_CONFIDENCE_TO_LOCK):
            self.is_locked = True
            self.locked_face = face
            self.lost_frame_count = 0
            self.lock_start_time = time.time()

            # Initialize action history
            self.history = ActionHistory(self.target_identity)
            self.history.log_action(
                "lock_acquired",
                f"Locked onto {self.target_identity} (confidence={match_result.similarity:.3f})",
                match_result.similarity
            )

            print(f"\n🔒 LOCKED onto: {self.target_identity}")
            print(f"   Confidence: {match_result.similarity:.3f}")
            print(f"   Time: {datetime.now().strftime('%H:%M:%S')}\n")

            return True

        return False

    def update_lock(self, faces: List[FaceDet], embedder: ArcFaceEmbedderONNX,
                    matcher: FaceDBMatcher, frame: np.ndarray) -> Optional[FaceDet]:
        """
        Update lock state based on current frame.

        Returns:
            The locked face if lock is maintained, None otherwise
        """
        if not self.is_locked:
            return None

        # Try to find the target identity in current faces
        target_face = None
        best_similarity = 0.0

        for face in faces:
            # Get embedding and match
            aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
            emb = embedder.embed(aligned)
            match_result = matcher.match(emb)

            # Check if this is our target
            if (match_result.name == self.target_identity and
                    match_result.similarity > best_similarity):
                target_face = face
                best_similarity = match_result.similarity

        if target_face is not None:
            # Found target - maintain lock
            self.locked_face = target_face
            self.lost_frame_count = 0

            # Detect actions while locked
            if self.history is not None:
                self.action_detector.detect_movement(target_face, self.history)
                self.action_detector.detect_blink(target_face.kps, self.history)
                self.action_detector.detect_smile(target_face.kps, self.history)

            return target_face
        else:
            # Target not found - increment lost counter
            self.lost_frame_count += 1

            # Release lock if lost for too long
            if self.lost_frame_count > MAX_LOST_FRAMES:
                self.release_lock()
                return None

            # Maintain lock (tolerance for brief failures)
            return self.locked_face

    def release_lock(self, manual: bool = False) -> None:
        """Release the current lock and save history"""
        if not self.is_locked:
            return

        # Calculate lock duration
        if self.lock_start_time is not None:
            duration = time.time() - self.lock_start_time
            self.total_lock_duration += duration

        # Log release
        if self.history is not None:
            reason = "Manual release" if manual else f"Face lost for {self.lost_frame_count} frames"
            self.history.log_action(
                "lock_released",
                reason,
                self.lost_frame_count
            )

            # Save history to file
            self.history.save_to_file()

        print(f"\n🔓 UNLOCKED: {self.target_identity}")
        print(f"   Duration: {duration:.2f}s" if self.lock_start_time else "")
        print(f"   Actions recorded: {len(self.history.records) if self.history else 0}\n")

        # Reset state
        self.is_locked = False
        self.locked_face = None
        self.lost_frame_count = 0
        self.lock_start_time = None
        self.history = None
        self.action_detector = ActionDetector()  # Reset detector state

    def get_lock_info(self) -> Dict:
        """Get current lock status information"""
        if not self.is_locked:
            return {
                "locked": False,
                "target": self.target_identity,
                "duration": 0.0,
                "actions": 0
            }

        duration = time.time() - self.lock_start_time if self.lock_start_time else 0.0

        return {
            "locked": True,
            "target": self.target_identity,
            "duration": duration,
            "actions": len(self.history.records) if self.history else 0,
            "lost_frames": self.lost_frame_count
        }


# -------------------------
# Visualization Helpers
# -------------------------
def draw_locked_face(vis: np.ndarray, face: FaceDet, lock_info: Dict) -> None:
    """Draw special visualization for locked face"""
    # Draw thick blue border for locked face
    cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), (255, 128, 0), 3)

    # Draw 5-point landmarks
    for (x, y) in face.kps.astype(int):
        cv2.circle(vis, (int(x), int(y)), 3, (255, 128, 0), -1)

    # Draw "LOCKED" banner
    banner_height = 30
    cv2.rectangle(
        vis,
        (face.x1, face.y1 - banner_height),
        (face.x2, face.y1),
        (255, 128, 0),
        -1
    )

    lock_text = f"🔒 LOCKED: {lock_info['target']}"
    cv2.putText(
        vis,
        lock_text,
        (face.x1 + 5, face.y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # Draw lock duration and action count
    duration_text = f"Duration: {lock_info['duration']:.1f}s | Actions: {lock_info['actions']}"
    cv2.putText(
        vis,
        duration_text,
        (face.x1, face.y2 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 128, 0),
        2
    )


def draw_unlocked_face(vis: np.ndarray, face: FaceDet, match_result: MatchResult,
                       is_target: bool) -> None:
    """Draw visualization for unlocked faces"""
    # Green for target identity, yellow for others
    color = (0, 255, 0) if is_target else (0, 255, 255)

    cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), color, 2)

    for (x, y) in face.kps.astype(int):
        cv2.circle(vis, (int(x), int(y)), 2, color, -1)

    # Label
    label = match_result.name if match_result.name else "Unknown"
    if is_target:
        label += " (TARGET)"

    cv2.putText(
        vis,
        label,
        (face.x1, max(0, face.y1 - 28)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    # Confidence
    conf_text = f"sim={match_result.similarity:.3f}"
    cv2.putText(
        vis,
        conf_text,
        (face.x1, max(0, face.y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2
    )


def draw_status_overlay(vis: np.ndarray, lock_manager: FaceLockManager,
                        matcher: FaceDBMatcher, fps: Optional[float]) -> None:
    """Draw system status overlay"""
    h, w = vis.shape[:2]

    lock_info = lock_manager.get_lock_info()

    # Status panel background
    panel_height = 120
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
    vis[:] = cv2.addWeighted(vis, 0.3, overlay, 0.7, 0)

    # Status text
    y_offset = 25
    line_height = 25

    # Line 1: Lock status
    if lock_info["locked"]:
        status_text = f"🔒 LOCKED: {lock_info['target']}"
        status_color = (0, 255, 0)
    else:
        status_text = f"🔓 UNLOCKED - Searching for: {lock_manager.target_identity}"
        status_color = (0, 165, 255)

    cv2.putText(vis, status_text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Line 2: Stats
    y_offset += line_height
    if lock_info["locked"]:
        stats_text = f"Duration: {lock_info['duration']:.1f}s | Actions: {lock_info['actions']} | Lost frames: {lock_info['lost_frames']}/{MAX_LOST_FRAMES}"
    else:
        stats_text = f"Enrolled IDs: {len(matcher._names)} | Threshold: {matcher.dist_thresh:.2f}"

    cv2.putText(vis, stats_text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Line 3: Controls
    y_offset += line_height
    controls_text = "Controls: [q]uit & save | [s]ave now | [l]ock toggle | [+/-]threshold | [r]eload DB"
    cv2.putText(vis, controls_text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Line 4: FPS
    y_offset += line_height
    if fps is not None:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(vis, fps_text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


# -------------------------
# Main Application
# -------------------------
def main():
    """
    Main face locking application.

    This extends the recognition system with:
    1. Lock state management
    2. Action detection and recording
    3. Persistent history files
    """
    # Initialize paths
    db_path = Path("data/db/face_db.npz")

    # Load database first to show available identities
    db = load_db_npz(db_path)
    if len(db) == 0:
        print("\n❌ ERROR: No enrolled identities found in database.")
        print(f"   Database location: {db_path}")
        print(f"\n   Please enroll at least one person using: python -m src.enroll\n")
        return

    # Prompt user for target identity
    print(f"\n{'=' * 60}")
    print(f"Face Locking System - Term 02 Week 04")
    print(f"{'=' * 60}")
    print(f"\nEnrolled identities in database:")
    for i, name in enumerate(sorted(db.keys()), 1):
        print(f"  {i}. {name}")
    print()

    # Get target identity from user
    while True:
        target_identity = input("Enter the name of the identity to lock onto: ").strip()

        if not target_identity:
            print("❌ Error: Name cannot be empty. Please try again.\n")
            continue

        if target_identity not in db:
            print(f"❌ Error: '{target_identity}' not found in database.")
            print(f"   Available identities: {', '.join(sorted(db.keys()))}")
            print(f"   Please enter an exact name from the list above.\n")
            continue

        # Valid identity found
        break

    # Display configuration
    print(f"\n{'=' * 60}")
    print(f"Configuration")
    print(f"{'=' * 60}")
    print(f"Target Identity: {target_identity}")
    print(f"Lock Tolerance: {MAX_LOST_FRAMES} frames (~{MAX_LOST_FRAMES / 30:.1f}s at 30fps)")
    print(f"Action Detection: Movement, Blinks, Smiles")
    print(f"History Output: {HISTORY_DIR}/")
    print(f"{'=' * 60}\n")

    # Initialize components
    det = HaarFaceMesh5pt(min_size=(70, 70), debug=False)
    embedder = ArcFaceEmbedderONNX(
        model_path="models/embedder_arcface.onnx",
        input_size=(112, 112),
        debug=False
    )
    matcher = FaceDBMatcher(db=db, dist_thresh=0.34)
    lock_manager = FaceLockManager(target_identity=target_identity)  # Use prompted name

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not available")

    # FPS tracking
    t0 = time.time()
    frames = 0
    fps: Optional[float] = None

    print("System ready. Press 'q' to quit and save history.\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Detect all faces
            faces = det.detect(frame, max_faces=5)
            vis = frame.copy()

            # Update lock state
            if lock_manager.is_locked:
                # Already locked - update and track
                locked_face = lock_manager.update_lock(faces, embedder, matcher, frame)

                if locked_face is not None:
                    # Draw locked face
                    lock_info = lock_manager.get_lock_info()
                    draw_locked_face(vis, locked_face, lock_info)

                    # Draw other faces as unlocked
                    for face in faces:
                        # Skip the locked face
                        if (face.x1 == locked_face.x1 and face.y1 == locked_face.y1 and
                                face.x2 == locked_face.x2 and face.y2 == locked_face.y2):
                            continue

                        aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
                        emb = embedder.embed(aligned)
                        match_result = matcher.match(emb)
                        draw_unlocked_face(vis, face, match_result, False)
            else:
                # Not locked - try to lock or draw all faces
                for face in faces:
                    aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
                    emb = embedder.embed(aligned)
                    match_result = matcher.match(emb)

                    # Try to acquire lock
                    if lock_manager.try_lock(face, match_result):
                        # Lock acquired - will be drawn in next frame
                        pass
                    else:
                        # Draw as unlocked
                        is_target = (match_result.name == LOCK_TARGET_IDENTITY)
                        draw_unlocked_face(vis, face, match_result, is_target)

            # Calculate FPS
            frames += 1
            dt = time.time() - t0
            if dt >= 1.0:
                fps = frames / dt
                frames = 0
                t0 = time.time()

            # Draw status overlay
            draw_status_overlay(vis, lock_manager, matcher, fps)

            # Show frame
            cv2.imshow("Face Locking System", vis)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                # Quit and save
                if lock_manager.is_locked:
                    lock_manager.release_lock(manual=True)
                break

            elif key == ord("s"):
                # Manual save
                if lock_manager.is_locked and lock_manager.history:
                    filepath = lock_manager.history.save_to_file()
                    print(f"[MANUAL SAVE] History saved to: {filepath}")
                else:
                    print("[WARNING] No active lock - nothing to save")

            elif key == ord("l"):
                # Toggle lock (manual override)
                if lock_manager.is_locked:
                    lock_manager.release_lock(manual=True)
                    print("[MANUAL] Lock released")
                else:
                    print("[MANUAL] Cannot manually lock - system auto-locks on target detection")

            elif key == ord("r"):
                # Reload database
                matcher.reload_from(db_path)
                print(f"[RELOAD] Database reloaded: {len(matcher._names)} identities")

            elif key in (ord("+"), ord("=")):
                # Increase threshold
                matcher.dist_thresh = float(min(1.20, matcher.dist_thresh + 0.01))
                print(f"[THRESHOLD] dist={matcher.dist_thresh:.2f} (sim~{1.0 - matcher.dist_thresh:.2f})")

            elif key == ord("-"):
                # Decrease threshold
                matcher.dist_thresh = float(max(0.05, matcher.dist_thresh - 0.01))
                print(f"[THRESHOLD] dist={matcher.dist_thresh:.2f} (sim~{1.0 - matcher.dist_thresh:.2f})")

    finally:
        # Cleanup
        if lock_manager.is_locked:
            lock_manager.release_lock(manual=True)

        cap.release()
        cv2.destroyAllWindows()

        print("\n" + "=" * 60)
        print("Face Locking Session Ended")
        print("=" * 60)
        print(f"Total lock duration: {lock_manager.total_lock_duration:.2f}s")
        print(f"History files saved to: {HISTORY_DIR}/")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()