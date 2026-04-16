from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np


def _xy(lm, w: int, h: int):
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def _dist(a, b) -> float:
    return float(np.linalg.norm(a - b))


def _mid(a, b):
    return (a + b) / 2.0


def _angle_deg(v1, v2) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(math.degrees(math.acos(cosang)))


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _norm_bad(x, good, bad) -> float:
    if bad <= good:
        return 0.0
    return float(np.clip((x - good) / (bad - good), 0.0, 1.0))


def _safe_mean(values: List[float], default=0.0):
    return float(np.mean(values)) if values else float(default)


@dataclass
class NeckEffortConfig:
    """Clinically-calibrated neck effort thresholds.

    References:
    - CVA (Craniovertebral Angle): Yip 2008, Lau 2010 — cutoff 48.8°
    - Head lateral tilt: Ferrario 1995 — normal < 3°
    - Shoulder elevation: Kibler 1998, Grandjean ergonomics
    - MTD markers: Roy 2003, Rubin 2007, Koufman functional dysphonia
    """
    smooth_window: int = 20
    ema_alpha: float = 0.25
    calibration_frames: int = 45
    min_visibility: float = 0.45
    min_confidence_to_score: float = 0.45
    hysteresis_margin: float = 0.08

    # --- Craniovertebral Angle (CVA) — forward head posture ---
    # Measured as angle of ear-shoulder vector from vertical.
    # Clinical CVA is measured from horizontal (90° - neck_tilt).
    # Good CVA ≥ 52°, bad CVA < 44°
    # So good neck_tilt = 90-52 = 38°, bad neck_tilt = 90-44 = 46°
    # BUT in webcam frontal view, forward head is partially invisible,
    # so we use Z-depth from MediaPipe as primary signal.
    forward_head_z_good: float = 0.02   # z-depth diff ears vs shoulders (relative)
    forward_head_z_bad: float = 0.18

    # --- Head lateral tilt (Ferrario 1995) ---
    # Angle of eye line from horizontal.
    # Good < 3°, moderate 6-10°, bad > 10°
    head_tilt_good: float = 3.0
    head_tilt_bad: float = 12.0

    # --- Shoulder asymmetry (angle of shoulder line from horizontal) ---
    # Kibler 1998 — significant > 5°, clinical > 15°
    shoulder_angle_good: float = 3.0
    shoulder_angle_bad: float = 15.0

    # --- Shoulder elevation (relative to calibrated baseline) ---
    # Measured as y-position rise / torso_length
    shoulder_raise_good: float = 0.02
    shoulder_raise_bad: float = 0.15

    # --- Head instability (variance over recent frames) ---
    head_instability_good: float = 0.008
    head_instability_bad: float = 0.07

    # --- Mouth open (excessive opening from jaw tension) ---
    # Normal mouth open during /a/: 0.20-0.30. Jaw tension: > 0.40 or < 0.15
    mouth_open_good: float = 0.30
    mouth_open_bad: float = 0.50

    # --- Head rotation (yaw from frontal) ---
    head_rotation_good: float = 0.03
    head_rotation_bad: float = 0.15

    # --- Feature weights for weighted sum (must sum to ~1.0) ---
    # Forward head and head tilt are primary (most validated clinically).
    w_forward_head: float = 0.25
    w_head_tilt: float = 0.20
    w_shoulder_raise: float = 0.18
    w_shoulder_asym: float = 0.12
    w_head_instability: float = 0.10
    w_head_rotation: float = 0.08
    w_mouth_open: float = 0.07

    # Calibration blending weight (0 = no calibration, 1 = pure delta)
    calibration_weight: float = 0.55


@dataclass
class NeckEffortState:
    calibrated: bool = False
    calibration_count: int = 0
    calibration_buffer: List[Dict[str, float]] = field(default_factory=list)
    baseline: Dict[str, float] = field(default_factory=dict)
    feature_history: deque = field(default_factory=lambda: deque(maxlen=20))
    head_anchor_history: deque = field(default_factory=lambda: deque(maxlen=20))
    ema_continuous_score: Optional[float] = None
    stable_score: Optional[int] = None
    last_valid_result: Optional[Dict[str, Any]] = None


class NeckEffortEngine:
    def __init__(self, config: Optional[NeckEffortConfig] = None):
        self.cfg = config or NeckEffortConfig()
        self.state = NeckEffortState(
            feature_history=deque(maxlen=self.cfg.smooth_window),
            head_anchor_history=deque(maxlen=self.cfg.smooth_window),
        )

        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def reset_session(self):
        self.state = NeckEffortState(
            feature_history=deque(maxlen=self.cfg.smooth_window),
            head_anchor_history=deque(maxlen=self.cfg.smooth_window),
        )

    def is_calibrated(self) -> bool:
        return self.state.calibrated

    def calibration_progress(self) -> float:
        return min(1.0, self.state.calibration_count / max(1, self.cfg.calibration_frames))

    def process_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        if frame_bgr is None or frame_bgr.size == 0:
            return self._error_result("empty_frame")

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        pose_res = self.pose.process(rgb)
        face_res = self.face.process(rgb)

        if not pose_res.pose_landmarks or not face_res.multi_face_landmarks:
            return self._no_score_result(
                reason="no_face_or_pose",
                confidence=0.0,
                debug={},
            )

        try:
            pose_lm = pose_res.pose_landmarks.landmark
            face_lm = face_res.multi_face_landmarks[0].landmark

            visibility = self._estimate_visibility(pose_lm)
            feats = self._extract_features(pose_lm, face_lm, w, h)
            view_quality = self._estimate_view_quality(feats, visibility)
            confidence = float(_clamp(0.55 * visibility + 0.45 * view_quality, 0.0, 1.0))

            self.state.feature_history.append(feats)
            self.state.head_anchor_history.append(feats["head_anchor"])

            smoothed = self._smooth_features()

            if not self.state.calibrated:
                self._update_calibration(smoothed, confidence)
                return {
                    "ok": True,
                    "mode": "calibrating",
                    "calibrated": False,
                    "calibration_progress": self.calibration_progress(),
                    "score": None,
                    "score_label": None,
                    "confidence": round(confidence, 3),
                    "features": self._jsonify_features(smoothed),
                    "reasons": ["Hold a relaxed neutral posture for calibration"],
                }

            if confidence < self.cfg.min_confidence_to_score:
                return self._no_score_result(
                    reason="low_confidence_view",
                    confidence=confidence,
                    debug={"view_quality": view_quality, "visibility": visibility},
                )

            continuous, contribs = self._compute_continuous_effort(smoothed)
            continuous = self._apply_calibration_delta(continuous, smoothed)
            continuous = float(_clamp(continuous, 0.0, 1.0))

            if self.state.ema_continuous_score is None:
                self.state.ema_continuous_score = continuous
            else:
                alpha = self.cfg.ema_alpha
                self.state.ema_continuous_score = (
                    alpha * continuous + (1 - alpha) * self.state.ema_continuous_score
                )

            cont_smooth = float(self.state.ema_continuous_score)
            raw_score = self._continuous_to_discrete(cont_smooth)
            stable_score = self._apply_hysteresis(raw_score, cont_smooth)
            reasons = self._build_reasons(contribs, smoothed)

            result = {
                "ok": True,
                "mode": "running",
                "calibrated": True,
                "calibration_progress": 1.0,
                "score": stable_score,
                "score_label": self._score_label(stable_score),
                "confidence": round(confidence, 3),
                "continuous_score": round(cont_smooth, 4),
                "features": self._jsonify_features(smoothed),
                "reasons": reasons,
            }
            self.state.last_valid_result = result
            return result
        except Exception as exc:
            return self._error_result(f"processing_exception: {str(exc)}")

    def _estimate_visibility(self, pose_lm) -> float:
        ids = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_EAR,
            self.mp_pose.PoseLandmark.RIGHT_EAR,
            self.mp_pose.PoseLandmark.NOSE,
        ]
        vals = []
        for idx in ids:
            try:
                vals.append(float(pose_lm[idx].visibility))
            except Exception:
                vals.append(0.0)
        mean_vis = _safe_mean(vals, 0.0)
        return float(np.clip((mean_vis - 0.2) / 0.8, 0.0, 1.0))

    def _extract_features(self, pose_lm, face_lm, w: int, h: int) -> Dict[str, float]:
        # Pose landmarks in pixel coordinates
        l_sh = _xy(pose_lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER], w, h)
        r_sh = _xy(pose_lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER], w, h)
        l_ear = _xy(pose_lm[self.mp_pose.PoseLandmark.LEFT_EAR], w, h)
        r_ear = _xy(pose_lm[self.mp_pose.PoseLandmark.RIGHT_EAR], w, h)
        nose = _xy(pose_lm[self.mp_pose.PoseLandmark.NOSE], w, h)

        # Pose landmark Z-coordinates (depth, normalized, lower = closer to camera)
        # MediaPipe gives z for each pose landmark. We use it to detect forward head.
        try:
            l_sh_z = float(pose_lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].z)
            r_sh_z = float(pose_lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].z)
            l_ear_z = float(pose_lm[self.mp_pose.PoseLandmark.LEFT_EAR].z)
            r_ear_z = float(pose_lm[self.mp_pose.PoseLandmark.RIGHT_EAR].z)
            shoulder_z = (l_sh_z + r_sh_z) / 2.0
            ear_z = (l_ear_z + r_ear_z) / 2.0
            # Forward head = ears closer to camera than shoulders (more negative z)
            forward_head_z = max(0.0, shoulder_z - ear_z)
        except Exception:
            forward_head_z = 0.0

        shoulder_mid = _mid(l_sh, r_sh)
        ear_mid = _mid(l_ear, r_ear)
        shoulder_width = max(_dist(l_sh, r_sh), 1e-6)

        # Torso length approximation: shoulder-mid to pose body y-axis
        # Used for shoulder_raise normalization (distance-independent)
        try:
            l_hip = _xy(pose_lm[23], w, h)  # LEFT_HIP
            r_hip = _xy(pose_lm[24], w, h)  # RIGHT_HIP
            hip_mid = _mid(l_hip, r_hip)
            torso_length = max(_dist(shoulder_mid, hip_mid), shoulder_width * 1.2)
        except Exception:
            torso_length = shoulder_width * 1.5

        # Face landmarks
        lip_top = _xy(face_lm[13], w, h)
        lip_bottom = _xy(face_lm[14], w, h)
        chin = _xy(face_lm[152], w, h)
        face_nose = _xy(face_lm[1], w, h)
        eye_l = _xy(face_lm[33], w, h)
        eye_r = _xy(face_lm[263], w, h)
        eye_mid = _mid(eye_l, eye_r)
        eye_dist = max(_dist(eye_l, eye_r), 1e-6)

        # --- Head lateral tilt (Ferrario 1995) ---
        # Angle of the eye line from horizontal.
        eye_vec = eye_r - eye_l
        horizontal = np.array([1.0, 0.0], dtype=np.float32)
        raw_eye_angle = _angle_deg(eye_vec, horizontal)
        head_tilt_deg = min(raw_eye_angle, 180.0 - raw_eye_angle)

        # --- Shoulder line angle from horizontal (Kibler 1998) ---
        shoulder_vec = r_sh - l_sh
        raw_shoulder_angle = _angle_deg(shoulder_vec, horizontal)
        shoulder_angle_deg = min(raw_shoulder_angle, 180.0 - raw_shoulder_angle)

        # --- Shoulder elevation (y-position relative to torso length) ---
        # Primary signal comes from calibration delta (handled later).
        # Raw metric: how high are shoulders in the frame (scaled by torso).
        shoulder_raise_metric = float(shoulder_mid[1] / max(torso_length, 1.0))

        # --- Mouth opening (face-relative, for jaw tension detection) ---
        mouth_open = _dist(lip_top, lip_bottom) / eye_dist

        # --- Head rotation (yaw) — frontal nose vs eye midline ---
        head_rotation = abs(float(face_nose[0] - eye_mid[0])) / eye_dist

        # --- Head stability anchor (for variance tracking) ---
        head_anchor = _dist(nose, shoulder_mid) / shoulder_width

        # --- Chin drop (unused but kept for extensibility) ---
        chin_drop = abs(float(chin[1] - lip_bottom[1])) / eye_dist

        return {
            "forward_head_z": float(forward_head_z),
            "head_tilt_deg": float(head_tilt_deg),
            "shoulder_angle_deg": float(shoulder_angle_deg),
            "shoulder_raise_metric": float(shoulder_raise_metric),
            "mouth_open": float(mouth_open),
            "head_rotation": float(head_rotation),
            "head_anchor": float(head_anchor),
            "chin_drop": float(chin_drop),
            # Legacy names kept for backwards compatibility in reasons/UI:
            "forward_head": float(forward_head_z),
            "neck_tilt_deg": float(head_tilt_deg),
            "shoulder_asym": float(shoulder_angle_deg) / 90.0,
            "shoulder_raise_proxy": 0.0,  # computed from calibration delta
        }

    def _estimate_view_quality(self, feats: Dict[str, float], visibility: float) -> float:
        penalties = []
        penalties.append(_norm_bad(feats["head_rotation"], 0.10, 0.28))
        penalties.append(_norm_bad(feats["neck_tilt_deg"], 25.0, 45.0))

        weird_mouth = 0.0
        if feats["mouth_open"] < 0.0 or feats["mouth_open"] > 0.35:
            weird_mouth = 1.0
        penalties.append(weird_mouth)

        penalty = _safe_mean(penalties, 0.0)
        return float(_clamp((1.0 - penalty) * (0.6 + 0.4 * visibility), 0.0, 1.0))

    def _smooth_features(self) -> Dict[str, float]:
        hist = list(self.state.feature_history)
        keys = hist[0].keys()
        out = {}
        for key in keys:
            out[key] = float(np.mean([item[key] for item in hist]))

        head_anchors = list(self.state.head_anchor_history)
        out["head_instability"] = float(np.std(head_anchors)) if len(head_anchors) >= 3 else 0.0
        return out

    def _update_calibration(self, feats: Dict[str, float], confidence: float):
        # Require higher confidence during calibration
        if confidence < 0.65:
            return

        # Reject frames where posture is clearly not neutral:
        # - Head rotation > 8° (not facing camera)
        # - Head tilt > 8° (tilted head)
        # - Shoulder angle > 8° (asymmetric)
        # This ensures baseline is built from genuinely neutral frames.
        if feats.get("head_rotation", 0) > 0.10:  # ~8° rotation
            return
        if feats.get("head_tilt_deg", 0) > 8.0:
            return
        if feats.get("shoulder_angle_deg", 0) > 8.0:
            return

        self.state.calibration_buffer.append(
            {
                "forward_head_z": feats.get("forward_head_z", 0.0),
                "head_tilt_deg": feats.get("head_tilt_deg", 0.0),
                "shoulder_angle_deg": feats.get("shoulder_angle_deg", 0.0),
                "shoulder_raise_metric": feats.get("shoulder_raise_metric", 0.0),
                "mouth_open": feats["mouth_open"],
                "head_rotation": feats["head_rotation"],
                "head_instability": feats.get("head_instability", 0.0),
            }
        )
        self.state.calibration_count += 1

        if self.state.calibration_count >= self.cfg.calibration_frames:
            keys = self.state.calibration_buffer[0].keys()
            # Use median — robust to brief posture dips during calibration
            self.state.baseline = {
                key: float(np.median([item[key] for item in self.state.calibration_buffer]))
                for key in keys
            }
            self.state.calibrated = True

    def _compute_continuous_effort(self, feats: Dict[str, float]):
        """Clinical scoring: max individual severity dominates, evidence boosts.

        score_raw = 0.80 * max_norm + 0.20 * combined_weighted_sum

        This ensures:
        - One severe metric (max=1.0) → raw ≥ 0.80 → score 3
        - One moderate metric (max=0.6) → raw ≥ 0.48 → score 2
        - One mild metric (max=0.25) → raw ≥ 0.20 → score 1
        """
        cfg = self.cfg

        nf = _norm_bad(feats["forward_head_z"], cfg.forward_head_z_good, cfg.forward_head_z_bad)
        ht = _norm_bad(feats["head_tilt_deg"], cfg.head_tilt_good, cfg.head_tilt_bad)
        sa = _norm_bad(feats["shoulder_angle_deg"], cfg.shoulder_angle_good, cfg.shoulder_angle_bad)

        # Shoulder raise — derived from baseline delta if available
        sr = 0.0
        if self.state.calibrated and self.state.baseline:
            sh_base = self.state.baseline.get("shoulder_raise_metric", None)
            sh_now = feats.get("shoulder_raise_metric", None)
            if sh_base is not None and sh_now is not None:
                # Lower y = higher on screen = raised shoulders
                raise_delta = max(0.0, sh_base - sh_now)
                sr = _norm_bad(raise_delta, cfg.shoulder_raise_good, cfg.shoulder_raise_bad)

        ni = _norm_bad(feats.get("head_instability", 0.0), cfg.head_instability_good, cfg.head_instability_bad)
        nm = _norm_bad(feats["mouth_open"], cfg.mouth_open_good, cfg.mouth_open_bad)
        nh = _norm_bad(feats["head_rotation"], cfg.head_rotation_good, cfg.head_rotation_bad)

        norms = {
            "forward_head": nf,
            "head_tilt": ht,
            "shoulder_asym": sa,
            "shoulder_raise": sr,
            "head_instability": ni,
            "mouth_open": nm,
            "head_rotation": nh,
        }

        weights = {
            "forward_head": cfg.w_forward_head,
            "head_tilt": cfg.w_head_tilt,
            "shoulder_asym": cfg.w_shoulder_asym,
            "shoulder_raise": cfg.w_shoulder_raise,
            "head_instability": cfg.w_head_instability,
            "mouth_open": cfg.w_mouth_open,
            "head_rotation": cfg.w_head_rotation,
        }

        contribs = {name: weights[name] * norms[name] for name in norms}

        combined_sum = float(sum(contribs.values()))
        max_severity = float(max(norms.values()))

        # 80% max severity + 20% combined evidence
        continuous = 0.80 * max_severity + 0.20 * combined_sum
        continuous = float(_clamp(continuous, 0.0, 1.0))

        return continuous, contribs

    def _apply_calibration_delta(self, continuous: float, feats: Dict[str, float]) -> float:
        """Calibration now refines rather than replaces the clinical score.

        Shoulder raise is already computed from baseline in _compute_continuous_effort.
        Here we optionally boost the score if user has deteriorated significantly
        from their own neutral baseline (since some users have baseline issues).
        """
        if not self.state.calibrated or not self.state.baseline:
            return continuous

        baseline = self.state.baseline
        deltas = []

        # Measure deterioration relative to baseline for each metric
        fh_delta = max(0.0, feats.get("forward_head_z", 0) - baseline.get("forward_head_z", 0))
        ht_delta = max(0.0, feats.get("head_tilt_deg", 0) - baseline.get("head_tilt_deg", 0))
        sa_delta = max(0.0, feats.get("shoulder_angle_deg", 0) - baseline.get("shoulder_angle_deg", 0))

        deltas.append(_norm_bad(fh_delta, 0.02, 0.12))
        deltas.append(_norm_bad(ht_delta, 2.0, 8.0))
        deltas.append(_norm_bad(sa_delta, 2.0, 8.0))

        delta_score = float(np.mean(deltas)) if deltas else 0.0

        # Gentle boost only when user has significantly deteriorated
        # Limit boost to 10% to avoid inflating clinical scores
        boost = 0.10 * delta_score
        return float(max(continuous, min(continuous + boost, 1.0)))

    def _continuous_to_discrete(self, x: float) -> int:
        # Calibrated boundaries based on synthetic clinical tests:
        # 0 (normal):   max_severity < 0.15
        # 1 (mild):     0.15 – 0.45 (mild single-metric or accumulated)
        # 2 (moderate): 0.45 – 0.80 (moderate severity)
        # 3 (severe):   ≥ 0.80 (severe — maxed out primary metric)
        if x < 0.15:
            return 0
        if x < 0.45:
            return 1
        if x < 0.80:
            return 2
        return 3

    def _apply_hysteresis(self, raw_score: int, cont: float) -> int:
        if self.state.stable_score is None:
            self.state.stable_score = raw_score
            return raw_score

        prev = self.state.stable_score
        bounds = {
            0: (0.00, 0.15),
            1: (0.15, 0.45),
            2: (0.45, 0.80),
            3: (0.80, 1.00),
        }

        lo, hi = bounds[prev]
        if prev == 0:
            if cont < hi + self.cfg.hysteresis_margin:
                return 0
        elif prev == 3:
            if cont > lo - self.cfg.hysteresis_margin:
                return 3
        elif (lo - self.cfg.hysteresis_margin) <= cont <= (hi + self.cfg.hysteresis_margin):
            return prev

        self.state.stable_score = raw_score
        return raw_score

    def _build_reasons(self, contribs: Dict[str, float], feats: Dict[str, float]) -> List[str]:
        ranked = sorted(contribs.items(), key=lambda item: item[1], reverse=True)
        reasons = []
        for name, value in ranked[:3]:
            if value < 0.03:
                continue
            if name == "forward_head":
                reasons.append("head_forward_posture")
            elif name == "head_tilt":
                reasons.append("neck_tilt")
            elif name == "shoulder_asym":
                reasons.append("shoulder_imbalance")
            elif name == "shoulder_raise":
                reasons.append("raised_shoulders")
            elif name == "head_instability":
                reasons.append("head_instability")
            elif name == "mouth_open":
                reasons.append("excess_mouth_opening")
            elif name == "head_rotation":
                reasons.append("head_rotation")

        if not reasons:
            reasons = ["minimal_visible_effort"]
        return reasons

    def _score_label(self, score: int) -> str:
        return {
            0: "appears_relaxed",
            1: "possible_mild_effort",
            2: "possible_moderate_effort",
            3: "possible_significant_effort",
        }.get(score, "unknown")

    def _jsonify_features(self, feats: Dict[str, float]) -> Dict[str, float]:
        out = {}
        for key, value in feats.items():
            if key == "head_anchor":
                continue
            out[key] = round(float(value), 4)
        return out

    def _no_score_result(self, reason: str, confidence: float, debug: Dict[str, Any]) -> Dict[str, Any]:
        result = {
            "ok": True,
            "mode": "no_score",
            "calibrated": self.state.calibrated,
            "calibration_progress": self.calibration_progress(),
            "score": None,
            "score_label": None,
            "confidence": round(float(confidence), 3),
            "features": None,
            "reasons": [reason],
            "debug": debug,
        }
        if self.state.last_valid_result is not None:
            result["last_valid_score"] = self.state.last_valid_result.get("score")
        return result

    def _error_result(self, reason: str) -> Dict[str, Any]:
        return {
            "ok": False,
            "mode": "error",
            "calibrated": self.state.calibrated,
            "calibration_progress": self.calibration_progress(),
            "score": None,
            "score_label": None,
            "confidence": 0.0,
            "features": None,
            "reasons": [reason],
        }