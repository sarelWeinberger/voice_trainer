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
    smooth_window: int = 20
    ema_alpha: float = 0.25
    calibration_frames: int = 45
    min_visibility: float = 0.45
    min_confidence_to_score: float = 0.45
    hysteresis_margin: float = 0.08
    forward_head_good: float = 0.03
    forward_head_bad: float = 0.18
    neck_tilt_good: float = 4.0
    neck_tilt_bad: float = 22.0
    shoulder_asym_good: float = 0.015
    shoulder_asym_bad: float = 0.11
    shoulder_raise_good: float = 0.05
    shoulder_raise_bad: float = 0.75
    head_instability_good: float = 0.008
    head_instability_bad: float = 0.07
    mouth_open_good: float = 0.025
    mouth_open_bad: float = 0.16
    head_rotation_good: float = 0.02
    head_rotation_bad: float = 0.13
    w_forward_head: float = 0.22
    w_neck_tilt: float = 0.20
    w_shoulder_asym: float = 0.10
    w_shoulder_raise: float = 0.18
    w_head_instability: float = 0.12
    w_mouth_open: float = 0.10
    w_head_rotation: float = 0.08
    calibration_weight: float = 0.45


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
        l_sh = _xy(pose_lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER], w, h)
        r_sh = _xy(pose_lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER], w, h)
        l_ear = _xy(pose_lm[self.mp_pose.PoseLandmark.LEFT_EAR], w, h)
        r_ear = _xy(pose_lm[self.mp_pose.PoseLandmark.RIGHT_EAR], w, h)
        nose = _xy(pose_lm[self.mp_pose.PoseLandmark.NOSE], w, h)

        shoulder_mid = _mid(l_sh, r_sh)
        ear_mid = _mid(l_ear, r_ear)
        shoulder_width = max(_dist(l_sh, r_sh), 1e-6)

        lip_top = _xy(face_lm[13], w, h)
        lip_bottom = _xy(face_lm[14], w, h)
        chin = _xy(face_lm[152], w, h)
        face_nose = _xy(face_lm[1], w, h)
        eye_l = _xy(face_lm[33], w, h)
        eye_r = _xy(face_lm[263], w, h)
        eye_mid = _mid(eye_l, eye_r)
        eye_dist = max(_dist(eye_l, eye_r), 1e-6)

        forward_head = abs(float(ear_mid[0] - shoulder_mid[0])) / shoulder_width
        neck_vec = ear_mid - shoulder_mid
        vertical_up = np.array([0.0, -1.0], dtype=np.float32)
        neck_tilt_deg = _angle_deg(neck_vec, vertical_up)
        shoulder_asym = abs(float(l_sh[1] - r_sh[1])) / shoulder_width
        neck_length = _dist(shoulder_mid, ear_mid) / shoulder_width
        shoulder_raise_proxy = max(0.0, 0.42 - neck_length) / 0.20
        mouth_open = _dist(lip_top, lip_bottom) / eye_dist
        head_rotation = abs(float(face_nose[0] - eye_mid[0])) / eye_dist
        head_anchor = _dist(nose, shoulder_mid) / shoulder_width
        chin_drop = abs(float(chin[1] - lip_bottom[1])) / eye_dist

        return {
            "forward_head": float(forward_head),
            "neck_tilt_deg": float(neck_tilt_deg),
            "shoulder_asym": float(shoulder_asym),
            "shoulder_raise_proxy": float(np.clip(shoulder_raise_proxy, 0.0, 1.5)),
            "mouth_open": float(mouth_open),
            "head_rotation": float(head_rotation),
            "head_anchor": float(head_anchor),
            "chin_drop": float(chin_drop),
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
        if confidence < 0.55:
            return

        self.state.calibration_buffer.append(
            {
                "forward_head": feats["forward_head"],
                "neck_tilt_deg": feats["neck_tilt_deg"],
                "shoulder_asym": feats["shoulder_asym"],
                "shoulder_raise_proxy": feats["shoulder_raise_proxy"],
                "mouth_open": feats["mouth_open"],
                "head_rotation": feats["head_rotation"],
                "head_instability": feats["head_instability"],
            }
        )
        self.state.calibration_count += 1

        if self.state.calibration_count >= self.cfg.calibration_frames:
            keys = self.state.calibration_buffer[0].keys()
            self.state.baseline = {
                key: float(np.mean([item[key] for item in self.state.calibration_buffer]))
                for key in keys
            }
            self.state.calibrated = True

    def _compute_continuous_effort(self, feats: Dict[str, float]):
        cfg = self.cfg

        nf = _norm_bad(feats["forward_head"], cfg.forward_head_good, cfg.forward_head_bad)
        nt = _norm_bad(feats["neck_tilt_deg"], cfg.neck_tilt_good, cfg.neck_tilt_bad)
        ns = _norm_bad(feats["shoulder_asym"], cfg.shoulder_asym_good, cfg.shoulder_asym_bad)
        nr = _norm_bad(feats["shoulder_raise_proxy"], cfg.shoulder_raise_good, cfg.shoulder_raise_bad)
        ni = _norm_bad(feats["head_instability"], cfg.head_instability_good, cfg.head_instability_bad)
        nm = _norm_bad(feats["mouth_open"], cfg.mouth_open_good, cfg.mouth_open_bad)
        nh = _norm_bad(feats["head_rotation"], cfg.head_rotation_good, cfg.head_rotation_bad)

        contribs = {
            "forward_head": cfg.w_forward_head * nf,
            "neck_tilt": cfg.w_neck_tilt * nt,
            "shoulder_asym": cfg.w_shoulder_asym * ns,
            "shoulder_raise": cfg.w_shoulder_raise * nr,
            "head_instability": cfg.w_head_instability * ni,
            "mouth_open": cfg.w_mouth_open * nm,
            "head_rotation": cfg.w_head_rotation * nh,
        }

        return float(sum(contribs.values())), contribs

    def _apply_calibration_delta(self, continuous: float, feats: Dict[str, float]) -> float:
        if not self.state.calibrated or not self.state.baseline:
            return continuous

        baseline = self.state.baseline
        deltas = []

        def rel_up(name, scale):
            return max(0.0, (feats[name] - baseline.get(name, feats[name])) / max(scale, 1e-6))

        deltas.append(rel_up("forward_head", 0.08))
        deltas.append(rel_up("neck_tilt_deg", 10.0))
        deltas.append(rel_up("shoulder_asym", 0.05))
        deltas.append(rel_up("shoulder_raise_proxy", 0.35))
        deltas.append(rel_up("head_instability", 0.03))

        delta_term = float(np.mean(deltas))
        return continuous * (1.0 - self.cfg.calibration_weight) + delta_term * self.cfg.calibration_weight

    def _continuous_to_discrete(self, x: float) -> int:
        if x < 0.22:
            return 0
        if x < 0.46:
            return 1
        if x < 0.72:
            return 2
        return 3

    def _apply_hysteresis(self, raw_score: int, cont: float) -> int:
        if self.state.stable_score is None:
            self.state.stable_score = raw_score
            return raw_score

        prev = self.state.stable_score
        bounds = {
            0: (0.00, 0.22),
            1: (0.22, 0.46),
            2: (0.46, 0.72),
            3: (0.72, 1.00),
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
            if value < 0.05:
                continue
            if name == "forward_head":
                reasons.append("head_forward_posture")
            elif name == "neck_tilt":
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
            0: "minimal",
            1: "mild",
            2: "moderate",
            3: "severe",
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