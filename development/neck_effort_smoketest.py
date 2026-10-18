from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from neck_effort_engine import NeckEffortEngine


def _apply_variant(frame: np.ndarray, variant: str) -> np.ndarray:
    if variant == "baseline":
        return frame
    if variant == "dark":
        return cv2.convertScaleAbs(frame, alpha=0.55, beta=-25)
    if variant == "blur":
        return cv2.GaussianBlur(frame, (9, 9), 0)
    if variant == "tight_crop":
        h, w = frame.shape[:2]
        x0 = int(w * 0.15)
        x1 = int(w * 0.85)
        y0 = int(h * 0.05)
        y1 = int(h * 0.72)
        cropped = frame[y0:y1, x0:x1]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    raise ValueError(f"Unknown variant: {variant}")


def run_smoketest(video_path: Path, variant: str, frame_step: int, max_frames: int | None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    engine = NeckEffortEngine()
    engine.start_recording_session()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    mode_counts = Counter()
    score_counts = Counter()
    confidences = []
    frames_seen = 0
    processed = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frames_seen += 1
            if frame_step > 1 and (frames_seen - 1) % frame_step:
                continue

            frame = _apply_variant(frame, variant)
            result = engine.process_frame(frame, timestamp=(frames_seen-1)/fps)
            processed += 1

            mode_counts[result["mode"]] += 1
            if result.get("score") is not None:
                score_counts[result["score"]] += 1
            confidences.append(float(result.get("confidence") or 0.0))

            if max_frames is not None and processed >= max_frames:
                break
        final_result = engine.finalize()
    finally:
        cap.release()
        engine.close()

    return {
        "video": str(video_path),
        "variant": variant,
        "frames_seen": frames_seen,
        "frames_processed": processed,
        "mode_counts": dict(mode_counts),
        "score_counts": dict(score_counts),
        "mean_confidence": round(float(np.mean(confidences)) if confidences else 0.0, 4),
        "calibrated": engine.is_calibrated(),
        "last_result": final_result,
    }


def main():
    parser = argparse.ArgumentParser(description="Smoke-test neck effort analysis on a real video.")
    parser.add_argument("video", help="Path to a local video file")
    parser.add_argument(
        "--variant",
        choices=["baseline", "dark", "blur", "tight_crop"],
        action="append",
        help="Run one or more degraded variants. Default: all variants.",
    )
    parser.add_argument("--frame-step", type=int, default=5, help="Process every Nth frame")
    parser.add_argument("--max-frames", type=int, default=120, help="Maximum processed frames per variant")
    parser.add_argument("--json-out", help="Optional path to save JSON results")
    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    variants = args.variant or ["baseline", "dark", "blur", "tight_crop"]

    results = [
        run_smoketest(
            video_path=video_path,
            variant=variant,
            frame_step=max(1, args.frame_step),
            max_frames=args.max_frames if args.max_frames > 0 else None,
        )
        for variant in variants
    ]

    payload = {"results": results}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)

    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
