"""Export app results into a clinician review CSV for GRBAS and neck-effort labeling."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR.parent / "root" / "voice_app" / "app.db"
OUTPUT_PATH = BASE_DIR / "clinical_review_queue.csv"

FIELDNAMES = [
    "review_priority",
    "triage_reason",
    "result_id",
    "user_id",
    "username",
    "full_name",
    "joined_class_code",
    "recorded_at",
    "analysis_title",
    "recording_locator",
    "task_prompt",
    "target_voice_type",
    "needs_audio_review",
    "needs_video_review",
    "auto_model_type",
    "auto_grbas_g",
    "auto_grbas_r",
    "auto_grbas_b",
    "auto_grbas_a",
    "auto_grbas_s",
    "auto_neck_score",
    "auto_neck_label",
    "auto_neck_confidence",
    "auto_neck_reasons",
    "f0_mean",
    "f0_std",
    "f0_range",
    "jitter_percent",
    "shimmer_percent",
    "hnr",
    "spectral_centroid",
    "spectral_tilt",
    "hf_energy_ratio",
    "intensity_mean",
    "f1",
    "f2",
    "f3",
    "include_for_training",
    "sample_quality",
    "primary_diagnosis_or_note",
    "clinician_grbas_g",
    "clinician_grbas_r",
    "clinician_grbas_b",
    "clinician_grbas_a",
    "clinician_grbas_s",
    "clinician_neck_effort",
    "neck_effort_basis",
    "task_match",
    "suspected_compensation_pattern",
    "clinician_comments",
]


def _safe_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        return str(round(float(value), digits))
    except (TypeError, ValueError):
        return ""


def _safe_text(value: Any) -> str:
    return "" if value is None else str(value)


def _review_priority(payload: Dict[str, Any]) -> Tuple[int, str]:
    scores = payload.get("scores") or {}
    features = payload.get("features") or {}
    neck_effort = payload.get("neck_effort") or {}

    priority = 0
    reasons: List[str] = []

    positive_scores = sum(1 for value in scores.values() if isinstance(value, (int, float)) and value > 0)
    if positive_scores >= 2:
        priority += 3
        reasons.append("multiple_grbas_flags")
    elif positive_scores == 1:
        priority += 2
        reasons.append("single_grbas_flag")

    nonzero_features = sum(
        1
        for value in features.values()
        if isinstance(value, (int, float)) and abs(float(value)) > 0
    )
    if nonzero_features >= 8:
        priority += 2
        reasons.append("rich_acoustic_signal")
    elif nonzero_features >= 4:
        priority += 1
        reasons.append("partial_acoustic_signal")

    neck_score = neck_effort.get("score")
    neck_confidence = neck_effort.get("confidence")
    if isinstance(neck_score, (int, float)):
        priority += 2
        reasons.append("neck_effort_available")
    if isinstance(neck_confidence, (int, float)) and float(neck_confidence) >= 0.6:
        priority += 1
        reasons.append("neck_confidence_high")

    if (payload.get("model_type") or "").lower() == "heuristic":
        priority += 1
        reasons.append("heuristic_prediction")

    if not reasons:
        reasons.append("baseline_review")

    return priority, ";".join(reasons)


def _rows() -> Iterable[Dict[str, str]]:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    query = """
        SELECT
            results.id AS result_id,
            results.user_id AS user_id,
            results.created_at AS created_at,
            results.title AS title,
            results.raw_scores AS raw_scores,
            users.username AS username,
            users.full_name AS full_name,
            users.joined_class_code AS joined_class_code
        FROM results
        JOIN users ON users.id = results.user_id
        ORDER BY results.created_at DESC
    """

    try:
        for record in connection.execute(query):
            payload = json.loads(record["raw_scores"])
            scores = payload.get("scores") or {}
            features = payload.get("features") or {}
            neck_effort = payload.get("neck_effort") or {}
            priority, triage_reason = _review_priority(payload)

            yield {
                "review_priority": str(priority),
                "triage_reason": triage_reason,
                "result_id": str(record["result_id"]),
                "user_id": str(record["user_id"]),
                "username": _safe_text(record["username"]),
                "full_name": _safe_text(record["full_name"]),
                "joined_class_code": _safe_text(record["joined_class_code"]),
                "recorded_at": _safe_text(record["created_at"]),
                "analysis_title": _safe_text(record["title"]),
                "recording_locator": "",
                "task_prompt": "",
                "target_voice_type": "",
                "needs_audio_review": "yes",
                "needs_video_review": "yes" if neck_effort else "optional",
                "auto_model_type": _safe_text(payload.get("model_type")),
                "auto_grbas_g": _safe_text(scores.get("G")),
                "auto_grbas_r": _safe_text(scores.get("R")),
                "auto_grbas_b": _safe_text(scores.get("B")),
                "auto_grbas_a": _safe_text(scores.get("A")),
                "auto_grbas_s": _safe_text(scores.get("S")),
                "auto_neck_score": _safe_text(neck_effort.get("score")),
                "auto_neck_label": _safe_text(neck_effort.get("score_label")),
                "auto_neck_confidence": _safe_number(neck_effort.get("confidence")),
                "auto_neck_reasons": ";".join(neck_effort.get("reasons") or []),
                "f0_mean": _safe_number(features.get("f0_mean")),
                "f0_std": _safe_number(features.get("f0_std")),
                "f0_range": _safe_number(features.get("f0_range")),
                "jitter_percent": _safe_number(features.get("jitter_percent")),
                "shimmer_percent": _safe_number(features.get("shimmer_percent")),
                "hnr": _safe_number(features.get("hnr")),
                "spectral_centroid": _safe_number(features.get("spectral_centroid")),
                "spectral_tilt": _safe_number(features.get("spectral_tilt"), 6),
                "hf_energy_ratio": _safe_number(features.get("hf_energy_ratio"), 4),
                "intensity_mean": _safe_number(features.get("intensity_mean")),
                "f1": _safe_number(features.get("f1")),
                "f2": _safe_number(features.get("f2")),
                "f3": _safe_number(features.get("f3")),
                "include_for_training": "",
                "sample_quality": "",
                "primary_diagnosis_or_note": "",
                "clinician_grbas_g": "",
                "clinician_grbas_r": "",
                "clinician_grbas_b": "",
                "clinician_grbas_a": "",
                "clinician_grbas_s": "",
                "clinician_neck_effort": "",
                "neck_effort_basis": "",
                "task_match": "",
                "suspected_compensation_pattern": "",
                "clinician_comments": "",
            }
    finally:
        connection.close()


def main() -> None:
    rows = list(_rows())
    rows.sort(key=lambda row: (-int(row["review_priority"]), row["recorded_at"], row["result_id"]))

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} review rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
