"""Build synthetic-data manifests for GRBAS pretraining and publication targets.

This script does not claim clinical validity. It creates:
1. A synthetic TTS manifest for pretraining / pipeline testing.
2. A publication target table for the real clinician-labeled data still needed.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SYNTHETIC_PATH = BASE_DIR / "synthetic_grbas_manifest.csv"
DEFAULT_TARGETS_PATH = BASE_DIR / "publication_data_targets.csv"

TTS_MODELS = {
    "piper": {
        "model": "en_US-lessac-medium",
        "why": "Fast local inference for bulk synthetic generation.",
    },
    "xtts_v2": {
        "model": "tts_models/multilingual/multi-dataset/xtts_v2",
        "why": "Higher speaker/style diversity for synthetic pretraining.",
    },
}

PROMPTS = [
    {
        "prompt_id": "sustained_a_short",
        "prompt_text": "aaaaaa",
        "utterance_kind": "sustained_vowel",
        "language": "zxx",
    },
    {
        "prompt_id": "sustained_a_long",
        "prompt_text": "aaaaaaaaaaaa",
        "utterance_kind": "sustained_vowel",
        "language": "zxx",
    },
    {
        "prompt_id": "counting_plain",
        "prompt_text": "one two three four five",
        "utterance_kind": "connected_speech",
        "language": "en",
    },
    {
        "prompt_id": "easy_sentence",
        "prompt_text": "Today I speak with a calm and steady voice.",
        "utterance_kind": "connected_speech",
        "language": "en",
    },
    {
        "prompt_id": "challenge_sentence",
        "prompt_text": "We will record a longer sample to inspect roughness breathiness and strain.",
        "utterance_kind": "connected_speech",
        "language": "en",
    },
]


@dataclass(frozen=True)
class Phenotype:
    phenotype_id: str
    grbas: Dict[str, int]
    perturbation_spec: Dict[str, float]
    notes: str


def _make_profile(phenotype_id: str, r: int, b: int, a: int, s: int, notes: str) -> Phenotype:
    overall = max(r, b, a, s)
    if overall == 0:
        overall = 0
    elif sum(value > 0 for value in (r, b, a, s)) >= 2 and overall < 3:
        overall += 1

    perturbation_spec = {
        "pitch_jitter": round(0.003 + 0.008 * r, 4),
        "amplitude_shimmer": round(0.010 + 0.020 * r + 0.010 * s, 4),
        "aspiration_noise": round(0.002 + 0.018 * b + 0.006 * a, 4),
        "gain_reduction_db": round(1.5 * a, 2),
        "spectral_tilt_shift": round(0.0003 * (b - s), 6),
        "soft_clip_drive": round(0.12 * s, 3),
        "tempo_delta": round(-0.015 * a + 0.010 * s, 4),
    }
    return Phenotype(
        phenotype_id=phenotype_id,
        grbas={"G": min(overall, 3), "R": r, "B": b, "A": a, "S": s},
        perturbation_spec=perturbation_spec,
        notes=notes,
    )


PHENOTYPES = [
    _make_profile("control", 0, 0, 0, 0, "Clean control sample for baseline calibration."),
    _make_profile("rough_1", 1, 0, 0, 0, "Mild roughness proxy."),
    _make_profile("rough_2", 2, 0, 0, 0, "Moderate roughness proxy."),
    _make_profile("rough_3", 3, 0, 0, 0, "Severe roughness proxy."),
    _make_profile("breathy_1", 0, 1, 0, 0, "Mild breathiness proxy."),
    _make_profile("breathy_2", 0, 2, 0, 0, "Moderate breathiness proxy."),
    _make_profile("breathy_3", 0, 3, 0, 0, "Severe breathiness proxy."),
    _make_profile("asthenic_1", 0, 0, 1, 0, "Mild asthenia proxy."),
    _make_profile("asthenic_2", 0, 0, 2, 0, "Moderate asthenia proxy."),
    _make_profile("asthenic_3", 0, 0, 3, 0, "Severe asthenia proxy."),
    _make_profile("strained_1", 0, 0, 0, 1, "Mild strain proxy."),
    _make_profile("strained_2", 0, 0, 0, 2, "Moderate strain proxy."),
    _make_profile("strained_3", 0, 0, 0, 3, "Severe strain proxy."),
    _make_profile("rough_breathy_2", 2, 2, 0, 0, "Common rough plus breathy mixed proxy."),
    _make_profile("breathy_asthenic_2", 0, 2, 2, 0, "Weak breathy mixed proxy."),
    _make_profile("rough_strained_2", 2, 0, 0, 2, "Hyperfunctional mixed proxy."),
    _make_profile("global_3", 3, 2, 2, 3, "High-severity mixed dysphonia proxy."),
]


def _speaker_profile(index: int) -> Dict[str, str]:
    gender_hint = "female" if index % 2 == 0 else "male"
    if index % 6 in (0, 1):
        pitch_band = "low"
    elif index % 6 in (2, 3):
        pitch_band = "mid"
    else:
        pitch_band = "high"

    if index < 16:
        split = "train"
    elif index < 20:
        split = "validation"
    else:
        split = "test"

    return {
        "speaker_id": f"syn_{index:03d}",
        "speaker_gender_hint": gender_hint,
        "speaker_pitch_band": pitch_band,
        "split": split,
    }


def iter_synthetic_rows(tts_backend: str, speaker_count: int) -> Iterable[Dict[str, str]]:
    model_info = TTS_MODELS[tts_backend]
    for index in range(speaker_count):
        speaker = _speaker_profile(index)
        for prompt in PROMPTS:
            for phenotype in PHENOTYPES:
                sample_id = "_".join(
                    [
                        speaker["speaker_id"],
                        prompt["prompt_id"],
                        phenotype.phenotype_id,
                    ]
                )
                yield {
                    "sample_id": sample_id,
                    "split": speaker["split"],
                    "stage": "synthetic_pretrain_only",
                    "source_type": "synthetic_tts",
                    "tts_backend": tts_backend,
                    "tts_model": model_info["model"],
                    "tts_backend_rationale": model_info["why"],
                    "speaker_id": speaker["speaker_id"],
                    "speaker_gender_hint": speaker["speaker_gender_hint"],
                    "speaker_pitch_band": speaker["speaker_pitch_band"],
                    "prompt_id": prompt["prompt_id"],
                    "prompt_text": prompt["prompt_text"],
                    "utterance_kind": prompt["utterance_kind"],
                    "language": prompt["language"],
                    "proxy_grbas_g": str(phenotype.grbas["G"]),
                    "proxy_grbas_r": str(phenotype.grbas["R"]),
                    "proxy_grbas_b": str(phenotype.grbas["B"]),
                    "proxy_grbas_a": str(phenotype.grbas["A"]),
                    "proxy_grbas_s": str(phenotype.grbas["S"]),
                    "phenotype_profile": phenotype.phenotype_id,
                    "perturbation_spec_json": json.dumps(phenotype.perturbation_spec, sort_keys=True),
                    "output_audio_relpath": f"synthetic_audio/{sample_id}.wav",
                    "requires_clinician_review": "yes",
                    "eligible_for_publication_claims": "no",
                    "notes": phenotype.notes,
                }


def write_synthetic_manifest(output_path: Path, tts_backend: str, speaker_count: int) -> int:
    rows = list(iter_synthetic_rows(tts_backend, speaker_count))
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def iter_publication_target_rows() -> Iterable[Dict[str, str]]:
    for dimension in "GRBAS":
        for level in range(4):
            yield {
                "dataset_type": "real_audio_clinician_labeled",
                "dimension": dimension,
                "severity_level": str(level),
                "target_samples": "120",
                "minimum_unique_speakers": "40",
                "task_type": "sustained_vowel_a",
                "split_policy": "speaker_level_70_15_15",
                "publication_role": "primary_training_validation_test",
                "notes": "Required for publication-grade claims. Synthetic rows do not count toward this target.",
            }

    for level in range(4):
        yield {
            "dataset_type": "real_video_clinician_labeled",
            "dimension": "neck_effort",
            "severity_level": str(level),
            "target_samples": "75",
            "minimum_unique_speakers": "30",
            "task_type": "audio_plus_video_recording",
            "split_policy": "speaker_level_70_15_15",
            "publication_role": "secondary_multimodal_validation",
            "notes": "Needed for neck-effort validation with clinician review.",
        }

    for phenotype in PHENOTYPES:
        yield {
            "dataset_type": "synthetic_tts_proxy",
            "dimension": phenotype.phenotype_id,
            "severity_level": "proxy",
            "target_samples": "120",
            "minimum_unique_speakers": "24",
            "task_type": "mixed_vowel_and_connected_speech",
            "split_policy": "speaker_level_70_15_15",
            "publication_role": "pretraining_only",
            "notes": "Useful for feature regression, augmentation, and smoke tests; not evidence of clinical validity.",
        }


def write_publication_targets(output_path: Path) -> int:
    rows = list(iter_publication_target_rows())
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic GRBAS manifests and publication targets.")
    parser.add_argument("--tts-backend", choices=sorted(TTS_MODELS.keys()), default="piper")
    parser.add_argument("--speaker-count", type=int, default=24)
    parser.add_argument("--synthetic-output", type=Path, default=DEFAULT_SYNTHETIC_PATH)
    parser.add_argument("--targets-output", type=Path, default=DEFAULT_TARGETS_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    synth_count = write_synthetic_manifest(args.synthetic_output, args.tts_backend, args.speaker_count)
    target_count = write_publication_targets(args.targets_output)
    print(f"wrote {synth_count} synthetic rows to {args.synthetic_output}")
    print(f"wrote {target_count} target rows to {args.targets_output}")


if __name__ == "__main__":
    main()