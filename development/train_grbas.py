"""Two-stage GRBAS trainer: synthetic pretrain vs. real-data fine-tune.

Why two stages
--------------
Synthetic GRBAS audio (see render_synthetic_grbas_audio.py) is produced FROM
its own labels by a fixed DSP formula. A model fit only on synthetic data
learns to invert that formula -- it is not evidence of clinical accuracy.
Synthetic data is therefore used here for pretraining / augmentation only,
and every headline metric is computed on a held-out set of REAL, clinician-
labeled recordings.

Modes
-----
  pretrain  Train on synthetic data only -> grbas_pretrained.joblib.
            A pipeline / baseline artifact. NOT clinically valid, and never
            written to the production path the app loads.
  finetune  Train the production model. Real clinician-labeled data drives
            train/val/test; synthetic rows may be mixed into the TRAIN split
            only, down-sampled via --synthetic-ratio. Validation and test are
            real-only -> grbas_model.joblib (the file app.py loads).

Leakage guards (hard failures)
  * no speaker_id shared across train/val/test
  * no synthetic sample in the val or test split
  * no sample_id duplicated across splits

Input CSVs
----------
Both CSVs are wide feature tables (one row per recording) with the columns in
grbas_model.FEATURE_NAMES plus identifiers and labels:
  synthetic  produced by render_synthetic_grbas_audio.py --extract-features;
             labels in proxy_grbas_{g,r,b,a,s}.
  real       one row per clinician-labeled recording; labels in
             clinician_grbas_{g,r,b,a,s}, plus a speaker_id column and
             (optionally) a split column. Rows without complete labels are
             skipped. See README "Model training" for how to build it.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# numpy and sklearn are imported lazily inside the functions that need them,
# so the loading / split / leakage-guard helpers (and their tests) stay usable
# in a minimal environment.

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SYNTHETIC_CSV = BASE_DIR / "synthetic_training_features.csv"
DEFAULT_REAL_CSV = BASE_DIR / "real_training_features.csv"
PRODUCTION_MODEL = BASE_DIR.parent / "root" / "voice_app" / "grbas_model.joblib"
PRETRAIN_MODEL = BASE_DIR / "grbas_pretrained.joblib"

LABEL_DIMS = "GRBAS"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _to_float(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _derive_speaker_id(row: Dict[str, str]) -> str:
    """Speaker id for split grouping. Real rows must carry speaker_id;
    synthetic sample ids look like 'syn_000_<prompt>_<phenotype>'."""
    speaker = (row.get("speaker_id") or "").strip()
    if speaker:
        return speaker
    sample_id = (row.get("sample_id") or "").strip()
    parts = sample_id.split("_")
    if len(parts) >= 2 and parts[0] == "syn":
        return f"{parts[0]}_{parts[1]}"
    return sample_id  # last resort: each sample is its own speaker


def _is_synthetic(row: Dict[str, str]) -> bool:
    if (row.get("source_type") or "").lower().startswith("synthetic"):
        return True
    return (row.get("sample_id") or "").startswith("syn_")


def _read_labels(row: Dict[str, str]) -> Optional[Dict[str, int]]:
    """Return {G,R,B,A,S: int 0-3} or None if any label is missing."""
    labels: Dict[str, int] = {}
    for dim in LABEL_DIMS:
        raw = None
        for key in (f"clinician_grbas_{dim.lower()}",
                    f"proxy_grbas_{dim.lower()}",
                    f"grbas_{dim.lower()}", dim):
            if key in row and str(row[key]).strip() != "":
                raw = row[key]
                break
        if raw is None:
            return None
        try:
            labels[dim] = max(0, min(3, int(round(float(raw)))))
        except (TypeError, ValueError):
            return None
    return labels


def load_feature_table(path: Path) -> List[dict]:
    """Load a feature CSV into sample dicts. Rows without complete GRBAS
    labels are silently skipped (e.g. clinician review not yet done)."""
    if not Path(path).exists():
        return []
    samples: List[dict] = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            labels = _read_labels(row)
            if labels is None:
                continue
            samples.append({
                "sample_id": (row.get("sample_id") or "").strip(),
                "speaker_id": _derive_speaker_id(row),
                "split": (row.get("split") or "").strip().lower(),
                "is_synthetic": _is_synthetic(row),
                "labels": labels,
                "raw": row,
            })
    return samples


# ---------------------------------------------------------------------------
# Splitting & leakage guards
# ---------------------------------------------------------------------------

def assert_no_speaker_overlap(splits: Dict[str, List[dict]]) -> None:
    seen: Dict[str, str] = {}
    for name, rows in splits.items():
        for speaker in {r["speaker_id"] for r in rows}:
            if speaker in seen and seen[speaker] != name:
                raise ValueError(
                    f"speaker leakage: '{speaker}' appears in both "
                    f"'{seen[speaker]}' and '{name}' splits")
            seen[speaker] = name


def assert_no_synthetic_in_eval(splits: Dict[str, List[dict]]) -> None:
    for name in ("val", "test"):
        leaked = [r["sample_id"] for r in splits.get(name, []) if r["is_synthetic"]]
        if leaked:
            preview = ", ".join(leaked[:5]) + (" ..." if len(leaked) > 5 else "")
            raise ValueError(
                f"synthetic data leaked into the '{name}' set: {preview}")


def assert_no_sample_id_overlap(splits: Dict[str, List[dict]]) -> None:
    seen: Dict[str, str] = {}
    for name, rows in splits.items():
        for row in rows:
            sid = row["sample_id"]
            if sid in seen and seen[sid] != name:
                raise ValueError(
                    f"sample_id '{sid}' appears in '{seen[sid]}' and '{name}'")
            seen[sid] = name


def check_splits(splits: Dict[str, List[dict]]) -> None:
    assert_no_speaker_overlap(splits)
    assert_no_synthetic_in_eval(splits)
    assert_no_sample_id_overlap(splits)


def split_samples(samples: List[dict], seed: int) -> Dict[str, List[dict]]:
    """Return {'train','val','test'}.

    Honors an existing 'split' column when every row has a valid value;
    otherwise does a speaker-grouped 70/15/15 split so no speaker is shared
    across splits.
    """
    valid = {"train", "val", "validation", "test"}
    if samples and all(s["split"] in valid for s in samples):
        out: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
        for s in samples:
            key = "val" if s["split"] in ("val", "validation") else s["split"]
            out[key].append(s)
        return out

    import numpy as np
    from sklearn.model_selection import GroupShuffleSplit

    groups = np.array([s["speaker_id"] for s in samples])
    n_speakers = len(set(groups))
    if n_speakers < 3:
        raise SystemExit(
            f"cannot make a speaker-grouped split from {n_speakers} unique "
            f"speaker(s); add a 'split' column or collect more speakers")

    indices = np.arange(len(samples))
    rest_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        .split(indices, groups=groups))
    rest = [samples[i] for i in rest_idx]
    # 0.1765 of the remaining 85% is ~15% of the whole.
    train_idx, val_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=seed)
        .split(np.arange(len(rest)), groups=groups[rest_idx]))
    return {
        "train": [rest[i] for i in train_idx],
        "val": [rest[i] for i in val_idx],
        "test": [samples[i] for i in test_idx],
    }


def downsample_synthetic(synthetic: List[dict], n_target: int, seed: int) -> List[dict]:
    if n_target <= 0 or not synthetic:
        return []
    if len(synthetic) <= n_target:
        return list(synthetic)
    return random.Random(seed).sample(list(synthetic), n_target)


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def build_xy(samples: List[dict]):
    """Build the feature matrix and per-dimension label arrays. Column order
    matches grbas_model.FEATURE_NAMES so the model is consistent with the
    feature vector app.py builds at predict time."""
    import numpy as np
    from grbas_model import FEATURE_NAMES

    X = np.array(
        [[_to_float(s["raw"].get(name)) for name in FEATURE_NAMES]
         for s in samples],
        dtype=float,
    )
    y = {dim: np.array([s["labels"][dim] for s in samples], dtype=int)
         for dim in LABEL_DIMS}
    return X, y


def evaluate(classifier, samples: List[dict]) -> dict:
    """Per-dimension accuracy, MAE and quadratic-weighted kappa (QWK -- the
    standard agreement metric for ordinal GRBAS ratings)."""
    if not samples:
        return {}
    import numpy as np
    from sklearn.metrics import accuracy_score, mean_absolute_error, cohen_kappa_score

    X, y = build_xy(samples)
    metrics: dict = {}
    accuracies, maes = [], []
    for dim in LABEL_DIMS:
        y_true = y[dim]
        y_pred = np.clip(classifier.models[dim].predict(X), 0, 3).astype(int)
        try:
            qwk = float(cohen_kappa_score(
                y_true, y_pred, weights="quadratic", labels=[0, 1, 2, 3]))
        except Exception:
            qwk = float("nan")
        accuracy = float(accuracy_score(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))
        metrics[dim] = {"accuracy": accuracy, "mae": mae,
                        "qwk": qwk, "n": int(len(y_true))}
        accuracies.append(accuracy)
        maes.append(mae)
    metrics["_mean"] = {"accuracy": float(np.mean(accuracies)),
                        "mae": float(np.mean(maes))}
    return metrics


def _print_metrics(title: str, metrics: dict) -> None:
    if not metrics:
        print(f"  {title}: (no samples)")
        return
    n = metrics["G"]["n"]
    print(f"  {title} (n={n})")
    print(f"    {'dim':<5}{'acc':>8}{'mae':>8}{'qwk':>8}")
    for dim in LABEL_DIMS:
        m = metrics[dim]
        qwk = "  n/a" if m["qwk"] != m["qwk"] else f"{m['qwk']:.3f}"
        print(f"    {dim:<5}{m['accuracy']:>8.3f}{m['mae']:>8.3f}{qwk:>8}")
    mean = metrics["_mean"]
    print(f"    {'mean':<5}{mean['accuracy']:>8.3f}{mean['mae']:>8.3f}")


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_pretrain(args: argparse.Namespace, forced: bool = False) -> None:
    synthetic = load_feature_table(args.synthetic_csv)
    if not synthetic:
        raise SystemExit(f"no synthetic data found at {args.synthetic_csv}")

    splits = split_samples(synthetic, args.seed)
    # The eval set IS synthetic here, so the no-synthetic-in-eval guard does
    # not apply -- but speaker / sample disjointness still must hold.
    assert_no_speaker_overlap(splits)
    assert_no_sample_id_overlap(splits)

    from grbas_model import GRBASClassifier, FEATURE_NAMES

    X, y = build_xy(splits["train"])
    classifier = GRBASClassifier(args.model_type)
    classifier.fit(X, y)
    test_metrics = evaluate(classifier, splits["test"])

    classifier.metadata = {
        "stage": "synthetic_pretrain_only",
        "clinical_use": "not_valid_pretraining_artifact_only",
        "eval_set": "synthetic_holdout",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": args.model_type,
        "feature_count": len(FEATURE_NAMES),
        "n_synthetic_train": len(splits["train"]),
        "n_synthetic_test": len(splits["test"]),
        "seed": args.seed,
        "test_metrics": test_metrics,
    }

    out = Path(args.out) if args.out else PRETRAIN_MODEL
    if out.resolve() == PRODUCTION_MODEL.resolve():
        raise SystemExit(
            "refusing to write a synthetic-only model to the production path "
            "grbas_model.joblib -- the app would mislabel it as 'trained'")
    classifier.save(out)

    print("=== pretrain (synthetic only) ===")
    print(f"  train: {len(splits['train'])} synthetic samples")
    _print_metrics("synthetic test", test_metrics)
    print(f"  saved -> {out}")
    print("  NOTE: synthetic-only metrics measure how well the model inverts "
          "the DSP perturbation formula, NOT clinical accuracy.")
    if forced:
        print("  This artifact was produced by --allow-synthetic-only and is "
              "NOT the production model.")


def stage_finetune(args: argparse.Namespace) -> None:
    real = load_feature_table(args.real_csv)
    synthetic = load_feature_table(args.synthetic_csv)

    if not real:
        if args.allow_synthetic_only and synthetic:
            print("WARNING: no real clinician-labeled data found. Falling "
                  "back to a SYNTHETIC-ONLY fit (--allow-synthetic-only). "
                  "This is NOT clinically valid.\n")
            stage_pretrain(args, forced=True)
            return
        raise SystemExit(
            f"no real labeled data at {args.real_csv}. The production model "
            f"requires real clinician-labeled recordings -- see README "
            f"'Model training' for the expected CSV format. To produce a "
            f"non-production pretrain artifact instead, pass "
            f"--allow-synthetic-only.")

    splits = split_samples(real, args.seed)
    n_real_train = len(splits["train"])

    n_synth_target = (int(round(args.synthetic_ratio * n_real_train))
                      if synthetic else 0)
    synth_train = downsample_synthetic(synthetic, n_synth_target, args.seed)
    for s in synth_train:
        s["split"] = "train"  # synthetic is only ever allowed in train

    full = {
        "train": splits["train"] + synth_train,
        "val": splits["val"],
        "test": splits["test"],
    }
    check_splits(full)

    if len(splits["test"]) < args.min_real_test:
        raise SystemExit(
            f"real test set has only {len(splits['test'])} sample(s) "
            f"(< --min-real-test {args.min_real_test}). Refusing to write the "
            f"production model on an under-powered test set. Collect more "
            f"clinician-labeled recordings, or lower --min-real-test only if "
            f"you understand the metrics will be noisy.")

    from grbas_model import GRBASClassifier, FEATURE_NAMES

    X, y = build_xy(full["train"])
    classifier = GRBASClassifier(args.model_type)
    classifier.fit(X, y)

    val_metrics = evaluate(classifier, splits["val"])
    test_metrics = evaluate(classifier, splits["test"])

    classifier.metadata = {
        "stage": "real_finetune",
        "clinical_use": "research_only",
        "eval_set": "real_holdout",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": args.model_type,
        "feature_count": len(FEATURE_NAMES),
        "n_real_train": n_real_train,
        "n_real_val": len(splits["val"]),
        "n_real_test": len(splits["test"]),
        "n_synthetic_train": len(synth_train),
        "synthetic_ratio": args.synthetic_ratio,
        "seed": args.seed,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    out = Path(args.out) if args.out else PRODUCTION_MODEL
    classifier.save(out)

    print("=== finetune (real data drives evaluation) ===")
    print(f"  train: {n_real_train} real + {len(synth_train)} synthetic "
          f"(ratio {args.synthetic_ratio})")
    _print_metrics("real validation", val_metrics)
    _print_metrics("real test", test_metrics)
    print(f"  saved -> {out}")
    print("  Test metrics above are on REAL held-out, speaker-disjoint data.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["finetune", "pretrain"],
                        default="finetune",
                        help="finetune (default): production model on real "
                             "data; pretrain: synthetic-only baseline")
    parser.add_argument("--real-csv", type=Path, default=DEFAULT_REAL_CSV)
    parser.add_argument("--synthetic-csv", type=Path,
                        default=DEFAULT_SYNTHETIC_CSV)
    parser.add_argument("--model-type", default="ensemble",
                        help="random_forest|svm|gradient_boosting|xgboost|"
                             "lightgbm|mlp|ensemble")
    parser.add_argument("--synthetic-ratio", type=float, default=0.5,
                        help="finetune: synthetic train rows = ratio * real "
                             "train rows. 0 = real only.")
    parser.add_argument("--min-real-test", type=int, default=15,
                        help="finetune: minimum real test samples required to "
                             "write the production model")
    parser.add_argument("--allow-synthetic-only", action="store_true",
                        help="permit a synthetic-only fit when no real data "
                             "exists; saved as the non-production artifact")
    parser.add_argument("--out", type=Path, default=None,
                        help="output model path (default depends on mode)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.synthetic_ratio < 0:
        raise SystemExit("--synthetic-ratio must be >= 0")
    if args.mode == "pretrain":
        stage_pretrain(args)
    else:
        stage_finetune(args)


if __name__ == "__main__":
    main()
