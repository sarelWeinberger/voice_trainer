"""Tests for the two-stage GRBAS trainer -- focused on the leakage guards,
split logic and CSV loading. These do not require sklearn; the heavy model
import is exercised only in the (skipped-if-unavailable) end-to-end test."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_grbas as tg  # noqa: E402


def _sample(sample_id, speaker_id, split="", synthetic=False, labels=None):
    return {
        "sample_id": sample_id,
        "speaker_id": speaker_id,
        "split": split,
        "is_synthetic": synthetic,
        "labels": labels or {d: 0 for d in "GRBAS"},
        "raw": {},
    }


class SpeakerIdTests(unittest.TestCase):
    def test_explicit_speaker_id_wins(self):
        self.assertEqual(
            tg._derive_speaker_id({"speaker_id": "p17", "sample_id": "x"}),
            "p17")

    def test_synthetic_speaker_from_sample_id(self):
        row = {"sample_id": "syn_004_sustained_a_long_rough_2"}
        self.assertEqual(tg._derive_speaker_id(row), "syn_004")

    def test_is_synthetic_detection(self):
        self.assertTrue(tg._is_synthetic({"source_type": "synthetic_tts"}))
        self.assertTrue(tg._is_synthetic({"sample_id": "syn_000_x"}))
        self.assertFalse(tg._is_synthetic({"sample_id": "result_42",
                                           "source_type": "real_audio"}))


class LabelTests(unittest.TestCase):
    def test_clinician_labels_parsed_and_clamped(self):
        row = {f"clinician_grbas_{d.lower()}": v
               for d, v in zip("GRBAS", ["3", "2", "0", "1", "9"])}
        self.assertEqual(tg._read_labels(row),
                         {"G": 3, "R": 2, "B": 0, "A": 1, "S": 3})

    def test_incomplete_labels_return_none(self):
        row = {"clinician_grbas_g": "1", "clinician_grbas_r": ""}
        self.assertIsNone(tg._read_labels(row))


class GuardTests(unittest.TestCase):
    def test_speaker_overlap_raises(self):
        splits = {
            "train": [_sample("a", "spk1")],
            "test": [_sample("b", "spk1")],
        }
        with self.assertRaisesRegex(ValueError, "speaker leakage"):
            tg.assert_no_speaker_overlap(splits)

    def test_disjoint_speakers_pass(self):
        splits = {
            "train": [_sample("a", "spk1")],
            "val": [_sample("b", "spk2")],
            "test": [_sample("c", "spk3")],
        }
        tg.assert_no_speaker_overlap(splits)  # must not raise

    def test_synthetic_in_test_raises(self):
        splits = {
            "train": [_sample("a", "spk1")],
            "test": [_sample("syn_0_x", "syn_0", synthetic=True)],
        }
        with self.assertRaisesRegex(ValueError, "synthetic data leaked"):
            tg.assert_no_synthetic_in_eval(splits)

    def test_synthetic_in_train_is_fine(self):
        splits = {
            "train": [_sample("syn_0_x", "syn_0", synthetic=True)],
            "test": [_sample("a", "spk1")],
        }
        tg.assert_no_synthetic_in_eval(splits)  # must not raise

    def test_sample_id_overlap_raises(self):
        splits = {
            "train": [_sample("dup", "spk1")],
            "val": [_sample("dup", "spk2")],
        }
        with self.assertRaisesRegex(ValueError, "appears in"):
            tg.assert_no_sample_id_overlap(splits)


class SplitTests(unittest.TestCase):
    def test_explicit_split_column_is_honored(self):
        samples = [
            _sample("a", "spk1", split="train"),
            _sample("b", "spk2", split="validation"),
            _sample("c", "spk3", split="test"),
        ]
        splits = tg.split_samples(samples, seed=0)
        self.assertEqual([s["sample_id"] for s in splits["train"]], ["a"])
        self.assertEqual([s["sample_id"] for s in splits["val"]], ["b"])
        self.assertEqual([s["sample_id"] for s in splits["test"]], ["c"])

    def test_downsample_synthetic_caps_count(self):
        synthetic = [_sample(f"syn_{i}", f"syn_{i}", synthetic=True)
                     for i in range(100)]
        kept = tg.downsample_synthetic(synthetic, n_target=10, seed=1)
        self.assertEqual(len(kept), 10)

    def test_downsample_zero_target_returns_empty(self):
        synthetic = [_sample("syn_0", "syn_0", synthetic=True)]
        self.assertEqual(tg.downsample_synthetic(synthetic, 0, seed=1), [])


class LoadTableTests(unittest.TestCase):
    def test_unlabeled_rows_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "real.csv"
            fields = ["sample_id", "speaker_id", "f0_mean",
                      *[f"clinician_grbas_{d.lower()}" for d in "GRBAS"]]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({"sample_id": "r1", "speaker_id": "spk1",
                                 "f0_mean": "180",
                                 **{f"clinician_grbas_{d.lower()}": "1"
                                    for d in "GRBAS"}})
                writer.writerow({"sample_id": "r2", "speaker_id": "spk2",
                                 "f0_mean": "200"})  # no labels -> skipped
            samples = tg.load_feature_table(path)
            self.assertEqual([s["sample_id"] for s in samples], ["r1"])

    def test_missing_file_returns_empty(self):
        self.assertEqual(tg.load_feature_table(Path("/no/such/file.csv")), [])


if __name__ == "__main__":
    unittest.main()
