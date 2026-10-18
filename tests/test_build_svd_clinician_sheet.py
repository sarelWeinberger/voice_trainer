"""Tests for the SVD clinician-sheet builder -- column detection, healthy/
pathological classification and stratified cohort selection. No heavy deps."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "development"))


ROOT = (Path(__file__).resolve().parents[1] / "root" / "voice_app")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_svd_clinician_sheet as bs  # noqa: E402


def _session(sid, pathology, gender):
    return {"session_id": sid, "pathology": pathology or "healthy",
            "age": "40", "gender": gender,
            "_healthy": bs.is_healthy(pathology)}


class ColumnDetectionTests(unittest.TestCase):
    def test_exact_match_preferred(self):
        fields = ["Recording ID", "Session ID", "Age"]
        self.assertEqual(bs.detect_column(fields, "id"), "Session ID")

    def test_substring_fallback(self):
        self.assertEqual(
            bs.detect_column(["Diagnose (DE)", "Alter"], "pathology"),
            "Diagnose (DE)")

    def test_no_match_returns_none(self):
        self.assertIsNone(bs.detect_column(["foo", "bar"], "gender"))


class ClassificationTests(unittest.TestCase):
    def test_healthy_tokens(self):
        for token in ("", "healthy", "Normal", "gesund", "-", "no pathology"):
            self.assertTrue(bs.is_healthy(token), token)

    def test_pathological(self):
        for token in ("Vocal fold paralysis", "Laryngitis", "Polyp"):
            self.assertFalse(bs.is_healthy(token), token)

    def test_gender_normalization(self):
        self.assertEqual(bs._normalize_gender("w"), "F")
        self.assertEqual(bs._normalize_gender("Male"), "M")
        self.assertEqual(bs._normalize_gender("männlich"), "M")
        self.assertEqual(bs._normalize_gender("?"), "")


class CohortSelectionTests(unittest.TestCase):
    def _population(self):
        rows = []
        for i in range(60):
            rows.append(_session(f"h{i}", "", "F" if i % 2 else "M"))
        for path in ("paralysis", "laryngitis", "polyp", "nodules"):
            for i in range(40):
                rows.append(_session(f"{path}{i}", path,
                                     "F" if i % 2 else "M"))
        return rows

    def test_respects_total_and_healthy_fraction(self):
        cohort = bs.select_cohort(self._population(), n_total=100,
                                  healthy_frac=0.4, seed=1)
        self.assertEqual(len(cohort), 100)
        n_healthy = sum(r["_healthy"] for r in cohort)
        self.assertEqual(n_healthy, 40)

    def test_pathology_spread(self):
        cohort = bs.select_cohort(self._population(), n_total=100,
                                  healthy_frac=0.4, seed=1)
        patho = [r["pathology"] for r in cohort if not r["_healthy"]]
        self.assertEqual(set(patho),
                         {"paralysis", "laryngitis", "polyp", "nodules"})
        # round-robin keeps diagnoses within one of each other
        counts = [patho.count(p) for p in set(patho)]
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_gender_balance(self):
        cohort = bs.select_cohort(self._population(), n_total=100,
                                  healthy_frac=0.4, seed=3)
        females = sum(r["gender"] == "F" for r in cohort)
        self.assertTrue(40 <= females <= 60, females)

    def test_deterministic(self):
        pop = self._population()
        a = [r["session_id"] for r in bs.select_cohort(pop, 80, 0.4, 7)]
        b = [r["session_id"] for r in bs.select_cohort(pop, 80, 0.4, 7)]
        self.assertEqual(a, b)

    def test_caps_at_available_population(self):
        cohort = bs.select_cohort(self._population(), n_total=10_000,
                                  healthy_frac=0.4, seed=1)
        self.assertEqual(len(cohort), 220)


class LoadAndBuildTests(unittest.TestCase):
    def test_load_metadata_and_dedup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "svd.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Session ID", "Pathology", "Age", "Gender"])
                writer.writerow(["1", "Laryngitis", "33", "w"])
                writer.writerow(["2", "", "41", "m"])
                writer.writerow(["1", "Laryngitis", "33", "w"])  # duplicate
            rows, resolved = bs.load_metadata(path, dict.fromkeys(
                ("id", "pathology", "age", "gender")))
            self.assertEqual(len(rows), 2)
            self.assertEqual(resolved["id"], "Session ID")
            self.assertTrue(rows[1]["_healthy"])

    def test_audio_filename(self):
        rows = bs.build_rows([_session("17", "polyp", "M")], "a", "n")
        self.assertEqual(rows[0]["audio_file"], "17-a_n.wav")
        self.assertIn("sustained /a/", rows[0]["task"])


if __name__ == "__main__":
    unittest.main()
