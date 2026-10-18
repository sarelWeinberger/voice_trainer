"""Clinician ratings -> real_training_features.csv.

The end-to-end case runs the real feature extractor over two generated tones,
so a schema drift between this script and train_grbas.py fails here first.
"""

import csv
import math
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "development"))

ROOT = (Path(__file__).resolve().parents[1] / "root" / "voice_app")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_real_training_features as builder  # noqa: E402

RATING_COLUMNS = ["session_id", "diagnosis", "group", "G", "R", "B", "A", "S",
                  "usable", "quality", "comment"]


def write_ratings(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RATING_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in RATING_COLUMNS})


def rating(session_id, g=1, r=1, b=0, a=0, s=2, **extra):
    row = {"session_id": session_id, "diagnosis": "healthy", "group": "healthy",
           "G": g, "R": r, "B": b, "A": a, "S": s, "usable": "yes",
           "quality": "good", "comment": ""}
    row.update(extra)
    return row


def write_tone(path, freq=140.0, seconds=1.0, sample_rate=16000):
    frames = b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * t / sample_rate)))
        for t in range(int(sample_rate * seconds))
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(frames)


class TriageTests(unittest.TestCase):
    def test_keeps_only_rated_usable_unique_rows(self):
        rows = [
            rating("voice001"),
            rating("voice002", usable="no"),          # clinician rejected it
            rating("voice003", B=""),                 # incomplete rating
            rating("voice001", G=3),                  # duplicate session
            rating("", G=0),                          # blank id
            rating("voice004", G="4"),                # out of the 0-3 scale
        ]
        kept, dropped = builder.triage(rows)

        self.assertEqual([k["session_id"] for k in kept], ["voice001"])
        self.assertEqual(dropped["marked_unusable"], ["voice002"])
        self.assertEqual(dropped["duplicate"], ["voice001"])
        self.assertEqual(sorted(dropped["not_rated"]), ["voice003", "voice004"])
        self.assertEqual(dropped["no_session_id"], ["<blank>"])

    def test_accepts_the_workbook_column_names(self):
        rows = [{"session_id": "voice009", "G": 0, "R": 0, "B": 0, "A": 0, "S": 0,
                 "sample_usable": "no", "sample_quality": "noisy"},
                {"session_id": "voice010", "G": 1, "R": 1, "B": 1, "A": 1, "S": 1,
                 "sample_usable": "yes", "sample_quality": "good"}]
        kept, dropped = builder.triage(rows)
        self.assertEqual(dropped["marked_unusable"], ["voice009"])
        self.assertEqual([k["session_id"] for k in kept], ["voice010"])
        self.assertEqual(kept[0]["quality"], "good")

    def test_labels_are_read_as_ints(self):
        self.assertEqual(builder.read_labels({"G": "2", "R": 0, "B": "0", "A": 1, "S": "3"}),
                         {"G": 2, "R": 0, "B": 0, "A": 1, "S": 3})
        self.assertIsNone(builder.read_labels({"G": 1, "R": 1, "B": 1, "A": 1}))


class FeatureTableTests(unittest.TestCase):
    def test_writes_a_table_train_grbas_can_load(self):
        from grbas_model import FEATURE_NAMES
        import train_grbas

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            audio_dir = tmp / "audio"
            audio_dir.mkdir()
            write_tone(audio_dir / "voice001.wav", freq=140.0)
            write_tone(audio_dir / "voice002.wav", freq=210.0)

            ratings = tmp / "ratings.csv"
            write_ratings(ratings, [
                rating("voice001", g=1, r=2, b=0, a=0, s=1),
                rating("voice002", g=3, r=3, b=2, a=1, s=2),
                rating("voice003"),                    # rated but no audio
            ])

            kept, _ = builder.triage(builder.load_ratings(ratings))
            out = tmp / "real_training_features.csv"
            written, failures = builder.build_feature_table(kept, audio_dir, None, out)

            self.assertEqual(written, 2)
            self.assertEqual(len(failures), 1)
            self.assertIn("voice003", failures[0])

            with out.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            for name in FEATURE_NAMES:
                self.assertIn(name, rows[0])
            self.assertEqual(rows[0]["speaker_id"], rows[0]["sample_id"])
            self.assertEqual(rows[0]["source_type"], "real")

            # The trainer must read the labels back and see real (not synthetic) rows.
            samples = train_grbas.load_feature_table(out)
            self.assertEqual(len(samples), 2)
            self.assertEqual(samples[0]["labels"], {"G": 1, "R": 2, "B": 0, "A": 0, "S": 1})
            self.assertFalse(any(s["is_synthetic"] for s in samples))
            self.assertEqual({s["speaker_id"] for s in samples}, {"voice001", "voice002"})


if __name__ == "__main__":
    unittest.main()
