"""Build a GRBAS clinician-labeling package from the VOICED database
(voice-icar-federico-ii-database, PhysioNet 1.0.0).

VOICED ships diagnoses (healthy / hyperkinetic dysphonia / hypokinetic
dysphonia / reflux laryngitis, plus subtypes) but NO perceptual GRBAS labels.
Every recording is a sustained /a/ vowel at comfortable pitch -- exactly the
app's task -- so one recording == one speaker == one row for the clinician.

This script:
  1. reads every ``voiceNNN-info.txt`` for ID / age / gender / diagnosis,
  2. selects a stratified, gender-balanced cohort (all 208 by default),
  3. converts each selected recording to a 16-bit WAV the clinician can play
     (reading the plain-text signal ``voiceNNN.txt`` + sample rate from the
     ``voiceNNN.hea`` WFDB header -- no wfdb dependency required),
  4. writes an .xlsx workbook with pre-filled metadata, empty G/R/B/A/S
     dropdowns (0-3) and an instructions sheet.

Once the clinician returns the filled workbook, feature extraction turns it
into real_training_features.csv for train_grbas.py --mode finetune.

Usage
-----
    python build_voiced_clinician_sheet.py \
        --db-dir voiced_database/voice-icar-federico-ii-database-1.0.0

Add ``--n-total 60`` for a smaller pilot cohort.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import struct
import wave
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

BASE_DIR = Path(__file__).resolve().parent

GRBAS_DIMS = [
    ("G", "Grade — overall severity of dysphonia"),
    ("R", "Roughness — irregularity of vocal fold vibration"),
    ("B", "Breathiness — audible air escape"),
    ("A", "Asthenia — vocal weakness / hypofunction"),
    ("S", "Strain — effort / hyperfunction"),
]

HEALTHY_TOKENS = {"healthy", "normal", "no pathology", "control", ""}


# ---------------------------------------------------------------------------
# VOICED metadata
# ---------------------------------------------------------------------------

def _parse_info(path: Path) -> Dict[str, str]:
    """Parse a tab-separated ``voiceNNN-info.txt`` into a flat dict.

    Keys are lower-cased and stripped of a trailing colon, e.g.
    ``"Diagnosis:" -> "diagnosis"``.
    """
    fields: Dict[str, str] = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if "\t" not in line:
            continue
        key, _, value = line.partition("\t")
        key = key.strip().rstrip(":").strip().lower()
        value = value.strip()
        if key:
            fields[key] = value
    return fields


def _normalize_gender(value: str) -> str:
    token = (value or "").strip().lower()
    if token in ("f", "w", "female", "weiblich"):
        return "F"
    if token in ("m", "male", "männlich", "maennlich"):
        return "M"
    return ""


def _diagnosis_family(diagnosis: str) -> str:
    """Collapse VOICED subtypes to a broad family for balanced stratification.

    "hyperkinetic dysphonia (polyps)" -> "hyperkinetic dysphonia"
    "reflux laryngitis"               -> "reflux laryngitis"
    "healthy"                         -> "healthy"
    """
    base = re.sub(r"\(.*?\)", "", diagnosis or "").strip().lower()
    base = re.sub(r"\s+", " ", base)
    return base or "unknown"


def is_healthy(diagnosis: str) -> bool:
    return _diagnosis_family(diagnosis) in HEALTHY_TOKENS


def load_metadata(db_dir: Path) -> List[dict]:
    """Load every VOICED recording's metadata into normalized session dicts."""
    rows: List[dict] = []
    for info_path in sorted(db_dir.glob("voice*-info.txt")):
        fields = _parse_info(info_path)
        # The filename stem is authoritative -- it matches the .txt/.dat/.hea
        # audio files. A few VOICED -info.txt files carry a wrong internal "ID"
        # (e.g. voice006-info.txt says ID=voice005), so we do NOT trust it.
        session_id = info_path.name.split("-info")[0].strip()
        diagnosis = fields.get("diagnosis") or "healthy"
        rows.append({
            "session_id": session_id,
            "diagnosis": diagnosis,
            "family": _diagnosis_family(diagnosis),
            "age": fields.get("age", ""),
            "gender": _normalize_gender(fields.get("gender", "")),
            "vhi": fields.get("voice handicap index (vhi) score", ""),
            "rsi": fields.get("reflux symptom index (rsi) score", ""),
            "_healthy": is_healthy(diagnosis),
        })
    if not rows:
        raise SystemExit(
            f"no 'voice*-info.txt' files found in {db_dir}. "
            f"Point --db-dir at the extracted VOICED database folder.")
    return rows


# ---------------------------------------------------------------------------
# Cohort selection (stratified, gender-balanced)
# ---------------------------------------------------------------------------

def _interleave_by_gender(pool: List[dict], rng: random.Random) -> List[dict]:
    """Order a pool so genders alternate -- any prefix stays roughly balanced."""
    females = [r for r in pool if r["gender"] == "F"]
    males = [r for r in pool if r["gender"] == "M"]
    other = [r for r in pool if r["gender"] not in ("F", "M")]
    for group in (females, males, other):
        rng.shuffle(group)
    ordered: List[dict] = []
    while females or males:
        if len(females) >= len(males) and females:
            ordered.append(females.pop())
        elif males:
            ordered.append(males.pop())
    ordered.extend(females or males)
    ordered.extend(other)
    return ordered


def select_cohort(rows: List[dict], n_total: int, healthy_frac: float,
                  seed: int) -> List[dict]:
    """Pick a stratified cohort: healthy controls fill ``healthy_frac`` of it,
    pathological speakers fill the rest spread round-robin across diagnosis
    families. Both groups are gender-balanced."""
    rng = random.Random(seed)
    healthy = [r for r in rows if r["_healthy"]]
    pathological = [r for r in rows if not r["_healthy"]]

    n_healthy = min(len(healthy), round(n_total * healthy_frac))
    n_patho = min(len(pathological), n_total - n_healthy)
    if n_healthy + n_patho < n_total:  # one pool short -> top up from the other
        spare = n_total - n_healthy - n_patho
        if len(healthy) > n_healthy:
            n_healthy = min(len(healthy), n_healthy + spare)
        elif len(pathological) > n_patho:
            n_patho = min(len(pathological), n_patho + spare)

    selected_healthy = _interleave_by_gender(healthy, rng)[:n_healthy]

    by_family: Dict[str, List[dict]] = {}
    for row in pathological:
        by_family.setdefault(row["family"], []).append(row)
    queues = [_interleave_by_gender(group, rng) for group in by_family.values()]
    rng.shuffle(queues)
    selected_patho: List[dict] = []
    while len(selected_patho) < n_patho and any(queues):
        for queue in queues:
            if queue and len(selected_patho) < n_patho:
                selected_patho.append(queue.pop())

    selected = selected_healthy + selected_patho
    selected.sort(key=lambda r: (r["_healthy"] is False, r["family"],
                                 r["session_id"]))
    return selected


# ---------------------------------------------------------------------------
# Audio: VOICED .txt signal + .hea sample rate -> 16-bit WAV
# ---------------------------------------------------------------------------

def _read_sample_rate(hea_path: Path, default: int = 8000) -> int:
    """WFDB header line 1: '<record> <n_signals> <fs> <n_samples>'."""
    try:
        first = hea_path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        return int(first.split()[2])
    except (OSError, IndexError, ValueError):
        return default


def convert_to_wav(session_id: str, db_dir: Path, audio_out: Path) -> bool:
    """Write ``<session_id>.wav`` from the plain-text VOICED signal. Returns
    True on success, False if the source files are missing."""
    txt_path = db_dir / f"{session_id}.txt"
    if not txt_path.exists():
        return False
    sample_rate = _read_sample_rate(db_dir / f"{session_id}.hea")

    signal = np.loadtxt(txt_path, dtype=np.float64)
    if signal.ndim > 1:           # safety: take first channel if multi-column
        signal = signal[:, 0]

    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak > 0:
        signal = signal / peak * 0.98   # normalize, leave a little headroom
    pcm = np.clip(np.round(signal * 32767.0), -32768, 32767).astype("<i2")

    audio_out.mkdir(parents=True, exist_ok=True)
    with wave.open(str(audio_out / f"{session_id}.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return True


# ---------------------------------------------------------------------------
# Workbook
# ---------------------------------------------------------------------------

HEADERS = [
    "row_id", "session_id", "group", "diagnosis", "age", "gender",
    "vhi", "rsi", "audio_file", "task",
    "G", "R", "B", "A", "S",
    "sample_usable", "sample_quality", "clinician_comments",
]
# Folder name the WAVs live in, relative to the workbook (used for the
# clickable audio_file hyperlinks and the delivery package layout).
AUDIO_SUBDIR = "voiced_clinician_audio"

GRBAS_COL_START = 11  # column K
GRBAS_COL_END = 15    # column O
USABLE_COL = 16       # column P
QUALITY_COL = 17      # column Q


def build_rows(selected: List[dict]) -> List[dict]:
    task = "sustained /a/ vowel, comfortable pitch"
    out = []
    for index, row in enumerate(selected, start=1):
        out.append({
            "row_id": index,
            "session_id": row["session_id"],
            "group": "healthy" if row["_healthy"] else "pathological",
            "diagnosis": row["diagnosis"],
            "age": row["age"],
            "gender": row["gender"],
            "vhi": row["vhi"],
            "rsi": row["rsi"],
            "audio_file": f"{row['session_id']}.wav",
            "task": task,
        })
    return out


def write_workbook(sheet_rows: List[dict], out_path: Path) -> None:
    """Write the clinician .xlsx. Falls back to CSV if openpyxl is missing."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
        from openpyxl.worksheet.datavalidation import DataValidation
    except ImportError:
        csv_path = out_path.with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=HEADERS)
            writer.writeheader()
            writer.writerows(sheet_rows)
        print(f"openpyxl not installed -- wrote plain CSV instead: {csv_path}")
        print("install openpyxl (pip install openpyxl) for the .xlsx workbook")
        return

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "GRBAS Ratings"

    header_fill = PatternFill("solid", fgColor="1F4E78")
    grbas_fill = PatternFill("solid", fgColor="FFF2CC")
    header_font = Font(bold=True, color="FFFFFF")

    sheet.append(HEADERS)
    for col in range(1, len(HEADERS) + 1):
        cell = sheet.cell(row=1, column=col)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center")

    for row in sheet_rows:
        sheet.append([row.get(name, "") for name in HEADERS])

    last_row = len(sheet_rows) + 1

    # Make each audio_file cell a clickable link to its WAV, so the clinician
    # can play a recording straight from the workbook (works once the package
    # is unzipped with the voiced_clinician_audio folder beside the workbook).
    from openpyxl.styles import Font as _Font
    audio_col = HEADERS.index("audio_file") + 1
    link_font = _Font(color="0563C1", underline="single")
    for offset, row in enumerate(sheet_rows, start=2):
        cell = sheet.cell(row=offset, column=audio_col)
        cell.hyperlink = f"{AUDIO_SUBDIR}/{row['audio_file']}"
        cell.font = link_font

    for col in range(GRBAS_COL_START, GRBAS_COL_END + 1):
        for r in range(2, last_row + 1):
            sheet.cell(row=r, column=col).fill = grbas_fill

    if last_row >= 2:
        grade_dv = DataValidation(
            type="list", formula1='"0,1,2,3"', allow_blank=True,
            showErrorMessage=True, showInputMessage=True,
            errorTitle="Invalid GRBAS value",
            error="Use 0 (normal), 1 (slight), 2 (moderate) or 3 (severe).")
        grade_dv.prompt = "0 normal · 1 slight · 2 moderate · 3 severe"
        grade_dv.promptTitle = "GRBAS rating"
        sheet.add_data_validation(grade_dv)
        first = get_column_letter(GRBAS_COL_START)
        last = get_column_letter(GRBAS_COL_END)
        grade_dv.add(f"{first}2:{last}{last_row}")

        usable_col = get_column_letter(USABLE_COL)
        usable_dv = DataValidation(type="list", formula1='"yes,no"',
                                   allow_blank=True)
        sheet.add_data_validation(usable_dv)
        usable_dv.add(f"{usable_col}2:{usable_col}{last_row}")

        quality_col = get_column_letter(QUALITY_COL)
        quality_dv = DataValidation(type="list", formula1='"good,fair,poor"',
                                    allow_blank=True)
        sheet.add_data_validation(quality_dv)
        quality_dv.add(f"{quality_col}2:{quality_col}{last_row}")

    widths = {
        "A": 8, "B": 12, "C": 13, "D": 34, "E": 6, "F": 8, "G": 6, "H": 6,
        "I": 16, "J": 30, "K": 5, "L": 5, "M": 5, "N": 5, "O": 5,
        "P": 14, "Q": 15, "R": 40,
    }
    for column, width in widths.items():
        sheet.column_dimensions[column].width = width
    sheet.freeze_panes = "A2"

    _write_instructions(workbook)
    workbook.save(out_path)


def _write_instructions(workbook) -> None:
    from openpyxl.styles import Font

    sheet = workbook.create_sheet("Instructions")
    lines = [
        ("GRBAS perceptual rating — instructions", True),
        ("", False),
        ("Listen to each audio file (column 'audio_file', in the "
         "voiced_clinician_audio folder) and rate the voice on the five "
         "GRBAS dimensions.", False),
        ("", False),
        ("Severity scale for every dimension:", True),
        ("  0 = normal      1 = slight      2 = moderate      3 = severe", False),
        ("", False),
        ("Dimensions:", True),
    ]
    for letter, description in GRBAS_DIMS:
        lines.append((f"  {letter} — {description}", False))
    lines += [
        ("", False),
        ("Notes:", True),
        ("  • All recordings are a sustained /a/ vowel. Rate the voice quality "
         "only; ignore recording loudness.", False),
        ("  • G usually reflects the most affected dimension(s); it need not "
         "equal their sum.", False),
        ("  • Recordings are 8 kHz — fine for G/R/B/A/S, but treat very "
         "high-frequency detail with caution.", False),
        ("  • If a file is noisy, clipped or wrong-task, set sample_usable=no "
         "and add a comment — you may leave G/R/B/A/S blank.", False),
        ("  • sample_quality (good/fair/poor) is your confidence in the "
         "recording, not the voice.", False),
        ("  • The 'diagnosis' column is the database's medical label, NOT a "
         "GRBAS hint — please rate by ear, blind to it where you can.", False),
        ("  • Please do not change the metadata columns A–J.", False),
    ]
    for index, (text, bold) in enumerate(lines, start=1):
        cell = sheet.cell(row=index, column=1, value=text)
        if bold:
            cell.font = Font(bold=True)
    sheet.column_dimensions["A"].width = 95


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--db-dir", type=Path, required=True,
        help="extracted VOICED database folder (holds voiceNNN-info.txt etc.)")
    parser.add_argument("--out", type=Path,
                        default=BASE_DIR / "voiced_clinician_grbas_sheet.xlsx")
    parser.add_argument("--n-total", type=int, default=208,
                        help="cohort size (default 208 -- the whole database)")
    parser.add_argument("--healthy-frac", type=float, default=0.4,
                        help="fraction of healthy controls (default 0.4)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audio-out", type=Path,
                        default=BASE_DIR / "voiced_clinician_audio")
    parser.add_argument("--no-audio", action="store_true",
                        help="skip WAV conversion (build the workbook only)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.healthy_frac <= 1.0:
        raise SystemExit("--healthy-frac must be between 0 and 1")

    rows = load_metadata(args.db_dir)
    print(f"loaded {len(rows)} recordings "
          f"({sum(r['_healthy'] for r in rows)} healthy, "
          f"{sum(not r['_healthy'] for r in rows)} pathological)")

    selected = select_cohort(rows, args.n_total, args.healthy_frac, args.seed)
    if len(selected) < args.n_total:
        print(f"NOTE: only {len(selected)} recordings available "
              f"(requested {args.n_total})")

    sheet_rows = build_rows(selected)
    write_workbook(sheet_rows, args.out)

    n_healthy = sum(r["group"] == "healthy" for r in sheet_rows)
    n_patho = len(sheet_rows) - n_healthy
    n_families = len({r["family"] for r in selected if not r["_healthy"]})
    print(f"selected {len(sheet_rows)} recordings: "
          f"{n_healthy} healthy, {n_patho} pathological "
          f"across {n_families} diagnosis families")
    print(f"workbook -> {args.out}")

    if not args.no_audio:
        copied = missing = 0
        for row in selected:
            if convert_to_wav(row["session_id"], args.db_dir, args.audio_out):
                copied += 1
            else:
                missing += 1
        print(f"audio: wrote {copied} WAVs, missing {missing} -> {args.audio_out}")


if __name__ == "__main__":
    main()
