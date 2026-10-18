"""Select a clinician-labeling cohort from the Saarbrücken Voice Database (SVD)
and build a GRBAS rating workbook.

SVD ships no perceptual GRBAS labels -- only diagnoses. This script picks a
stratified set of speakers (healthy + pathological, gender-balanced), takes one
recording per speaker (the sustained /a/ at normal pitch, which matches the
app's task), and writes an .xlsx with empty GRBAS columns (0-3 dropdowns) plus
an instructions sheet -- ready to hand to a clinician.

One recording per speaker keeps the clinician's job to one row per voice and
makes later speaker-level train/val/test splitting leakage-free by construction.

Input
-----
The SVD metadata table (CSV/TSV): the "Export" from
https://stimmdb.coli.uni-saarland.de/ or the svd_information.csv used in SVD
research repos. It must have, per recording session, at least: a session id,
a pathology/diagnosis field, age and gender. Column names are auto-detected;
override with --id-col / --pathology-col / --age-col / --gender-col.

Output
------
A workbook with two sheets:
  "GRBAS Ratings"  one row per selected recording; pre-filled metadata and
                   empty G/R/B/A/S columns (0-3 dropdowns) for the clinician.
  "Instructions"   a short GRBAS rating guide.

Once the clinician returns the filled workbook, feature extraction turns it
into real_training_features.csv for train_grbas.py --mode finetune.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent

# Tokens that mark a session as a healthy / no-pathology control.
HEALTHY_TOKENS = {
    "", "-", "--", "n/a", "na", "none", "no", "healthy", "health",
    "normal", "gesund", "kein befund", "ohne befund", "no pathology",
    "control", "kontrolle",
}

# Header-name hints for auto-detecting columns (matched as substrings, lowercased).
COLUMN_HINTS = {
    "id": ["session id", "recording session", "speaker id", "speakerid",
           "session", "id"],
    "pathology": ["pathology", "pathologies", "diagnosis", "diagnose",
                  "classification", "klassifikation", "befund", "disease"],
    "age": ["age", "alter"],
    "gender": ["gender", "sex", "geschlecht"],
}

GRBAS_DIMS = [
    ("G", "Grade — overall severity of dysphonia"),
    ("R", "Roughness — irregularity of vocal fold vibration"),
    ("B", "Breathiness — audible air escape"),
    ("A", "Asthenia — vocal weakness / hypofunction"),
    ("S", "Strain — effort / hyperfunction"),
]


# ---------------------------------------------------------------------------
# Metadata loading & column detection
# ---------------------------------------------------------------------------

def detect_column(fieldnames: List[str], role: str) -> Optional[str]:
    """Return the best-matching column name for a role, or None."""
    lowered = {name: (name or "").strip().lower() for name in fieldnames}
    for hint in COLUMN_HINTS[role]:
        for name, low in lowered.items():
            if low == hint:
                return name
    for hint in COLUMN_HINTS[role]:
        for name, low in lowered.items():
            if hint in low:
                return name
    return None


def _sniff_reader(handle):
    sample = handle.read(4096)
    handle.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    return csv.DictReader(handle, dialect=dialect)


def _normalize_gender(value: str) -> str:
    token = (value or "").strip().lower()
    if token in ("f", "w", "female", "weiblich", "frau"):
        return "F"
    if token in ("m", "male", "männlich", "maennlich", "mann"):
        return "M"
    return ""


def is_healthy(pathology: str) -> bool:
    return (pathology or "").strip().lower() in HEALTHY_TOKENS


def load_metadata(path: Path, cols: Dict[str, Optional[str]]) -> List[dict]:
    """Load SVD metadata into normalized session dicts."""
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = _sniff_reader(handle)
        fieldnames = reader.fieldnames or []
        resolved = {
            role: cols.get(role) or detect_column(fieldnames, role)
            for role in COLUMN_HINTS
        }
        if not resolved["id"]:
            raise SystemExit(
                f"could not find a session-id column in {path}. "
                f"Columns seen: {fieldnames}. Pass --id-col explicitly.")

        rows: List[dict] = []
        seen_ids = set()
        for raw in reader:
            session_id = (raw.get(resolved["id"]) or "").strip()
            if not session_id or session_id in seen_ids:
                continue  # blank or duplicate session
            seen_ids.add(session_id)
            pathology = (raw.get(resolved["pathology"]) or "").strip() \
                if resolved["pathology"] else ""
            rows.append({
                "session_id": session_id,
                "pathology": pathology or "healthy",
                "age": (raw.get(resolved["age"]) or "").strip()
                       if resolved["age"] else "",
                "gender": _normalize_gender(raw.get(resolved["gender"], ""))
                          if resolved["gender"] else "",
                "_healthy": is_healthy(pathology),
            })
    return rows, resolved


# ---------------------------------------------------------------------------
# Cohort selection
# ---------------------------------------------------------------------------

def _interleave_by_gender(pool: List[dict], rng: random.Random) -> List[dict]:
    """Order a pool so genders alternate, so any prefix is roughly balanced."""
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
    """Pick a stratified cohort.

    Healthy controls fill `healthy_frac` of the cohort. Pathological speakers
    fill the rest, spread evenly across distinct pathologies (round-robin) so
    no single diagnosis dominates. Both groups are gender-balanced.
    """
    rng = random.Random(seed)
    healthy = [r for r in rows if r["_healthy"]]
    pathological = [r for r in rows if not r["_healthy"]]

    n_healthy = min(len(healthy), round(n_total * healthy_frac))
    n_patho = min(len(pathological), n_total - n_healthy)
    # If one pool is short, top up from the other.
    if n_healthy + n_patho < n_total:
        spare = n_total - n_healthy - n_patho
        if len(healthy) > n_healthy:
            n_healthy = min(len(healthy), n_healthy + spare)
        elif len(pathological) > n_patho:
            n_patho = min(len(pathological), n_patho + spare)

    selected_healthy = _interleave_by_gender(healthy, rng)[:n_healthy]

    by_pathology: Dict[str, List[dict]] = {}
    for row in pathological:
        by_pathology.setdefault(row["pathology"], []).append(row)
    queues = [_interleave_by_gender(group, rng)
              for group in by_pathology.values()]
    rng.shuffle(queues)
    selected_patho: List[dict] = []
    while len(selected_patho) < n_patho and any(queues):
        for queue in queues:
            if queue and len(selected_patho) < n_patho:
                selected_patho.append(queue.pop())

    selected = selected_healthy + selected_patho
    selected.sort(key=lambda r: (r["_healthy"] is False, r["pathology"],
                                 r["session_id"]))
    return selected


# ---------------------------------------------------------------------------
# Workbook
# ---------------------------------------------------------------------------

def _audio_filename(session_id: str, vowel: str, pitch: str) -> str:
    """SVD vowel recordings are named '<session>-<vowel>_<pitch>.wav'
    (e.g. '1-a_n.wav' for the /a/ at normal pitch)."""
    return f"{session_id}-{vowel}_{pitch}.wav"


HEADERS = [
    "row_id", "session_id", "group", "pathology", "age", "gender",
    "audio_file", "task",
    "G", "R", "B", "A", "S",
    "sample_usable", "sample_quality", "clinician_comments",
]
GRBAS_COL_START = 9   # column I
GRBAS_COL_END = 13    # column M


def build_rows(selected: List[dict], vowel: str, pitch: str) -> List[dict]:
    task = f"sustained /{vowel}/ vowel, {'normal' if pitch == 'n' else pitch} pitch"
    out = []
    for index, row in enumerate(selected, start=1):
        out.append({
            "row_id": index,
            "session_id": row["session_id"],
            "group": "healthy" if row["_healthy"] else "pathological",
            "pathology": row["pathology"],
            "age": row["age"],
            "gender": row["gender"],
            "audio_file": _audio_filename(row["session_id"], vowel, pitch),
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

    # Tint the columns the clinician fills in.
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

        usable_dv = DataValidation(type="list", formula1='"yes,no"',
                                   allow_blank=True)
        sheet.add_data_validation(usable_dv)
        usable_dv.add(f"N2:N{last_row}")

        quality_dv = DataValidation(type="list", formula1='"good,fair,poor"',
                                    allow_blank=True)
        sheet.add_data_validation(quality_dv)
        quality_dv.add(f"O2:O{last_row}")

    widths = {
        "A": 8, "B": 12, "C": 13, "D": 26, "E": 6, "F": 8, "G": 18,
        "H": 28, "I": 5, "J": 5, "K": 5, "L": 5, "M": 5,
        "N": 14, "O": 15, "P": 40,
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
        ("Listen to each audio file (column 'audio_file') and rate the voice "
         "on the five GRBAS dimensions.", False),
        ("", False),
        ("Severity scale for every dimension:", True),
        ("  0 = normal      1 = slight      2 = moderate      3 = severe",
         False),
        ("", False),
        ("Dimensions:", True),
    ]
    for letter, description in GRBAS_DIMS:
        lines.append((f"  {letter} — {description}", False))
    lines += [
        ("", False),
        ("Notes:", True),
        ("  • Rate from the sustained vowel only; ignore recording loudness.",
         False),
        ("  • G usually reflects the most affected dimension(s); it need not "
         "equal their sum.", False),
        ("  • If a file is noisy, clipped or wrong-task, set sample_usable=no "
         "and add a comment — you may leave G/R/B/A/S blank.", False),
        ("  • sample_quality (good/fair/poor) is your confidence in the "
         "recording, not the voice.", False),
        ("  • Please do not change the metadata columns A–H.", False),
    ]
    for index, (text, bold) in enumerate(lines, start=1):
        cell = sheet.cell(row=index, column=1, value=text)
        if bold:
            cell.font = Font(bold=True)
    sheet.column_dimensions["A"].width = 95


def copy_audio(selected_rows: List[dict], audio_src: Path, audio_out: Path,
               vowel: str, pitch: str) -> tuple[int, int]:
    """Copy the selected recordings into one folder for the clinician.
    Matches any extension so .nsp or .wav both work. Returns (copied, missing)."""
    audio_out.mkdir(parents=True, exist_ok=True)
    copied = missing = 0
    for row in selected_rows:
        stem = _audio_filename(row["session_id"], vowel, pitch).rsplit(".", 1)[0]
        matches = list(audio_src.rglob(f"{stem}.*"))
        if matches:
            shutil.copy2(matches[0], audio_out / matches[0].name)
            copied += 1
        else:
            missing += 1
    return copied, missing


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", type=Path, required=True,
                        help="SVD metadata CSV/TSV export")
    parser.add_argument("--out", type=Path,
                        default=BASE_DIR / "svd_clinician_grbas_sheet.xlsx")
    parser.add_argument("--n-total", type=int, default=250,
                        help="cohort size (default 250 -- a good-model target)")
    parser.add_argument("--healthy-frac", type=float, default=0.4,
                        help="fraction of healthy controls (default 0.4)")
    parser.add_argument("--vowel", default="a", choices=["a", "i", "u"],
                        help="SVD vowel to label (default a)")
    parser.add_argument("--pitch", default="n",
                        choices=["n", "l", "h"],
                        help="pitch: n normal, l low, h high (default n)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--pathology-col", default=None)
    parser.add_argument("--age-col", default=None)
    parser.add_argument("--gender-col", default=None)
    parser.add_argument("--audio-src", type=Path, default=None,
                        help="if given, copy selected recordings from this "
                             "folder tree into --audio-out")
    parser.add_argument("--audio-out", type=Path,
                        default=BASE_DIR / "svd_clinician_audio")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.healthy_frac <= 1.0:
        raise SystemExit("--healthy-frac must be between 0 and 1")

    rows, resolved = load_metadata(args.metadata, {
        "id": args.id_col, "pathology": args.pathology_col,
        "age": args.age_col, "gender": args.gender_col,
    })
    print(f"columns used: {resolved}")
    print(f"loaded {len(rows)} sessions "
          f"({sum(r['_healthy'] for r in rows)} healthy, "
          f"{sum(not r['_healthy'] for r in rows)} pathological)")

    selected = select_cohort(rows, args.n_total, args.healthy_frac, args.seed)
    if len(selected) < args.n_total:
        print(f"WARNING: only {len(selected)} sessions available "
              f"(requested {args.n_total})")

    sheet_rows = build_rows(selected, args.vowel, args.pitch)
    write_workbook(sheet_rows, args.out)

    n_healthy = sum(r["group"] == "healthy" for r in sheet_rows)
    n_patho = len(sheet_rows) - n_healthy
    n_path_kinds = len({r["pathology"] for r in sheet_rows
                        if r["group"] == "pathological"})
    print(f"selected {len(sheet_rows)} recordings: "
          f"{n_healthy} healthy, {n_patho} pathological "
          f"across {n_path_kinds} diagnoses")
    print(f"workbook -> {args.out}")

    if args.audio_src:
        copied, missing = copy_audio(selected, args.audio_src, args.audio_out,
                                     args.vowel, args.pitch)
        print(f"audio: copied {copied}, missing {missing} -> {args.audio_out}")


if __name__ == "__main__":
    main()
