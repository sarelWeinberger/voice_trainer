"""Build a single self-contained HTML GRBAS rating tool from the VOICED cohort.

Unlike the .xlsx + Drive route, this bundles every recording (as base64 audio)
plus the rating table and instructions into ONE .html file. The clinician opens
it in any browser, plays each recording inline, picks G/R/B/A/S from dropdowns,
and clicks "Export results" to download a CSV to send back. Work auto-saves to
the browser's localStorage so nothing is lost if the tab closes.

Usage:
    python build_voiced_clinician_html.py \
        --db-dir voiced_database/voice-icar-federico-ii-database-1.0.0
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import wave
from pathlib import Path

import numpy as np

import build_voiced_clinician_sheet as sheet  # reuse metadata + cohort logic

BASE_DIR = Path(__file__).resolve().parent

GRBAS_HELP = {
    "G": "Grade — overall severity",
    "R": "Roughness",
    "B": "Breathiness",
    "A": "Asthenia (weakness)",
    "S": "Strain (effort)",
}


def _wav_data_uri(session_id: str, db_dir: Path) -> str | None:
    """Build a base64 'data:audio/wav' URI from the VOICED .txt signal."""
    txt_path = db_dir / f"{session_id}.txt"
    if not txt_path.exists():
        return None
    sample_rate = sheet._read_sample_rate(db_dir / f"{session_id}.hea")
    signal = np.loadtxt(txt_path, dtype=np.float64)
    if signal.ndim > 1:
        signal = signal[:, 0]
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak > 0:
        signal = signal / peak * 0.98
    pcm = np.clip(np.round(signal * 32767.0), -32768, 32767).astype("<i2")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def build_html(rows: list[dict], db_dir: Path) -> str:
    cards = []
    skipped = 0
    for row in rows:
        uri = _wav_data_uri(row["session_id"], db_dir)
        if uri is None:
            skipped += 1
            continue
        sid = html.escape(row["session_id"])
        diagnosis = html.escape(row["diagnosis"])
        group = "healthy" if row["_healthy"] else "pathological"
        meta = f"{group} · {html.escape(str(row['age']))}{html.escape(row['gender'])}"
        selects = "".join(
            f'<td><select data-dim="{dim}" data-id="{sid}" '
            f'class="grbas" title="{GRBAS_HELP[dim]}">'
            f'<option value=""></option><option>0</option><option>1</option>'
            f'<option>2</option><option>3</option></select></td>'
            for dim in ("G", "R", "B", "A", "S")
        )
        cards.append(f"""
        <tr data-id="{sid}" data-diagnosis="{diagnosis}" data-group="{group}">
          <td class="sid">{sid}</td>
          <td class="meta">{meta}</td>
          <td><audio controls preload="none" src="{uri}"></audio></td>
          {selects}
          <td><select data-field="usable" data-id="{sid}" class="aux">
            <option value="yes" selected>yes</option><option value="no">no</option>
          </select></td>
          <td><select data-field="quality" data-id="{sid}" class="aux">
            <option value=""></option><option>good</option><option>fair</option>
            <option>poor</option></select></td>
          <td><input type="text" data-field="comment" data-id="{sid}"
            class="comment" placeholder="…"></td>
        </tr>""")

    if skipped:
        print(f"WARNING: {skipped} recordings had no audio and were skipped")

    rows_html = "".join(cards)
    total = len(cards)
    return _PAGE_TEMPLATE.replace("__ROWS__", rows_html).replace("__TOTAL__", str(total))


_PAGE_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VOICED — GRBAS clinician rating</title>
<style>
  :root { font-family: system-ui, Segoe UI, Arial, sans-serif; }
  body { margin: 0; color: #1a1a1a; background: #f4f6f9; }
  header { position: sticky; top: 0; z-index: 10; background: #1F4E78;
    color: #fff; padding: 12px 20px; box-shadow: 0 2px 6px rgba(0,0,0,.2); }
  header h1 { margin: 0 0 4px; font-size: 18px; }
  .bar { display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
    font-size: 14px; }
  .bar button { background: #fff; color: #1F4E78; border: 0; border-radius: 6px;
    padding: 7px 14px; font-weight: 600; cursor: pointer; }
  .bar button.secondary { background: #2c5f8f; color: #fff; }
  #progress { font-weight: 600; }
  details { margin: 14px 20px; background: #fff; border-radius: 8px; padding: 8px 16px; }
  summary { cursor: pointer; font-weight: 600; }
  details ul { line-height: 1.6; }
  table { border-collapse: collapse; width: calc(100% - 40px); margin: 14px 20px 60px;
    background: #fff; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
  th, td { border: 1px solid #e2e6ea; padding: 6px 8px; text-align: center;
    font-size: 13px; }
  th { background: #eaf0f6; position: sticky; top: 64px; z-index: 5; }
  td.sid { font-weight: 600; white-space: nowrap; }
  td.meta { color: #555; white-space: nowrap; font-size: 12px; }
  audio { height: 34px; width: 230px; }
  select.grbas { width: 46px; font-size: 14px; }
  select.grbas:invalid, select.grbas.empty { background: #fff7e6; }
  tr.done { background: #f1faf1; }
  input.comment { width: 130px; }
  .grbas-head { background: #fff2cc !important; }
</style>
</head>
<body>
<header>
  <h1>VOICED — GRBAS perceptual rating</h1>
  <div class="bar">
    <span id="progress">0 / __TOTAL__ rated</span>
    <button onclick="exportCsv()">⬇ Export results (CSV)</button>
    <button class="secondary" onclick="saveNow()">Save progress</button>
    <span id="saved" style="font-size:12px;opacity:.85"></span>
  </div>
</header>

<details open>
  <summary>Instructions — how to rate (click to collapse)</summary>
  <p>Listen to each recording (a sustained <b>/a/</b> vowel) and rate the voice on
     the five GRBAS dimensions. Severity scale for every dimension:
     <b>0</b> normal · <b>1</b> slight · <b>2</b> moderate · <b>3</b> severe.</p>
  <ul>
    <li><b>G</b> — Grade: overall severity of dysphonia (usually reflects the
        most affected dimension; need not equal their sum).</li>
    <li><b>R</b> — Roughness: irregularity of vocal fold vibration.</li>
    <li><b>B</b> — Breathiness: audible air escape.</li>
    <li><b>A</b> — Asthenia: vocal weakness / hypofunction.</li>
    <li><b>S</b> — Strain: effort / hyperfunction.</li>
  </ul>
  <ul>
    <li>Recordings are 8 kHz — fine for G/R/B/A/S; treat very high-frequency
        detail with caution. Ignore recording loudness.</li>
    <li>If a recording is noisy/clipped/wrong-task, set <b>usable = no</b>, add a
        comment, and you may leave G/R/B/A/S blank.</li>
    <li>The <b>diagnosis</b> column is the database's medical label, not a GRBAS
        hint — rate by ear, blind to it where you can.</li>
    <li>Your work auto-saves in this browser. When finished, click
        <b>Export results (CSV)</b> and send the downloaded file back.</li>
  </ul>
</details>

<table>
  <thead><tr>
    <th>recording</th><th>info</th><th>audio</th>
    <th class="grbas-head">G</th><th class="grbas-head">R</th>
    <th class="grbas-head">B</th><th class="grbas-head">A</th>
    <th class="grbas-head">S</th>
    <th>usable</th><th>quality</th><th>comment</th>
  </tr></thead>
  <tbody>__ROWS__</tbody>
</table>

<script>
const KEY = "voiced_grbas_ratings_v1";
const TOTAL = __TOTAL__;

function load() {
  let data = {};
  try { data = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
  document.querySelectorAll("select[data-id], input[data-id]").forEach(el => {
    const id = el.dataset.id;
    const rec = data[id]; if (!rec) return;
    const k = el.dataset.dim || el.dataset.field;
    if (rec[k] !== undefined && rec[k] !== null) el.value = rec[k];
  });
  refresh();
}

function collect() {
  const data = {};
  document.querySelectorAll("tr[data-id]").forEach(tr => {
    const id = tr.dataset.id;
    const rec = { diagnosis: tr.dataset.diagnosis, group: tr.dataset.group };
    tr.querySelectorAll("select, input").forEach(el => {
      const k = el.dataset.dim || el.dataset.field;
      rec[k] = el.value;
    });
    data[id] = rec;
  });
  return data;
}

function refresh() {
  let done = 0;
  document.querySelectorAll("tr[data-id]").forEach(tr => {
    const grbas = [...tr.querySelectorAll("select.grbas")];
    const usable = tr.querySelector('select[data-field="usable"]').value;
    const filled = grbas.every(s => s.value !== "");
    const complete = filled || usable === "no";
    tr.classList.toggle("done", complete);
    grbas.forEach(s => s.classList.toggle("empty", s.value === "" && usable !== "no"));
    if (complete) done++;
  });
  document.getElementById("progress").textContent = done + " / " + TOTAL + " rated";
}

function saveNow() {
  localStorage.setItem(KEY, JSON.stringify(collect()));
  const s = document.getElementById("saved");
  const t = new Date().toLocaleTimeString();
  s.textContent = "saved " + t;
  refresh();
}

function csvCell(v) {
  v = (v === undefined || v === null) ? "" : String(v);
  return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
}

function exportCsv() {
  saveNow();
  const data = collect();
  const cols = ["session_id","diagnosis","group","G","R","B","A","S",
                "usable","quality","comment"];
  const lines = [cols.join(",")];
  Object.keys(data).sort().forEach(id => {
    const r = data[id];
    lines.push([id, r.diagnosis, r.group, r.G, r.R, r.B, r.A, r.S,
                r.usable, r.quality, r.comment].map(csvCell).join(","));
  });
  const blob = new Blob([lines.join("\n")], {type: "text/csv;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "voiced_grbas_clinician_ratings.csv";
  a.click();
}

document.addEventListener("change", e => {
  if (e.target.matches("select[data-id], input[data-id]")) saveNow();
});
window.addEventListener("DOMContentLoaded", load);
</script>
</body>
</html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db-dir", type=Path, required=True,
                        help="extracted VOICED database folder")
    parser.add_argument("--out", type=Path,
                        default=BASE_DIR / "voiced_clinician_rating.html")
    parser.add_argument("--n-total", type=int, default=208)
    parser.add_argument("--healthy-frac", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = sheet.load_metadata(args.db_dir)
    selected = sheet.select_cohort(rows, args.n_total, args.healthy_frac, args.seed)
    print(f"selected {len(selected)} recordings")
    page = build_html(selected, args.db_dir)
    args.out.write_text(page, encoding="utf-8")
    size_mb = args.out.stat().st_size / 1e6
    print(f"wrote {args.out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
