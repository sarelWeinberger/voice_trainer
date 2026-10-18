"""Build a LINK-HOSTED GRBAS rating page from the VOICED cohort.

Why this exists (vs. build_voiced_clinician_html.py):
    The single-file HTML embeds all 208 recordings as base64 -> a 21 MB file.
    That file does not survive being sent over WhatsApp/email: the messenger
    can't open a 21 MB `.html`, so the clinician sees "the file won't open".

    This builder instead emits a SMALL page that loads each recording from a
    URL on demand, plus a folder of the WAVs. Both go under the app's `static/`
    dir, so the deployed FastAPI app (HF Space) serves them at a public URL and
    the clinician just clicks a LINK -- nothing to download or open. Works on a
    phone or a computer; audio streams only when a row is played.

    The rating UI, localStorage autosave and CSV export are unchanged, so the
    returned CSV is identical to the embedded-HTML route.

Usage:
    python build_voiced_clinician_web.py \
        --db-dir voiced_database/voice-icar-federico-ii-database-1.0.0

Outputs (default):
    static/voiced_rating.html          # the page (small, no embedded audio)
    static/voiced_audio/voiceNNN.wav   # one WAV per recording

Then deploy with ./deploy_hf.sh and send the clinician:
    https://<space-host>/voiced_rating.html
"""

from __future__ import annotations

import argparse
import html
import shutil
from pathlib import Path

import build_voiced_clinician_sheet as sheet  # reuse metadata + cohort logic

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR.parent / "root" / "voice_app" / "static"

GRBAS_HELP = {
    "G": "Grade — overall severity",
    "R": "Roughness",
    "B": "Breathiness",
    "A": "Asthenia (weakness)",
    "S": "Strain (effort)",
}


def _ensure_audio(session_id: str, db_dir: Path, existing: Path,
                  audio_out: Path) -> bool:
    """Put ``<session_id>.wav`` into ``audio_out``.

    Prefer copying an already-converted WAV from ``existing`` (the
    voiced_clinician_audio folder); fall back to converting from the raw
    VOICED signal. Returns True if a WAV now exists in ``audio_out``.
    """
    dst = audio_out / f"{session_id}.wav"
    src = existing / f"{session_id}.wav"
    audio_out.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copyfile(src, dst)
        return True
    return sheet.convert_to_wav(session_id, db_dir, audio_out)


def build_html(rows: list[dict], db_dir: Path, existing: Path,
               audio_out: Path, audio_base: str) -> str:
    cards = []
    skipped = 0
    for row in rows:
        sid = row["session_id"]
        if not _ensure_audio(sid, db_dir, existing, audio_out):
            skipped += 1
            continue
        sid_e = html.escape(sid)
        diagnosis = html.escape(row["diagnosis"])
        group = "healthy" if row["_healthy"] else "pathological"
        meta = f"{group} · {html.escape(str(row['age']))}{html.escape(row['gender'])}"
        src = f"{audio_base}/{sid_e}.wav"
        selects = "".join(
            f'<td><select data-dim="{dim}" data-id="{sid_e}" '
            f'class="grbas" title="{GRBAS_HELP[dim]}">'
            f'<option value=""></option><option>0</option><option>1</option>'
            f'<option>2</option><option>3</option></select></td>'
            for dim in ("G", "R", "B", "A", "S")
        )
        cards.append(f"""
        <tr data-id="{sid_e}" data-diagnosis="{diagnosis}" data-group="{group}">
          <td class="sid">{sid_e}</td>
          <td class="meta">{meta}</td>
          <td><audio controls preload="none" src="{src}"></audio></td>
          {selects}
          <td><select data-field="usable" data-id="{sid_e}" class="aux">
            <option value="yes" selected>yes</option><option value="no">no</option>
          </select></td>
          <td><select data-field="quality" data-id="{sid_e}" class="aux">
            <option value=""></option><option>good</option><option>fair</option>
            <option>poor</option></select></td>
          <td><input type="text" data-field="comment" data-id="{sid_e}"
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
  .note { margin: 14px 20px 0; padding: 10px 16px; background: #fff4e5;
    border: 1px solid #f0c98a; border-radius: 8px; font-size: 13px; }
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

<div class="note">
  <b>Please rate on a computer with headphones</b> for accurate perceptual
  judgement. Your ratings save automatically in this browser as you go — but
  when you finish (or before closing), click <b>Export results (CSV)</b> and
  send that one small file back.
</div>

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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db-dir", type=Path, required=True,
                        help="extracted VOICED database folder")
    parser.add_argument("--out", type=Path,
                        default=STATIC_DIR / "voiced_rating.html")
    parser.add_argument("--audio-out", type=Path,
                        default=STATIC_DIR / "voiced_audio",
                        help="where the served WAVs are written")
    parser.add_argument("--existing-audio", type=Path,
                        default=BASE_DIR / "voiced_clinician_audio",
                        help="already-converted WAVs to copy from (fast path)")
    parser.add_argument("--audio-base", default="voiced_audio",
                        help="URL path the page loads audio from "
                             "(relative to the page)")
    parser.add_argument("--n-total", type=int, default=208)
    parser.add_argument("--healthy-frac", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = sheet.load_metadata(args.db_dir)
    selected = sheet.select_cohort(rows, args.n_total, args.healthy_frac, args.seed)
    print(f"selected {len(selected)} recordings")
    page = build_html(selected, args.db_dir, args.existing_audio,
                      args.audio_out, args.audio_base)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    n_wav = len(list(args.audio_out.glob("*.wav")))
    size_kb = args.out.stat().st_size / 1e3
    print(f"wrote {args.out} ({size_kb:.0f} KB)")
    print(f"wrote {n_wav} WAVs -> {args.audio_out}")
    print("\nNext: ./deploy_hf.sh   then send the clinician:")
    print("  https://<space-host>/voiced_rating.html")


if __name__ == "__main__":
    main()
