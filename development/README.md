# Development files

The running app lives in `root/voice_app`. This directory preserves offline
training scripts and datasets, clinician-review deliverables, prior experiments,
and obsolete files removed from the runtime folder. User recordings remain in
`root/voice_app/data`; the database and local environment files were not moved.

`cleanup_moves.json` records the original and current paths. Historical research
reports retain their original commands and paths; adjust those paths before
reproducing an old experiment. Nothing here is included in the runtime manifest
or Docker context. `README.original.md` preserves the previous training guide.
Detailed neck-analysis documentation is in `NECK_ANALYSIS_V1.md`.

Install optional tools with:

```bash
root/voice_app/venv/bin/pip install -r development/requirements.txt
```

Run tests from the repository root with `./check.sh`. To run an offline tool:

```bash
PYTHONPATH=root/voice_app root/voice_app/venv/bin/python development/train_grbas.py --help
```

Training data defaults are now relative to this directory. The production model
output still points to the runtime app; synthetic/pretraining outputs stay here.
The published clinician page and its audio remain under the app's `static/`.
