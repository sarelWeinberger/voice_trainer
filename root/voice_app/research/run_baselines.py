"""Run acoustic falsification and speaker-held-out Lombard proxy experiments.
Usage: python research/run_baselines.py /tmp/lombardgrid_audio.zip
"""
import io
import json
import re
import sys
import zipfile
from pathlib import Path
import numpy as np
import soundfile as sf
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observable_effort import acoustic_timeline


def features(x, sr):
    rows = [r for r in acoustic_timeline(x, sr) if r['usable']]
    return [np.mean([r['rms_dbfs'] for r in rows]),
            np.mean([r['high_frequency_energy_ratio'] for r in rows])] if rows else None


def main():
    sr = 16000
    x = .1 * np.sin(2 * np.pi * 200 * np.arange(sr * 2) / sr)
    a, b = features(x, sr), features(x * 2, sr)
    report = {'gain_falsification': {'rms_change_db': b[0] - a[0],
              'spectral_ratio_change': b[1] - a[1],
              'conclusion': 'Recording gain changes intensity without any change in vocal production; intensity cannot identify effort.'}}
    if len(sys.argv) > 1:
        X, y, groups = [], [], []
        with zipfile.ZipFile(sys.argv[1]) as archive:
            for name in sorted(archive.namelist()):
                match = re.search(r's(\d+)_(?:\d+_\d+_\d+_)?([lrp])_', name)
                if not name.endswith('.wav') or not match:
                    continue
                samples, rate = sf.read(io.BytesIO(archive.read(name)))
                row = features(samples, rate)
                if row is not None and np.isfinite(row).all():
                    X.append(row); y.append(int(match[2] == 'l')); groups.append(match[1])
        X, y = np.array(X), np.array(y)
        if len(set(groups)) < 5:
            raise ValueError('Insufficient speakers or filename format mismatch')
        results = {}
        for name, indices in [('intensity_only', [0]), ('spectral_balance_only', [1]), ('combined', [0, 1])]:
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=0))
            scores = cross_val_score(model, X[:, indices], y, groups=groups,
                                     cv=GroupKFold(5), scoring='balanced_accuracy')
            results[name] = {'fold_balanced_accuracy': scores.tolist(), 'mean': float(scores.mean())}
        report['lombard_proxy'] = {'samples': len(y), 'speakers': len(set(groups)),
                                  'class_counts': np.bincount(y).tolist(), 'baselines': results,
                                  'limitation': 'Lombard condition is not muscle tension, self-perceived effort, strain or clinical diagnosis. No visual or physiological validation.'}
    out = Path(__file__).with_name('baseline_results.json')
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(out.read_text())

if __name__ == '__main__':
    main()
