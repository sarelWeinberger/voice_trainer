"""Validate generated image-tracker traces; no provider calls or clinical cutoffs."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from head_motion import analyze_payload

folder=Path(__file__).resolve().parent
reports={}
for filename in ('synthetic_traces.json','recovery_traces.json'):
    for name,payload in json.loads((folder/filename).read_text()).items():
        report=analyze_payload(json.dumps(payload));reports[name]=report
        measured=[w for w in report['windows'] if w['status']=='measured']
        assert measured,(name,report)
        amplitudes=[w['amplitude_percent_image_width'] for w in measured]
        if name=='shake':
            assert min(amplitudes)>.2
            assert all(abs(w['dominant_frequency_hz']-4.4)<.15 for w in measured)
        else:assert max(amplitudes)<.05,(name,amplitudes)
        if name.endswith('recovery'):
            assert any(w.get('reason')=='tracking_reinitialized' for w in report['windows'])
        print(name, 'measured windows',len(measured),'amplitude range',min(amplitudes),max(amplitudes))
(folder/'control_reports.json').write_text(json.dumps(reports,indent=2))
