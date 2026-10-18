"""Controlled falsification tests; these do not calibrate clinical effort grades."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
folder=Path(__file__).resolve().parent;root=folder.parents[1];sys.path.insert(0,str(root))
from head_motion import analyze_payload

def payload(rows):return dict(version='head-motion-trace-v2',source='browser_patch_tracking',samples=rows)

def sinusoid(hz,duration=6,drift=False):
 return payload([[float(t),float(.003*np.sin(2*np.pi*hz*t)+(.02*np.sin(2*np.pi*.5*t) if drift else 0)),0,0,0,1,0] for t in np.arange(0,duration,1/30)])

reports={};noise_cases=0;false_frequencies=0
for seed in range(12):
 rng=np.random.default_rng(seed);noise=rng.normal(0,.0004,(180,2))
 for correlated in (False,True):
  values=noise.copy()
  if correlated:
   for i in range(1,len(values)):values[i]=.65*values[i-1]+values[i]
  rows=[[i/30,float(v[0]),float(v[1]),0,0,1,0] for i,v in enumerate(values)]
  report=analyze_payload(json.dumps(payload(rows)));noise_cases+=1
  false_frequencies+=sum(w.get('dominant_frequency_hz') is not None for w in report['windows'])
assert false_frequencies==0
for name,data in [('fast',sinusoid(4.4)),('fast_with_slow_posture',sinusoid(4.4,drift=True)),('slow',sinusoid(.5,8)),('very_slow',sinusoid(.3,12)),('high',sinusoid(10))]:
 report=analyze_payload(json.dumps(data));assert any(w.get('periodicity_status')=='supported' for w in report['windows']),name
 reports[name]=report
for filename in ('synthetic_traces.json','recovery_traces.json'):
 if (folder/filename).exists():
  for name,data in json.loads((folder/filename).read_text()).items():reports['image_'+name]=analyze_payload(json.dumps(data))
summary=dict(noise_cases=noise_cases,noise_windows_with_published_frequency=false_frequencies,
 source_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'head_motion.py',root/'static/neck-motion.js',root/'static/neck-motion-analysis.js']},reports=reports)
(folder/'experiment_results.json').write_text(json.dumps(summary,indent=2))
print('Noise cases:',noise_cases,'published noise frequencies:',false_frequencies)
for name,r in reports.items():print(name,'windows',len(r['windows']),'supported',sum(w.get('periodicity_status')=='supported' for w in r['windows']))
