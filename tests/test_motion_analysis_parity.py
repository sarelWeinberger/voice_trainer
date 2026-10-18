"""The offline fallback must apply the same gates as the server."""
import json,subprocess,sys,unittest
from pathlib import Path
import numpy as np
ROOT=(Path(__file__).resolve().parents[1] / "root" / "voice_app");sys.path.insert(0,str(ROOT))
from head_motion import analyze_payload

class ParityTests(unittest.TestCase):
 def test_server_and_browser_agree_for_signal_noise_loss_and_multiple_rates(self):
  cases=[]
  for hz,fps,duration in ((4.4,30,6),(.5,30,8),(10,30,6),(4.4,15,6)):
   rows=[[float(t),float(.003*np.sin(2*np.pi*hz*t)),0,0,0,1,0] for t in np.arange(0,duration,1/fps)]
   cases.append(dict(version='head-motion-trace-v2',source='browser_patch_tracking',samples=rows))
  rng=np.random.default_rng(1234)
  cases.append(dict(version='head-motion-trace-v2',source='browser_patch_tracking',samples=[[i/30,float(v),0,0,0,1,0] for i,v in enumerate(rng.normal(0,.0004,180))]))
  lost=json.loads(json.dumps(cases[0]));
  for i,row in enumerate(lost['samples']):
   if i>=45:row[6]=1
  cases.append(lost)
  script="const {analyze}=require(process.argv[1]);let input='';process.stdin.on('data',c=>input+=c);process.stdin.on('end',()=>process.stdout.write(JSON.stringify(JSON.parse(input).map(analyze))));"
  local=json.loads(subprocess.run(['node','-e',script,str(ROOT/'static/neck-motion-analysis.js')],input=json.dumps(cases),capture_output=True,text=True,check=True).stdout)
  for payload,client in zip(cases,local):
   server=analyze_payload(json.dumps(payload));self.assertEqual(server['status'],client['status'])
   self.assertEqual(len(server['windows']),len(client['windows']))
   for a,b in zip(server['windows'],client['windows']):
    for key in ('status','reason','periodicity_status','dominant_frequency_hz'):
     self.assertEqual(a.get(key),b.get(key),(key,a,b))
    for key in ('amplitude_percent_image_width','periodic_fraction','half_coherence','half_amplitude_ratio'):
     if a.get(key) is None:self.assertIsNone(b.get(key))
     else:self.assertAlmostEqual(a[key],b[key],places=3,msg=(key,a,b))

if __name__=='__main__':unittest.main()
