"""Replay identical decoded frames through the actual browser tracker (no API).

Images are temporary; only derived coordinates/diagnostics persist.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2
import mediapipe as mp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from head_motion import analyze_payload


def replay(video, output, module):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = 256, round(256 * cap.get(4) / cap.get(3))
    rows, grays = [], []
    with mp.solutions.face_mesh.FaceMesh(refine_landmarks=False) as detector:
        i, last_seed = 0, -2
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            anchors = None
            if t - last_seed >= .9:
                result = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if result.multi_face_landmarks:
                    points = result.multi_face_landmarks[0].landmark
                    anchors = [[points[k].x, points[k].y] for k in (10,151,168,33,263,1)]
                    last_seed = t
            gray = cv2.cvtColor(cv2.resize(frame, (width, height)), cv2.COLOR_BGR2GRAY)
            grays.append(gray.tobytes())
            rows.append(dict(t=t, anchors=anchors))
            i += 1
    cap.release()
    with tempfile.TemporaryDirectory(prefix='neck-replay-') as directory:
        path = Path(directory)
        (path/'gray.bin').write_bytes(b''.join(grays))
        (path/'frames.json').write_text(json.dumps(dict(width=width,height=height,rows=rows)))
        script = r'''
const fs=require('fs'),{MotionTracker,match,backgroundModel}=require(process.argv[1]);
const meta=JSON.parse(fs.readFileSync(process.argv[2]+'/frames.json'));
const data=fs.readFileSync(process.argv[2]+'/gray.bin'),size=meta.width*meta.height;
const tracker=new MotionTracker(meta.width,meta.height,meta.rows.find(r=>r.anchors).anchors),debug=[];
for(let i=0;i<meta.rows.length;i++){
 const row=meta.rows[i],gray=new Uint8Array(data.subarray(i*size,(i+1)*size));
 if(tracker.previous){
  const matches=tracker.background.map(p=>match(tracker.previous,gray,meta.width,meta.height,p)).filter(Boolean);
  debug.push({t:row.t,points:tracker.background.length,matches:matches.length,
   zones:[...new Set(matches.map(p=>(p.origin[0]<meta.width/2?0:1)+(p.origin[1]<meta.height/2?0:2)))],
   deltas:matches.map(p=>p.delta),model:!!backgroundModel(matches,meta.width,meta.height,[128,60])});
 }
 tracker.push(gray,row.t,row.anchors);
}
console.log(JSON.stringify({trace:{version:'head-motion-trace-v2',source:'browser_patch_tracking',samples:tracker.rows,
 tracking_diagnostics:{head_missing_samples:tracker.headMissing,background_missing_samples:tracker.backgroundMissing,
 reinitializations:tracker.reinitializations,background_rejected_samples:tracker.backgroundRejected}},debug}));
'''
        completed = subprocess.run(['node','-e',script,str(module.resolve()),str(path)],capture_output=True,text=True,check=True)
    result=json.loads(completed.stdout)
    result.update(source_file=video.name,fps=fps,report=analyze_payload(json.dumps(result['trace'])))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(source=video.name,status=result['report']['status'],samples=len(rows),
        measured=sum(w['status']=='measured' for w in result['report'].get('windows',[])),
        diagnostics=result['trace']['tracking_diagnostics'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--module',type=Path,default=ROOT/'static/neck-motion.js')
    args=parser.parse_args()
    replay(args.video,args.output,args.module)
