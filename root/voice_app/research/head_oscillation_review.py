"""Exploratory dense head-motion audit; not a validated tremor/effort detector."""
import argparse,json,sys
from pathlib import Path
import cv2
import numpy as np
from scipy.signal import detrend,welch,butter,sosfiltfilt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from neck_v1_cv import CVProcessor

def audit(video,output):
 cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS)
 processor=CVProcessor();rows=[];previous=None;old_anchors=None;bg_total=np.zeros(2);face_total=np.zeros(2)
 try:
  while True:
   ok,frame=cap.read()
   if not ok:break
   frame=cv2.resize(frame,(640,round(frame.shape[0]*640/frame.shape[1])))
   gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);i=len(rows);t=i/fps
   row=processor.process(frame,t);anchors=np.array(row.get('landmarks',{}).get('face_anchors',[]),dtype=np.float32)
   bg_delta=None;face_delta=None
   if previous is not None:
    mask=np.zeros_like(gray);mask[:,:gray.shape[1]//8]=255;mask[:,-gray.shape[1]//8:]=255
    corners=cv2.goodFeaturesToTrack(previous,80,.02,8,mask=mask)
    if corners is not None:
     tracked,status,_=cv2.calcOpticalFlowPyrLK(previous,gray,corners,None)
     valid=status[:,0]==1
     if valid.sum()>=8:bg_delta=np.median((tracked-corners)[valid,0,:],axis=0)
    if old_anchors is not None and len(old_anchors)==6:
     tracked,status,_=cv2.calcOpticalFlowPyrLK(previous,gray,old_anchors[:,None,:],None)
     back,back_status,_=cv2.calcOpticalFlowPyrLK(gray,previous,tracked,None)
     valid=(status[:,0]==1)&(back_status[:,0]==1)&(np.linalg.norm(back[:,0,:]-old_anchors,axis=1)<.8)
     if valid.sum()>=4:face_delta=np.median(tracked[valid,0,:]-old_anchors[valid],axis=0)
   if bg_delta is not None:bg_total+=bg_delta
   if face_delta is not None and bg_delta is not None:face_total+=face_delta-bg_delta
   valid=bool(len(anchors)==6 and (i==0 or (bg_delta is not None and face_delta is not None)))
   rows.append(dict(t=t,valid=valid,landmark_center_px=(anchors.mean(0)-bg_total).tolist() if len(anchors)==6 else None,
      tracked_head_relative_px=face_total.tolist() if valid else None,
      background_translation_px=bg_total.tolist(),head_pitch_deg=row['features'].get('head_pitch_deg'),
      head_roll_deg=row['features'].get('head_roll_deg')))
   previous=gray;old_anchors=anchors if len(anchors)==6 else None
 finally:cap.release();processor.close()
 stats={};band_traces={};valid=np.array([r['valid'] for r in rows])
 # Do not interpolate invalid spans or manufacture oscillation from missing tracking.
 if valid.all() and len(rows)>fps*3:
  for key in ['landmark_center_px','tracked_head_relative_px','background_translation_px']:
   values=np.array([r[key] for r in rows]);clean=detrend(values,axis=0)
   band=sosfiltfilt(butter(3,[2,min(10,fps*.4)],btype='bandpass',fs=fps,output='sos'),clean,axis=0)
   f,power=welch(clean,fs=fps,nperseg=min(len(rows),128),axis=0)
   chosen=(f>=2)&(f<=min(10,fps*.4));combined=power.sum(1)
   stats[key]=dict(exploratory_band_hz=[2,min(10,fps*.4)],band_rms_px=float(np.sqrt(np.mean(np.sum(band*band,axis=1)))),
       band_peak_hz=float(f[chosen][np.argmax(combined[chosen])]),peak_is_not_a_tremor_diagnosis=True)
   band_traces[key]=band.tolist()
 output.parent.mkdir(parents=True,exist_ok=True)
 result=dict(source_file=video.name,fps=fps,frame_count=len(rows),time_basis='container mean FPS; not independent clock validation',
    valid_frame_fraction=float(valid.mean()),statistics=stats,rows=rows,band_traces=band_traces,
    limitations=['No clinical tremor or muscle tension ground truth','Image motion can include normal movement, articulation and tracking noise',
      '2–10 Hz band is exploratory, not a clinical cutoff','Static border translation cannot fully remove camera rotation or rolling shutter',
      'No neural network trained; no production scores modified'])
 output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['rows','band_traces']}))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('video',type=Path);p.add_argument('--output',required=True,type=Path)
 args=p.parse_args();audit(args.video,args.output)
