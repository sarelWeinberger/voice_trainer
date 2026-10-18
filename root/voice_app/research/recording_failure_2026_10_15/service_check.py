"""Actual authenticated app routes + default OpenAI transport, isolated database.

Sends the user's authorized clip; retains derived results only. No test doubles.
"""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import cv2
import imageio_ffmpeg
from dotenv import load_dotenv

ROOT=Path(__file__).resolve().parents[2]
FOLDER=Path(__file__).resolve().parent


def main():
    load_dotenv(ROOT.parents[1]/'.env',override=True)
    assert os.environ.get('OPENAI_API_KEY'), 'Server key is not configured'
    with tempfile.TemporaryDirectory(prefix='neck-service-check-') as directory:
        os.environ['DATABASE_URL']='sqlite:///'+str(Path(directory)/'test.sqlite')
        sys.path.insert(0,str(ROOT))
        from fastapi.testclient import TestClient
        import app
        from database import SessionLocal,User,NeckRecording
        with SessionLocal() as db:
            user=User(username='local_recording_check',hashed_password='unused-test-login',role=3)
            db.add(user);db.commit();db.refresh(user);user_id=user.id
        token=app.create_access_token({'sub':'local_recording_check'})
        headers={'Authorization':'Bearer '+token}
        video=ROOT/'data/WIN_20261015_21_04_57_Pro.mp4'
        wav=Path(directory)/'audio.wav'
        conversion=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(video),'-vn','-ac','1','-ar','16000','-y',str(wav)],capture_output=True)
        assert conversion.returncode==0,'Audio extraction failed'
        cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps
        frames=[]
        for n in range(int(duration*5)):
            cap.set(cv2.CAP_PROP_POS_FRAMES,round(n/5*fps));ok,image=cap.read()
            assert ok,'Video decoding failed'
            image=cv2.resize(image,(480,270))
            ok,jpeg=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,95]);assert ok
            frames.append((n/5,jpeg.tobytes()))
        cap.release()
        with TestClient(app.app) as client:
            started=client.post('/api/neck-effort/recording/start',headers=headers)
            assert started.status_code==200
            session=started.json()['session_id']
            for i in range(0,len(frames),10):
                batch=frames[i:i+10]
                response=client.post('/api/neck-effort/frames',headers=headers,
                    data={'session_id':session,'timestamps':json.dumps([p[0] for p in batch])},
                    files=[('frames',('frame.jpg',p[1],'image/jpeg')) for p in batch])
                assert response.status_code==200, response.status_code
                assert response.json()['accepted_frames']==len(batch)
            before=time.monotonic()
            response=client.post('/api/neck-effort/recording/stop',headers=headers,
                data={'session_id':session,'audio_offset_seconds':'0',
                      'motion_trace':(FOLDER/'desktop_normal_trace.json').read_text()},
                files={'audio':('audio.wav',wav.read_bytes(),'audio/wav')})
            elapsed=time.monotonic()-before
            assert response.status_code==200,response.status_code
            packet=response.json()['neck_effort'];analysis=packet['neckAnalysis']
            result=dict(source_file=video.name,transport='default production interpret(), real OpenAI',
                database='isolated temporary SQLite',stop_seconds=elapsed,http_status=response.status_code,
                packet=packet)
            with SessionLocal() as db:
                saved=db.query(NeckRecording).filter_by(user_id=user_id,session_id=session).one()
                result['saved_status']=json.loads(saved.summary)['neckAnalysis']['status']
                result['saved_frame_count']=saved.frame_count
            (FOLDER/'service_after.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
            print(json.dumps(dict(http_status=response.status_code,status=analysis['status'],stop_seconds=elapsed,
                captured_frames=result['saved_frame_count'],saved_status=result['saved_status'],
                motion_status=analysis['movement_stability']['status'],reasons=packet.get('reasons')),ensure_ascii=False),flush=True)
            assert analysis['status']=='complete',packet.get('reasons')
            assert result['saved_status']=='complete'
            assert analysis['movement_stability']['status']=='measured'
        for engine in app.neck_effort_engines.values():engine.close()


if __name__=='__main__':main()
