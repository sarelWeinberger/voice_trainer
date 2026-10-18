"""Native MP4/WAV decode and real tracker/VAD insufficient-quality integration."""
import sys
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
import soundfile as sf
sys.path.insert(0,str((Path(__file__).resolve().parents[1] / "root" / "voice_app")))
from observable_effort import analyze_video_audio, neck_audio_timing
from neck_v1_scoring import NeckAnalysis

class MediaTests(unittest.TestCase):
    def test_real_decode_no_person_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            video=Path(directory)/'blank.mp4';audio=Path(directory)/'silent.wav'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),10,(320,240))
            self.assertTrue(writer.isOpened())
            for _ in range(20):writer.write(np.zeros((240,320,3),np.uint8))
            writer.release();sf.write(audio,np.zeros(32000),16000)
            result=analyze_video_audio(video,audio,.1)
            parsed=NeckAnalysis.model_validate(result['neckAnalysis'])
            self.assertEqual(parsed.status,'insufficient_quality')
            self.assertIsNone(parsed.overall.category)
            self.assertEqual(len(parsed.dimensions),5)
            self.assertTrue(all(item.score is None for item in parsed.dimensions.values()))
            self.assertEqual(result['synchronization']['audio_offset_seconds'],.1)

    def test_vad_silence_not_speech(self):
        result=neck_audio_timing(np.zeros(16000),16000,.25)
        self.assertFalse(result[0]['speech_active'])
        self.assertEqual(result[0]['t'],.25)

if __name__=='__main__':unittest.main()
