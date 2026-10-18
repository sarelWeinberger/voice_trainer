import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "root" / "voice_app")))
from observable_effort import ObservableEffortEngine, acoustic_timeline, decode_live_neck_audio
from neck_effort_engine import NeckEffortEngine


class ObservableTests(unittest.TestCase):
    def test_gain_is_not_effort(self):
        x = .1 * np.sin(2 * np.pi * 200 * np.arange(16000) / 16000)
        a = acoustic_timeline(x, 16000)[0]
        b = acoustic_timeline(x * 2, 16000)[0]
        self.assertAlmostEqual(b['rms_dbfs'] - a['rms_dbfs'], 6.02, places=2)
        self.assertIsNone(a['effort'])

    def test_silence_and_clipping_abstain(self):
        for samples in (np.zeros(8000), np.ones(8000)):
            self.assertFalse(acoustic_timeline(samples, 8000)[0]['usable'])

    def test_timing_and_partial_window(self):
        rows = acoustic_timeline(np.zeros(20000), 8000)
        self.assertEqual([r['t'] for r in rows], [0, 1, 2])
        self.assertEqual(rows[-1]['duration'], .5)

    def test_invalid_audio_rejected(self):
        with self.assertRaises(ValueError):
            acoustic_timeline([np.nan], 8000)

    def test_compatibility_name_selects_v1(self):
        self.assertIs(ObservableEffortEngine, NeckEffortEngine)

    def test_live_audio_limits_and_missing_vad_remain_unknown(self):
        import io
        import soundfile as sf
        data=io.BytesIO();sf.write(data,np.zeros(16000),16000,format='WAV',subtype='PCM_16')
        wav=data.getvalue()
        with patch.dict('sys.modules',{'webrtcvad':None}):
            result=decode_live_neck_audio(wav,.1)
        self.assertEqual(result[0]['t'],.1)
        self.assertFalse(result[0]['vad_available'])
        self.assertFalse(result[0]['speech_active'])
        for audio,offset in [(b'',0),(b'not wav',0),(wav,float('nan')),(wav,2)]:
            with self.assertRaises((ValueError,RuntimeError)):decode_live_neck_audio(audio,offset)
        long=io.BytesIO();sf.write(long,np.zeros(16000*26),16000,format='WAV',subtype='PCM_16')
        with self.assertRaises(ValueError):decode_live_neck_audio(long.getvalue())

    def test_empty_vad_module_is_unavailable_without_failing_valid_audio(self):
        import io
        import soundfile as sf
        data=io.BytesIO();sf.write(data,np.zeros(16000),16000,format='WAV',subtype='PCM_16')
        with patch.dict('sys.modules',{'webrtcvad':SimpleNamespace()}):
            result=decode_live_neck_audio(data.getvalue(),0)
        self.assertFalse(result[0]['vad_available'])
        self.assertFalse(result[0]['speech_active'])
        self.assertEqual(result[0]['speech_method'],'unavailable')

if __name__ == '__main__':
    unittest.main()
