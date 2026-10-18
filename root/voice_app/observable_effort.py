"""Research baseline: measured correlates, never inferred muscle activation."""
import math
import io
import numpy as np
import soundfile as sf
from neck_effort_engine import NeckEffortEngine

VERSION = 'observable-correlates-v1'
UNAVAILABLE = {
    'visual_neck_effort': 'No validated RGB-to-muscle-effort mapping',
    'extralaryngeal_activity': 'Requires physiological reference such as sEMG',
    'acoustic_vocal_effort': 'No calibrated effort model or speaker reference',
    'respiratory_phonatory_effort': 'No airflow or pressure measurement',
    'fused_effort': 'No paired validated effort labels',
    'clinical_diagnosis': 'Outside this construct',
}


# Compatibility name for existing neck-only callers.
ObservableEffortEngine = NeckEffortEngine


def acoustic_timeline(samples, sample_rate, window_seconds=1.0):
    """Uncalibrated dBFS and spectral balance; no strain/effort score."""
    x = np.asarray(samples, dtype=float)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.ndim != 1 or sample_rate <= 0 or not np.isfinite(x).all():
        raise ValueError('Invalid audio samples or sample rate')
    width = max(1, int(sample_rate * window_seconds))
    rows = []
    for begin in range(0, len(x), width):
        chunk = x[begin:begin + width]
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        power = abs(np.fft.rfft(chunk * np.hanning(len(chunk)))) ** 2
        freq = np.fft.rfftfreq(len(chunk), 1 / sample_rate)
        total = float(power.sum())
        clipped = float(np.mean(abs(chunk) >= .999))
        rows.append({'t': begin / sample_rate, 'duration': len(chunk) / sample_rate,
                     'rms_dbfs': round(20 * math.log10(max(rms, 1e-12)), 3),
                     'high_frequency_energy_ratio': float(power[freq >= 1000].sum() / total) if total > 1e-15 else None,
                     'clipped_fraction': clipped,
                     'usable': rms > 1e-5 and clipped < .01,
                     'limitations': ['microphone_gain_distance_noise_and_phonetic_content_confounded'],
                     'speech_active': rms > 1e-4 and clipped < .01,
                     'speech_method': 'energy_activity_only_not_verified_speech',
                     'effort': None})
    return rows


def neck_audio_timing(samples, sample_rate, offset_seconds=0.0):
    """Neck-only timing with WebRTC VAD; raw audio is never sent to OpenAI."""
    from scipy.signal import resample_poly
    x = np.asarray(samples, dtype=float)
    if x.ndim == 2:
        x = x.mean(axis=1)
    rows = acoustic_timeline(x, sample_rate)
    try:
        import webrtcvad
        vad = webrtcvad.Vad(2)
    except ImportError:
        vad = None
    divisor = math.gcd(int(sample_rate), 16000)
    normalized = resample_poly(x, 16000 // divisor, int(sample_rate) // divisor)
    pcm = (np.clip(normalized, -1, 1) * 32767).astype('<i2')
    for row in rows:
        begin = round(row['t'] * 16000)
        end = min(len(pcm), round((row['t'] + row['duration']) * 16000))
        votes = []
        if vad is not None:
            for index in range(begin, end - 479, 480):
                votes.append(vad.is_speech(pcm[index:index+480].tobytes(), 16000))
        row['speech_active'] = bool(votes and np.mean(votes) >= .2 and row['usable'])
        row['speech_method'] = 'webrtcvad_30ms_mode2' if vad else 'unavailable'
        row['vad_available'] = vad is not None
        row['t'] += offset_seconds
    return rows


def decode_live_neck_audio(data, offset_seconds=0.0):
    """Bounded in-memory WAV decoding for provisional live speech context."""
    if not data or len(data)>5*1024*1024 or not math.isfinite(offset_seconds) or abs(offset_seconds)>1:
        raise ValueError('Invalid live audio size or offset')
    with sf.SoundFile(io.BytesIO(data)) as file:
        if (file.format!='WAV' or not 8000<=file.samplerate<=96000 or
                not 1<=file.channels<=2 or not 0<file.frames<=file.samplerate*25):
            raise ValueError('Invalid live WAV recording')
        samples=file.read(dtype='float32')
        if not np.isfinite(samples).all():
            raise ValueError('Invalid audio samples')
        return neck_audio_timing(samples,file.samplerate,offset_seconds)


def analyze_observable_recording(audio_path, visual=None, synchronized=False):
    samples, sr = sf.read(audio_path)
    return {'model_version': VERSION, 'construct': 'Observable Neck/Vocal Effort',
            'status': 'research_baseline_unvalidated', 'audio_timeline': acoustic_timeline(samples, sr),
            'visual_timeline': visual or [], 'synchronization': 'shared_media_clock' if synchronized else 'unverified_do_not_fuse',
            'outputs': {key: None for key in UNAVAILABLE}, 'unavailable': UNAVAILABLE,
            'uncertainty': {'calibrated_effort_interval': None, 'reason': 'No held-out physiological validation'}}


def analyze_video_audio(video_path, audio_path=None, audio_offset_seconds=0.0, max_seconds=60):
    """Existing upload flow, now using the same Neck Analysis V1 engine."""
    import cv2
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    engine = None
    try:
        if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
            raise ValueError('Video cannot be decoded or has invalid frame rate')
        engine = NeckEffortEngine()
        engine.start_recording_session()
        frame_index = 0
        stride = max(1, round(fps / 5))
        while frame_index / fps < max_seconds:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                engine.process_frame(frame, timestamp=frame_index/fps)
            frame_index += 1
        audio = None
        if audio_path is not None:
            with sf.SoundFile(audio_path) as file:
                sr=file.samplerate
                samples=file.read(frames=int(sr*(max_seconds+abs(audio_offset_seconds))))
            audio = neck_audio_timing(samples, sr, audio_offset_seconds)
        result = engine.finalize(audio=audio, timing='caller_declared_offset' if audio is not None else 'unverified')
        result['synchronization'] = dict(kind='caller_declared_offset', audio_offset_seconds=audio_offset_seconds,
            limitations='Constant-frame-rate timebase; offset is declared, not verified; device drift is not corrected')
        result['visual_duration_analyzed'] = frame_index/fps
        result['truncated'] = frame_index/fps >= max_seconds
        return result
    finally:
        capture.release()
        if engine is not None:
            engine.close()
