"""Render synthetic GRBAS proxy audio from the synthetic manifest.

This pipeline is intended for pretraining, augmentation, and smoke tests.
It does not make publication-grade clinical claims on its own.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import math
import wave
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import librosa
import numpy as np
import soundfile as sf
from scipy import signal


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = BASE_DIR / "synthetic_grbas_manifest.csv"
DEFAULT_AUDIO_DIR = BASE_DIR / "synthetic_audio"
DEFAULT_FEATURES_CSV = BASE_DIR / "synthetic_training_features.csv"
DEFAULT_SAMPLE_RATE = 24000

VOICE_POOLS = {
    "female": [
        "en-US-AvaNeural",
        "en-US-JennyNeural",
        "en-GB-SoniaNeural",
        "en-AU-NatashaNeural",
    ],
    "male": [
        "en-US-AndrewNeural",
        "en-US-BrianNeural",
        "en-GB-RyanNeural",
        "en-AU-WilliamNeural",
    ],
}


def _peak_normalize(y: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak < 1e-8:
        return y.astype(np.float32)
    return (y * (target_peak / peak)).astype(np.float32)


def _speaker_seed(speaker_id: str) -> int:
    total = 0
    for idx, char in enumerate(speaker_id):
        total += (idx + 1) * ord(char)
    return total & 0xFFFFFFFF


def _speaker_voice(row: Dict[str, str]) -> Tuple[str, str, int]:
    gender = row.get("speaker_gender_hint", "female")
    pool = VOICE_POOLS.get(gender, VOICE_POOLS["female"])
    seed = _speaker_seed(row["speaker_id"])
    voice = pool[seed % len(pool)]
    pitch_band = row.get("speaker_pitch_band", "mid")
    if pitch_band == "low":
        pitch_hz = 120 if gender == "male" else 170
    elif pitch_band == "high":
        pitch_hz = 185 if gender == "male" else 250
    else:
        pitch_hz = 145 if gender == "male" else 210
    return gender, voice, pitch_hz


def _edge_rate_and_pitch(row: Dict[str, str]) -> Tuple[str, str]:
    pitch_band = row.get("speaker_pitch_band", "mid")
    if pitch_band == "low":
        pitch = "-4Hz"
    elif pitch_band == "high":
        pitch = "+5Hz"
    else:
        pitch = "+0Hz"

    seed = _speaker_seed(row["speaker_id"])
    rate = [-8, -4, 0, 4, 8][seed % 5]
    return f"{rate:+d}%", pitch


def _base_text(row: Dict[str, str]) -> str:
    if row.get("utterance_kind") == "sustained_vowel":
        return "ah"
    return row["prompt_text"]


def _vowel_formants(gender: str) -> List[Tuple[float, float]]:
    if gender == "male":
        return [(730.0, 90.0), (1090.0, 110.0), (2440.0, 170.0)]
    return [(860.0, 100.0), (1220.0, 120.0), (2810.0, 180.0)]


def _generate_sustained_vowel(duration_sec: float, sample_rate: int, f0_hz: int, gender: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_sec * sample_rate), dtype=np.float64) / sample_rate
    envelope = np.ones_like(t)
    attack = min(int(0.08 * sample_rate), len(t))
    release = min(int(0.12 * sample_rate), len(t))
    if attack > 0:
        envelope[:attack] = np.linspace(0.0, 1.0, attack)
    if release > 0:
        envelope[-release:] *= np.linspace(1.0, 0.0, release)

    waveform = np.zeros_like(t)
    for harmonic in range(1, 16):
        phase = rng.uniform(0.0, 2.0 * math.pi)
        waveform += (1.0 / (harmonic ** 1.25)) * np.sin(2.0 * math.pi * f0_hz * harmonic * t + phase)

    waveform *= envelope

    for center_hz, bandwidth_hz in _vowel_formants(gender):
        q = max(center_hz / max(bandwidth_hz, 1.0), 0.1)
        b, a = signal.iirpeak(center_hz / (sample_rate / 2.0), Q=q)
        waveform = signal.lfilter(b, a, waveform)

    breath = rng.normal(0.0, 0.004, size=len(waveform))
    waveform = waveform + breath * envelope
    return _peak_normalize(waveform)


async def _edge_tts_wav_bytes(text: str, voice: str, rate: str, pitch: str) -> bytes:
    import edge_tts

    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
    audio_buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_buffer.write(chunk["data"])

    y, sr = librosa.load(io.BytesIO(audio_buffer.getvalue()), sr=None, mono=True)
    y = _peak_normalize(y.astype(np.float32))

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        pcm = np.clip(y * 32767.0, -32768, 32767).astype(np.int16)
        handle.writeframes(pcm.tobytes())
    return wav_buffer.getvalue()


def _breath_noise(length: int, strength: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=length)
    b, a = signal.butter(2, [0.08, 0.85], btype="bandpass")
    return signal.lfilter(b, a, noise) * strength


def _apply_spectral_tilt(y: np.ndarray, tilt_shift: float) -> np.ndarray:
    if abs(tilt_shift) < 1e-9:
        return y
    coeff = float(np.clip(tilt_shift * 850.0, -0.92, 0.92))
    if coeff >= 0.0:
        return signal.lfilter([1.0, -coeff], [1.0], y)
    return signal.lfilter([1.0], [1.0, coeff], y)


def _apply_tempo(y: np.ndarray, tempo_delta: float) -> np.ndarray:
    if abs(tempo_delta) < 1e-6:
        return y
    rate = float(np.clip(1.0 + tempo_delta, 0.88, 1.12))
    return librosa.effects.time_stretch(y.astype(np.float32), rate=rate)


def _apply_shimmer(y: np.ndarray, sample_rate: int, strength: float, seed: int) -> np.ndarray:
    if strength <= 0.0:
        return y
    rng = np.random.default_rng(seed)
    control_hz = 12
    control = rng.normal(0.0, strength, size=max(2, int(len(y) / sample_rate * control_hz) + 2))
    envelope = np.interp(
        np.linspace(0.0, len(control) - 1, len(y)),
        np.arange(len(control)),
        control,
    )
    return y * np.clip(1.0 + envelope, 0.15, 2.5)


def _apply_pitch_jitter_vowel(y: np.ndarray, sample_rate: int, strength: float, seed: int) -> np.ndarray:
    if strength <= 0.0:
        return y
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y), dtype=np.float64)
    jitter = rng.normal(0.0, strength * 12.0, size=len(y))
    warped = np.clip(indices + np.cumsum(jitter), 0.0, len(y) - 1.0)
    return np.interp(indices, warped, y).astype(np.float32)


def _apply_proxy_effects(y: np.ndarray, sample_rate: int, spec: Dict[str, float], utterance_kind: str, seed: int) -> np.ndarray:
    out = y.astype(np.float32)
    out = _apply_tempo(out, float(spec.get("tempo_delta", 0.0)))
    out = _apply_shimmer(out, sample_rate, float(spec.get("amplitude_shimmer", 0.0)), seed + 1)

    aspiration = float(spec.get("aspiration_noise", 0.0))
    if aspiration > 0.0:
        out = out + _breath_noise(len(out), aspiration, seed + 2)

    if utterance_kind == "sustained_vowel":
        out = _apply_pitch_jitter_vowel(out, sample_rate, float(spec.get("pitch_jitter", 0.0)), seed + 3)

    out = _apply_spectral_tilt(out, float(spec.get("spectral_tilt_shift", 0.0)))

    drive = float(spec.get("soft_clip_drive", 0.0))
    if drive > 0.0:
        saturation = 1.0 + drive * 8.0
        out = np.tanh(out * saturation) / np.tanh(saturation)

    gain_reduction_db = float(spec.get("gain_reduction_db", 0.0))
    out = out * (10.0 ** (-gain_reduction_db / 20.0))
    return _peak_normalize(out)


def _read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _base_key(row: Dict[str, str]) -> str:
    return f"{row['speaker_id']}__{row['prompt_id']}"


async def _ensure_base_audio(row: Dict[str, str], base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    output_path = base_dir / f"{_base_key(row)}.wav"
    if output_path.exists():
        return output_path

    gender, voice, pitch_hz = _speaker_voice(row)
    seed = _speaker_seed(row["speaker_id"])

    if row.get("utterance_kind") == "sustained_vowel":
        duration = 2.2 if row.get("prompt_id", "").endswith("short") else 3.8
        waveform = _generate_sustained_vowel(duration, DEFAULT_SAMPLE_RATE, pitch_hz, gender, seed)
        sf.write(output_path, waveform, DEFAULT_SAMPLE_RATE)
        return output_path

    rate, pitch = _edge_rate_and_pitch(row)
    wav_bytes = await _edge_tts_wav_bytes(_base_text(row), voice, rate, pitch)
    output_path.write_bytes(wav_bytes)
    return output_path


def _feature_rows_from_rendered(manifest_rows: Iterable[Dict[str, str]], extract_features_fn) -> List[Dict[str, str]]:
    fielded_rows: List[Dict[str, str]] = []
    for row in manifest_rows:
        wav_path = BASE_DIR / row["output_audio_relpath"]
        features = extract_features_fn(str(wav_path))
        flat = {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "source_type": row["source_type"],
            "proxy_grbas_g": row["proxy_grbas_g"],
            "proxy_grbas_r": row["proxy_grbas_r"],
            "proxy_grbas_b": row["proxy_grbas_b"],
            "proxy_grbas_a": row["proxy_grbas_a"],
            "proxy_grbas_s": row["proxy_grbas_s"],
            "audio_path": row["output_audio_relpath"],
        }
        for key, value in features.items():
            if key.startswith("_"):
                continue
            flat[key] = value
        fielded_rows.append(flat)
    return fielded_rows


def _write_features_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


async def render_dataset(manifest_path: Path, audio_dir: Path, limit: int | None, extract_features: bool, features_csv: Path | None) -> Tuple[int, Path | None]:
    rows = _read_manifest(manifest_path)
    if limit is not None:
        rows = rows[:limit]

    base_dir = audio_dir / "_base"
    audio_dir.mkdir(parents=True, exist_ok=True)

    base_keys = {}
    for row in rows:
        base_keys.setdefault(_base_key(row), row)

    for row in base_keys.values():
        await _ensure_base_audio(row, base_dir)

    rendered_rows: List[Dict[str, str]] = []
    for row in rows:
        base_wav = base_dir / f"{_base_key(row)}.wav"
        y, sr = librosa.load(base_wav, sr=None, mono=True)
        spec = json.loads(row["perturbation_spec_json"])
        seed = _speaker_seed(row["sample_id"])
        rendered = _apply_proxy_effects(y, sr, spec, row.get("utterance_kind", "connected_speech"), seed)
        output_path = BASE_DIR / row["output_audio_relpath"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, rendered, sr)
        rendered_rows.append(row)

    features_path = None
    if extract_features:
        from grbas_model import extract_features as extract_features_fn

        features_rows = _feature_rows_from_rendered(rendered_rows, extract_features_fn)
        features_path = features_csv or DEFAULT_FEATURES_CSV
        _write_features_csv(features_path, features_rows)

    return len(rendered_rows), features_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render synthetic GRBAS audio from a manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--extract-features", action="store_true")
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rendered_count, features_path = asyncio.run(
        render_dataset(
            manifest_path=args.manifest,
            audio_dir=args.audio_dir,
            limit=args.limit,
            extract_features=args.extract_features,
            features_csv=args.features_csv,
        )
    )
    print(f"rendered {rendered_count} synthetic wav files into {args.audio_dir}")
    if features_path is not None:
        print(f"wrote feature table to {features_path}")


if __name__ == "__main__":
    main()