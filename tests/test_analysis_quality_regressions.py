"""Regression checks for false certainty and missed events; no provider calls."""
import ast
import asyncio
import os
from types import SimpleNamespace
from fastapi import HTTPException
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import parselmouth
from parselmouth.praat import call
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'root/voice_app'))
from neck_v1_cv import select_frames
from neck_v1_scoring import NeckEffortConfig, aggregate, DIMENSIONS
from test_neck_effort_engine import prediction, context, row
from neck_effort_engine import evidence_context

# Isolate the production acoustic scorer from optional training dependencies.
source = ROOT / 'root/voice_app/grbas_model.py'
function = next(node for node in ast.parse(source.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == 'calculate_grbas')
namespace = dict(np=np, parselmouth=parselmouth, call=call,
                 _clamp=lambda value, lo, hi: max(lo, min(hi, value)))
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)


class QualityRegressions(unittest.TestCase):
    def score(self, audio):
        with tempfile.NamedTemporaryFile(suffix='.wav') as file:
            sf.write(file.name, audio, 16000)
            return namespace['calculate_grbas'](file.name)

    def test_silence_and_unvoiced_noise_do_not_receive_grades(self):
        for audio in (np.zeros(48000), np.random.default_rng(1).normal(0, .1, 48000)):
            with self.subTest(signal=float(np.std(audio))), self.assertRaises(ValueError):
                self.score(audio)

    def test_pure_high_pitch_alone_does_not_raise_strain(self):
        t = np.arange(48000) / 16000
        for frequency in (150, 350):
            scores = self.score(.2 * np.sin(2 * np.pi * frequency * t))
            self.assertEqual(scores['S'], 0)

    def test_severe_shimmer_branch_is_reachable(self):
        original = namespace['call']
        def measured(*args):
            if args[1] == 'Get jitter (local)': return .001
            if args[1] == 'Get shimmer (local)': return .09
            return call(*args)
        namespace['call'] = measured
        try:
            scores = self.score(.2 * np.sin(2 * np.pi * 150 * np.arange(48000) / 16000))
            self.assertEqual(scores['R'], 2)
        finally:
            namespace['call'] = original

    def test_breathiness_noise_signal_changes_b_instead_of_staying_zero(self):
        t = np.arange(48000) / 16000
        tone = .2 * np.sin(2*np.pi*150*t) + .07 * np.sin(2*np.pi*300*t)
        noise = np.random.default_rng(7).normal(0, 1, len(t))
        quiet = self.score(tone + .002 * noise)
        noisy = self.score(tone + .04 * noise)
        self.assertGreater(noisy['B'], quiet['B'])

    def test_inconsistent_model_scores_are_not_silently_downgraded(self):
        output = prediction()
        output.dimensions.jaw_compensation.continuous_score = 0.0
        result = aggregate(output, context())
        self.assertIsNone(result['dimensions']['jaw_compensation']['score'])
        self.assertIn('Inconsistent', result['diagnostics']['dimensions']['jaw_compensation']['rejection_reason'])

    def test_voice_route_returns_422_for_unusable_audio(self):
        tree = ast.parse((ROOT / 'root/voice_app/app.py').read_text())
        function = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == 'analyze')
        function.decorator_list = []
        function.returns = None
        function.args.defaults = []
        for argument in function.args.args: argument.annotation = None
        async def read(): return b'invalid-audio'
        async def run_inline(fn, *args): return fn(*args)
        def predict(features): raise ValueError('אין די קול מחזורי')
        env = dict(tempfile=tempfile, os=os, HTTPException=HTTPException,
                   _ensure_wav=lambda path, data: path, run_in_threadpool=run_inline,
                   extract_features=lambda path: {}, classifier=SimpleNamespace(predict=predict))
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'app.py', 'exec'), env)
        with self.assertRaises(HTTPException) as error:
            asyncio.run(env['analyze'](SimpleNamespace(read=read), None, None, None))
        self.assertEqual(error.exception.status_code, 422)

    def test_default_sampling_includes_all_distinct_event_families(self):
        rows = [row(i / 5) for i in range(100)]
        indices = [7, 11, 33, 47, 73, 87]
        keys = ['anterior_motion', 'lateral_motion', 'jaw_opening', 'head_motion',
                'shoulder_motion', 'shoulders_head_relative_motion']
        for key, index in zip(keys, indices): rows[index]['features'][key] = 100
        selected = select_frames(rows, NeckEffortConfig().max_evidence_frames)
        self.assertTrue(set(indices).issubset(selected))
        self.assertIn(0, selected)
        self.assertIn(99, selected)

    def test_zero_cannot_be_based_only_on_first_two_adjacent_frames(self):
        rows = [row(i / 5) for i in range(30)]
        ctx = evidence_context(rows, [0, 1, 15, 29], NeckEffortConfig())
        result = aggregate(prediction((0, 0, 0, 0, 0)), ctx)
        self.assertTrue(all(result['dimensions'][key]['score'] is None for key in DIMENSIONS))
        output = prediction((0, 0, 0, 0, 0))
        for key in DIMENSIONS: getattr(output.dimensions, key).frame_indices = [0, 15, 29]
        result = aggregate(output, ctx)
        self.assertTrue(all(result['dimensions'][key]['score'] == 0 for key in DIMENSIONS))


if __name__ == '__main__': unittest.main()
