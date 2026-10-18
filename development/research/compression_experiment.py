"""Offline paired compression experiment. No OpenAI, audio, scores or persisted images.

OpenCV JPEG quality is a controlled approximation of browser toBlob quality.
The app computes CV BEFORE its second JPEG encoding; both stages are measured here.
PNG references quantify perturbation at each resolution, not physiological truth.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neck_v1_cv import CVProcessor, select_frames
from neck_effort_engine import evidence_context
from neck_v1_scoring import NeckEffortConfig

FEATURES = ('jaw_opening', 'head_roll_deg', 'head_pitch_deg', 'head_motion',
            'anterior_motion', 'lateral_motion', 'shoulder_motion', 'shoulders_head_relative_motion')


def encode(image, quality):
    extension = '.png' if quality is None else '.jpg'
    settings = [] if quality is None else [cv2.IMWRITE_JPEG_QUALITY, quality]
    ok, encoded = cv2.imencode(extension, image, settings)
    if not ok:
        raise ValueError('Image encoding failed')
    return encoded, cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def psnr(a, b):
    mse = float(np.mean((a.astype(float)-b.astype(float))**2))
    return 10*math.log10(255**2/mse) if mse else None  # None means identical pixels.


def load_frames(path, start, seconds):
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not math.isfinite(fps) or fps <= 0:
            raise ValueError('Source video unavailable or invalid FPS')
        frames = []
        for n in range(round(seconds*5)):
            index = round((start+n/5)*fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = cap.read()
            if not ok:
                raise ValueError('Source video shorter than requested segment')
            frames.append((index/fps-start, image))
        return frames, fps
    finally:
        cap.release()


def run(path, start, seconds, repeats):
    frames, fps = load_frames(path, start, seconds)
    specs = {f'{edge}_{name}': (edge, q1, q2) for edge in (480, 960)
             for name, q1, q2 in [('png_reference', None, None),
                                   ('jpeg80_88', 80, 88), ('jpeg95_95', 95, 95)]}
    samples = {key: [] for key in specs}
    for repeat in range(repeats):
        order = list(specs) if repeat % 2 == 0 else list(reversed(specs))
        for name in order:
            edge, first_quality, second_quality = specs[name]
            tracker = CVProcessor()
            rows, first_sizes, final_sizes, first_psnr, final_psnr = [], [], [], [], []
            elapsed, encode_elapsed, sizes = [], [], []
            try:
                for timestamp, frame in frames:
                    h, w = frame.shape[:2]
                    scale = min(1, edge/max(h, w))
                    source = cv2.resize(frame, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_LINEAR)
                    encode_start = time.perf_counter()
                    encoded, cv_input = encode(source, first_quality)
                    encode_elapsed.append(time.perf_counter()-encode_start)
                    before = time.perf_counter()
                    rows.append(tracker.process(cv_input, timestamp))
                    elapsed.append(time.perf_counter()-before)
                    final, model_input = encode(cv_input, second_quality)
                    first_sizes.append(len(encoded)); final_sizes.append(len(final))
                    first_psnr.append(psnr(source, cv_input)); final_psnr.append(psnr(source, model_input))
                    sizes.append(list(source.shape[1::-1]))
            finally:
                tracker.close()
            samples[name].append(dict(rows=rows, cv_seconds=elapsed, encode_seconds=encode_elapsed,
                upload_bytes=first_sizes, model_image_bytes=final_sizes, first_psnr=first_psnr,
                final_psnr=final_psnr, dimensions=sizes[0]))
            print(f'{name}: repeat {repeat+1}/{repeats}, {len(rows)} frames, '
                  f'warm mean CV {np.mean(elapsed[1:]):.4f}s', flush=True)

    # Fix evidence indices to the same reference sequence for every comparison.
    indices = select_frames(samples['960_png_reference'][0]['rows'], 8)
    conditions = {}
    for name, runs in samples.items():
        def mean_field(key, warm=False):
            values = [v for r in runs for v in r[key][1 if warm else 0:] if v is not None]
            return round(float(np.mean(values)), 6) if values else None
        quality = [evidence_context(r['rows'], indices, NeckEffortConfig())['dimension_quality'] for r in runs]
        deviations = {}
        ref = samples[f'{specs[name][0]}_png_reference']
        for feature in FEATURES:
            differences = [abs(a['features'][feature]-b['features'][feature])
                for run, reference in zip(runs, ref) for a, b in zip(run['rows'], reference['rows'])
                if feature in a['features'] and feature in b['features']
                and min(a['visibility'].get('face',0),b['visibility'].get('face',0))>=.25]
            deviations[feature] = dict(paired_frames=len(differences),
                mean_absolute_difference=round(float(np.mean(differences)),6) if differences else None)
        conditions[name] = dict(image_dimensions=runs[0]['dimensions'],
            face_detected_fraction=round(float(np.mean([r['visibility'].get('face',0)>0 for run in runs for r in run['rows']])),4),
            dimension_quality={k:round(float(np.mean([q[k] for q in quality])),4) for k in quality[0]},
            warm_mean_cv_seconds=mean_field('cv_seconds',True),
            mean_cv_seconds=mean_field('cv_seconds'), mean_first_encode_seconds=mean_field('encode_seconds'),
            mean_upload_bytes=mean_field('upload_bytes'), mean_model_image_bytes=mean_field('model_image_bytes'),
            mean_cv_input_psnr_db=mean_field('first_psnr'), mean_model_input_psnr_db=mean_field('final_psnr'),
            cv_feature_difference_from_same_resolution_png=deviations)
    return dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source_file=path.name,
        source_fps=fps, source_dimensions=list(frames[0][1].shape[1::-1]),
        segment_start=start, segment_seconds=seconds, frame_count=len(frames), repeats=repeats,
        selected_indices=indices, conditions=conditions,
        limitations=['No effort scores, OpenAI judgments or physiological ground truth',
            'OpenCV JPEG/canvas scaling approximate, not byte-equivalent to browser encoding',
            'Input video already compressed; PNG does not undo source compression',
            'PNG is a within-resolution reference, not evidence of medical accuracy',
            'Tracking-quality rules explicitly depend on scale and resolution',
            'Timings depend on hardware/load; cold first frames excluded only from warm mean'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--seconds', type=float, default=5)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not (math.isfinite(args.start) and args.start>=0 and 1<=args.seconds<=20 and 1<=args.repeats<=3):
        parser.error('Use finite nonnegative start, 1–20 seconds and 1–3 repeats')
    result = run(args.video, args.start, args.seconds, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(f'Derived results saved to {args.output}', flush=True)
