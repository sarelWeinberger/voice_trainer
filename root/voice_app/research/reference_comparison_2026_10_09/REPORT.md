# Initial reference and later observable changes — 2026-10-09

## Implemented

The app now computes `neck-observable-change-v1` locally for every completed recording. It uses the initial five seconds for recordings whose sampled duration is at least ten seconds; shorter recordings use up to two seconds or one third of the sampled duration. This automatic initial reference is **not known to be relaxed**. Selecting a different reference interval is not yet a user-facing option.

Later two-second windows report median values, changes from reference, sample IQR, valid frame counts and sampling coverage. Signals are image-based head pitch/roll, normalized jaw opening, shoulder-to-chin vertical distance, and anterior/lateral optical-flow speed. Shoulder-to-chin distance is affected by head/jaw movement and is not isolated shoulder elevation. Flow is not a measure of static contour prominence, anatomical muscles, force or tension. No new effort score is computed from these differences.

Samples require appropriate region visibility, lighting and camera stability. At least three usable observations and 50% of the target 5fps sampling coverage are required. Missing window measurements or an unavailable reference remain null, including when too few frames arrive. These are engineering quality rules, not calibrated sensitivity or accuracy cutoffs. Sparse gaps remain insufficient data even if the few available samples are clear. Values/dispersion are descriptive, not clinical confidence intervals. Live and container audio timing retains its existing uncertainty; VAD is contextual and does not gate these general observable measurements as speech-specific effort.

Selection now reserves `max(2, limit−2)` temporal anchors (six of the eight normal evidence frames) before selecting extrema. The first and last frames are retained. Extrema cannot consume all later temporal coverage. It remains dependent on CV measurements for the two adaptive slots; comparisons across preprocessing conditions must reuse the recorded `selected_frame_indices`, as supported by the research harness `--indices`. Selected timestamps/indices and selection version are retained in diagnostics.

Prompt `neck-rubric-v1.4` receives the explicit reference/window comparison, states that the initial reference is not relaxed by assumption, and distinguishes visually separable local change from biological independence. Simultaneous jaw/head movement alone does not rule out visually separable tissue change. Ordinary articulation, geometric differences and flow cannot alone justify a positive score. The same model, severity weights, score cutoffs, local-separability gate and provider budget remain in place. Production still makes one provider request per recording, with no automatic repeat requests.

Desktop/mobile show the reference measurements and later deltas separately above the recording-level observational scores. Missing comparisons remain explicit. The local comparison survives provider errors; raw media is still cleared. Saved older results without the new field retain their prior display. No historical observations are fabricated.

## Actual recording and real provider checks

The supplied `WIN_20261008_23_28_11_Pro.mp4` was analyzed as a full sequence at 5fps, 480-pixel maximum edge and JPEG95. No reported effort onset labels were sent to the provider. Reference: 0–5 seconds; 89 sampled frames; evidence indices `[0, 18, 35, 48, 53, 70, 77, 88]`. Later comparisons cover 5–7, 7–9, 9–11, 11–13, 13–15, 15–17 and the final partial window. Window ends estimate sample support up to the last frame plus one capped interval, so the final endpoint is not an exact container duration.

Reference median jaw opening was .0487 eye-distance units. Later jaw-opening changes were +.1338 (11–13s), +.1762 (13–15s) and +.2048 (15–17s). Head pitch differences were +15.728°, +16.2841° and +14.1522°, respectively. These image-based values demonstrate observable differences, not increasing muscular effort. Local motion-speed medians did not increase monotonically; that negative finding is preserved rather than converted into an effort trend.

Two actual provider requests used **identical context, images, prompt, model and temperature**. Both accepted posture category 1, lateral category 2, anterior category 2 and jaw category 2, with shoulders unavailable. The continuous scores also matched (1.1, 1.7, 1.6, 1.8 for those four dimensions). This is consistency on one input pair, not clinical correctness or general repeatability. It does not validate the participant-reported effort progression or establish causal/local muscular activation. No independent human labels or physiological reference are available.

First provider latency was 14.92s; second was 11.55s. Total reported tokens across the two verification requests: 39,315, with cached input on the second request. Local CV processing during live recordings is concurrent with capture; this offline full-video test does not establish a browser end-to-end latency guarantee. The prompt's extra derived context can increase input tokens and response latency.

Raw derived results: [video_results.json](video_results.json). The repeatability check is an offline research option (`--repeats`), not a user-session confidence calibration mechanism. The recorded score thresholds were not lowered and positive scores were not forced.

## Verification

Derived measurement figure (no source images): [observable_changes.png](observable_changes.png). Lines connect window medians; they are not continuous physiological measurements or calibrated confidence intervals. Missing measurements would appear as gaps.

58 neck Python tests passed, including reference/delta computations, short recordings, missing-reference/null behavior, gaps, camera instability, temporal coverage and all existing scoring/API checks. Frontend tests cover reference tables, signed deltas, unavailable comparisons and retained local results on provider failure. Actual Chromium desktop/mobile viewport tests rendered the real derived recording result, confirmed all 42 measurement/window rows, horizontal scrolling, explicit missing shoulder score, and local comparison display on a simulated provider error, without JavaScript errors. See [browser_results.json](browser_results.json).

The local app was restarted on port 5000. No Hugging Face deployment was performed. Professional validation remains incomplete; clinician-reviewed positive and negative recordings, repeatability across people and devices, and physiological references are required before interpreting these observations as measured effort.
