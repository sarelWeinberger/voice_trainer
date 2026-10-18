# Review of the supplied 17.7-second effort progression recording

Source: `WIN_20261008_23_28_11_Pro.mp4`, 1280×720, approximately 14.99 fps, 17.745 seconds. The participant reports effort beginning at five seconds and worsening thereafter. This timing was **not sent to the provider** and is not a physiological reference label.

## Method

Three contiguous segments (0–5, 5–10, 10–17.745 seconds) and the full recording were analyzed with the current CV processor, prompt, pinned provider model, and unchanged scoring thresholds. Sampling was 5 fps. JPEG quality was 95 for both encodings. Audio was extracted locally to mono 16 kHz WAV for VAD and acoustic measurements; temporary WAVs were deleted. Selected stills, CV summaries and derived audio timing were sent to OpenAI; no raw audio or video was sent. No images are stored with these result artifacts.

Container timing remains an estimate, not independently verified synchronization, and retains the same .8 uncertainty factor as live estimated timing. Six real provider calls completed. Initial sandbox calls failed without provider responses; the completed experiment was rerun outside the sandbox. Successful responses reported **95,380 total tokens** in aggregate, including input and output. This is the six-call investigation total, not the cost of one ordinary app recording. Provider response times ranged from 7.37 to 11.78 seconds.

## Accepted scores

| Run | Posture | Lateral neck | Anterior/submental | Shoulders | Jaw |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0–5 seconds, 480 edge | 0 | 0 | 0 | 0 | 0 |
| 5–10 seconds, 480 edge | 1 | 0 | 0 | 0 | 0 |
| 10–17.745 seconds, 480 edge | 0 | 0 | 0 | 0 | 0 |
| Full recording, 480 edge | 0 | 2 | 2 | unavailable | 0 |
| Full recording, 960 edge, independently selected frames | 0 | 0 | 0 | 0 | 0 |
| Full recording, 960 edge, matched 480 frame indices | 0 | unavailable | 0 | 0 | 0 |

The full 480 run attributed lateral/anterior changes to indices 46 and 48 (approximately 9.2 and 9.6 seconds). It accepted the two scores with evidence-confidence indices approximately .334 each. The matched 960 run described mild lateral prominence (proposed continuous .7) but marked local activity as not separable from head/jaw mechanics, so lateral remained null with its observation retained. Its anterior proposal was zero. All thresholds (.15, .10, .05) applied offline to each fixed response gave the same dimension scores for that response: threshold reduction does not solve this case.

## Measured changes and negative findings

Median normalized jaw opening was .0487, .1824 and .2037 across the three stages. Median estimated image pose pitch was −7.92°, 5.57° and 7.75°. These are observable geometry estimates, not validated cervical angles or effort scores. Flow-magnitude medians decreased across stages; more effort reported by the participant did not correspond to more residual image motion. This is a negative result for using flow magnitude as an effort proxy in this recording.

The last stage had only 2.444 seconds within VAD-active coarse windows, despite the participant describing increasing effort. Its recorded level ranged roughly −42 to −60 dBFS. VAD activity is not a phonation/effort reference and these windows are not exact voiced durations. There is insufficient evidence here to distinguish quieter phonation from noise or a VAD miss without an independent audio review. Gain, distance and acoustic content remain confounds.

Segmented scores did not track the reported progression. Seeing the full sequence yielded positive scores at 480 but these were not robust to resolution/context changes. The original 480/960 comparison used different selected frames; it cannot isolate resolution effects. Matching full-frame indices removed that selection difference but CV metadata, crops and decoded image details still differed, and there is no repeat-to-repeat reliability estimate. Even at temperature zero, these runs do not establish deterministic or calibrated judgments. A score of zero cannot be interpreted as absence of the participant's actual effort.

## Decision

Do not lower production thresholds or switch to 960 based on this single case. The current V1 output is not a reliable time-resolved estimate of the reported effort increase. The next meaningful improvement to evaluate is a separate baseline-to-later observable-change analysis with explicit temporal coverage, supported by reviewed frame annotations and repeatability checks. Keep that output separate from physiological effort. Do not promote the single positive 480 run to clinical ground truth or tune the prompt to the participant's reported onset.

Reproducible harness: `../staged_video_review.py`; results: `results.json`, `whole_480.json`, `whole_960.json`, `whole_960_matched_frames.json`. The imageio-ffmpeg package was installed locally for this research decoder; it was not added to the application's production requirements. A regression test covers conservative container timing. No production capture settings, scoring thresholds, provider prompt or Hugging Face deployment were changed by this investigation.
