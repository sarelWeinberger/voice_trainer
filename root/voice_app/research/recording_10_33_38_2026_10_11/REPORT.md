# New clip and dedicated-network feasibility — 2026-10-11

Source: `WIN_20261011_10_33_38_Pro.mp4`, duration 9.137 s, container average frame rate 28.126 fps. The participant clarified the missed target as head/neck shaking throughout the recording and perceived neck tension. Oscillatory image motion and muscle tension remain separate targets. Do not call a prediction a confirmed false negative without an operational target and reference annotation.

## Experiments performed

Processed 46 frames at target 5 fps, JPEG quality 95. Initial reference was 0–2 s under the existing short-recording rule, not a verified relaxed baseline. Two successful real OpenAI requests were made without participant labels:

| Setting | Evidence indices | Accepted regional grades | Provider latency | Input + output tokens |
| --- | --- | --- | ---: | ---: |
| Max edge 480 | 0, 2, 9, 18, 27, 36, 44, 45 | All five 0 | 7.043 s | 15,674 + 1,058 |
| Max edge 960, matched indices | Same | All five 0 | 8.121 s | 19,800 + 839 |

Derived results: `video_results.json` and `matched_960_results.json`. Total usage: 37,371 tokens. Confidence after current engineering rules was about 39–46%, not a calibrated medical accuracy probability. No score was suppressed by thresholds. Thus changing the thresholds cannot turn these particular model proposals positive. Higher resolution alone did not rescue this clip. This is not a pure compression ablation: CV features and derived crop geometry were recomputed at each resolution. One model response per condition does not establish repeatability.

Local geometry changed modestly: the last 8–9.195 s sampled-support window had head pitch median +2.4546 degrees and head roll +2.1009 degrees relative to the initial reference. Anterior/lateral residual flow deltas were +0.0949/+0.0859 normalized eye-distances/s; earlier windows were variable. These are descriptive differences, not effort labels. The final support endpoint slightly exceeds container duration. Audio/video alignment is inferred from the container, not independently measured. Local processing per sample was 0.0681 s at edge 480 and 0.1635 s at 960; this is not measured live-browser throughput.

A local six-frame preview was inspected. Clothing covers the lower neck/torso and illumination differs across sides. The preview is insufficient to independently establish the user's reported target or its onset. No diagnosis was made, and no source frames/audio were retained in these report artifacts.

## Dedicated network: a testable next experiment

### Update after participant clarification: oscillatory head motion

Ran `research/head_oscillation_review.py` locally on all 257 source frames at container mean 28.126 fps, resized to width 640. No provider request was needed. The script compares facial-landmark center displacement with forward/backward-checked image-point optical tracking, both relative to border-point background translation. All frame pairs met the script's tracking support requirements. Derived traces are in `dense_head_motion.json`; plots in `head_oscillation.png`. No source images are retained in these artifacts.

After linear detrending, an exploratory 2–10 Hz band had peak frequency about 4.395 Hz in both head traces. Band RMS displacement was 1.676 pixels for landmarks and 1.775 pixels for image tracking, versus 0.075 pixels for background translation at image width 640. The two head measures share face-anchor locations and are corroborating estimates, not fully independent measurements. Filter boundary effects, face articulation, tracking error, compression, camera rotation/rolling shutter and container timing remain potential confounds. The band and pixel amplitude are descriptive engineering quantities, not calibrated clinical tremor thresholds.

This supports the presence of an oscillatory image-motion component in the head relative to the background. It does not separately prove neck-local oscillation or physiological muscle tension. The reported subjective tension is recorded separately in the annotation template. Source mean-rate timing is approximate and no hardware synchronization was verified.

Sampling at 5 fps has a Nyquist limit of 2.5 Hz and cannot faithfully reconstruct a 4.395 Hz component. Median two-second posture values further discard alternating movements. Increasing still-image resolution therefore addresses a different problem from preserving this temporal signal. The current rubric also lacks an explicit oscillatory-motion output. These are concrete reasons to investigate a separate head/neck movement-stability measure instead of simply relabelling generic effort grades.

A next implementation should preserve a dense motion stream with timestamps and background controls, optionally computed locally in the browser to avoid multiplying image uploads. Audit it against independently annotated shaking/stable clips and report movement stability separately from effort. Do not increase all server image uploads automatically without measuring CPU throughput and request latency. A temporal neural network is an additional comparator after appropriate labels exist; it cannot fix a temporal signal lost before inference. No production capture rate or scores were changed during this audit.

A video network could process temporally contiguous, stabilized neck/chin/shoulder crops rather than rely on eight selected stills. That is a candidate improvement, not evidence that a new network will recover subjective effort or hidden muscle tension.

1. Define and annotate separate outputs: visible posture/movement, visible neck-appearance change, and jaw behavior, each ordinal 0–3 with an explicit unassessable state. Record subjective vocal effort separately. Preserve lighting, swallowing and head/jaw confounds. Do not label every reported difficult phonation as visually abnormal.
2. Obtain independent human annotations with times and descriptions, preferably two reviewers and adjudication of disagreement. Include ordinary speech, visible changes, reported effort without visible changes, and movement without reported effort. Current clips alone are not a validated training population. `annotation_template.json` provides empty fields; no labels were fabricated.
3. First compare a simple supervised baseline on current local features against the existing OpenAI grader. Then compare a pretrained image/video encoder with a temporal head (for example a small temporal convolutional network) on chronological neck crops. Keep head/jaw features as context, and separate optional audio features from the visual task. A generic video encoder's action-recognition weights are initialization, not neck-effort weights.
4. Hold out people and recording sessions, keeping every window from a person on the same side of the split. Evaluate nonzero-grade recall, false positives in confounded ordinary movements, ordinal error, abstention/coverage and calibration. Compare RGB-only, local-feature-only, audio-only and multimodal versions. Measure speed and memory on the deployment CPU before integration.
5. Integrate only after the model improves on independent held-out labels. For claims about muscle activity, collect an appropriately synchronized physiological reference such as sEMG; do not equate sEMG, subjective effort, external appearance or pathology.

No neural model was trained or substituted into production. There are no independent labels for this clip and no trained task-specific weights to validate. No app thresholds or production resolution were changed to fit this example.

## Sources inspected

- [Torchvision R3D-18 documentation](https://docs.pytorch.org/vision/0.26/models/generated/torchvision.models.video.r3d_18.html): provides a video-network implementation and Kinetics-400 action-recognition weights. It is a possible technical starting point, not a validated neck-effort estimator.
- [University of Missouri laryngeal sEMG dataset](https://vigir.missouri.edu/Research/sEMG_dataset.html): describes submental/infrahyoid sEMG and simultaneous acoustics from 61 female participants, with a data-request agreement. The page does not establish paired ordinary RGB videos with the intended visible-effort annotations. It cannot be treated as a ready RGB training set. No request was submitted or external messages sent.

This was a focused feasibility search, not an exhaustive dataset review. Identifying a suitable labelled RGB dataset remains open. The participant's specification of what was missed is required to decide which labels and comparisons should be added next.
