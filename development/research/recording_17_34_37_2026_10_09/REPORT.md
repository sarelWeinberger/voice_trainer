# Recording review — 2026-10-09, 17:34:37

Source: `WIN_20261009_17_34_37_Pro.mp4`, 21.217 seconds.
Participant reported an increase in perceived effort around 7–8 seconds. This label was recorded for comparison only and was not sent to OpenAI; it is not physiological ground truth.

## Procedure and execution

Current engine, 106 samples at target 5 fps, max edge 480, JPEG quality 95. Initial 0–5 seconds used as reference without assuming relaxation. Subsequent two-second windows describe geometry and residual motion. Selected frames: 0, 21, 42, 43, 63, 84, 88, 105; timestamps 0, 4.216, 8.397, 8.604, 12.613, 16.794, 17.589, 21.010 seconds. Eight full frames plus contextual crops were sent; raw video/audio were not sent.

First request exceeded the 15-second provider timeout. Its local results and failure are retained in `video_results.json`. A research-only second attempt allowed a 30-second socket timeout and completed in 9.212 provider seconds; total processing was 21.979 seconds. Production timeout was not changed. Added the research CLI option `--provider-timeout` (default remains 15). Successful response usage: 20,061 input + 977 output = 21,038 tokens. Usage for the timed-out request is unknown; it may still have incurred usage. This is not a two-response repeatability experiment.

## Temporal findings

| Window, seconds | Head pitch median delta, degrees | Anterior residual motion delta | Lateral residual motion delta |
| --- | ---: | ---: | ---: |
| 5–7 | +0.6018 | -0.0358 | -0.0245 |
| 7–9 | +3.2552 | +0.1570 | +0.0858 |
| 9–11 | +9.5074 | +0.0492 | +0.0351 |
| 11–13 | +10.8674 | -0.0733 | -0.0432 |
| 15–17 | +9.9457 | -0.0631 | -0.0314 |
| 19–21 | +10.2709 | +0.0305 | +0.0156 |

Motion deltas are normalized eye-distances/second, not muscle activation. Head-angle change rises around the reported transition and remains elevated later. Residual neck flow rises in 7–9 but is not monotonically increasing afterward. Jaw opening becomes smaller later; its median is not a measure of fixation or vocal effort. Shoulder-to-chin distance also changes, but with no torso reference it cannot isolate shoulder elevation. The last 21.0–21.21 window has only one sample and is deliberately unassessable, not zero.

## Real model and scoring result

| Dimension | Accepted score | Adjusted confidence | Interpretation |
| --- | ---: | ---: | --- |
| Head/neck posture | 2/3 | 39% | Model described sustained head elevation from sample 42 (8.397 s), compared with the beginning. |
| Lateral neck / SCM appearance | unavailable | — | Model proposed mild contour prominence at sample 88 (17.589 s), but marked it inseparable from head mechanics; scorer retained an ambiguous observation and rejected an effort score. |
| Anterior/submental appearance | 0/3 | 35% | No abnormal external change identified; model also noted partial collar occlusion. |
| Shoulders | 1/3 | 31% | Model described elevation with head movement; general body/head movement remains a confound. |
| Jaw compensation | 0/3 | 37% | No abnormal jaw compensation identified. |

Partial aggregate: 0.5667/3, confidence 26.99%, based on four of five dimensions. These engineering confidence values are not calibrated probabilities of medical correctness.

## Limits and conclusion

The current comparison and model detected head/posture change in a window overlapping the reported effort increase. Temporal correspondence does not prove that the change measures vocal effort or muscle tension. Neither the local optical flow nor the model's single recording-level grade demonstrates recovery of a progressively increasing effort signal. Lateral recruitment remains ambiguous, anterior recruitment remains undetected, and shoulders lack a torso reference. No scoring thresholds were lowered or fitted to this recording.

Audio/video share container timing, not independently verified synchronization. This review uses offline OpenCV encoding rather than a live browser capture. There is no blinded human grading, physiological reference, clinical validation, or repeated successful model response. The next useful check is independent review of the beginning, 7–9 seconds, and later segments for visible changes; simply forcing positive local scores would hide the remaining uncertainty.
