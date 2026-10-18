# New recording review — 2026-10-09

Source: `WIN_20261009_16_25_13_Pro.mp4`, duration 20.785 seconds.
Derived measurements and the real provider response are in `video_results.json`.
The source video was not modified. No images or raw audio are retained in this report.

## Method

Processed 104 frames at a target of 5 frames/second, with max edge 480 and JPEG quality 95. Compared the initial 0–5 second reference to subsequent two-second windows. The reference is not assumed to be relaxed. Selected eight frames spanning the recording plus contextual crops using the current production engine. Extracted audio locally for speech-window context; container audio/video alignment was not independently verified. No participant-reported onset or increasing-effort labels were sent to the provider.

The initial sandboxed attempt produced no successful provider response. The subsequent network-enabled attempt completed with a real OpenAI response: 15.232 seconds at the provider, 19,652 input tokens and 872 output tokens (20,524 total). This was one successful inference, not a repeatability experiment or a complete live browser test.

## Findings

The model proposed **zero** for all five dimensions. The scoring engine accepted those proposals, with adjusted confidence approximately 29–33%. No positive proposed score was suppressed by confidence thresholds. Lowering the thresholds therefore cannot change these particular zero proposals into positive scores.

Local geometry still changed. Selected window medians relative to the initial reference:

| Window, seconds | Head pitch delta, degrees | Jaw opening delta, eye-distance ratio | Anterior residual motion delta, eye-distances/second | Lateral residual motion delta, eye-distances/second |
| --- | ---: | ---: | ---: | ---: |
| 5–7 | +0.0760 | +0.0012 | +0.0082 | +0.0179 |
| 11–13 | +1.4753 | +0.0152 | +0.0543 | +0.0576 |
| 15–17 | +2.0976 | +0.0023 | +0.0284 | +0.0198 |
| 17–19 | +2.8522 | -0.0004 | -0.0671 | -0.0237 |
| 19–20.804 | +2.3610 | -0.0002 | +0.0588 | +0.0383 |

The final comparison endpoint estimates sample support and slightly exceeds the source container duration. Full window values and dispersion are in the JSON.

Head pitch shifted gradually in later windows. Jaw opening peaked around 11–13 seconds and returned close to its reference value. Residual neck flow did not increase consistently. Shoulder-to-chin geometry changed, but it cannot isolate shoulder elevation from head/body mechanics; a torso reference was unavailable.

## Interpretation and next comparison

These results do not demonstrate that the participant experienced no effort. They show that the current image-based grader did not identify abnormal external recruitment from its selected evidence. Numeric zero is an assessment of visible behavior, not a measurement of internal muscle tension or subjective vocal effort.

The descriptive comparison detected some movement changes, but there is no established mapping from their magnitudes to effort. Neither the zero proposals nor the head-angle changes establish clinical accuracy. The reported onset and whether the initial five seconds were comfortable ordinary speech need confirmation before comparison with the participant's experience. Independent human review of the same temporal segments would help distinguish visually missed changes from effort without a clearly visible external signal. No thresholds or production scoring rules were changed to fit this recording.
