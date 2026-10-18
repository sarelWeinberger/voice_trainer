# Paired compression experiment on the supplied recording

Source: `WIN_20261008_18_21_53_Pro.mp4`, 1280×720, approximately 20 fps and 8.4 seconds. Derived results: [user_video_results.json](user_video_results.json). Original video was read locally; no OpenAI calls or tokens were used. No extracted images are stored with these results.

## Method

The same 25 frames from seconds 0–4.8 were processed at 5 fps, in six conditions, twice (300 CV frame evaluations). Conditions used maximum image edges of 480 and 960 pixels, with PNG references, JPEG 80 followed by JPEG 88, or JPEG 95 followed by JPEG 95. CV runs before the second encoding, matching the application. Each run starts with a fresh tracker; condition order reverses on the second repetition.

OpenCV resizing and JPEG encoding approximate the browser pipeline; this is not an end-to-end browser test. The source recording is already compressed. PNG is a reference for additional encoding changes, not ground truth for anatomy or muscle activity. Frames were inspected for framing; speech timing was not validated.

## Results

All conditions detected a face in all frames. Warm CV times exclude each run's first frame and do not include network or OpenAI latency.

| Condition | Actual dimensions | Mean upload/image | Warm CV/frame | Anterior motion deviation | Lateral motion deviation |
| --- | --- | ---: | ---: | ---: | ---: |
| 480 JPEG 80/88 | 480×270 | 17.2 KiB | 60.1 ms | 0.206286 | 0.121623 |
| 480 JPEG 95/95 | 480×270 | 40.6 KiB | 60.5 ms | 0.130110 | 0.079803 |
| 960 JPEG 80/88 | 960×540 | 51.7 KiB | 139.9 ms | 0.040944 | 0.036745 |
| 960 JPEG 95/95 | 960×540 | 125.7 KiB | 141.2 ms | 0.018063 | 0.024781 |

Deviations are mean absolute differences from the PNG reference **at the same resolution**, in the CV feature's units. Cross-resolution values have different references and should not be treated as physiological accuracy comparisons.

At 480 pixels, JPEG 95 reduced anterior and lateral motion deviations by approximately 37% and 34%, respectively, at about 2.36× upload size. CV time changed by only 0.4 ms/frame in this run. At 960 pixels, the corresponding reductions were approximately 56% and 33%, at 2.43× upload size. The 960-pixel conditions took about 2.3× as long to process as the 480-pixel conditions.

Internal dimension-quality values were approximately 0.52–0.54 at 480 pixels and 0.94–0.95 at 960 pixels. The quality formula itself rewards resolution and apparent face size, so this increase does not establish improved confidence calibration or effort accuracy.

## Negative findings and limitations

- Estimated head pitch was highly compression-sensitive at 480 pixels: mean absolute deviation from PNG was 15.89° for JPEG 80 and 15.19° for JPEG 95. Higher JPEG quality alone did not resolve this. At 960 pixels, deviations were 0.61° and 0.23°. These are image-based pose estimates, not cervical joint angles; investigate pose solver stability and angular conventions before relying on this signal.
- No paired shoulder-motion feature values were available in any condition. This feature requires pose tracking prerequisites; a shoulder quality value alone does not prove shoulder motion was measured. This experiment cannot compare that feature.
- Repeating conditions gives paired observations but this summary does not separately quantify repeat-to-repeat variability or statistical uncertainty. One person and one five-second segment do not establish generalization.
- No OpenAI judgments, final effort scores, clinical labels, sEMG measurements, audio synchronization validation, or browser capture/network tests were included.

## Decision

JPEG 95 at the current 480-pixel size is a reasonable candidate for a controlled browser comparison: it preserved several local CV signals better with little measured CV cost. Do not automatically switch production to 960 pixels based on this single recording, and do not lower rejection thresholds to obtain scores. First compare actual browser JPEG encoding, recording throughput, and pose stability; separately assess whether OpenAI judgments change on identical source frames. Production capture settings were not changed by this experiment.
