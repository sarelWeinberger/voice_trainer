# Offline compression and resolution experiment — 2026-10-08

## Method

Source: public Mixkit asset `17679-720.mp4`, downloaded from
https://assets.mixkit.co/videos/17679/17679-720.mp4 . Decoded source dimensions 1280×720.
The SHA-256 and FPS are recorded in each JSON result. This is stock footage, not a patient recording or a physiological validation dataset.

Two segments from the same video: 0–5 seconds (head facing down toward laptop) and 12–17 seconds (head raised, phone conversation). Each segment uses the same 25 decoded frames at approximately 5fps for every condition. Six conditions per segment: four JPEG settings (480/960 maximum edge × first JPEG quality 80/95), plus lossless PNG references at both resolutions. Each condition ran twice, with condition order reversed in the second pass: 600 processed frames total.

The current-like condition uses quality 80 before local CV and quality 88 for the later model image. The high-quality condition uses 95 at both stages. This matches the application's stage order: its second JPEG encoding occurs **after** CV, so it cannot affect the already computed CV measurements. OpenCV JPEG quality and linear scaling approximate browser JPEG/canvas processing; they are not byte-equivalent to it. The source MP4 is already compressed, so PNG references do not recover lost source information.

Evidence indices are fixed using the 960 PNG reference. CV feature perturbations are paired to the PNG reference **at the same resolution**, rather than treating differences across image scales as compression error. Feature comparisons include frames with face quality >=.25 in both conditions. Missing detection counts are reported separately; a smaller feature difference alone does not capture missing frames. No OpenAI calls, acoustic analysis, effort grades or raw patient images were involved. The experiment persists derived JSON only.

## Results

The phone segment gave these measurements. Timings exclude the first frame of each run and depend on local CPU load. Image size refers to the first encoded upload, in KiB.

| Condition | Face detected | Mean CV/frame | Upload/image | Anterior motion MAD vs same-size PNG | Lateral motion MAD vs same-size PNG |
| --- | ---: | ---: | ---: | ---: | ---: |
| 480 JPEG80/88 | 100% | .0651s | 20.1KiB | .074515 | .039720 |
| 480 JPEG95/95 | 100% | .0631s | 41.6KiB | .039586 | .017783 |
| 960 JPEG80/88 | 100% | .2084s | 52.1KiB | .079619 | .056932 |
| 960 JPEG95/95 | 100% | .1636s | 113.0KiB | .023044 | .011806 |

MAD is mean absolute deviation in arbitrary normalized motion-feature units; it is not clinical error. At 480, quality 95 reduced anterior/lateral perturbation relative to PNG by about 47%/55%, respectively, while roughly doubling upload size. The 960/95 shoulder feature was **farther** from its PNG reference than 960/80 (.266029 vs .171337); higher quality did not monotonically improve every derived feature.

In the head-down segment, the fraction of frames with any detected face was 72% at 480/80, 92% at 480/95, and 100% for 480 PNG. At 960, both JPEG conditions detected a face in 88% of frames, versus 100% for PNG. In the phone segment, all conditions detected faces in every frame. These are detection-presence figures, not calibrated landmark accuracy.

## Negative results and decision

Anterior/lateral region quality was zero in the head-down segment for all conditions. In the phone segment, neck dimension-quality indices were about .075–.112, still below the application's .25 acceptance threshold, including PNG references. Compression/resolution changes therefore did not make this stock clip sufficient for accepted neck-region scores. The quality formula itself depends on image scale and resolution, so increases in that index cannot establish clinical accuracy.

960 substantially increased CV time and bytes; one condition's mean exceeded the .2s budget associated with 5fps. Hardware/load and detection behavior affect timing; this offline experiment does not measure browser/network deadlines. Higher-resolution defaults are not justified by this clip alone.

Next candidate: test 480 JPEG95/95 on a representative high-quality original recording with visible face, neck and shoulders. Run matched comparisons on exactly the same frames and audio, first verifying the real browser encoding, then assessing the provider's interpretations against observable evidence. More returned scores or confidence alone do not count as an improvement. Production defaults remain unchanged.

## Reproduce

From the repository root, with a local source video:

```bash
root/voice_app/venv/bin/python root/voice_app/research/compression_experiment.py /path/to/source.mp4 --seconds 5 --repeats 2 --output /tmp/compression-results.json
```

For the second tested segment add `--start 12`. Results: `results.json` and `phone_segment_results.json`. The experiment ran successfully on both real-media segments; the script also passed Python compilation.
