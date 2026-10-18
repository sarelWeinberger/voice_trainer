# Tracking recovery: implemented October 12, verified October 15, 2026

## Reproduced defect

The old tracker permanently deleted every patch that failed to match. Once fewer than three head or four background points remained, valid coordinates could never resume. The application also paused FaceMesh throughout recording, preventing a new face reference from arriving.

A controlled static textured-image sequence with one blank frame reproduced permanent failure: 14 usable samples before the interruption, zero usable samples after the original image returned, and zero head/background points remaining. The user's saved live report independently showed approximately 31Hz camera sampling but only 12 usable samples over approximately 0.4 seconds, followed by no usable tracking. That historical report cannot identify the exact image event or failed point type because dense images/traces were not retained.

## Implementation

- Select up to 16 textured, spatially separated patches around the initial upper-face landmark bounds instead of depending on six exact landmark pixels. These measure visible image motion, not isolated neck muscles.
- Extend the integer search radius from four to six pixels at 256px image width. Compare patches after removing uniform brightness offset; reject poor matches and textureless target patches rather than matching blank images.
- Request the existing FaceMesh detector when head patches are insufficient during recording, throttled to one request per 500ms. Healthy tracking does not trigger extra recognition. Apply fresh detector references to reinitialize a lost head track. The same hooks are present in desktop and mobile.
- Reacquire border-background patches when enough textured candidates return and head patches can still be tracked. A genuinely textureless background can still make relative movement unavailable.
- Start a new segment on every reinitialization. The browser sends bounded seven-column rows with segment ID in `head-motion-trace-v2`; the server continues accepting six-column V1 traces. A two-second analysis window containing usable data from multiple segments is unavailable with reason `tracking_reinitialized`. Resetting coordinate origin must not generate a false oscillation.
- Persist counts of missing head/background matches, remaining points and reinitializations. Explain unavailable windows in the existing renderer; keep group grades and provider-error behavior unchanged.

## Verification

Python head-motion tests cover segment-boundary abstention, known frequency with missing frames, stale-format compatibility, payload bounds and diagnostic filtering. API tests exercise V2 reception, persistence of recovery diagnostics, session enforcement and local measurement with a failing mock provider. JavaScript tests cover desktop/mobile camera callbacks, reference-request throttling, head reacquisition, automatic background recovery, brightness change and refusal to track a persistently blank image.

The generated image controls, passed through the actual tracker and Python estimator, produced approximately 0% image-width amplitude for a still image, 0.497% for known head oscillation, 0.002–0.0033% for slow movement, and 0.012–0.027% residual for camera motion. Three of five camera-motion windows were usable; recovery boundaries intentionally excluded the others. Head and background recovery cases returned four measurable windows, all zero amplitude, without treating reset jumps as shaking. See `control_reports.json`, `recovery_traces.json`, and `validate_controls.py`. These bounds are test expectations, not clinical thresholds.

Chromium used local playback of `data/WIN_20261011_10_33_38_Pro.mp4` at desktop and mobile viewport sizes. An injected blank grayscale interval at approximately 0.7–0.85 seconds forced loss of both tracks. The test supplies FaceMesh reference fixtures computed offline from this same clip when the browser requests a fresh reference. It exercises real browser decoding, tracking, recovery requests, payload analysis and rendering. It does not exercise a physical webcam, mobile hardware or live in-browser FaceMesh inference. Separate callback tests confirm that the application's desktop/mobile recording callbacks invoke detection when requested.

| Case | Browser samples | Measured windows | Reinitializations |
| --- | ---: | ---: | ---: |
| Desktop, original clip | 181 | 8/8 | 0 |
| Desktop, injected occlusion | 148 | 4/8 | 2 |
| Mobile viewport, original clip | 185 | 8/8 | 0 |
| Mobile viewport, injected occlusion | 152 | 6/8 | 2 |

After injected loss, both viewports resumed usable tracking beyond three seconds. All browser cases preserved group grades, hid administrator diagnostics for ordinary users, displayed local measurements on simulated provider failure, and produced no JavaScript errors. Detailed results and derived traces are in `browser_results.json` and the four case trace files. No OpenAI request was needed for this change.

## Limits and live verification

This fixes the reproduced permanent-loss mechanism. Repeated occlusion, rapid movement outside the search range, poor lighting, expression, lack of background texture, face-detection failure or CPU contention can still interrupt tracking. Clinical muscle tension cannot be established by this image-motion metric. Border features are geometric background candidates, not a semantic guarantee of stationary scenery.

The latest build was started locally after the backend change. It has not been deployed to Hugging Face. A refreshed browser and a new recording from the user's camera are required to check the original live environment. Historical records cannot be recomputed without discarded dense traces. The new report does not guarantee that every window will be measurable.
