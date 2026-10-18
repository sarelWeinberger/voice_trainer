# Motion analysis hardening — October 15, 2026

This change addresses the six issues identified in the code audit. It improves observable image movement measurement and result delivery. It does not validate a muscle-effort score or infer internal laryngeal tension.

| Issue | Implemented | Remaining limit |
| --- | --- | --- |
| Random noise receives a frequency | Periodicity strength, split-half phase/amplitude consistency, minimum numerical amplitude and frequency-band guard checks; unsupported frequencies remain null | Gates are engineering controls. No clinical sensitivity/specificity or statistical-confidence calibration is established |
| Points drift while count stays healthy | Forward/backward patch consistency; periodic real FaceMesh references; head-region verification and replenishment; stale/absent references invalidate tracking; corrections begin new segments | Drift between checks and detector errors remain possible |
| Background may move | Exclude an estimated upper-body region; require distributed background consensus; compensate translation and small rotation; reject background-dominated periodicity | A coherent moving scene can be indistinguishable from camera movement. This is not a semantic person mask or a guarantee of stationary scenery |
| Fixed 2–8Hz range | Separate 2/6/10-second windows with sampling-dependent bands and boundary guards | Roughly 0.3–12Hz is available only with sufficient recording length, frame rate and quality. Below/above that support, no claim is made |
| Server failure hides local movement | Equivalent local JS/worker estimator; display before server completion; retain measurement on offline/timeout/auth failure; fallback if workers are blocked | The result remains in the current page's memory. Offline archival and automatic later synchronization to server history are not implemented |
| Unvalidated effort grades/confidence | Explicit validation metadata in success/error packets; regular-user display identifies experimental grades and hides uncalibrated percentages; administrator diagnostics retain evidence indices | Clinical grade accuracy and probability calibration are **not solved** without independent reference labels |

## Signal estimation

`head-movement-stability-v3` operates on actual observed timestamps, without inventing coordinates inside missing frames. Existing temporal coverage, usable-sample, segment and sampling-density gates remain. A window crossing a tracking reinitialization is unavailable.

Windows cover different bands: 2 seconds for frequencies starting at 1.5Hz, 6 seconds for 0.5–1.5Hz, and 10 seconds for 0.3–0.5Hz. The fast-band upper bound is min(12Hz, 0.4 × observed sampling support). Three observed cycles are required in the published band. Guard frequencies beyond its edges prevent slow off-grid movement being forced onto the fast band's lower edge. A 0.05Hz search step is numerical granularity, not a claim of frequency accuracy.

For each band, constant/linear drift and the first two lower-frequency nuisance harmonics are projected out. This preserves faster oscillation when larger slow posture movement is present. Least-squares sinusoid candidates are checked against residual energy and independently fitted halves. Publication requires explained fraction at least 0.55, half-vector coherence at least 0.8, half-amplitude ratio at least 0.5, and RMS amplitude at least 3e-6 of image width. These are explicit exploratory gates, not calibrated p-values or medical confidence. At the selected frequency, background RMS above half of head RMS marks the window camera-confounded; frequency and head oscillation amplitude are withheld.

`not_established` means periodicity did not pass the gates, not absence of perceived effort. Zero amplitude refers to negligible residual motion in the tested band, not all movement or muscle activity. Unknown values remain null. The UI shows the tested band and approximate frequency. Longer recordings permit slow-band windows; the application does not extend a recording automatically.

## Tracking and performance

Head patches are cross-checked against new FaceMesh references approximately once per second while healthy and requested more often after loss. A missing face or reference older than three seconds invalidates head tracking. Forward/backward patch matching rejects inconsistent correspondences. Spatially separated background points must agree on translation/small rotation; otherwise measurement can abstain.

The matcher uses a coarse grid followed by local refinement to reduce pixel comparisons while preserving matching checks. FaceMesh now omits iris refinement, which the existing neck/facial metrics do not use. Initial live browser tests exposed timing gaps under load; the lighter detector, matcher optimization and correct recording-tab activation improved usability. Slow hardware and repeated occlusion can still prevent adequate sampling. No frame-rate guarantee is made.

## Independent local result delivery

`static/neck-motion-analysis.js` implements the same estimator as `head_motion.py`. Browser workers compute from bounded derived coordinates, not new uploaded images. If worker loading/creation fails, computation falls back to the already-loaded function. Desktop and mobile stop flows preserve their recording session, show the local result while waiting, and keep it when the server request fails. A late response from a different recording cannot overwrite the current recording display.

Server results remain authoritative when received. Local motion can be attached when the server result lacks that field. API credentials, quotas and provider interpretation are not bypassed; a failed provider still produces no group grades. No OpenAI request or additional API tokens were used to test this change.

## Validation performed

- 61 Python neck tests passed, including V1/V2 input compatibility, session enforcement, persistence, provider failure and validation metadata.
- 14 head-motion tests passed. They include random and correlated noise, isolated dropped frames, slow off-grid motion, fast oscillation with larger slow movement, camera dominance, reinitialization boundaries and payload validation.
- The Python/Node parity test passed for signal, noise, missing data, rates and multiple time scales. This brings the Python test total to 76, not 76 independent recordings.
- JavaScript capture, motion, recovery, integrity and renderer tests passed. They cover face-reference expiry/absence, healthy-count drift, spatial background consensus, camera rotation, blocked workers, timeout/offline delivery, pending states and role-based confidence display.
- Controlled experiments used 24 noise traces: twelve seeds each for white noise and an AR(1) process with coefficient 0.65. None published a frequency in the resulting windows. This does not establish a false-positive rate for arbitrary real-world artifacts.
- Known 4.4Hz, 10Hz, 0.5Hz and 0.3Hz inputs were recovered in their appropriate bands. The 4.4Hz input remained detectable with larger 0.5Hz posture motion. Generated image-tracker controls covered still images, head oscillation, slow movement and whole-frame camera motion. Camera controls published no head periodicity in this run. See `experiment_results.json` and `experiments.py`.

## Browser verification and negative results

Actual Chromium playback used `data/WIN_20261011_10_33_38_Pro.mp4`, real in-browser FaceMesh inference, the actual desktop/mobile result functions and the local analysis worker. This closes the earlier gap where recovery references were supplied by offline fixtures. It does **not** test a physical webcam or phone.

A first live mobile run failed all windows because of dropped frames. The strict temporal gates were retained. Testing then used the lighter FaceMesh configuration and optimized matching. The recording tab is activated before each case, matching the user workflow; a hidden video after the previous results screen is an invalid capture setup. All four final capture cases passed. Tracking-loss injection blanks the motion-tracker image briefly; FaceMesh itself reads the original video. Real detector absence is separately covered by the reference-absence tests.

The captured live-detector traces were replayed through the final estimator in a browser with networking actually disabled. Python and browser results agreed, and local motion remained visible when the real stop-request fetch failed offline. Final results:

| Case | Usable measurement windows | Windows passing periodicity gates | Offline result visible |
| --- | ---: | ---: | --- |
| Desktop, original clip | 6/9 | 2 | Yes |
| Desktop, injected tracking loss | 7/9 | 5 | Yes |
| Mobile viewport, original clip | 7/9 | 5 | Yes |
| Mobile viewport, injected tracking loss | 7/9 | 6 | Yes |

Unavailable windows were not filled or changed to zero. A usable window can still lack established periodicity. These counts are from one real clip and controlled perturbations, with no independent clinical labels. They do not establish that the same proportions generalize. `browser_results.json` contains capture/profiling results; `final_replay_results.json` contains final-estimator/offline results; `browser_check.py` and `replay_final.py` reproduce the checks.

## Work that cannot yet be completed

Clinical calibration requires independently rated recordings with positive and low-observation examples, more than one rater, varied cameras/participants and a held-out participant split. The empty `validation_manifest_template.csv` provides fields for these labels without fabricated examples. Self-reported effort must be distinguished from expert observation and physiological measurements. Any muscle-activity claim additionally needs an appropriate external physiological reference and a scientifically justified target; this code does not establish one.

Physical webcam/phone validation, arbitrary scene/camera separation, behavior on very slow devices, calibration of grades/confidence and robustness to all possible noise remain open. Provider service availability and end-to-end latency cannot be guaranteed by local tracking fixes. Existing approximate live audio/video timing was not changed by this work.

The changes are local to this workspace and have not been deployed to Hugging Face. Refresh the browser and make a new recording to use them. Existing saved reports cannot be recomputed without discarded dense traces.
