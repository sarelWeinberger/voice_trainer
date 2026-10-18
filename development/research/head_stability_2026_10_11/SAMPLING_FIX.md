# Sampling and missing-frame repair — 2026-10-11

The user's saved report contains 135 samples at median 15.62Hz and one measurable window of nine. V1 stored only a generic rejection reason, not the raw browser trace, so the exact reasons for each historical window cannot be recovered. A defect was identified in the browser: accepting a frame only when elapsed media time exceeds 1/31 seconds halves a 31.25fps source (32ms intervals) to 15.625fps. Acquisition now advances a fixed 30Hz time grid. Regression tests cover 25, 30, 31.25 and 60fps camera clocks.

After repairing acquisition, real Chromium playback still produced isolated missing frames. The V1 continuous-filter algorithm rejected a whole two-second window for any gap above 120ms. Diagnostics on replay showed tracking coverage around 97–100% but gaps of 128–368ms; this was a timing rejection rather than failure to track the head. The first retest still measured only one desktop and two mobile windows. Increasing sample rate alone was insufficient.

## V2 measurement

`head-movement-stability-v2` fits observed coordinates at their actual timestamps. No coordinates are interpolated across missing frames. For each two-second window, least-squares models use a constant, linear drift, nuisance sin/cos components at 0.5, 1 and 1.5Hz, and a candidate oscillatory sin/cos component. Candidate frequencies range from 2Hz to min(8Hz, 0.4 × minimum of median sampling rate and observed sample density), in 0.05Hz steps. The candidate that most reduces head residual sum of squares is reported. Numerical frequency-grid spacing is not empirical frequency accuracy.

Amplitude is RMS of the dominant fitted sinusoidal component, expressed as percent image width. This differs from V1's full band-filtered RMS; historical V1 reports remain supported, and V2 explicitly identifies its amplitude type. Background amplitude is evaluated at the selected head frequency. The measurement describes image movement, not muscle tension or clinical tremor. Trends, head rotation, expression, torso movement and tracking errors remain possible confounds.

Windows need at least 20 usable samples, at least 80% tracking coverage, at least 1.7 seconds of tracked span, at least 80% occupied 100ms time bins, density at least 12 samples/second, and no usable-sample gap above 400ms. These engineering gates limit missing-data estimation; they are not diagnostic thresholds. Significant gaps or sparse bursts still abstain. Reports include rejection reasons, received/usable counts, temporal coverage, span and largest gap. Ordinary users see a concise missing-window reason and time coverage; detailed counts are in the administrator panel.

## Results

- Six head-motion unit tests passed, including recovery of a known 4.4Hz sinusoid with isolated frame losses and rejection of sparse burst sampling.
- All 60 neck Python tests passed, including mock-provider failure persistence and session validation.
- Capture, tracker and frontend tests passed, including backward rendering of V1, V2 time coverage and specific missing-window reasons.
- Synthetic image-tracker controls, evaluated by V2: static image 0%; head oscillation approximately 0.497%; slow translation approximately 0.0011–0.0015%; camera translation residual approximately 0.0015–0.0094%. See `synthetic_reports_v2.json`. These generated controls do not establish accuracy on normal patients.
- Latest actual Chromium retest of the same real clip: 236 samples in each viewport, median 31.217Hz (actual mean delivery is lower because of dropped frames and the sampling cap). Eight windows of eight measured in both desktop and mobile. Dominant peaks varied approximately 4.05–5.10Hz, rather than one fixed frequency for the entire clip. Group grades, provider-failure rendering and administrator visibility checks passed without JavaScript errors. `browser_results.json` and browser trace files now contain this latest V2 run; the V1 counts in `REPORT.md` are historical.

The retest uses local playback of the existing video and first-frame anchors, not a physical webcam or mobile device. It makes no OpenAI call. A new recording from the user's camera is needed to assess improvement in that environment. Old records cannot be retroactively reanalyzed without their discarded dense traces. The server was restarted to apply the updated analysis.
