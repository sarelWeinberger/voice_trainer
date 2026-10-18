# Recording failure reproduced and corrected — October 15, 2026

Source: `data/WIN_20261015_21_04_57_Pro.mp4`, 109 decoded frames, approximately 7.25 seconds, mean source rate 15.04fps. The latest pasted live report covers a different, five-second acquisition; the database diagnostics and supplied file were examined separately.

## Findings and changes

1. **Background point loss repeatedly reset the entire motion track.** The latest saved live acquisition had 75 dense samples, one head-missing sample, 33 background-missing samples and seven reinitializations. The preceding acquisition had zero head-missing samples and 44 background-missing samples. Head detection was not the main failure. On the newly supplied file, deterministic replay of the previous tracker reproduced zero usable windows and ten reinitializations.

   Fixed background sample locations missed useful scenery detail or selected edges that did not constrain two-dimensional motion. When consensus fails, the tracker now searches for spatially separated corners in the outer image quarters, outside the estimated person region. A structure-tensor score rejects one-dimensional edges. Crucially, it retries tracking **from the previous actual image to the current image**. Successful recovery retains the coordinates and segment. It does not interpolate a missing displacement. The existing six-inlier, 70% agreement and three-quadrant background requirements remain.

   True tracking discontinuities still start a new segment and invalidate windows crossing that discontinuity. The original fixed-point route remains the first choice: an initial all-corner implementation degraded the synthetic camera-motion control to 150/180 usable samples. Retaining fixed points and using corner selection as recovery restored that control. This negative result was not resolved by weakening quality gates.

2. **The provider timeout discarded a valid response.** A real request using the updated local key, the same pinned model, eight full images and two neck-context crops completed in 22.36 seconds. It used 14,831 input and 1,018 output tokens. The production socket timeout had been 15 seconds, and the browser stopped waiting after 20 seconds.

   The provider socket timeout is now 45 seconds and the neck stop-request deadline is 55 seconds, on desktop and mobile. There are still no automatic provider retries. These are bounded waiting policies, not a guarantee of a successful response within 20 seconds or a strict overall socket deadline. Independent local motion remains visible while waiting and after service failure. The model, image quality, evidence count and scoring acceptance thresholds were preserved.

3. **A usable irregular movement trace had no visible numerical amplitude.** The existing analyzer supplies `motion_rms_percent_image_width` even when its single-frequency periodicity gates fail. The result table now displays this RMS dispersion after removal of the window's constant/linear trend and two slower nuisance components. It is not the fitted sinusoid amplitude or a muscle-effort grade. Compare windows of the same length/tested band; tracking noise may also contribute. Dominant frequency is still withheld unless the original periodicity gates pass.

4. **The failure message overemphasized a single head miss.** When background loss predominates, the UI identifies it as the main failure rather than describing both sources equally and directing the user only to face lighting. Exact loss counts remain in administrator diagnostics.

## Verification

| Check | Result |
| --- | --- |
| Identical decoded clip, previous tracker | 0/7 usable windows; 10 reinitializations |
| Identical decoded clip, final tracker | 7/7 usable windows; no head/background loss; no reinitializations |
| Chromium desktop, real in-browser FaceMesh | 99 dense samples; 7/7 usable windows; no head/background loss or resets |
| Chromium mobile viewport, real in-browser FaceMesh | 101 dense samples; 7/7 usable windows; no head/background loss or resets |
| Local result after simulated failed stop request | Remained visible in both browser views |
| Authenticated application routes, default production OpenAI transport, temporary isolated database | HTTP 200; analysis `complete`; 36 sampled images accepted; motion `measured`; complete result persisted; stop call 10.12s |
| Focused Python provider/API checks | 16 passed |
| JavaScript checks | Image-motion controls, tracking recovery, integrity/corner-recovery regression, desktop/mobile request construction, renderer and timeout contract passed |

The first real API request used a research transport with a 45-second timeout to establish the cause. The second used the unmodified application request path and the **default production** `interpret()` transport, with real authentication and multipart uploads via FastAPI's in-process HTTP test client. Only the latter used an isolated temporary database; existing user records were not altered. Browser checks used actual clip playback and actual FaceMesh inference, with a simulated server disconnect specifically for the local fallback check. These are distinct checks, not a claim that a physical webcam plus browser plus real API were tested in one continuous recording.

No coherent single dominant frequency passed the gates on this clip. In desktop playback, two-second-window RMS dispersion ranged from about 0.19% to 0.44% of image width; the six-second window was about 0.48%. This describes observed image movement, not independently verified tremor or effort. In both real provider runs, the five regional rubric grades were zero. Thus successful delivery and recovered tracking **do not demonstrate that the rubric detects the user's felt neck tension**. That remains an unresolved modeling/validation issue.

## Reproduction and artifacts

- `replay_clip.py`: decodes the local file, obtains fresh face references, runs the actual JS tracker and stores derived data only. Temporary grayscale images are removed automatically. `--module` permits comparison with a saved tracker version.
- `tracking_before.json`, `tracking_after.json`: deterministic replay traces, correspondence diagnostics and server measurements. `tracking_candidate.json` is an intermediate candidate, not the final implementation.
- `provider_before.json`: first real provider experiment and token/latency measurements; no retained media.
- `browser_results.json`, `desktop_normal_trace.json`, `mobile_normal_trace.json`: live browser detector/capture results, processed by both local and Python estimators.
- `service_check.py`, `service_after.json`: real authenticated route/provider/persistence check with an isolated database and no test doubles for interpretation.

Browser reproduction uses `research/motion_hardening_2026_10_15/browser_check.py --video <clip> --output-dir <folder> --normal-only`. The new CLI options preserve the previous script defaults. An API-backed check incurs API usage and sends selected identifiable frames, as authorized for this investigation; running the local replay does neither.

The official [latency guide](https://developers.openai.com/api/docs/guides/latency-optimization) and [image-input documentation](https://developers.openai.com/api/docs/guides/images-vision) were checked while investigating the request. They do not establish a 15-second response guarantee; actual end-to-end times above are measurements from these runs.

## Remaining limits

No clinical validation, grade calibration, neck-muscle ground truth, independent tremor labels, or physical webcam/phone test was performed. Background consensus cannot prove scenery is stationary, and glare, occlusion, slow devices, frame gaps or camera movement can still prevent measurement. Waiting longer reduces this particular premature cutoff but cannot ensure provider availability. The dense metric measures head-image motion relative to scenery and does not isolate neck muscle tension.

Changes are local and have not been deployed to Hugging Face. The local application was restarted with the updated key and timeout. Refresh the page before making a new live recording; existing stored reports do not recompute automatically.
