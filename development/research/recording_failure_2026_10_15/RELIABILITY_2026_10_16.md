# Capture reliability and the zero-grade investigation — October 16, 2026

## Outcome

Short upload interruptions and lost acknowledgements now recover without processing images twice. Dense head-motion evidence now reaches the interpreter before grading. On the supplied clip, a controlled real OpenAI comparison still returned all-zero regional grades. That negative result remains unresolved; no acceptance threshold or grade was changed to force a positive result.

## Capture transport

`static/neck-capture.js` retains up to 150 queued JPEGs, with a separate 16MiB queue byte bound. This covers 125 images at the supported maximum 25-second audio length under normal image sizes. In-flight upload/encoding buffers are additional bounded buffers. Overflow is counted; older baseline images are no longer silently evicted. Encoding already started before stop is allowed to finish within the drain deadline.

Uploads are serialized in batches of at most ten images. Each batch has a stable ID and immutable timestamps/JPEGs across retries. Network errors, temporary server errors and invalid receipts permit up to two retries, with 250/750ms backoff and an eight-second request timeout. Permanent HTTP errors terminate transfer. Stop allows up to twelve seconds to drain pending images, using a monotonic clock. Extended disconnection, persistent errors or exceeding the memory/drain bounds can still leave missing images; there is no unlimited buffering or waiting.

The backend caches a bounded per-session receipt and SHA-256 digest of batch contents. A repeated accepted batch returns its receipt without running CV again. Reusing an ID with different contents is rejected. A partially processed batch can resume at its first unprocessed frame; later batches cannot overtake it. Unsupported frame limits and failed processing do not receive a false full acknowledgement. Cache entries contain derived receipts/digests, not JPEGs, and session reset clears them.

Capture counters accompany the stop request and are reconciled against the number of frames actually held by the server. A lost acknowledgement can therefore be distinguished from a missing server image. A partial analysis is labeled in the regular UI; technical counters remain in administrator diagnostics. Existing clients without the optional new fields retain their route contract.

## Why the grades were zero

The prior raw model outputs already contained zero grades; confidence thresholds were not turning positive grades into zero. Also, the dense tracking result was previously attached after `finalize()`, so it never reached the interpreter.

The stop route now recomputes the supplied dense trace on the server and passes it into `finalize()`. `neck_capture_context.py` creates bounded comparisons of first, last and largest movement windows, grouped by equal duration and tested band, with coverage, RMS dispersion, periodicity status and supported frequency. Unsupported frequencies and missing windows remain unavailable. Raw image motion is not converted into an effort grade.

The same pinned `gpt-4.1-2025-04-14`, images, image quality, evidence selection, output schema, weights and acceptance thresholds remain. Prompt version `neck-rubric-v1.5` tells the interpreter to consider dense movement context and distinguish measured irregular movement from posture compensation or local neck recruitment. The result view likewise explains why a zero posture-compensation grade can coexist with measurable head movement.

## Real provider experiment and negative result

`compare_motion_context.py` submitted identical image evidence under the same updated prompt, once without dense context and once with it. No participant-reported effort timing or desired grade was supplied. The image fingerprints and selected frame indices matched between arms.

| Arm | Provider seconds | Input/output tokens | Accepted regional grades |
| --- | ---: | ---: | --- |
| Without dense context | 8.79 | 14,952 / 868 | All five 0 |
| With measured dense context | 8.30 | 16,681 / 1,065 | All five 0 |

The second response explicitly acknowledged measured irregular head movement in its confounds while retaining zero posture-compensation grading. This demonstrates that the new input reaches the interpretation, but **does not demonstrate improved effort sensitivity**. A single call per arm cannot establish repeatability or a latency improvement. Measurable movement and the participant's felt effort do not supply an independent clinical reference for the rubric grade.

An initial comparison failed to include usable dense summaries because native NumPy scalar values were excluded by an exact Python type check. That integration error was fixed; a regression now uses the actual server estimator's output. The incomplete comparison is preserved in `motion_context_incomplete_comparison.json` and excluded from the table. Four real provider requests were made in total: the incomplete pair and the corrected pair. `motion_context_comparison.json` contains the corrected evidence/results and token measurements. No additional patient images were retained.

The official [GPT-4.1 documentation](https://developers.openai.com/api/docs/models/gpt-4.1) identifies image input and no native video input. The application supplies selected images and derived timing/motion context. The [evaluation guidance](https://developers.openai.com/api/docs/guides/evaluation-best-practices) informed the controlled comparison; this experiment does not substitute for independent target-specific labels.

## Browser and regression verification

`browser_transport_check.py` used real Chromium playback of `WIN_20261015_21_04_57_Pro.mp4`, real in-browser FaceMesh, actual JPEG encoding and the application's authenticated multipart endpoints through an isolated FastAPI test client. The first batch was processed by the real CV pipeline; its response was held for 6.2 seconds and deliberately lost. Retrying exercised the backend's actual receipt cache. Interpretation alone used a controlled zero response, so this check measures capture and routing, not real provider accuracy.

| Browser view | Captured | Received/acknowledged | Retries | Duplicate timestamps | Dense context at interpretation |
| --- | ---: | ---: | ---: | --- | --- |
| Desktop | 37 | 37 | 1 | None | Measured |
| Mobile viewport | 37 | 37 | 1 | None | Measured |

No uncaught browser errors occurred. The application returned complete result packets, with no partial-transfer warning. `browser_transport_results.json` stores the counters, HTTP statuses and derived context. This was video playback in Chromium, not a physical phone/webcam test.

68 Python neck tests passed. New checks cover duplicate and conflicting batch IDs, session reset, partial-batch resume, capture-counter validation/reconciliation, native estimator scalars, motion reaching the interpreter, preserving zero grades despite motion, and motion availability on provider failure. JavaScript capture/renderer checks passed, including 125 retained frames during a pending request, byte/count overflow reporting, lost acknowledgements, permanent failures, stop during JPEG encoding, bounded draining, desktop/mobile metadata and role-appropriate warnings. Existing JavaScript syntax and whitespace checks passed.

## Remaining work

The transfer fix handles the tested failures within explicit bounds. Sustained poor connectivity can still produce a clearly labeled partial result. Audio/video timing remains approximate.

Reliable grading of the neck effort described by the participant has **not** been established. Resolving it requires independently annotated examples of the intended observable behavior, including positive and ordinary speech examples, with separate labels for visible instability, local contour change and felt effort. Current evidence does not establish that lowering thresholds, changing weights or training on synthetic effort labels would improve accuracy. The existing empty validation manifest is available under `research/motion_hardening_2026_10_15/validation_manifest_template.csv`; its labels have not been fabricated.

The changes are local and have not been deployed to Hugging Face. The application must run the updated backend and refreshed browser scripts together. Historical reports cannot recover missing discarded frames.
