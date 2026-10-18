# Hugging Face deployment verification — 2026-10-18

Space: https://huggingface.co/spaces/Voice-trainer/voice-trainer

Verified running commit: `0d56509a72e6d8be4ef63a0b34a9833aff9dbc17`.
Application update: `2dd19967b35c21f9af7779dd5d6fcf98d64d9fab`.

## Service failures found and fixed

1. The Space initially had no `OPENAI_API_KEY` Secret. The existing local working key was installed through the authenticated Hugging Face secrets API. Its value was never printed, committed, included in browser code, or written into these reports.
2. The first deployed recording check returned HTTP 500 before reaching OpenAI. Runtime logs identified `AttributeError: module 'webrtcvad' has no attribute 'Vad'` in audio timing. The existing Docker patch opened `webrtcvad.py` for writing before reading it, truncating the module. The patch now reads the source first. The Docker build also instantiates VAD and processes a silent audio frame, so an empty but importable module fails the build.
3. Audio timing now treats an unavailable or unloadable VAD implementation as unknown speech timing rather than crashing. It does not reinterpret unavailable VAD as silence.

## Evidence

- 25 focused tests passed across deployment configuration, observable audio metrics, and neck API behavior. Regression tests execute the actual Docker patch against a temporary module and exercise an importable module lacking `Vad`.
- The Space reported RUNNING with the exact verified commit. Seven deployed HTML/JavaScript resources matched local sources byte for byte. OpenAPI included frame batch receipts and capture transport diagnostics.
- Chromium smoke checks of the public desktop and mobile pages passed: HTTP 200, expected updated capture scripts, no JavaScript exceptions or failed script/style assets. These were page-load checks, not a physical webcam recording.
- A real remote recording check used the previously supplied `WIN_20261015_21_04_57_Pro.mp4`: 36 JPEG frames at 5 fps and its mono audio, sent to the deployed recording endpoints. The actual OpenAI integration was used; interpretation was not mocked.
- Four frame uploads and recording finalization returned HTTP 200. All 36 frames were acknowledged, transport was complete, head motion was measured, and dense motion context was supplied. Analysis status was `complete`, with no failure reasons. Finalization took 10.728 seconds; total API workflow took 19.859 seconds. These are observations from one clip, not a latency guarantee.

Sanitized machine-readable evidence is stored alongside this report. Recordings, tokens, secrets, database contents, and research artifacts were excluded from the public deployment.

After the user reported entering a new key, a repeat real remote check also completed successfully: HTTP 200 for all uploads and finalization, 36/36 frames accepted, motion measured, no service failure reasons, 9.802 seconds to finalize and 15.443 seconds for the API workflow. All internal regional grades remained zero. `analysis_check.json` now contains this latest check. No secret values were inspected or replaced during the repeat check; success verifies the server's current configuration, not the identity of a particular key.

## Remaining limits

- The real model returned zero for all five internal regional grades on this clip. Restoring service availability does not resolve sensitivity to the reported increasing effort or establish grading accuracy. The earlier negative motion-context comparison remains relevant.
- Grades describe experimental observable behavior. They have not been clinically calibrated and do not measure internal muscle tension.
- No physical webcam recording on the user's device was tested in this cloud verification.
- The Space has no configured persistent storage or external database; this deployment does not establish durable recording/account storage. Authentication provisioning was not changed.

The previous failed recording report remains a historical result. A new analysis is required to use the deployed fix.
