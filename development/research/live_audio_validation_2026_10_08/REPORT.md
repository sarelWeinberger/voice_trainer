# Live neck audio context validation — 2026-10-08

Live recording stop previously finalized with unknown speech status and no audio windows. It now receives the same bounded WAV as voice analysis, the matching neck session ID and an estimated audio offset. Local WebRTC VAD context reaches the neck interpreter; raw WAV is not sent to OpenAI. The browser offset remains a heuristic, with the existing .8 timing-uncertainty factor retained.

## Verified

- 45 neck Python tests and 6 acoustic/compatibility Python tests passed. Controlled outputs verified provider context, window/video overlap, preserved null scores, silence skipping inference, invalid audio and stale-session rejection.
- Node frontend tests passed for desktop/mobile WAV/session/offset submission, explicit partial coverage, safe rendering and provisional audio timing.
- Real Chromium with synthetic media and the actual isolated API, ordinary network timing:

| View | Accepted frames | Observed span | Effective FPS | Estimated audio offset | VAD status | Mean CV seconds/frame | Uncaught JS errors |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Desktop | 25 | 4.8s | 5 | -.05088s | active | .0243 | 0 |
| Mobile viewport | 25 | 4.8s | 5 | -.08788s | active | .0248 | 0 |

Both browser requests used actual multipart parsing, WAV decoding and local VAD. Visual input contained no human, so OpenAI was skipped and no scores were fabricated. See `ordinary_network_results.json` for derived diagnostics.

## Negative result and limits

An earlier stress run inserted 1.7s delay before each batch submission. Desktop delivered only 15 frames across 3.198s, effective FPS 4.378, mean server CV time .2292s/frame. Audio context arrived (`active`, VAD available, offset -.02088s). The run failed the required >=20-frame check before testing mobile. Upload/drain deadlines can drop evidence under load; a previous successful stress result is not a guarantee. No production timeout was increased to conceal this result.

One-second VAD windows provide provisional context; their overlap span is not exact voiced duration. Estimated offsets do not verify synchronization, compensate for device drift, or prove speech causes movement. Synthetic media and controlled provider outputs cannot establish patient-level interpretation quality or clinical accuracy. No real OpenAI interpretation of a new patient recording was run for this change.
