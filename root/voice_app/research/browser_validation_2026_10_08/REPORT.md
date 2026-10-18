# Browser verification after preprocessing fixes

Tests used current local application code, headless Chromium, and an isolated SQLite database at port 5099. No production database or Hugging Face deployment was changed.

## Completed

- Desktop and mobile viewports: login, synthetic camera/audio acquisition, five-second recording, frame batches, audio arrival/VAD, voice analysis, neck finalization, and button recovery. Each received 25 frames spanning 4.801 seconds, at 4.999 fps. No JavaScript page errors. Synthetic camera depicts no person, so OpenAI was correctly skipped. These are Chromium viewport checks, not physical phone camera tests. Results: desktop_mobile.json.
- File upload of a generated blank MP4 and silent WAV, actual local decoding, report rendering, JSON download, and an explicitly simulated HTTP 502 error. The failure cleared the download and restored the button. No JavaScript page errors. Results: upload.json.
- User recording frames: first 25 sampled source frames loaded in Chromium, passed through actual NeckCapture canvas/JPEG 0.95 encoding, actual authenticated batch endpoints, real CV processor, finalization, and actual renderer. Received all 25 frames spanning 4.799 seconds at 5 fps. Pitch available in 25 frames; head-relative shoulder displacement in 24 temporal pairs. Mean server CV time 63.9 ms/frame. No JavaScript page errors. This replay validates browser encoding/transport, not physical webcam acquisition or audio synchronization on the supplied recording. Source frames were resized to 640×360 before replay and then browser capture used its 480-pixel cap. Results: user_video_browser.json; reproducible harness: user_video_browser_test.py.

## Negative finding and correction to test setup

The first user-frame replay decoded original-resolution PNG data one frame at a time with the preview tracker still running. It received 25 frames over 44.488 seconds (0.539 fps). The application correctly abstained due to sparse sampling. Preserved in slow_test_harness.json. The corrected harness preloads decoded images and pauses preview tracking, matching the application's recording behavior. This removes a test setup bottleneck; it is not evidence that every slow-device or network failure is eliminated.

## Real OpenAI attempt and remaining blocker

No OPENAI_API_KEY exists in this execution environment or the requested local .env file. The user-frame run reached the provider branch and returned the explicit missing-key error, which rendered correctly. No real OpenAI requests occurred, no provider response was simulated for that run, and no tokens were used. A valid server-side key is required to complete actual provider latency, schema validation, aggregation, and successful result rendering checks.

The user authorized a real provider test with their recording. It should be run once the key is made available; do not interpret the browser transport pass as a provider or clinical validation pass. The .env paths are now excluded from Git. The test server does not automatically load a subsequently created .env; it must be restarted using that file or an equivalent server environment.

Update: the user identified an existing `.env` in the project root. It contains OPENAI_API_KEY. Started a separate test server on port 5100 with that environment file and repeated the source-frame browser test. This made an actual OpenAI request, with the application's configured model, prompt, structured-output schema, images and 15-second provider timeout. OpenAI returned HTTP 401; the app rendered the sanitized authentication failure correctly. Provider finalization took 0.774 seconds, not a timeout. The response's HTTP 200 is the app endpoint status; the analysis itself has status `error` and must not be counted as a successful model result.

All 25 frames arrived at 5 fps over 4.8 seconds; pitch was available for all 25 and head-relative shoulder motion for 24 temporal pairs. CV averaged 57.9 ms/frame. No JavaScript errors. No synchronized audio was supplied in this source-frame replay. The key has the expected prefix and no surrounding/embedded whitespace or label; its value was never printed. A replacement valid key is necessary for successful provider output, scoring and rendering verification. No model output or token usage was returned, so this report makes no assertion about billed usage for the failed request. Results: real_openai_attempt.json.

## Successful retry after replacing the key

The user replaced the root .env key. A new isolated server on port 5101 loaded the updated environment. The same source-frame browser replay made a real provider request and completed successfully. Verified `analysis_status=complete`, no error reasons, all five dimensions accepted by aggregation, actual browser rendering, and no JavaScript errors. Results: real_openai_success.json.

25 frames were received across 4.801 seconds at 4.999 fps. Mean CV processing was 54.7 ms/frame. Finalization, including the real provider response, validation and rendering, took 10.381 seconds after frame transfer. The observation summary was 0/3 with adjusted evidence confidence 31.21%; this is an unvalidated observable-behavior result, not medical accuracy or absence of internal tension. No original audio was provided in this replay, so speech timing remained unknown/unverified. Separate synthetic live-audio tests cover audio transport/VAD but do not validate synchronization of this user's original video/audio. Token usage was not captured by this harness.

Successful completion resolves the authentication blocker for this test. It does not establish reliability across recordings, devices or network conditions. Production server settings and Hugging Face were not changed by the verification.
