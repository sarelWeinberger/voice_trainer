# Analysis wait regression — 2026-10-06

Deployed commit 2933da3 to Voice-trainer/voice-trainer; Hugging Face runtime reports RUNNING at that exact commit. Public app.js bytes match the local fixed file.

37 Python tests passed (34 existing neck/API/media/observable tests plus 3 new output-budget, HTTP-error and timeout tests). Existing Node renderer tests and JavaScript syntax checks passed.

Chromium tests on desktop and mobile used actual application pages and isolated SQLite login/session routes, with synthetic camera/microphone. Simulated neck and failing voice responses demonstrated independent request starts (desktop 0.067 seconds apart; mobile 0.083), a visible neck result despite audio failure, automatic default 5-second recording stop, re-enabled recording button and a visible 20-second request timeout. No uncaught JavaScript errors. Slow-provider success after the deadline did not turn the missing result into a score.

A separate actual HTTP test, without response interception, completed recording start/frame/stop and audio analysis with HTTP 200 for synthetic input. Nonhuman images correctly abstained. No real OpenAI call was made; latency or scoring accuracy of successful inference with the production key is not established. Eight selected frames plus up to two crops replace 24 frames plus six crops; reduced temporal evidence is an accuracy tradeoff requiring later evaluation. The pinned GPT-4.1 model, image detail, scoring weights and scientific abstention rules are unchanged.

The frontend 20-second timeout limits waiting, not successful inference latency or guaranteed server execution duration. The provider uses a 15-second socket timeout; DNS, queueing, CV and database work are not covered by a total server deadline. Result may be unavailable at deadline. An external service cannot guarantee every request succeeds within five seconds.

Saved harnesses expect Playwright/Chromium and the isolated local server on port 5099; do not run them against production data. Artifacts contain only synthetic/mock results.
