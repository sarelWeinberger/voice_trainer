# Neck V1 browser validation — 2026-10-06

Passed in real headless Chromium 149 (Playwright 1.63.0), against the FastAPI application served locally on port 5099 with an isolated temporary SQLite database. Desktop viewport: 1280×900. Mobile HTML viewport: 390×844. Browser-provided synthetic camera and microphone streams were used; no patient recordings were used.

## Observed results

- Actual UI login succeeded on both pages.
- Actual recording start, image frame uploads, recording stop, audio analysis and result display succeeded on both pages (HTTP 200).
- Synthetic camera has no human neck: the real backend correctly returned unavailable dimensions and a null overall score. UI displayed “לא ניתן להעריך”, without converting missing scores into zero.
- All five dimensions, score categories, overall score and confidence displayed using explicitly simulated result data. Literal HTML in evidence remained text, without creating an image element.
- Simulated unavailable-service result displayed its message without zero-valued effort scores.
- No uncaught JavaScript errors. No horizontal document overflow at either tested viewport (1280/1280 and 390/390).
- Actual multipart upload of a three-second blank MP4 and silent WAV succeeded. JSON download parsed and retained a null overall score.
- Simulated upload HTTP 502 displayed its message, hid the download button and re-enabled submission.

## Boundaries

This is browser integration validation, not clinical validation. No OPENAI_API_KEY was configured, so no real OpenAI inference was tested. Non-null scores were injected directly into the shared renderer to test presentation; the real API path was exercised only for insufficient-quality input. Mobile viewport testing does not establish behavior on physical Android/iOS devices. Safari, Firefox, real camera/microphone input and clinician-rated recordings remain untested.

No production application code changes were needed. Playwright was installed in the local virtual environment only; production requirements were not changed for this test. Temporary Chromium libraries were extracted under /tmp without system installation.

## Evidence and reproduction

`results.json` and `upload-results.json` contain assertions and observed HTTP statuses. Screenshots contain explicitly simulated score data. `uploaded-report.json` is from the real synthetic-media upload.

The saved Python scripts use the repository's seeded test account and expect an isolated local server at 127.0.0.1:5099. They are browser test harnesses, not production modules. They expect Playwright/Chromium and write artifacts to `/tmp/neck-browser-results` (create it before running). They must not be pointed at a production database.
