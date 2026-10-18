# Local fixes and repeat validation

The initial experiment is preserved in USER_VIDEO_REPORT.md. Subsequent local changes:

- Browser JPEG quality increased from 0.8 to 0.95; server re-encoding increased from 88 to 95. Maximum capture edge remains 480 pixels to preserve throughput.
- Perspective pose now uses downward-positive image Y coordinates, a frontal initialization, positive-depth checks, reprojection error checks and plausible orientation limits. Failed fits omit pitch and emit a warning. This remains an uncalibrated image pose estimate, not a cervical angle.
- When hips are not tracked, a separately named shoulder displacement relative to the face transform is provided. It does not claim torso referencing or isolated shoulder recruitment. The original torso-referenced feature remains unavailable. Poor shoulder tracking cannot produce the torso-referenced feature.
- Tracking loss clears temporal body state. Evidence context includes pose/shoulder warnings and feature availability counts; the UI explains the limitations in Hebrew.
- Browser cache versions advanced for both desktop and mobile.

## Verification

48 neck Python tests passed, including synthetic pose perturbation and rejection tests. Capture queue and frontend JavaScript tests passed. Repeated the paired video experiment on the same first five seconds, six conditions and two repetitions: [user_video_after_fixes.json](user_video_after_fixes.json).

At 480 pixels, mean pitch deviation from the corresponding PNG reference changed from 15.89° to 2.05° for JPEG 80, and from 15.19° to 1.05° for JPEG 95. All 50 observations had pitch estimates in these conditions. Head-relative shoulder motion was available in all 48 eligible temporal pairs, while torso-referenced shoulder motion correctly remained unavailable. JPEG 95 head-relative shoulder deviation was 0.100651 versus 0.196309 for JPEG 80.

Warm processing time in the repeat experiment was 111 ms/frame for 480 JPEG 95 and 256 ms/frame for 960 JPEG 95. Host load differed from the initial run, so these are not controlled before/after performance estimates. Keeping 480 avoids the observed 960 processing cost exceeding the 200 ms sampling interval.

No OpenAI requests or clinical validation were performed. Actual browser encoding, camera acquisition and network throughput have not been revalidated after these changes. No Hugging Face deployment or running-server restart was performed. Other causes, including provider quotas, device latency and insufficient observable local activity, cannot be eliminated by these preprocessing fixes.
