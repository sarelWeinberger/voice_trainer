# Observable Neck/Vocal Effort: feasibility and baseline report

Research date: 2026-10-01. Status: implemented research baseline, **not a validated professional effort estimator**. The request supplied to this project ends at “vocal hyper”; no unseen requirements are assumed. Existing GRBAS analysis is separate and is not used to train or validate this construct.

## Operational targets

| Construct | Operational reference | Current RGB/audio output |
|---|---|---|
| External visible neck effort | Blinded ratings of visible muscle prominence/deformation during specified tasks, with rater agreement | Not estimated: landmark posture does not measure muscle prominence |
| Extralaryngeal/perilaryngeal activity | Task-normalized regional sEMG, with placement and cross-talk controls | Unavailable; no electrodes or learned paired reference |
| Vocal effort | Speaker self-rating per task/window, separately from listener perception | Unavailable; no calibrated supervised model |
| Acoustic manifestations of strain | Blinded auditory strain ratings and validated acoustic measures for matched speech tasks | Raw acoustic correlates only; no strain probability |
| Head/neck compensation | Image-plane kinematics relative to a reference; compensation requires task/context evidence | Head/shoulder angles, reference changes and head-distance variability; compensation not inferred |
| Respiratory/phonatory effort | Airflow, pressure and respiratory references | Unavailable |
| MTD/pathology | Independent clinical assessment, including relevant examination | Excluded |

Separate outputs are preferable. Fusion would hide differences between subjective effort, posture, acoustic properties and physiology. Null means unknown/unvalidated, never zero effort. Current `confidence` is a heuristic tracking-quality value, not accuracy probability. No calibrated effort interval can be supplied without held-out reference labels; the API exposes that explicitly.

## Evidence ledger

Searches used PubMed, primary papers and originating dataset pages for sEMG, supra/infrahyoid muscles, SCM, perilaryngeal tension, MTD, vocal hyperfunction, vocal effort and audiovisual datasets. This is a targeted feasibility review, not a systematic review with exhaustive screening. Some publisher/PMC pages blocked full-text access; abstracts and accessible primary PDFs were used where available. Publication dates were checked against the research date; search-engine relative dates were not accepted as evidence.

1. [Stepp, 2012: sEMG measurement, analysis and interpretation](https://pubmed.ncbi.nlm.nih.gov/22232412/), [author's tutorial PDF](https://sites.bu.edu/stepplab/files/2015/02/Stepp.Tutorial.2012.pdf). Electrode placement, recording and interpretation matter in the speech/swallowing anatomy. Regional sEMG is not equivalent to selective internal laryngeal muscle measurement. Engineering implication: normalize physiological labels within participant/session and report electrode regions; do not claim pixel-to-internal-tension recovery.
2. [Wang & Yiu, 2023: suprahyoid and SCM in pitch/loudness control](https://pubmed.ncbi.nlm.nih.gov/37215173/). Twenty-nine screened participants performed vowels with pitch/loudness manipulations. Lower pitch/loudness reduced suprahyoid activation; higher levels did not significantly increase activation relative to baseline. /i/ differed from other vowels, and SCM showed no significant changes across these manipulations. This directly challenges a simple loudness-to-neck-activation rule. It does not prove absence of activation in all tasks or clinical populations.
3. [McKenna et al., 2019: physiology and self-perceived effort](https://pubmed.ncbi.nlm.nih.gov/30969902/), [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC6802880/). Twenty-six healthy adults performed /ifi/ tasks. A combined model retained pressure, supraglottal compression and suprahyoid activation, with adjusted R² .60. These were measured with specialized physiological instruments, not ordinary video/audio. This supports a multi-component construct and leaves substantial variance unexplained; it does not validate this application.
4. [McKenna & Stepp, 2018: acoustic/perceptual effort](https://sites.bu.edu/stepplab/files/2018/09/JASA_2018.pdf). Acoustic relationships depend on whether effort is self-rated or listener-rated. RFF effects and controlled task conditions motivate future task-specific analysis. Laryngoscopy during recording and ordered effort elicitation limit direct generalization to casual patient video. Current baseline does not claim to reproduce their acoustic regression.
5. [Hočevar-Boltežar et al., 1998: sEMG in MTD](https://pubmed.ncbi.nlm.nih.gov/9840515/). Pilot observations of elevated regional activity are historical evidence, not a general diagnostic threshold or RGB scoring rule.
6. [Fatigue-related perilaryngeal sEMG changes](https://pubmed.ncbi.nlm.nih.gov/36580552/). Vocal loading/fatigue is another target distinct from instantaneous perceived effort. Future longitudinal validation must model task duration rather than equate fatigue with posture.
7. [Eulerian video magnification pilot](https://pubmed.ncbi.nlm.nih.gov/30917886/). Relevant hypothesis for subtle external motion; pilot diagnostic work is not sufficient to deploy a tension estimator. Full text was not obtained in this run. No efficacy number is inferred from the title.
8. [Motion capture of extrinsic laryngeal movement](https://pmc.ncbi.nlm.nih.gov/articles/PMC10724377/). Retrieved indexing describes motion capture in pMTD and controls. Specialized motion capture measures kinematics, distinct from activation. Full text was blocked in this run; no quantitative effect is claimed.

Inference from these sources: ordinary RGB plus uncalibrated audio can expose kinematic/acoustic correlates, but this evidence does not establish identifiable internal tension or a transferable fused effort score. That is a feasibility limitation, not proof that every future supervised method must fail.

## Dataset audit

| Dataset | Reference | Utility | Limitations/status |
|---|---|---|---|
| Lombard GRID | [Originating lab](https://spandh.dcs.shef.ac.uk/avlombard/), [Zenodo record](https://zenodo.org/records/3736465), [corpus paper](https://pubmed.ncbi.nlm.nih.gov/29960497/) | 54 talkers, plain/Lombard conditions, synchronized front/profile recordings; acoustic condition proxy experiment | Head-mounted views and no paired neck sEMG/effort truth; proxy accuracy cannot validate muscle effort. Audio downloaded and experiment executed; video archive not downloaded |
| ViGIR laryngeal sEMG v1 | [Dataset owner](https://vigir.missouri.edu/Research/sEMG_dataset.html) | Simultaneous regional submental/infrahyoid sEMG and audio, MVC and vocal effort metadata, 61 women | Requires access request; restricted demographic coverage; public page does not describe RGB video. Not downloaded; no external messages sent |
| Original GRID | [Zenodo](https://zenodo.org/records/3625687) | Audiovisual tracking/speech control, 34 speakers | No physiological effort ground truth; not downloaded |
| Existing application recordings | Local user sessions | Could support consented annotation after protocol approval | Existing `consent_research=False` data were not used for research training |

Lombard GRID attribution: Najwa Alghamdi, Steve Maddock, Ricard Marxer, Jon Barker and Guy J. Brown, University of Sheffield/Université de Toulon. Origin page states CC BY 4.0. Public audio remains in `/tmp`, not committed with source. The archive's recorded MD5 is `fa0cfc739705323b53ba50e148b3a144`.

## Implemented experiments and negative results

`run_baselines.py` produces `baseline_results.json`. Three logistic regression baselines use intensity only, spectral balance only and both; scaling is fitted inside each training fold. Five GroupKFold partitions hold speakers out completely. Labels are plain/Lombard condition, not effort or disease. Balanced accuracy and fold results are reported. Observed archive: 5,390 accepted audio files from 54 speakers (2,691 plain, 2,699 Lombard). Mean held-out balanced accuracy: intensity 0.7480; spectral balance 0.7028; combined 0.7981. The combined fold values span 0.7455–0.8548. The chance baseline is 0.5. This is condition discrimination, not validated vocal effort. Archive MD5 was verified against the publisher.

These models are research artifacts and are deliberately not used as a clinical application model.

Gain falsification: double the amplitude of exactly the same 200 Hz signal. RMS changes by 6.02 dB, spectral balance does not change. Vocal production has not changed. Therefore recorded intensity alone fails an identifiability test for physiological effort. Synthetic signal tests establish software properties only.

Visual falsification: feed a 40° head angle and a 30° shoulder angle to the new result layer. Observable angles remain available while effort and diagnosis scores remain null. A tilted head cannot, by itself, establish muscular effort. No supervised visual accuracy result is available: paired video/physiological targets were not obtained.

Technical negative results: initial download failed inside the network sandbox, then a permitted transfer timed out and was resumed. Initial dataset ingestion assumed the original long filename convention, but the distributed archive uses shortened filenames; this was corrected before the reported experiment. No results from the failed ingestion are treated as evidence.

## Application behavior

`observable_effort.py` reuses existing landmark tracking but overrides the previous clinical severity calculation. The production import now selects this engine. Legacy scoring code remains as a historical implementation and its regression tests; it is not the selected runtime engine. Old stored scores are not retroactively reinterpreted.

The live camera widget shows image-plane angles and tracking quality. The audio analysis response includes one-second RMS dBFS, high-frequency energy ratio, clipping checks and unavailability reasons. Quiet/clipped windows are marked unusable; noise is not reliably excluded, and usable is not a speech detector. The spectral ratio depends on phonetic content, microphone response and sample rate. No sound pressure level is inferred.

`POST /api/observable-effort/recording` accepts authenticated multipart `video`, WAV `audio`, and `audio_offset_seconds` (audio start on video timeline). It analyzes up to 60 seconds of constant-frame-rate video at roughly five frames/second, returns separate timelines, and exposes caller-declared timing. It does not independently verify synchronization or correct variable frame rate/device drift. Uploads are limited to 50 MB each and temporary files are removed after processing. The UI offers uploads and JSON download. Live server-receipt timestamps are expressly not certified synchronized with audio.

Tracking uses monocular estimates internally for view/calibration compatibility. Estimated forward depth is removed from result features and is never presented as physical neck displacement. Head/shoulder angles depend on camera roll and viewpoint. The neutral reference is a short session baseline, not a validated resting-muscle reference. Calibration and failed observations are retained as timeline gaps rather than interpreted as low effort. More rigorous camera-quality, occlusion and motion uncertainty quantification remains necessary.

## Validation needed before a professional estimator

Collect consented synchronized CFR neck/face video, audio, regional sEMG (suprahyoid, infrahyoid, SCM), normalized reference contractions and speaker effort ratings; add blinded external visual and auditory ratings separately. Use comfortable/raised effort, matched vowels/connected speech, pitch/loudness changes, deliberate posture changes without effort, camera rotation, gain/distance changes, noise and fatigue tasks. Respiratory claims require airflow/pressure references. Include a range of neck anatomy, skin appearance, age, sex, language and recording devices.

Keep all recordings for each participant in one split. Fit normalization on training speakers only. Compare constant/subject-reference, visual-only, acoustic-only and multimodal models; evaluate within-speaker change, held-out-speaker error, rank correlations, rater agreement, task/device subgroup performance and abstention coverage. Use clustered bootstrap by participant and calibrated prediction intervals on a distinct calibration split. Include modality missingness and counterfactual camera/gain controls. Fusion should be promoted only if it improves held-out physiological/perceptual targets over each modality, without worsening uncertainty or subgroup failures.

Current acceptance: software baseline implemented and tested; genuine proxy experiment executed. Professional effort estimation, calibrated uncertainty, direct external muscle visibility measurement and clinical validation remain **unachieved**. They cannot be completed by relabeling posture or by importing disease labels. Access to paired physiological reference recordings is the concrete remaining dependency.

## Verification performed

- Seven existing landmark/scoring regression tests and five new observable-output tests passed.
- Real OpenCV MP4 decode plus WAV read, MediaPipe inference on no-person frames, explicit audio offset and abstention passed. This is a software integration check, not human visual validation.
- Actual FastAPI route callable, selected runtime engine, OpenAPI registration and response schema passed using a temporary SQLite database.
- Python compilation, both JavaScript syntax checks and `git diff --check` passed. No browser visual review was performed.
- Full repository unittest discovery did not complete and was interrupted; a full-suite pass is not claimed.
