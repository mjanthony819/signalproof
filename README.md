# SignalProof

**A research-only ECG artifact lab, not a diagnostic tool or clinical signal-quality gate.**

SignalProof compares an actually trained 816-parameter 1D CNN against a trained spectral logistic-regression baseline. It includes browser-local inference, a waveform workspace, controlled stress tests, evidence and confusion matrices, a searchable error explorer, and a local research notebook with validated backup/restore.

## Run the working product

Open `index.html` in a modern desktop browser. The entire app, trained weights, research examples and measured results are embedded. No install, login, network, paid API, telemetry or backend is needed. The six routes also adapt to phones.

`SignalProof_Complete.zip` contains the full editable project, training and build code, saved checkpoint, exported model weights, split manifest, prediction exports, tests, source checksums, offline app and detailed DOCX interview manual. Extract it before following development commands. This archive is source code and research artifacts, not an opaque executable. The readable training script and result summary are also supplied at repository root for inspection.

## Problem and who has it

Students and research engineers need to compare ECG artifact models without hiding data leakage, uncertainty or failed predictions behind one accuracy number. SignalProof makes the evaluation protocol and its limits part of the product. It is not intended for clinicians or patients to diagnose disease, reject ECGs, decide triage or replace review.

## Method

Public MIT-BIH reference ECGs and MIT-BIH Noise Stress Test waveforms from PhysioNet are used to build four semi-synthetic labels: Reference (no added noise), baseline wander, muscle artifact and electrode motion. Reference is not a claim that the original ECG is clean or normal. Whole-window RMS noise is added at -6, 0, 6 or 12 dB. This simple SNR definition is not identical to the original NSTDB beat-specific noise generator.

- First channel; polyphase resampling 360 to 128 Hz (16/45).
- 150 non-overlapping 8-second windows per reference record, stride 10 seconds.
- Train records 100, 101, 103, 105, 106, 107: **3,600** constructed examples.
- Validation records 108, 109: **1,200** examples.
- Held-out test records 111, 112, 113: **1,800** examples.
- All variants of a reference window stay in the same split.
- Noise time regions: train 0–880 s, validation 910–1300 s, test 1330–1790 s, with gaps.
- The same noise recording identities remain shared. Within-split noise segments can repeat or overlap.
- Baseline: 12 Welch/morphology features, training-only standardization, L2 multinomial logistic regression.
- CNN: Conv1D(1→8, k9, stride4), ReLU; Conv1D(8→12, k7, stride4), ReLU; global mean pooling; linear 12→4. **816 parameters**, trained from scratch on CPU.
- Seed 42; AdamW .003, weight decay .001, batch128, 32 epochs. Choose minimum validation log-loss checkpoint.
- Separate scalar temperatures fitted on validation only. Test does not choose weights or calibration.

## Evaluation

Actual held-out results from the executed pipeline:

| Metric | Spectral baseline | 1D CNN |
|---|---:|---:|
| Accuracy | 46.61% | 50.78% |
| Macro F1 | 0.411 | 0.485 |
| Log loss, lower is better | 1.152 | 1.176 |
| 10-bin ECE, lower is better | 0.141 | 0.109 |
| Multiclass Brier, lower is better | 0.635 | 0.614 |

The CNN improves macro F1, but **has worse log loss**. This is a modest semi-synthetic result, not clinical validation. At the exploratory 70% threshold, CNN coverage is only 6.78%; selective accuracy on that small accepted subset must never be presented as overall accuracy. The default threshold is not a validated safety guarantee.

Every test prediction, confusion matrix, per-record and SNR-sliced result is included in the complete source archive and downloadable from the app. SNR slices also include all reference windows, so slice sizes differ. Browser timing is measured live; recorded Python timing is specific to the development CPU.

## Reproduce

1. Download and extract `SignalProof_Complete.zip`; enter its project folder.
2. Use Python 3.10 or a compatible environment. Create a fresh virtual environment.
3. `pip install -r requirements.txt` (CPU PyTorch is sufficient; no paid service or GPU).
4. `python train.py` downloads the selected public PhysioNet source recordings, prepares the splits, trains both models, fits validation temperatures and writes results. The large raw recordings and prepared tensor cache are intentionally not bundled.
5. `python build.py` rebuilds the self-contained HTML from `web/template.html`, inference code and exported model JSON.
6. `node test.js` runs 19 core checks, including seven Python/JavaScript probability and feature parity fixtures (absolute tolerance 0.00002).
7. Optional UI test dependencies: `pip install playwright`, `python -m playwright install chromium`, then `python ui_test.py`.

`data_checksums.json` fingerprints downloaded source files. `split_manifest.json` includes every source record/offset, label, SNR and noise offset. `training.log`, `results.json`, `cnn.pt` and exact training package versions are included. Small floating-point differences across environments are possible.

UI checks cover all six routes at 320/390/1280 px, no horizontal overflow or JS errors, local inference, malformed CSV rejection, save/reload/replay, stress tests, evidence threshold, error filters and an actual offline export/reopen. Visual screenshots were inspected after fixes. See the detailed interview manual for architecture math and 32 questions with answers.

## Who benefits, and who could be harmed

Students can learn model comparison and leakage-aware experiment design. Research engineers can inspect and reproduce a small artifact-signature benchmark. Patients and clinicians could be harmed if outputs are mistaken for diagnosis, signal acceptance, normal rhythm or a safe clinical filter. Do not use it for those purposes. Only de-identified research signals should be imported. The app cannot verify consent, sampling rate or clinical meaning.

## Limitations

Only 11 selected reference records, 3 held out, and 3 shared noise source recordings are used. The test windows are correlated, not 1,800 independent patients. There are no expert clinical quality labels, device-external cohorts, demographic claims, mixed-artifact validation, disease targets or confidence guarantees. Original reference ECGs may contain artifacts or arrhythmias. An accurate injected-artifact label does not establish clinical usability. Stress-bench mixtures have no ground-truth quality labels. No statistical-significance or clinical-readiness claim is made.

## Another month

Add independent expert-labeled quality datasets with compatible terms, subject/device-external evaluation, noise-source holdout, record-level uncertainty, simultaneous artifacts and calibration-transfer checks. Compare stronger architectures using new held-out data, not repeated tuning against this test. Work with domain experts before considering any clinical use.

## Attribution and licenses

The code is MIT-licensed. The source databases and adapted signal excerpts retain **Open Data Commons Attribution License 1.0** terms. See `THIRD_PARTY.md` and the source archive. No endorsement by PhysioNet or the original authors is implied.

- MIT-BIH Arrhythmia Database v1.0.0: https://physionet.org/content/mitdb/1.0.0/
- License: https://physionet.org/content/mitdb/view-license/1.0.0/
- Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database*. IEEE Eng Med Biol Mag. 2001;20(3):45–50. DOI: 10.1109/51.932724.
- MIT-BIH Noise Stress Test Database v1.0.0: https://physionet.org/content/nstdb/1.0.0/
- License: https://physionet.org/content/nstdb/view-license/1.0.0/
- Moody GB, Muldrow WE, Mark RG. *A noise stress test for arrhythmia detectors*. Computers in Cardiology 1984;11:381–384.
- Goldberger AL et al. *PhysioBank, PhysioToolkit, and PhysioNet*. Circulation 2000;101:e215–e220. DOI: 10.1161/01.CIR.101.23.e215.

## AI assistance disclosure

Instinct AI assisted with concept, code, training scripts, analysis, tests, UI, documentation and demo narration. Reported results come from executed training and evaluation, not generated performance claims. The project author should study the manual and understand the method before presenting. No claim is made that the author independently wrote every generated line or clinically validated the system.
