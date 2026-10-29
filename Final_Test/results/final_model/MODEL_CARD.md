# Model card - final deepfake detector of the thesis

Frozen 2026-10-28 21:55:35 by FINALTEST.py from run **train-Joint_seed42_F1** (selected on validation data only).

## Model
- EfficientNet-B0 (ImageNet IMAGENET1K_V1 initialisation) per face frame -> 1-layer temporal transformer over the 16 frame positions -> attention pooling -> P(fake); 4,967,167 parameters; video only (no audio).
- Variant F1: F0 + degradation-aware Stage 2: unpaired 50/50 original/degraded clips, natural degradation mixture
- Selection: Joint F2 unless its mean validation score is more than 0.005 below Joint F1; best-validation seed of the chosen variant (validation data only). Mean validation score: F1 0.7772, F2 0.7566; chosen seed 42 (validation score 0.8030). F2 fell more than the tolerance below F1 on validation, so F1 was frozen (pre-registered fallback).

## Input and preprocessing
- 16 frames uniformly spaced over the video; frames normalised to a 360 px short side; OpenCV YuNet face box (margin 0.20, 3-frame smoothing); every face is passed through an 80 px bottleneck (removes the resolution shortcut) and cropped to 128 px RGB; the network sees 192 px (bilinear) with ImageNet normalisation.
- The preprocessing is the Test 5 code itself (9-10-26_test5_independent.py: extract_face_clip (imported at prediction time, never copied)); its source hash `ce57c0c8e3c402de...` is checked before every prediction.

## Decision rule
- P(fake) = isotonic calibration of the logit (fitted on 3,296 validation videos: originals + degraded versions).
- **FAKE** if P >= 0.560, **REAL** otherwise; **UNCERTAIN** if |P - 0.560| < 0.006 (band tuned for 90% coverage on validation; realised validation coverage 100.0%) or if no face is found.
- Temperature scaling (T = 1.53, threshold 0.69) is kept for reference.

## Test performance (Test 5 test split; never used for training, selection or calibration)
| test set | condition | n | no face | auc | bal. acc. (all) | real recognised | fake detected | coverage | bal. acc. (decided) | ECE (temperature) | ECE (isotonic) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BioDeepAV | original | 1420 | 39 | 0.958 | 0.937 | 0.899 | 0.976 | 0.973 | 0.932 | 0.105 | 0.054 |
| BioDeepAV | degraded | 4076 | 117 | 0.942 | 0.874 | 0.889 | 0.859 | 0.971 | 0.869 | 0.088 | 0.058 |
| FF++ | original | 2038 | 0 | 0.929 | 0.847 | 0.905 | 0.789 | 1.000 | 0.847 | 0.162 | 0.078 |
| FF++ | degraded | 2411 | 0 | 0.860 | 0.776 | 0.878 | 0.674 | 1.000 | 0.776 | 0.116 | 0.053 |

AUC is the ranking quality of the logit; 'decided' = videos outside the uncertain band (coverage = share decided).

## Explanations
- Grad-CAM of the last EfficientNet block for the predicted class (evidence for FAKE or for REAL), up-sampled to the face crop; temporal attention shows which frames carried the decision.
- Grad-CAM stability between original and degraded versions (Joint, seed mean of the chosen variant):
- gradcam top20 iou all BioDeepAV: 0.568
- explanation changed while stable pct BioDeepAV: 35.633
- clean reference iou BioDeepAV: 0.689
- gradcam top20 iou hard BioDeepAV: 0.486
- gradcam top20 iou all FF++: 0.663
- explanation changed while stable pct FF++: 23.394
- clean reference iou FF++: 0.664
- gradcam top20 iou hard FF++: 0.551
- Faithfulness (insertion / deletion gain over a random map; > 0 = the map points at evidence the model uses):
- gc del gain grey degraded ALL: +nan [+nan, +nan] - not distinguishable from random
- gc ins gain grey degraded ALL: +0.259 [+0.153, +0.352] - faithful (better than random)
- gc del gain grey original ALL: -0.234 [-0.333, -0.133] - worse than random
- gc ins gain grey original ALL: +0.297 [+0.192, +0.395] - faithful (better than random)
- gc del gain grey degraded FAKE: +nan [+nan, +nan] - not distinguishable from random
- gc ins gain grey degraded FAKE: +nan [+nan, +nan] - not distinguishable from random
- gc del gain grey original FAKE: +nan [+nan, +nan] - not distinguishable from random
- gc ins gain grey original FAKE: +nan [+nan, +nan] - not distinguishable from random
- gc del gain grey degraded REAL: +nan [+nan, +nan] - not distinguishable from random
- gc ins gain grey degraded REAL: +nan [+nan, +nan] - not distinguishable from random
- gc del gain grey original REAL: +nan [+nan, +nan] - not distinguishable from random
- gc ins gain grey original REAL: +0.407 [+0.332, +0.460] - faithful (better than random)
- A stable or faithful explanation is not proof that the decision is correct.

## Intended use and limits
- Research and decision support: screening face videos similar to BioDeepAV and FaceForensics++ (C23), including the degradation types of the thesis (blur, brightness, contrast, saturation, noise, compression, motion blur, frame drop, fps reduction, low resolution; three severities).
- Not a forensic or legal proof; not for identity decisions; not validated on other generators, other datasets, non-face content or audio deepfakes.
- Two datasets only; dataset-specific cues may remain (see the cross-dataset results of the final test); small faces (< 80 px) and videos without a detectable face are less reliable (the latter are always UNCERTAIN).
- Calibration and the uncertain band hold for the validation distribution; on new data, re-check calibration before trusting probabilities.

## Reproducibility
- final model file SHA-256 `6ccd2e5c988fab9713fce9ae9ab8f734a01241d47c018dc2f9bacdaec52f18c1`; weights SHA-256 `82f0e71d2a84042a4cf487cc72a4a0da92670b2895c84722f7e6870157080878`
- source checkpoint `E:\Code\run_FINALTEST\checkpoints\train-Joint\seed42\F1.pt` (SHA-256 `56bdb1209dedb1602cf0278b3d197ace7c29715c09f70732696278b632ddd903`)
- protocol SHA-256 `3291c1a2bd3b5692ec40bfd7b043845c80485acc87c2cc24ce4a6d902906331a`; split hash `6ae9fe14ca0dbd443ffa6a0bdc17167e95e69c8c8bd9bc439f264ef20a81f93a`; script SHA-256 `94be0af5c305514c0e78d1b898d4dfd1672fd05bf2dbe95e5a36ee9417103821`

## Use
```
python FINALTEST.py --predict "D:\video.mp4"        # or a folder; add --overlay-video for an MP4 with the heat-map
python FINALTEST.py --phase gui
```
Outputs: run_FINALTEST/predict/<video>_<time>/verdict.json + explanation.png (+ explanation_overlay.mp4).