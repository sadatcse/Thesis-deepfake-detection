# Test 5 (independent): hard-degradation-aware training and Grad-CAM explanation stability

_Generated 2026-10-16 02:52:48 | script `9-10-26_test5_independent.py` sha256 `e789a2a2a512ee07…` | data seed 20261009 | split hash `6ae9fe14ca0dbd44…`_

**Completion:** 24/24 Stage-2 runs finished. Seeds completed per variant: 5A: [42, 123, 2026], 5B: [42, 123, 2026], 5C: [42, 123, 2026], 5D: [42, 123, 2026].

## 1. Research question and hypotheses

Does targeted training on the degradation types with the least stable explanations in earlier work (brightness, saturation, contrast, blur) improve detection robustness and Grad-CAM explanation stability of the AudioVisualDeepFakeCNN?

- **H1** (5B - 5A): Hard-degradation sampling improves detection on hard degradations
- **H2** (5C - 5B): Grad-CAM consistency (with pairing) improves explanation stability
- **H3** (5D - 5C): Feature consistency + curriculum improve robustness / stability beyond 5C
- **H4** (5D - 5A): The full method (5D) improves robustness over the independent baseline (5A)

## 2. How independence was established

- New root `E:\Code\run_test5_independent`; fresh inventory with full-file SHA-256; new grouping; new split (seed 20261009); own caches (`t5_` namespace, provenance + inventory-derived keys); fresh model initialisation; no import of Test 4 code at run time.
- Artifact reads logged: 1140; blocked reads of Test 4 / old Test 5 artifacts: 0; read-only historical documentation reads: 0.
- Cache files without provenance: []; cache keys match current data: {'faceclean': True, 'facedeg': True}; hard/soft links to other runs: [] / [].

## 3. Dataset inventory and exclusions

| dataset | subset | videos | probe_ok | audio_stream | external_wav | median_width | median_height | median_fps | median_duration_s | degraded_available |
|---|---|---|---|---|---|---|---|---|---|---|
| BioDeepAV | fake | 1693 | 1693 | 1693 | 1693 | 512.000 | 512.000 | 25.000 | 10.200 | 5079.000 |
| BioDeepAV | real | 2010 | 2010 | 2010 | 2010 | 168.000 | 168.000 | 25.000 | 2.720 | 54312.000 |
| FF++ | DeepFakeDetection | 1000 | 1000 | 0 | 0 | 1920.000 | 1080.000 | 24.000 | 30.125 | 3000.000 |
| FF++ | Deepfakes | 1000 | 1000 | 0 | 0 | 854.000 | 480.000 | 25.000 | 16.653 | 2864.000 |
| FF++ | Face2Face | 1000 | 1000 | 0 | 0 | 832.000 | 480.000 | 25.000 | 16.640 | 2864.000 |
| FF++ | FaceShifter | 1000 | 1000 | 0 | 0 | 854.000 | 480.000 | 25.000 | 16.640 | 2864.000 |
| FF++ | FaceSwap | 1000 | 1000 | 0 | 0 | 854.000 | 480.000 | 25.000 | 13.833 | 2864.000 |
| FF++ | NeuralTextures | 1000 | 1000 | 0 | 0 | 832.000 | 480.000 | 25.000 | 13.833 | 2864.000 |
| FF++ | original | 1000 | 1000 | 0 | 0 | 854.000 | 480.000 | 25.000 | 16.640 | 2871.000 |

Originals 10703; manifest degraded outputs 79582 (valid link metadata 79582); orphan degraded files 6030; exact duplicate files 2007.

- BioDeepAV fake videos carry no identity/source metadata (UUID names): identity overlap with real videos cannot be verified from metadata and is screened by face-crop and audio duplicate checks instead.
- FF++ manipulation folders: degradation type/severity counts are identical across Deepfakes/Face2Face/FaceShifter/FaceSwap/NeuralTextures (same random assignment per sequence id) - documented, not leakage.

Excluded items by reason: `{"excluded: content link to source not verified": 62, "excluded: face-clip extraction failed": 1}` (details: data_audit/degraded_excluded.csv, data_audit/degraded_link_verification.csv).

## 4. Split and leakage validation

Gate **PASS**; confirmed cross-split leakage **0**.

| check | result |
|---|---|
| source inventory integrity verified | PASS |
| source-group split isolation verified | PASS |
| original/degraded grouping verified | PASS |
| exact cross-split duplicates investigated (none confirmed) | PASS |
| confirmed cross-split source leakage == 0 | PASS |
| unresolved duplicates documented | PASS |
| no Test 4 / old Test 5 caches or checkpoints loaded | PASS |
| fresh Test 5 cache provenance verified | PASS |
| every dataset has >= 20 real and fake videos in val and test | PASS |

Groups 787; content merges before splitting 18604; unmerged near-duplicate candidates 34983 (across splits, unresolved and documented: 2474); within-split exact duplicate sets 344. Exact screens find byte-identical files and identical decoded clips; the dHash/NCC screen finds re-encoded copies with nearly identical framing; it does NOT detect the same person filmed in different videos.

| dataset | subset | test | train | val |
|---|---|---|---|---|
| BioDeepAV | fake | 1006.000 | 453.000 | 233.000 |
| BioDeepAV | real | 414.000 | 1396.000 | 200.000 |
| FF++ | DeepFakeDetection | 838.000 | 63.000 | 99.000 |
| FF++ | Deepfakes | 200.000 | 700.000 | 100.000 |
| FF++ | Face2Face | 200.000 | 700.000 | 100.000 |
| FF++ | FaceShifter | 200.000 | 700.000 | 100.000 |
| FF++ | FaceSwap | 200.000 | 700.000 | 100.000 |
| FF++ | NeuralTextures | 200.000 | 700.000 | 100.000 |
| FF++ | original | 200.000 | 700.000 | 100.000 |

## 5. Preprocessing and model configuration

- Frames: 16 frames uniformly spaced over the decoded video (cv2 grab/retrieve; sequential fallback); model clips use every 2th cached frame (evaluation averages the 2 offsets; training draws 8 random sorted frames).
- Face: YuNet (OpenCV 5.0.0, model sha256 `None…`), margin 0.2, frames normalised to 360 px shorter side, effective face resolution 80 px, output 128 px; degraded crops reuse the source boxes.
- Audio: STFT n_fft 512, hop 256, win 512 Hann, log1p magnitude, bilinear 128x128, per-clip z-score clip +-3 /3; external .wav if present, else ffmpeg audio stream; tiled/cropped to 3 s; silent/no audio -> zero spectrogram + mask 0.
- Model: AudioVisualDeepFakeCNN, 584,978 parameters, PyTorch default initialisation, seeded by the training seed (no pretrained / Test 4 weights).
- Optimiser: AdamW (wd 0.0001), OneCycle; Stage 1 8 epochs lr 0.001; Stage 2 10 epochs lr 0.0003; 32 clips per step.
- Model selection: best epoch by validation score = mean(val original AUC, val degraded AUC) of the training dataset (Stage 1: val original AUC); threshold = validation balanced-accuracy optimum.
- Unavoidable differences: 5C introduces original/degraded pairing because Grad-CAM consistency needs pairs; 5C vs 5B therefore measures pairing + Grad-CAM consistency together; 5D adds feature consistency AND the curriculum; 5D vs 5C measures both together.

## 6. Results for all four variants (in-domain, seed mean ± SD across seeds, n = seeds)


**Clean AUC**

| train_on | test_dataset | 5A | 5B | 5C | 5D | S1 |
|---|---|---|---|---|---|---|
| BioDeepAV | BioDeepAV | 0.966 ± 0.032 (n=3) | 0.981 ± 0.002 (n=3) | 0.952 ± 0.038 (n=3) | 0.967 ± 0.008 (n=3) | 0.880 ± 0.069 (n=3) |
| FF++ | FF++ | 0.770 ± 0.014 (n=3) | 0.755 ± 0.020 (n=3) | 0.744 ± 0.037 (n=3) | 0.736 ± 0.046 (n=3) | 0.704 ± 0.054 (n=3) |

**Degraded AUC (pooled)**

| train_on | test_dataset | 5A | 5B | 5C | 5D | S1 |
|---|---|---|---|---|---|---|
| BioDeepAV | BioDeepAV | 0.939 ± 0.038 (n=3) | 0.962 ± 0.007 (n=3) | 0.922 ± 0.047 (n=3) | 0.903 ± 0.041 (n=3) | 0.795 ± 0.064 (n=3) |
| FF++ | FF++ | 0.705 ± 0.008 (n=3) | 0.688 ± 0.027 (n=3) | 0.694 ± 0.027 (n=3) | 0.704 ± 0.024 (n=3) | 0.631 ± 0.036 (n=3) |

**Worst-case type AUC**

| train_on | test_dataset | 5A | 5B | 5C | 5D | S1 |
|---|---|---|---|---|---|---|
| BioDeepAV | BioDeepAV | 0.858 ± 0.053 (n=3) | 0.894 ± 0.022 (n=3) | 0.840 ± 0.051 (n=3) | 0.759 ± 0.110 (n=3) | 0.558 ± 0.092 (n=3) |
| FF++ | FF++ | 0.570 ± 0.009 (n=3) | 0.565 ± 0.020 (n=3) | 0.559 ± 0.041 (n=3) | 0.533 ± 0.115 (n=3) | 0.441 ± 0.043 (n=3) |

**Prediction flip rate %**

| train_on | test_dataset | 5A | 5B | 5C | 5D | S1 |
|---|---|---|---|---|---|---|
| BioDeepAV | BioDeepAV | 6.461 ± 0.720 (n=3) | 6.575 ± 0.297 (n=3) | 8.701 ± 1.475 (n=3) | 11.122 ± 1.746 (n=3) | 14.246 ± 1.547 (n=3) |
| FF++ | FF++ | 15.097 ± 2.554 (n=3) | 16.134 ± 0.760 (n=3) | 15.664 ± 1.516 (n=3) | 13.037 ± 3.748 (n=3) | 13.369 ± 2.630 (n=3) |

**Hard-degradation AUC**

| train_on | test_dataset | 5A | 5B | 5C | 5D | S1 |
|---|---|---|---|---|---|---|
| BioDeepAV | BioDeepAV | 0.919 ± 0.040 (n=3) | 0.947 ± 0.008 (n=3) | 0.911 ± 0.045 (n=3) | 0.904 ± 0.026 (n=3) | 0.800 ± 0.072 (n=3) |
| FF++ | FF++ | 0.647 ± 0.018 (n=3) | 0.621 ± 0.039 (n=3) | 0.625 ± 0.023 (n=3) | 0.682 ± 0.024 (n=3) | 0.548 ± 0.037 (n=3) |

## 7. Results across seeds

| train_on | variant | metric | n_seeds | seed_42 | seed_123 | seed_2026 |
|---|---|---|---|---|---|---|
| BioDeepAV | 5A | clean_auc | 3 | 0.929 | 0.981 | 0.987 |
| BioDeepAV | 5B | clean_auc | 3 | 0.983 | 0.982 | 0.979 |
| BioDeepAV | 5C | clean_auc | 3 | 0.909 | 0.981 | 0.966 |
| BioDeepAV | 5D | clean_auc | 3 | 0.966 | 0.960 | 0.976 |
| BioDeepAV | S1 | clean_auc | 3 | 0.818 | 0.954 | 0.868 |
| FF++ | 5A | clean_auc | 3 | 0.755 | 0.782 | 0.775 |
| FF++ | 5B | clean_auc | 3 | 0.770 | 0.732 | 0.763 |
| FF++ | 5C | clean_auc | 3 | 0.748 | 0.705 | 0.778 |
| FF++ | 5D | clean_auc | 3 | 0.758 | 0.682 | 0.767 |
| FF++ | S1 | clean_auc | 3 | 0.729 | 0.642 | 0.741 |
| BioDeepAV | 5A | degraded_auc_pooled | 3 | 0.896 | 0.950 | 0.970 |
| BioDeepAV | 5B | degraded_auc_pooled | 3 | 0.968 | 0.963 | 0.954 |
| BioDeepAV | 5C | degraded_auc_pooled | 3 | 0.868 | 0.957 | 0.940 |
| BioDeepAV | 5D | degraded_auc_pooled | 3 | 0.880 | 0.878 | 0.950 |
| BioDeepAV | S1 | degraded_auc_pooled | 3 | 0.749 | 0.867 | 0.768 |
| FF++ | 5A | degraded_auc_pooled | 3 | 0.707 | 0.695 | 0.711 |
| FF++ | 5B | degraded_auc_pooled | 3 | 0.713 | 0.660 | 0.689 |
| FF++ | 5C | degraded_auc_pooled | 3 | 0.686 | 0.672 | 0.723 |
| FF++ | 5D | degraded_auc_pooled | 3 | 0.703 | 0.681 | 0.729 |
| FF++ | S1 | degraded_auc_pooled | 3 | 0.637 | 0.592 | 0.664 |
| BioDeepAV | 5A | mean_top20_iou [hard] | 3 | 0.440 | 0.420 | 0.447 |
| BioDeepAV | 5B | mean_top20_iou [hard] | 3 | 0.429 | 0.501 | 0.440 |
| BioDeepAV | 5C | mean_top20_iou [hard] | 3 | 0.442 | 0.458 | 0.476 |
| BioDeepAV | 5D | mean_top20_iou [hard] | 3 | 0.478 | 0.455 | 0.458 |
| BioDeepAV | S1 | mean_top20_iou [hard] | 3 | 0.428 | 0.450 | 0.532 |
| FF++ | 5A | mean_top20_iou [hard] | 3 | 0.351 | 0.389 | 0.397 |
| FF++ | 5B | mean_top20_iou [hard] | 3 | 0.373 | 0.423 | 0.396 |
| FF++ | 5C | mean_top20_iou [hard] | 3 | 0.608 | 0.645 | 0.632 |
| FF++ | 5D | mean_top20_iou [hard] | 3 | 0.586 | 0.667 | 0.544 |
| FF++ | S1 | mean_top20_iou [hard] | 3 | 0.334 | 0.345 | 0.331 |

## 8. Per-degradation detection robustness

| train_on | degradation | 5A | 5B | 5C | 5D | S1 |
|---|---|---|---|---|---|---|
| BioDeepAV | blur | 0.956 | 0.971 | 0.948 | 0.951 | 0.898 |
| BioDeepAV | brightness | 0.858 | 0.894 | 0.840 | 0.806 | 0.750 |
| BioDeepAV | compression | 0.957 | 0.971 | 0.940 | 0.951 | 0.867 |
| BioDeepAV | contrast | 0.952 | 0.980 | 0.935 | 0.945 | 0.842 |
| BioDeepAV | frame_drop | 0.958 | 0.981 | 0.947 | 0.966 | 0.873 |
| BioDeepAV | motion_blur | 0.952 | 0.976 | 0.945 | 0.955 | 0.864 |
| BioDeepAV | noise | 0.961 | 0.972 | 0.927 | 0.793 | 0.558 |
| BioDeepAV | saturation | 0.926 | 0.947 | 0.931 | 0.939 | 0.792 |
| FF++ | blur | 0.642 | 0.588 | 0.615 | 0.686 | 0.485 |
| FF++ | brightness | 0.674 | 0.603 | 0.628 | 0.678 | 0.561 |
| FF++ | compression | 0.678 | 0.703 | 0.717 | 0.693 | 0.657 |
| FF++ | contrast | 0.570 | 0.602 | 0.597 | 0.662 | 0.498 |
| FF++ | fps_reduction | 0.670 | 0.633 | 0.627 | 0.609 | 0.626 |
| FF++ | frame_drop | 0.788 | 0.768 | 0.776 | 0.741 | 0.726 |
| FF++ | low_resolution | 0.665 | 0.672 | 0.680 | 0.578 | 0.583 |
| FF++ | motion_blur | 0.809 | 0.795 | 0.775 | 0.805 | 0.736 |
| FF++ | noise | 0.756 | 0.724 | 0.740 | 0.720 | 0.690 |
| FF++ | saturation | 0.755 | 0.726 | 0.717 | 0.740 | 0.680 |

Brightness and saturation by severity: figures/fig_brightness_saturation_by_severity.png; full table metrics/degradation_metrics_test5.csv (sample counts per cell included).

## 9. Explanation stability and limitations

| train_on | test_dataset | 5A | 5B | 5C | 5D | S1 |
|---|---|---|---|---|---|---|
| BioDeepAV | BioDeepAV | 0.436 ± 0.014 (n=3) | 0.457 ± 0.039 (n=3) | 0.459 ± 0.017 (n=3) | 0.464 ± 0.013 (n=3) | 0.470 ± 0.055 (n=3) |
| FF++ | FF++ | 0.379 ± 0.025 (n=3) | 0.398 ± 0.025 (n=3) | 0.629 ± 0.019 (n=3) | 0.599 ± 0.063 (n=3) | 0.337 ± 0.008 (n=3) |

All types:

| train_on | test_dataset | 5A | 5B | 5C | 5D | S1 |
|---|---|---|---|---|---|---|
| BioDeepAV | BioDeepAV | 0.602 ± 0.021 (n=3) | 0.604 ± 0.031 (n=3) | 0.614 ± 0.013 (n=3) | 0.620 ± 0.021 (n=3) | 0.625 ± 0.030 (n=3) |
| FF++ | FF++ | 0.614 ± 0.026 (n=3) | 0.627 ± 0.026 (n=3) | 0.775 ± 0.020 (n=3) | 0.750 ± 0.049 (n=3) | 0.574 ± 0.021 (n=3) |

Explanation changed while prediction stable (%):

| train_on | test_dataset | 5A | 5B | 5C | 5D | S1 |
|---|---|---|---|---|---|---|
| BioDeepAV | BioDeepAV | 34.059 ± 1.191 (n=3) | 30.358 ± 5.753 (n=3) | 30.257 ± 1.219 (n=3) | 27.541 ± 2.574 (n=3) | 26.649 ± 5.880 (n=3) |
| FF++ | FF++ | 26.753 ± 4.261 (n=3) | 23.371 ± 4.342 (n=3) | 9.487 ± 2.072 (n=3) | 11.213 ± 3.023 (n=3) | 31.044 ± 4.182 (n=3) |


No ground-truth manipulation-region annotations were found; explanation CORRECTNESS is not established - only stability under degradation is measured. The clean-to-clean reference IoU (same video, different frame subset) in metrics/gradcam_summary_test5.csv gives the stability expected without any degradation.

## 10. Cross-dataset generalization

Clean AUC:

| train_on | test_dataset | 5A | 5B | 5C | 5D | S1 |
|---|---|---|---|---|---|---|
| BioDeepAV | FF++ | 0.542 ± 0.027 (n=3) | 0.514 ± 0.037 (n=3) | 0.490 ± 0.048 (n=3) | 0.498 ± 0.066 (n=3) | 0.477 ± 0.023 (n=3) |
| FF++ | BioDeepAV | 0.500 ± 0.000 (n=3) | 0.500 ± 0.000 (n=3) | 0.500 ± 0.000 (n=3) | 0.500 ± 0.000 (n=3) | 0.500 ± 0.001 (n=3) |

Degraded AUC:

| train_on | test_dataset | 5A | 5B | 5C | 5D | S1 |
|---|---|---|---|---|---|---|
| BioDeepAV | FF++ | 0.517 ± 0.021 (n=3) | 0.491 ± 0.033 (n=3) | 0.475 ± 0.042 (n=3) | 0.477 ± 0.045 (n=3) | 0.466 ± 0.013 (n=3) |
| FF++ | BioDeepAV | 0.500 ± 0.000 (n=3) | 0.500 ± 0.000 (n=3) | 0.500 ± 0.000 (n=3) | 0.500 ± 0.000 (n=3) | 0.501 ± 0.002 (n=3) |


Class-wise errors and prediction distributions (seed means):

| train_on | test_dataset | variant | condition | auc | balanced_accuracy | error_rate_real_fpr | error_rate_fake_fnr | predicted_fake_rate | mean_p_fake_real | mean_p_fake_fake |
|---|---|---|---|---|---|---|---|---|---|---|
| BioDeepAV | FF++ | 5A | clean | 0.542 | 0.514 | 0.815 | 0.158 | 0.840 | 0.385 | 0.440 |
| BioDeepAV | FF++ | 5A | degraded | 0.517 | 0.510 | 0.801 | 0.178 | 0.817 | 0.362 | 0.381 |
| BioDeepAV | FF++ | 5B | clean | 0.514 | 0.500 | 0.715 | 0.286 | 0.714 | 0.281 | 0.306 |
| BioDeepAV | FF++ | 5B | degraded | 0.491 | 0.491 | 0.694 | 0.324 | 0.680 | 0.259 | 0.260 |
| BioDeepAV | FF++ | 5C | clean | 0.490 | 0.498 | 0.767 | 0.238 | 0.762 | 0.416 | 0.399 |
| BioDeepAV | FF++ | 5C | degraded | 0.475 | 0.484 | 0.725 | 0.307 | 0.701 | 0.363 | 0.332 |
| BioDeepAV | FF++ | 5D | clean | 0.498 | 0.504 | 0.752 | 0.240 | 0.759 | 0.408 | 0.405 |
| BioDeepAV | FF++ | 5D | degraded | 0.477 | 0.484 | 0.669 | 0.364 | 0.644 | 0.336 | 0.307 |
| BioDeepAV | FF++ | S1 | clean | 0.477 | 0.486 | 0.858 | 0.169 | 0.834 | 0.436 | 0.414 |
| BioDeepAV | FF++ | S1 | degraded | 0.466 | 0.486 | 0.790 | 0.237 | 0.769 | 0.376 | 0.333 |
| FF++ | BioDeepAV | 5A | ablation_audio_masked | 0.550 | 0.511 | 0.490 | 0.488 | 0.505 | 0.677 | 0.689 |
| FF++ | BioDeepAV | 5A | ablation_audio_only | 0.500 | 0.500 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| FF++ | BioDeepAV | 5A | clean | 0.500 | 0.500 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| FF++ | BioDeepAV | 5A | degraded | 0.500 | 0.500 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| FF++ | BioDeepAV | 5B | ablation_audio_masked | 0.490 | 0.495 | 0.505 | 0.505 | 0.498 | 0.676 | 0.672 |
| FF++ | BioDeepAV | 5B | ablation_audio_only | 0.500 | 0.500 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| FF++ | BioDeepAV | 5B | clean | 0.500 | 0.500 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| FF++ | BioDeepAV | 5B | degraded | 0.500 | 0.500 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| FF++ | BioDeepAV | 5C | ablation_audio_masked | 0.479 | 0.454 | 0.365 | 0.728 | 0.299 | 0.673 | 0.666 |
| FF++ | BioDeepAV | 5C | ablation_audio_only | 0.500 | 0.500 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| FF++ | BioDeepAV | 5C | clean | 0.500 | 0.500 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| FF++ | BioDeepAV | 5C | degraded | 0.500 | 0.500 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| FF++ | BioDeepAV | 5D | ablation_audio_masked | 0.415 | 0.430 | 0.516 | 0.624 | 0.417 | 0.709 | 0.693 |
| FF++ | BioDeepAV | 5D | ablation_audio_only | 0.500 | 0.500 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| FF++ | BioDeepAV | 5D | clean | 0.500 | 0.500 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| FF++ | BioDeepAV | 5D | degraded | 0.500 | 0.500 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| FF++ | BioDeepAV | S1 | ablation_audio_masked | 0.501 | 0.504 | 0.701 | 0.290 | 0.708 | 0.512 | 0.513 |
| FF++ | BioDeepAV | S1 | ablation_audio_only | 0.500 | 0.500 | 0.737 | 0.262 | 0.737 | 0.737 | 0.738 |
| FF++ | BioDeepAV | S1 | clean | 0.500 | 0.500 | 0.737 | 0.262 | 0.737 | 0.737 | 0.738 |
| FF++ | BioDeepAV | S1 | degraded | 0.501 | 0.501 | 0.737 | 0.261 | 0.738 | 0.737 | 0.739 |


Audio masking / modality ablation (BioDeepAV test videos with audio; FF++ has no audio):

| train_on | test_dataset | variant | condition | auc | auc_normal_same_videos | flip_rate_vs_normal_pct |
|---|---|---|---|---|---|---|
| BioDeepAV | BioDeepAV | 5A | ablation_audio_masked | 0.948 | 0.966 | 5.540 |
| BioDeepAV | BioDeepAV | 5A | ablation_audio_only | 0.588 | 0.966 | 15.235 |
| BioDeepAV | BioDeepAV | 5B | ablation_audio_masked | 0.976 | 0.981 | 5.376 |
| BioDeepAV | BioDeepAV | 5B | ablation_audio_only | 0.661 | 0.981 | 20.587 |
| BioDeepAV | BioDeepAV | 5C | ablation_audio_masked | 0.935 | 0.952 | 6.761 |
| BioDeepAV | BioDeepAV | 5C | ablation_audio_only | 0.595 | 0.952 | 18.732 |
| BioDeepAV | BioDeepAV | 5D | ablation_audio_masked | 0.931 | 0.967 | 3.357 |
| BioDeepAV | BioDeepAV | 5D | ablation_audio_only | 0.595 | 0.967 | 20.892 |
| BioDeepAV | BioDeepAV | S1 | ablation_audio_masked | 0.867 | 0.880 | 9.554 |
| BioDeepAV | BioDeepAV | S1 | ablation_audio_only | 0.540 | 0.880 | 21.831 |
| FF++ | BioDeepAV | 5A | ablation_audio_masked | 0.550 | 0.500 | 49.460 |
| FF++ | BioDeepAV | 5A | ablation_audio_only | 0.500 | 0.500 | 0.000 |
| FF++ | BioDeepAV | 5B | ablation_audio_masked | 0.490 | 0.500 | 50.188 |
| FF++ | BioDeepAV | 5B | ablation_audio_only | 0.500 | 0.500 | 0.000 |
| FF++ | BioDeepAV | 5C | ablation_audio_masked | 0.479 | 0.500 | 70.094 |
| FF++ | BioDeepAV | 5C | ablation_audio_only | 0.500 | 0.500 | 0.000 |
| FF++ | BioDeepAV | 5D | ablation_audio_masked | 0.415 | 0.500 | 58.286 |
| FF++ | BioDeepAV | 5D | ablation_audio_only | 0.500 | 0.500 | 0.000 |
| FF++ | BioDeepAV | S1 | ablation_audio_masked | 0.501 | 0.500 | 36.033 |
| FF++ | BioDeepAV | S1 | ablation_audio_only | 0.500 | 0.500 | 0.000 |


Cross-dataset generalisation is claimed only where the cross-dataset AUC is clearly above 0.5 across seeds.

## 11. Statistical uncertainty

Sample uncertainty: cluster bootstrap over source groups (paired: same test videos and replicates for both models) of the seed-averaged difference. Seed uncertainty: per-seed differences, their SD and sign agreement. A difference is called 'supported' only if the 95 % CI excludes 0 AND all seeds agree in sign (>= 2 seeds).

| train_on | comparison | metric | n_seeds | mean_diff | sd_diff_across_seeds | ci95_low | ci95_high | verdict |
|---|---|---|---|---|---|---|---|---|
| BioDeepAV | 5B - 5A | clean_auc | 3 | 0.0158 | 0.0340 | -0.0045 | 0.0440 | inconclusive |
| BioDeepAV | 5B - 5A | degraded_auc | 3 | 0.0228 | 0.0449 | 0.0008 | 0.0446 | inconclusive |
| BioDeepAV | 5B - 5A | hard_auc | 3 | 0.0284 | 0.0474 | -0.0022 | 0.0477 | inconclusive |
| BioDeepAV | 5B - 5A | blur_auc | 3 | 0.0158 | 0.0209 | -0.0052 | 0.0439 | inconclusive |
| BioDeepAV | 5B - 5A | brightness_auc | 3 | 0.0358 | 0.0450 | -0.0195 | 0.0741 | inconclusive |
| BioDeepAV | 5B - 5A | contrast_auc | 3 | 0.0285 | 0.0551 | -0.0025 | 0.0397 | inconclusive |
| BioDeepAV | 5B - 5A | saturation_auc | 3 | 0.0208 | 0.0261 | -0.0020 | 0.0340 | inconclusive |
| BioDeepAV | 5B - 5A | worst_type_auc | 3 | 0.0358 | 0.0450 | -0.0180 | 0.0635 | inconclusive |
| BioDeepAV | 5B - 5A | flip_rate_pct | 3 | 0.1145 | 0.9035 | -2.2391 | 1.0585 | inconclusive |
| BioDeepAV | 5B - 5A | gradcam_iou_all | 3 | 0.0021 | 0.0484 | -0.0522 | 0.0292 | inconclusive |
| BioDeepAV | 5B - 5A | gradcam_iou_hard | 3 | 0.0209 | 0.0522 | -0.0487 | 0.0449 | inconclusive |
| BioDeepAV | 5B - 5A | gradcam_pearson_hard | 3 | 0.0122 | 0.0472 | -0.0598 | 0.0343 | inconclusive |
| BioDeepAV | 5C - 5B | clean_auc | 3 | -0.0294 | 0.0392 | -0.0687 | 0.0008 | inconclusive |
| BioDeepAV | 5C - 5B | degraded_auc | 3 | -0.0399 | 0.0523 | -0.0638 | -0.0010 | supported deterioration |
| BioDeepAV | 5C - 5B | hard_auc | 3 | -0.0365 | 0.0481 | -0.0576 | 0.0038 | inconclusive |
| BioDeepAV | 5C - 5B | blur_auc | 3 | -0.0236 | 0.0182 | -0.0548 | 0.0107 | inconclusive |
| BioDeepAV | 5C - 5B | brightness_auc | 3 | -0.0543 | 0.0318 | -0.0858 | 0.0076 | inconclusive |
| BioDeepAV | 5C - 5B | contrast_auc | 3 | -0.0457 | 0.0761 | -0.0672 | 0.0019 | inconclusive |
| BioDeepAV | 5C - 5B | saturation_auc | 3 | -0.0164 | 0.0368 | -0.0383 | 0.0141 | inconclusive |
| BioDeepAV | 5C - 5B | worst_type_auc | 3 | -0.0543 | 0.0318 | -0.0904 | 0.0242 | inconclusive |
| BioDeepAV | 5C - 5B | flip_rate_pct | 3 | 2.1263 | 1.7104 | -0.7660 | 5.4967 | inconclusive |
| BioDeepAV | 5C - 5B | gradcam_iou_all | 3 | 0.0101 | 0.0377 | -0.0173 | 0.0571 | inconclusive |
| BioDeepAV | 5C - 5B | gradcam_iou_hard | 3 | 0.0020 | 0.0411 | -0.0235 | 0.0630 | inconclusive |
| BioDeepAV | 5C - 5B | gradcam_pearson_hard | 3 | 0.0274 | 0.0588 | -0.0096 | 0.0921 | inconclusive |
| BioDeepAV | 5D - 5C | clean_auc | 3 | 0.0150 | 0.0392 | -0.0135 | 0.0309 | inconclusive |
| BioDeepAV | 5D - 5C | degraded_auc | 3 | -0.0190 | 0.0520 | -0.0351 | -0.0051 | inconclusive |
| BioDeepAV | 5D - 5C | hard_auc | 3 | -0.0066 | 0.0422 | -0.0456 | 0.0082 | inconclusive |
| BioDeepAV | 5D - 5C | blur_auc | 3 | 0.0031 | 0.0310 | -0.0130 | 0.0129 | inconclusive |
| BioDeepAV | 5D - 5C | brightness_auc | 3 | -0.0334 | 0.0329 | -0.1112 | 0.0017 | inconclusive |
| BioDeepAV | 5D - 5C | contrast_auc | 3 | 0.0100 | 0.0834 | -0.0469 | 0.0246 | inconclusive |
| BioDeepAV | 5D - 5C | saturation_auc | 3 | 0.0084 | 0.0212 | -0.0381 | 0.0390 | inconclusive |
| BioDeepAV | 5D - 5C | worst_type_auc | 3 | -0.0806 | 0.1037 | -0.1136 | -0.0198 | inconclusive |
| BioDeepAV | 5D - 5C | flip_rate_pct | 3 | 2.4207 | 2.4110 | -6.0118 | 5.1632 | inconclusive |
| BioDeepAV | 5D - 5C | gradcam_iou_all | 3 | 0.0060 | 0.0085 | -0.0090 | 0.0133 | inconclusive |
| BioDeepAV | 5D - 5C | gradcam_iou_hard | 3 | 0.0054 | 0.0279 | -0.0111 | 0.0188 | inconclusive |
| BioDeepAV | 5D - 5C | gradcam_pearson_hard | 3 | -0.0328 | 0.0379 | -0.0716 | -0.0024 | supported deterioration |
| BioDeepAV | 5D - 5A | clean_auc | 3 | 0.0014 | 0.0314 | -0.0237 | 0.0222 | inconclusive |
| BioDeepAV | 5D - 5A | degraded_auc | 3 | -0.0362 | 0.0317 | -0.0572 | -0.0035 | supported deterioration |
| BioDeepAV | 5D - 5A | hard_auc | 3 | -0.0147 | 0.0227 | -0.0635 | 0.0113 | inconclusive |
| BioDeepAV | 5D - 5A | blur_auc | 3 | -0.0047 | 0.0122 | -0.0327 | 0.0199 | inconclusive |
| BioDeepAV | 5D - 5A | brightness_auc | 3 | -0.0519 | 0.0248 | -0.1469 | 0.0018 | inconclusive |
| BioDeepAV | 5D - 5A | contrast_auc | 3 | -0.0072 | 0.0651 | -0.0514 | 0.0144 | inconclusive |
| BioDeepAV | 5D - 5A | saturation_auc | 3 | 0.0129 | 0.0117 | -0.0504 | 0.0574 | inconclusive |
| BioDeepAV | 5D - 5A | worst_type_auc | 3 | -0.0991 | 0.0745 | -0.1399 | -0.0069 | supported deterioration |
| BioDeepAV | 5D - 5A | flip_rate_pct | 3 | 4.6614 | 1.0304 | -2.0099 | 6.9733 | inconclusive |
| BioDeepAV | 5D - 5A | gradcam_iou_all | 3 | 0.0181 | 0.0127 | -0.0071 | 0.0244 | inconclusive |
| BioDeepAV | 5D - 5A | gradcam_iou_hard | 3 | 0.0283 | 0.0149 | -0.0072 | 0.0423 | inconclusive |
| BioDeepAV | 5D - 5A | gradcam_pearson_hard | 3 | 0.0068 | 0.0137 | -0.0288 | 0.0230 | inconclusive |
| FF++ | 5B - 5A | clean_auc | 3 | -0.0151 | 0.0325 | -0.0320 | -0.0040 | inconclusive |
| FF++ | 5B - 5A | degraded_auc | 3 | -0.0171 | 0.0214 | -0.0346 | -0.0038 | inconclusive |
| FF++ | 5B - 5A | hard_auc | 3 | -0.0258 | 0.0233 | -0.0517 | -0.0055 | inconclusive |
| FF++ | 5B - 5A | blur_auc | 3 | -0.0544 | 0.0375 | -0.1139 | 0.0665 | inconclusive |
| FF++ | 5B - 5A | brightness_auc | 3 | -0.0708 | 0.0205 | -0.1184 | -0.0203 | supported deterioration |
| FF++ | 5B - 5A | contrast_auc | 3 | 0.0321 | 0.0359 | -0.0370 | 0.0853 | inconclusive |
| FF++ | 5B - 5A | saturation_auc | 3 | -0.0292 | 0.0672 | -0.0939 | 0.0030 | inconclusive |
| FF++ | 5B - 5A | worst_type_auc | 3 | -0.0045 | 0.0284 | -0.0620 | 0.0417 | inconclusive |
| FF++ | 5B - 5A | flip_rate_pct | 3 | 1.0369 | 1.8348 | -0.0413 | 1.9475 | inconclusive |
| FF++ | 5B - 5A | gradcam_iou_all | 3 | 0.0136 | 0.0184 | 0.0007 | 0.0256 | inconclusive |
| FF++ | 5B - 5A | gradcam_iou_hard | 3 | 0.0182 | 0.0177 | -0.0056 | 0.0346 | inconclusive |
| FF++ | 5B - 5A | gradcam_pearson_hard | 3 | 0.0169 | 0.0453 | -0.0370 | 0.0508 | inconclusive |
| FF++ | 5C - 5B | clean_auc | 3 | -0.0114 | 0.0232 | -0.0187 | -0.0034 | inconclusive |
| FF++ | 5C - 5B | degraded_auc | 3 | 0.0061 | 0.0314 | -0.0040 | 0.0155 | inconclusive |
| FF++ | 5C - 5B | hard_auc | 3 | 0.0042 | 0.0492 | -0.0166 | 0.0215 | inconclusive |
| FF++ | 5C - 5B | blur_auc | 3 | 0.0270 | 0.0430 | -0.0438 | 0.0693 | inconclusive |
| FF++ | 5C - 5B | brightness_auc | 3 | 0.0244 | 0.0445 | -0.0347 | 0.0671 | inconclusive |
| FF++ | 5C - 5B | contrast_auc | 3 | -0.0049 | 0.0520 | -0.0538 | 0.0284 | inconclusive |
| FF++ | 5C - 5B | saturation_auc | 3 | -0.0088 | 0.0347 | -0.0435 | 0.0594 | inconclusive |
| FF++ | 5C - 5B | worst_type_auc | 3 | -0.0058 | 0.0592 | -0.0529 | 0.0401 | inconclusive |
| FF++ | 5C - 5B | flip_rate_pct | 3 | -0.4701 | 2.2719 | -2.6516 | 0.6793 | inconclusive |
| FF++ | 5C - 5B | gradcam_iou_all | 3 | 0.1477 | 0.0144 | 0.1312 | 0.1626 | supported improvement |
| FF++ | 5C - 5B | gradcam_iou_hard | 3 | 0.2310 | 0.0076 | 0.2079 | 0.2556 | supported improvement |
| FF++ | 5C - 5B | gradcam_pearson_hard | 3 | 0.3100 | 0.0432 | 0.2758 | 0.3699 | supported improvement |
| FF++ | 5D - 5C | clean_auc | 3 | -0.0081 | 0.0163 | -0.0471 | 0.0144 | inconclusive |
| FF++ | 5D - 5C | degraded_auc | 3 | 0.0104 | 0.0060 | -0.0365 | 0.0363 | inconclusive |
| FF++ | 5D - 5C | hard_auc | 3 | 0.0561 | 0.0294 | -0.0076 | 0.0888 | inconclusive |
| FF++ | 5D - 5C | blur_auc | 3 | 0.0709 | 0.0528 | -0.0654 | 0.1283 | inconclusive |
| FF++ | 5D - 5C | brightness_auc | 3 | 0.0499 | 0.0452 | -0.0605 | 0.1034 | inconclusive |
| FF++ | 5D - 5C | contrast_auc | 3 | 0.0651 | 0.0367 | -0.0102 | 0.1106 | inconclusive |
| FF++ | 5D - 5C | saturation_auc | 3 | 0.0226 | 0.0280 | -0.0310 | 0.0579 | inconclusive |
| FF++ | 5D - 5C | worst_type_auc | 3 | -0.0267 | 0.1214 | -0.0874 | 0.0328 | inconclusive |
| FF++ | 5D - 5C | flip_rate_pct | 3 | -2.6268 | 2.5083 | -3.7810 | -0.4585 | supported improvement |
| FF++ | 5D - 5C | gradcam_iou_all | 3 | -0.0255 | 0.0299 | -0.0366 | -0.0116 | inconclusive |
| FF++ | 5D - 5C | gradcam_iou_hard | 3 | -0.0295 | 0.0555 | -0.0440 | -0.0072 | inconclusive |
| FF++ | 5D - 5C | gradcam_pearson_hard | 3 | -0.0103 | 0.0592 | -0.0297 | 0.0096 | inconclusive |
| FF++ | 5D - 5A | clean_auc | 3 | -0.0347 | 0.0561 | -0.0862 | -0.0060 | inconclusive |
| FF++ | 5D - 5A | degraded_auc | 3 | -0.0006 | 0.0161 | -0.0609 | 0.0341 | inconclusive |
| FF++ | 5D - 5A | hard_auc | 3 | 0.0344 | 0.0408 | -0.0429 | 0.0721 | inconclusive |
| FF++ | 5D - 5A | blur_auc | 3 | 0.0436 | 0.0643 | -0.0785 | 0.1026 | inconclusive |
| FF++ | 5D - 5A | brightness_auc | 3 | 0.0035 | 0.0347 | -0.1191 | 0.0716 | inconclusive |
| FF++ | 5D - 5A | contrast_auc | 3 | 0.0923 | 0.0646 | -0.0317 | 0.1602 | inconclusive |
| FF++ | 5D - 5A | saturation_auc | 3 | -0.0154 | 0.0550 | -0.0788 | 0.0358 | inconclusive |
| FF++ | 5D - 5A | worst_type_auc | 3 | -0.0370 | 0.1133 | -0.1050 | 0.0224 | inconclusive |
| FF++ | 5D - 5A | flip_rate_pct | 3 | -2.0600 | 6.2985 | -3.4602 | -1.0478 | inconclusive |
| FF++ | 5D - 5A | gradcam_iou_all | 3 | 0.1358 | 0.0423 | 0.1226 | 0.1536 | supported improvement |
| FF++ | 5D - 5A | gradcam_iou_hard | 3 | 0.2197 | 0.0668 | 0.1964 | 0.2435 | supported improvement |
| FF++ | 5D - 5A | gradcam_pearson_hard | 3 | 0.3167 | 0.0412 | 0.2784 | 0.3518 | supported improvement |

## 12. Failure cases

| train_on | variant | worst_type | worst_type_auc | worst_type_severity | flip_rate |
|---|---|---|---|---|---|
| BioDeepAV | 5A | brightness | 0.858 | brightness/severe | 6.461 |
| BioDeepAV | 5B | brightness | 0.894 | brightness/severe | 6.575 |
| BioDeepAV | 5C | brightness | 0.840 | brightness/severe | 8.701 |
| BioDeepAV | 5D | brightness | 0.759 | noise/severe | 11.122 |
| BioDeepAV | S1 | noise | 0.558 | noise/severe | 14.246 |
| FF++ | 5A | contrast | 0.570 | fps_reduction/mild | 15.097 |
| FF++ | 5B | brightness | 0.565 | brightness/severe | 16.134 |
| FF++ | 5C | fps_reduction | 0.559 | brightness/severe | 15.664 |
| FF++ | 5D | low_resolution | 0.533 | low_resolution/medium | 13.037 |
| FF++ | S1 | blur | 0.441 | low_resolution/medium | 13.369 |

Per-video flips: metrics/predictions/*.csv.gz; Grad-CAM pairs with changed explanations: metrics/gradcam_stability_test5.csv.

## 13. Comparison with the historical Test 4 results

**Different evaluation splits** (Test 4: seed 42, its own grouping; Test 5: seed 20261009 with stricter DFD actor grouping and content-duplicate merging). This is NOT a controlled paired comparison; the Test 4 model was never evaluated on the Test 5 test set (its training data overlaps that test set).


## 14. Hypothesis verdicts

| hypothesis | comparison | verdict | evidence |
|---|---|---|---|
| H1 | 5B - 5A | inconclusive / not supported | BioDeepAV hard_auc: +0.0284 [-0.0022, 0.0477] (inconclusive, seeds 3); BioDeepAV brightness_auc: +0.0358 [-0.0195, 0.0741] (inconclusive, seeds 3); BioDeepAV saturation_auc: +0.0208 [-0.0020, 0.0340] (inconclusive, seeds 3); FF++ hard_auc: -0.0258 [-0.0517, -0.0055] (inconclusive, seeds 3); FF++ brightness_auc: -0.0708 [-0.1184, -0.0203] (supported deterioration, seeds 3); FF++ saturation_auc: -0.0292 [-0.0939, 0.0030] (inconclusive, seeds 3) |
| H2 | 5C - 5B | partially supported (some training sets) | BioDeepAV gradcam_iou_all: +0.0101 [-0.0173, 0.0571] (inconclusive, seeds 3); BioDeepAV gradcam_iou_hard: +0.0020 [-0.0235, 0.0630] (inconclusive, seeds 3); FF++ gradcam_iou_all: +0.1477 [0.1312, 0.1626] (supported improvement, seeds 3); FF++ gradcam_iou_hard: +0.2310 [0.2079, 0.2556] (supported improvement, seeds 3) |
| H3 | 5D - 5C | inconclusive / not supported | BioDeepAV degraded_auc: -0.0190 [-0.0351, -0.0051] (inconclusive, seeds 3); BioDeepAV flip_rate_pct: +2.4207 [-6.0118, 5.1632] (inconclusive, seeds 3); BioDeepAV gradcam_iou_hard: +0.0054 [-0.0111, 0.0188] (inconclusive, seeds 3); FF++ degraded_auc: +0.0104 [-0.0365, 0.0363] (inconclusive, seeds 3); FF++ flip_rate_pct: -2.6268 [-3.7810, -0.4585] (supported improvement, seeds 3); FF++ gradcam_iou_hard: -0.0295 [-0.0440, -0.0072] (inconclusive, seeds 3) |
| H4 | 5D - 5A | not supported (statistically supported deterioration) | BioDeepAV degraded_auc: -0.0362 [-0.0572, -0.0035] (supported deterioration, seeds 3); BioDeepAV hard_auc: -0.0147 [-0.0635, 0.0113] (inconclusive, seeds 3); BioDeepAV worst_type_auc: -0.0991 [-0.1399, -0.0069] (supported deterioration, seeds 3); BioDeepAV flip_rate_pct: +4.6614 [-2.0099, 6.9733] (inconclusive, seeds 3); BioDeepAV gradcam_iou_hard: +0.0283 [-0.0072, 0.0423] (inconclusive, seeds 3); FF++ degraded_auc: -0.0006 [-0.0609, 0.0341] (inconclusive, seeds 3); FF++ hard_auc: +0.0344 [-0.0429, 0.0721] (inconclusive, seeds 3); FF++ worst_type_auc: -0.0370 [-0.1050, 0.0224] (inconclusive, seeds 3); FF++ flip_rate_pct: -2.0600 [-3.4602, -1.0478] (inconclusive, seeds 3); FF++ gradcam_iou_hard: +0.2197 [0.1964, 0.2435] (supported improvement, seeds 3) |

## 15. Recommended thesis conclusions

- Report only effects marked 'supported' above; describe the rest as inconclusive at this sample size and seed count.
- Explanation stability is not explanation correctness: no localisation ground truth exists in this data.
- If 5B-5D do not beat 5A, state that hard-degradation weighting did not improve robustness under this protocol.
- Cross-dataset numbers indicate whether the detector transfers; do not claim generalisation where AUC ≈ 0.5.
- Test 4 vs Test 5 differences mix method and split changes and must be described as such.


---
Files: metrics_test5.csv, degradation_metrics_test5.csv, robustness_test5.csv, gradcam_stability_test5.csv, gradcam_summary_test5.csv, cross_dataset_test5.csv, seed_comparison.csv, statistics_test5.csv (metrics/); checkpoint hashes (checkpoints/checkpoint_hashes.csv); configs/experiment_config_5A-5D.json; reports/leakage_audit_report.json; splits/split_integrity_report.json.
