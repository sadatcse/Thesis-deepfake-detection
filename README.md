# Deepfake Video Detection Under Degradation - Thesis Code and Results

This repository holds the code and result files for every experiment in Chapter 5 of the thesis
**[Thesis_LitReview_and_Experiments_Final_Test.pdf](Thesis_LitReview_and_Experiments_Final_Test.pdf)**.
Each test has its own folder, named as in the thesis. Each folder contains `code/` (the script that was run) and, where the files still exist, `results/` (the output folder of that run).

## Tests

| Test | Thesis section | Folder | Script | What it tests |
|---|---|---|---|---|
| Test 1 | 5.2 | [`Test_1_Mixed_Degradation_BioDeepAV`](Test_1_Mixed_Degradation_BioDeepAV/) | `deepfake_biodeepav_mixed_degradation_av.py` | Mixed degradation training on BioDeepAV |
| Test 2 | 5.3 | [`Test_2_Two_Stage_Audio_Visual`](Test_2_Two_Stage_Audio_Visual/) | `30-9-26test.py` | Two-stage audio-visual training on BioDeepAV + FF++ |
| Test 3 | 5.4 | [`Test_3_Face_Crop_Balanced_Calibrated`](Test_3_Face_Crop_Balanced_Calibrated/) | `1-10-26.py` | Face-crop, balanced, threshold-calibrated detector |
| Test 4 | 5.5 | [`Test_4_Shortcut_Resistant_Explanation_Consistent`](Test_4_Shortcut_Resistant_Explanation_Consistent/) | `6-10-26.py` | Shortcut-resistant, explanation-consistent detector |
| Test 5 | 5.6 | [`Test_5_Independent_Replication`](Test_5_Independent_Replication/) | `9-10-26_test5_independent.py` | Independent replication with hard-degradation-aware training (5A-5D, 3 seeds) |
| Basic retest | 5.7 | [`Basic_Retest_151026`](Basic_Retest_151026/) | `151026.py` | Real / fake detector with explainable AI |
| Improvement suite | 5.8 | [`Improvement_Suite_171026`](Improvement_Suite_171026/) | `171026.py` | Reproduction, seeds, sampling, training length, calibration, XAI faithfulness |
| Test 6 | 5.9 | [`Test_6_Adaptive_Group_DRO_harddro20`](Test_6_Adaptive_Group_DRO_harddro20/) | `201026.py` | Adaptive Group-DRO sampling, 20 epochs (harddro20) |
| Test 7 | 5.10 | [`Test_7_MSTF_AV_Multi_Stream`](Test_7_MSTF_AV_Multi_Stream/) | `161026.py` | MSTF-AV multi-stream audio-visual detector (files say "Test 6", variants 6A-6D) |
| Final Test | 5.13 | [`Final_Test`](Final_Test/) | `FINALTEST.py` | Pretrained, degradation-aware, explanation-consistent detector with calibrated decisions |

The comparison of tests at the end of Chapter 5 draws on the folders above. Appendix A of the thesis lists the source file for every number.

## Datasets

The experiments use **BioDeepAV** and **FaceForensics++ (C23)**, plus degraded copies made with the DeepFake Bulk Video Degradation Tool (thesis Section 5.0). The datasets are **not** in this repository. Get them from their original providers under their licences. The scripts expect them under `E:\Datasets` by default; most scripts take the dataset and output paths as command-line options (run a script with `--help`).

## Environment

The runs used Windows 11, Python 3.13, PyTorch 2.14 (CUDA 12.6) and an NVIDIA RTX 3050 (6 GB). See `requirements.txt` for the main packages; the full `pip freeze` of the Test 6 run is in `Test_6_Adaptive_Group_DRO_harddro20/results/reports/pip_freeze.txt`.

```bash
pip install -r requirements.txt
# final test: self-test, then the full run
python Final_Test/code/FINALTEST.py --phase selftest
python Final_Test/code/FINALTEST.py --phase run
# predict with the frozen final model
python Final_Test/code/FINALTEST.py --predict path/to/video_or_folder
```

The Final Test reads the prepared Test 5 split and caches (`--t5-root`) and imports the Test 5 preprocessing (`--t5-script`), so Test 5 must be prepared first.

## Not included (size)

These files are excluded to keep the repository small (see `.gitignore`); the originals stay in the local run folders:

- frame / face / audio caches (`*.npy`, about 45 GB in total)
- training checkpoints (`*.pt`), **except** the frozen final model `Final_Test/results/final_model/FINAL_MODEL.pt` (20 MB)
- per-run raw predictions (`metrics/predictions/*.csv.gz`) and per-run Grad-CAM / faithfulness tables; the combined tables used in the thesis are included
- run logs larger than 5 MB, the data-audit inventories and earlier draft versions of the thesis PDF

The SHA-256 hashes of all checkpoints are kept in each `results/checkpoints/checkpoint_hashes.csv`.
