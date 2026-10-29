# Final Test - Thesis Section 5.13

Final test: pretrained (EfficientNet-B0 + temporal transformer), degradation-aware and explanation-consistent video detector with calibrated decisions (variants F0-F2, Joint / BioDeepAV / FF++, 3 seeds).

- Code: `code/FINALTEST.py`
- Results: `results/`

Note: `results/final_model/FINAL_MODEL.pt` is the frozen final detector (see `MODEL_CARD.md`). The 27 training checkpoints, the per-run prediction / Grad-CAM / faithfulness files and the two 9 MB training logs are not included.

Results layout (only the folders that exist for this test):

- `configs/`: experiment configurations
- `metrics/`: result tables (CSV/JSON) used in the thesis
- `figures/`, `xai/`: figures and Grad-CAM examples
- `logs/`: run logs, training histories (`history_*.csv`) and sampling weights (`sampling_*.csv`)
- `reports/`: the run report (`*.md`) and summary JSON files
