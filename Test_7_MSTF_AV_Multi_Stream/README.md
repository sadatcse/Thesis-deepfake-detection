# Test 7 - Thesis Section 5.10

MSTF-AV multi-stream audio-visual detector with consistency, Group-DRO and shortcut-adversarial training. The script and its files call this experiment "Test 6" (variants 6A-6D).

- Code: `code/161026.py`
- Results: `results/`

Note: 45 checkpoints and the per-run prediction / Grad-CAM files are not included. `checkpoint_hashes.csv` lists the hash of every checkpoint.

Results layout (only the folders that exist for this test):

- `configs/`: experiment configurations
- `metrics/`: result tables (CSV/JSON) used in the thesis
- `figures/`, `xai/`: figures and Grad-CAM examples
- `logs/`: run logs, training histories (`history_*.csv`) and sampling weights (`sampling_*.csv`)
- `reports/`: the run report (`*.md`) and summary JSON files
