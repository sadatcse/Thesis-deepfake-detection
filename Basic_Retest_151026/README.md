# Basic retest - Thesis Section 5.7

Basic retest: real / fake video detector with explainable AI (single seed, Grad-CAM faithfulness).

- Code: `code/151026.py`
- Results: `results/`

Note: The face cache (18.9 GB) and the checkpoints are not included.

Results layout (only the folders that exist for this test):

- `configs/`: experiment configurations
- `metrics/`: result tables (CSV/JSON) used in the thesis
- `figures/`, `xai/`: figures and Grad-CAM examples
- `logs/`: run logs, training histories (`history_*.csv`) and sampling weights (`sampling_*.csv`)
- `reports/`: the run report (`*.md`) and summary JSON files
