# Test 5 - Thesis Section 5.6

Independent replication with hard-degradation-aware training (variants 5A-5D, 3 seeds, strict leakage audit).

- Code: `code/9-10-26_test5_independent.py`
- Results: `results/`

Note: `code/earlier_version_8-10-26_TEST5_INDEPENDENT.py` is the first version. The face/audio caches (about 26 GB), the checkpoints and the per-run prediction files are not included.

Results layout (only the folders that exist for this test):

- `configs/`: experiment configurations
- `metrics/`: result tables (CSV/JSON) used in the thesis
- `figures/`, `xai/`: figures and Grad-CAM examples
- `logs/`: run logs, training histories (`history_*.csv`) and sampling weights (`sampling_*.csv`)
- `reports/`: the run report (`*.md`) and summary JSON files
