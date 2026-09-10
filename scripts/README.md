# Pipeline scripts

The scripts are grouped by research stage:

- `tse/`: manifest preparation, preprocessing, training, decoding, and base
  evaluation;
- `tse_candidate_gate/`: complementary WeSep candidate generation and
  enrollment-consistency selection;
- `tse_selected_evidence/`: clean selected-evidence generation, evaluation,
  validation, and table compilation;
- `tse_noisy_wham/`: Natural-WHAM!/controlled-SNR preparation, grounding,
  evaluation, statistics, reporting, and TEST freeze checks;
- `resource_guard.py`: shared CPU, RAM, disk, and GPU safety checks.

Cluster-specific `screen` launchers and historical experiment wrappers are not
included. Invoke the Python entry points explicitly and inspect `--help` before
using a new manifest or output directory. See `docs/PIPELINE.md` for the full
stage map.
