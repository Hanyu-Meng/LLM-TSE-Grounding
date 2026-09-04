# Internal-to-public code map

This repository is a deliberately compact extraction of the research code used
on the lab workstation. Machine orchestration and generated artifacts were not
copied. The public modules preserve the following method logic:

| Internal campaign source | Public module | Preserved behavior |
|---|---|---|
| `se_align/train/fsq_neighbors.py` | `fsq.py` | 3^8 ID factorization and Hamming geometry |
| `scripts/tse_candidate_gate/aggregate_candidate_gate.py` | `candidate_selection.py` | frozen Pool-D order and enrollment-cosine selection |
| `scripts/tse_noisy_wham/decode_noisy_grounding.py` | `csg.py` | per-step Hamming penalty and temporal tolerance |
| `scripts/tse_noisy_wham/decode_noisy_grounding.py` | `gnr.py` | Top-K/Hamming-ball/anchor intersection and immutable history contract |
| `scripts/tse_noisy_wham/build_noisy_difficulty.py` | `difficulty.py` | deployable residual-ratio feature and source thresholds |
| `scripts/tse_noisy_wham/paired_noisy_statistics.py` | `statistics.py` | paired bootstrap convention |

The public implementation uses NumPy so its invariants can be audited on CPU.
It does not include private absolute paths, scheduler wrappers, cached metrics,
data manifests, model checkpoints, audio, or third-party repositories.
