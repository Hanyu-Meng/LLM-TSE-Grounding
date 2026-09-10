# Compact-to-full code map

The repository contains two complementary implementation layers:

- `src/llm_tse_grounding/`: dependency-light NumPy operators for auditing core
  method invariants;
- `se_align/` and `scripts/`: the curated Torch/CosyVoice/WeSep research
  pipeline used to generate and evaluate the paper systems.

Machine orchestration, unrelated SE utilities, and generated artifacts are not
copied. The public names match the manuscript: CDCS-2, CDCS-5 direct, Qwen-TSE
UD, Qwen-TSE fixed CSG, and Qwen-TSE GNR. The mapping is:

| Full research implementation | Compact module | Preserved behavior |
|---|---|---|
| `se_align/train/fsq_neighbors.py` | `fsq.py` | 3^8 ID factorization and Hamming geometry |
| `scripts/tse_candidate_gate/aggregate_candidate_gate.py` | `candidate_selection.py` | frozen CDCS-5 order and enrollment-cosine selection |
| `scripts/tse_noisy_wham/decode_noisy_grounding.py` | `csg.py` | per-step Hamming penalty and temporal tolerance |
| `scripts/tse_noisy_wham/decode_noisy_grounding.py` | `gnr.py` | Top-K/Hamming-ball/anchor intersection and immutable history contract |
| `scripts/tse_noisy_wham/build_noisy_difficulty.py` | `difficulty.py` | deployable residual-ratio feature and source thresholds |
| `scripts/tse_noisy_wham/paired_noisy_statistics.py` | `statistics.py` | paired bootstrap convention |

The compact implementation uses NumPy so its invariants can be audited on CPU.
The full implementation remains under its original module paths so campaign
scripts and imports stay traceable. Neither layer includes private absolute
paths, scheduler wrappers, cached metrics, data manifests, model checkpoints,
bulk audio, or third-party repositories.

See `PIPELINE.md` for the stage-by-stage entry points.
