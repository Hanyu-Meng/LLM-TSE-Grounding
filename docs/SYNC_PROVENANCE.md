# Source synchronization provenance

Synchronization date: 2026-09-10

Source snapshot: the private LLM-TSE research project on lab5090.

Control path used for the read-only source audit:

```text
Mac Air -> jollibear -> lab5090
```

## Verification

The curated synchronization compared 95 source/config/test files by SHA-256:

- 91 files are byte-identical to the lab5090 source;
- 3 YAML files differ only because workstation-specific absolute paths were
  replaced with repository-relative placeholders;
- `freeze_noisy_test_protocol.py` differs only because three user-specific
  cache paths were replaced by `LLM_TSE_CACHE_ROOT`, defaulting to `~/.cache`.

All 91 synced Python files passed `compileall`. The dependency-light reference
suite passed 9/9 tests, and the minimal operator example completed. The TSE
Torch contract test is included but requires the optional full-pipeline
dependencies.

## Included scope

- Qwen/WavLM TSE model and data contract;
- CosyVoice3 S3 codec adapter;
- training and decode entry points;
- frozen WeSep candidate generation and Pool-D selection;
- Clean selected-evidence generation and evaluation;
- Natural-WHAM!/controlled-SNR preparation and evaluation;
- CSG/GNR decode logic, paired statistics, reporting, and TEST freeze checks.

## Excluded scope

- data, checkpoints, pretrained models, audio, embeddings, tokens, transcripts,
  manifests, caches, logs, and bulk results;
- third-party repositories;
- cluster job launchers, Screen sessions, orchestration, and internal agent
  files;
- failed recovery/learned-selector branches;
- post-TEST DEV selective-router and multi-lambda/token-level experiments.

No files on lab5090 were modified during synchronization.
