# Source synchronization provenance

Synchronization date: 2026-09-10

Source snapshot: the private LLM-TSE research project on lab5090.

Control path used for the read-only source audit:

```text
Mac Air -> jollibear -> lab5090
```

## Public-code normalization

The source snapshot was reduced to the code paths used by the manuscript.
General SE modules, workstation launch helpers, data, weights, and generated
experiment artifacts were removed. Internal lettered candidate-set names were
renamed to the manuscript terminology (`CDCS-2` and `CDCS-5`), and Qwen-TSE
systems now use the same public names as the paper. Portable configuration paths
remain repository-relative.

Every retained Python file is checked with `compileall`. The dependency-light
reference suite and minimal operator example are run before release. The TSE
Torch contract test is included but requires the optional full-pipeline
dependencies.

## Included scope

- Qwen/WavLM TSE model and data contract;
- CosyVoice3 S3 codec adapter;
- training and decode entry points;
- frozen WeSep candidate generation and CDCS-5 selection;
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
- general speech-enhancement training and evaluation modules not used in the
  manuscript.

No files on lab5090 were modified during synchronization.
