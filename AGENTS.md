# Repository working agreement

## Purpose

This repository is the curated, reproducible public-facing codebase for the
LLM-TSE grounding research project. Keep it concise and organized; it is not a
mirror of the lab machines.

## Sync policy

- After a meaningful, validated code or documentation improvement, create a
  clear git commit and push it to `origin/main`.
- Do not push incomplete experiments, transient debugging edits, generated
  caches, large datasets, checkpoints, raw evaluation outputs, or machine-local
  paths.
- Never commit credentials, SSH material, access tokens, private host details,
  participant data, or other secrets.
- Before each push, review `git diff`, run the relevant lightweight checks, and
  confirm that the repository remains reproducible and understandable.
- Preserve useful remote research code by curating only the reusable parts into
  this repository. Do not treat lab5090 or jollibear as folders to mirror.
- Experiment results should be added only when they are frozen, traceable to a
  documented configuration, and appropriate to share.

## Remote-compute boundary

Reading remote experiment state does not authorize changing or interrupting a
running experiment. Do not stop, restart, replace, or launch remote jobs unless
the user explicitly requests that action. Repository maintenance must not
disturb active lab workloads.
