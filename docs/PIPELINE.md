# Research pipeline

This document maps the synced experiment code to the paper workflow. Commands
are intentionally shown as entry points rather than cluster launch scripts.
Run them from the repository root and inspect each script's `--help` before a
full experiment.

## 1. Prepare the TSE data contract

```text
LibriMix mixture + clean sources + enrollment utterance
    -> manifest
    -> WeSep evidence waveform
    -> S3 evidence/target tokens
    -> prepared JSONL
```

Primary entry points:

- `scripts/tse/build_librimix_manifest.py`
- `scripts/tse/check_tse_assets.py`
- `scripts/tse/preprocess_tse_features.py`
- `scripts/tse/validate_tse_manifest.py`
- `se_align/data/tse_dataset.py`

The manifest stores file references and metadata, not audio or token arrays
inside this repository. Generated JSONL and feature artifacts are ignored.

## 2. Train and decode Qwen-TSE

Primary entry points:

- `se_align/tse/model.py`: multimodal Qwen/WavLM target-speaker model;
- `se_align/tse/build.py`: model construction;
- `scripts/tse/train_tse.py`: supervised training;
- `scripts/tse/decode_tse_tokens.py`: unrestricted or grounded token decoding;
- `scripts/tse/evaluate_stage1_audio.py`: waveform-domain evaluation;
- `scripts/tse/evaluate_stage1_asr.py`: ASR-consistency evaluation.

The model conditions on a target-speaker embedding, WavLM mixture features,
and evidence S3 tokens, then predicts target S3 tokens. Frozen model paths and
manifests are supplied through `configs/tse_train_qwen_wavlm_fsq.yaml`.

## 3. Generate CDCS candidates

```text
target enrollment
    -> full / early / middle / late views
    -> primary WeSep candidates
    + TF-map/context candidate
    -> frozen speaker embeddings
    -> CDCS-2 or CDCS-5 target-consistent selection
```

Primary entry points:

- `scripts/tse_candidate_gate/run_frozen_wesep_candidates.py`
- `scripts/tse_candidate_gate/extract_candidate_speaker_embeddings.py`
- `scripts/tse_candidate_gate/score_candidate_acoustics.py`
- `scripts/tse_candidate_gate/score_candidate_asr.py`
- `scripts/tse_candidate_gate/aggregate_candidate_gate.py`
- `scripts/tse_candidate_gate/validate_candidate_gate.py`

`aggregate_candidate_gate.py` implements the deployable enrollment-cosine
selector. Evaluation-only references may appear in analysis outputs but are
not selector inputs.

## 4. Assemble CDCS evidence

Primary entry points:

- `scripts/tse_selected_evidence/prepare_selected_evidence.py`
- `scripts/tse_selected_evidence/prepare_selected_evidence_test.py`
- `scripts/tse_selected_evidence/merge_candidate_metrics.py`
- `scripts/tse_selected_evidence/benchmark_candidate_selection.py`
- `scripts/tse_selected_evidence/validate_selected_smoke.py`

The output is a frozen per-trial CDCS evidence choice plus aligned
waveform/token references for `CDCS-5 direct`, `Qwen-TSE UD`, `Qwen-TSE fixed
CSG`, and the declared GNR ablation.

## 5. Tokenize, ground, and synthesize

Primary entry points:

- `scripts/tse_selected_evidence/tokenize_selected_evidence_safe.py`
- `scripts/tse_selected_evidence/decode_selected_evidence.py`
- `scripts/tse_selected_evidence/synthesize_selected_tokens.py`
- `scripts/tse_noisy_wham/decode_noisy_grounding.py`
- `se_align/train/fsq_neighbors.py`
- `se_align/codec/cosyvoice3_codec.py`

`decode_selected_evidence.py` and `decode_noisy_grounding.py` contain the
actual Torch decoding path used by the campaign. The smaller
`src/llm_tse_grounding/` modules are dependency-light reference operators for
auditing the same CDCS-5, FSQ, CSG, and GNR invariants.

## 6. Clean evaluation

Primary entry points:

- `scripts/tse_selected_evidence/evaluate_selected_audio_safe.py`
- `scripts/tse_selected_evidence/evaluate_selected_asr_safe.py`
- `scripts/tse_selected_evidence/evaluate_intelligibility_quality.py`
- `scripts/tse_selected_evidence/finalize_selected_evidence.py`
- `scripts/tse_selected_evidence/audit_icassp_test_results.py`
- `scripts/tse_selected_evidence/build_icassp_table.py`
- `scripts/tse_selected_evidence/build_icassp_test_table.py`
- `scripts/tse_selected_evidence/build_final_paper_tables.py`

The evaluation joins systems by trial ID and reports fidelity, speaker
identity, perceptual estimates, and paired uncertainty without changing the
frozen output waveforms.

## 7. Natural-noise and controlled-SNR evaluation

Primary entry points:

- `scripts/tse_noisy_wham/prepare_noisy_wham_data.py`
- `scripts/tse_noisy_wham/prepare_noisy_views.py`
- `scripts/tse_noisy_wham/prepare_noisy_candidates.py`
- `scripts/tse_noisy_wham/analyze_noisy_candidates.py`
- `scripts/tse_noisy_wham/finalize_noisy_evidence.py`
- `scripts/tse_noisy_wham/build_noisy_difficulty.py`
- `scripts/tse_noisy_wham/compile_noisy_split.py`
- `scripts/tse_noisy_wham/paired_noisy_statistics.py`
- `scripts/tse_noisy_wham/evaluate_noisy_extended_metric.py`
- `scripts/tse_noisy_wham/validate_noisy_outputs.py`
- `scripts/tse_noisy_wham/build_noisy_reports.py`

The natural-noise subset and controlled `{-5, 0, 5, 10, 15}` dB SNR subset
must remain distinct. The paper's primary noisy table uses 6,000 natural-noise
target trials per system; the 2,400 controlled-SNR trials are a separate
robustness analysis.

## 8. Freeze and verify TEST

Primary entry points:

- `scripts/tse_noisy_wham/freeze_noisy_test_protocol.py`
- `scripts/tse_noisy_wham/manage_noisy_test_execution.py`
- `scripts/tse_noisy_wham/verify_frozen_noisy_test.py`
- `scripts/tse_noisy_wham/exact_token_cache.py`

The freeze script hashes model, code, protocol, and input artifacts before the
one permitted Noisy TEST execution. It expects the complete private experiment
state and is therefore an audit/reference tool in a fresh clone. Set
`LLM_TSE_CACHE_ROOT` if Torch/Hugging Face caches are not under `~/.cache`.

## Deliberately excluded

- checkpoints and pretrained weights;
- LibriSpeech/LibriMix/WHAM! corpora;
- generated waveforms, tokens, embeddings, transcripts, and per-trial JSONL;
- Screen/job orchestration and workstation-specific launch wrappers;
- recovery-model and learned-selector branches that failed the DEV gate;
- post-TEST selective-router and multi-lambda/token-level method-development
  experiments;
- general speech-enhancement models and evaluators not used by this paper;
- internal agent, automation, or research-management files.

These exclusions keep the repository reviewable and prevent partial DEV
experiments from being mistaken for the frozen paper system.
