# Reproducibility boundary

## Evaluation split discipline

- Model and policy choices are made on DEV only.
- Clean TEST was executed once after freezing the clean protocol.
- The noisy study contains 6,000 natural-noisy and 2,400 controlled-noise trials
  per split. Controlled SNRs are -5, 0, 5, 10, and 15 dB.
- Noisy TEST is a one-run confirmation and must not be used for post-test tuning.

## Inference boundary

Pool-D selection may use only the mixture, target enrollment, frozen extractor
outputs, and target-enrollment speaker similarity. Clean target audio,
interferer audio, transcripts, WER, and SI-SDR are evaluation-only.

CSG receives the selected evidence tokens and changes only the audio-token
scores. GNR receives one teacher-forced logit pass over an immutable complete
anchor. Refined GNR tokens are never fed into later histories.

## Code and artifact boundary

The repository includes the curated TSE model, data contracts, candidate
selection, grounding, synthesis adapters, evaluation, statistics, and report
generation code. It does not redistribute LibriSpeech, LibriMix/WHAM!, model
weights, generated audio, speaker embeddings, token caches, ASR transcripts,
or third-party code. Obtain those assets from their respective owners and point
the portable example configs at their local locations.

Post-TEST selective routing and multi-lambda/token-level DEV experiments are
excluded from this snapshot. They are method-development evidence rather than
the frozen paper system and must not be represented as held-out TEST gains.

## Recommended checks

1. Run `python -m unittest discover -s tests -v`.
2. Verify exhaustive FSQ ID/coordinate round trips.
3. Verify CSG with lambda zero leaves logits and argmaxes unchanged.
4. Verify every accepted GNR edit has Hamming distance no greater than R.
5. Verify the anchor is present at every GNR position.
6. Verify selected candidates use enrollment similarity only.
7. Report switch counts and denominators together with rates.
8. Use paired resampling on common trial IDs for system comparisons.
9. Keep Natural Noisy and controlled-SNR aggregates separate.
10. Verify `execution_count=1` and `post_test_tuning=false` before citing Noisy
    TEST results.
