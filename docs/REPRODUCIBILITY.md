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

## What is intentionally absent

This repository does not redistribute LibriSpeech, LibriMix/WHAM, model weights,
generated audio, speaker embeddings, token caches, ASR transcripts, or
third-party code. Obtain those assets from their respective owners and use the
portable method operators here inside your own inference pipeline.

## Recommended checks

1. Run `python -m unittest discover -s tests -v`.
2. Verify exhaustive FSQ ID/coordinate round trips.
3. Verify CSG with lambda zero leaves logits and argmaxes unchanged.
4. Verify every accepted GNR edit has Hamming distance no greater than R.
5. Verify the anchor is present at every GNR position.
6. Verify selected candidates use enrollment similarity only.
7. Report switch counts and denominators together with rates.
8. Use paired resampling on common trial IDs for system comparisons.
