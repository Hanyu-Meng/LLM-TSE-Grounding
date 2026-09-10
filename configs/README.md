# Configuration

The three `tse_*.yaml` files are portable copies of the experiment structure.
Paths are repository-relative placeholders and must be adapted to the local
machine before running the full pipeline.

Expected ignored locations:

```text
external/CosyVoice/
external/wesep-real-tse/
external/LibriSpeech/
external/LibriMix/
external/wham_noise/
pretrained/Qwen2.5-0.5B-Instruct/
pretrained/wavlm-base-plus/
pretrained/Fun-CosyVoice3-0.5B/
pretrained/wesep/spk_emb_100/
manifests/
artifacts/
exp/
```

Run scripts from the repository root so relative paths resolve consistently.
CLI arguments override configuration values where supported. Never commit
local absolute paths, credentials, manifests, checkpoints, or generated data.
