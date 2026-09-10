"""Evaluate a trained Phase-2 token-LM: predict clean tokens -> decode -> metrics.

Pipeline per utt:
    noisy (tokens or wav) --[SEModel.generate]--> predicted clean CV3 tokens
        --strip specials (<6561)--> [Phase-1 CosyVoice3Codec.decode + prompt/spk_emb]
        --> 24 kHz wav --> resample 16k --> metrics vs clean reference

Compare the aggregate against results/reconstruction_ceiling.json (upper bound).

    python -m se_align.eval.eval_phase2 --train-config configs/train_t2t.yaml \
        --ckpt exp/t2t_qwen0.5b/model_epoch0.pt --cv3-config configs/vbdemand_cv3.yaml \
        --split test --limit 20 --prompt noisy
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from ..codec.cosyvoice3_codec import build_codec_from_config
from ..utils.audio import load_wav, resample
from ..utils.common import ensure_dir, get_logger, set_seed
from ..utils.config import load_config
from .metrics import MetricBundle, aggregate

LOG = get_logger("eval_phase2")


def _np16(wav, sr):
    w = resample(wav, sr, 16000) if sr != 16000 else wav
    return w.reshape(-1).cpu().numpy().astype(np.float32)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cv3-config", default="configs/vbdemand_cv3.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--prompt", choices=["noisy", "clean"], default="noisy")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--save-wavs", action="store_true")
    args = ap.parse_args(argv)

    tcfg = load_config(args.train_config)
    ccfg = load_config(args.cv3_config)
    set_seed(ccfg.get("seed", 1986))
    dtype = torch.bfloat16 if tcfg.get("bf16", True) else torch.float32
    task = tcfg["task_type"]

    # ---- Phase-2 model ----
    from ..train.build import build_model
    from ..train.dataset import SEDataset

    model, tokenizer, vocab = build_model(
        tcfg["llm_path"], task_type=task, whisper_name=tcfg.get("whisper_name", "large-v3"),
        whisper_ds_rate=tcfg.get("whisper_ds_rate", 5), dtype=dtype, device=args.device,
        freeze_encoder=True,
    )
    sd = torch.load(args.ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    LOG.info("loaded ckpt %s (missing=%d unexpected=%d)", args.ckpt, len(missing), len(unexpected))
    model.eval()

    ds = SEDataset(
        ccfg["tokenize.out_dir"], args.split, vocab, tokenizer, task_type=task,
        data_root=ccfg["data.root"], txt_subdir=ccfg["data.txt"][args.split],
        cv3_root=ccfg["cosyvoice3.cv3_root"], whisper_ds_rate=tcfg.get("whisper_ds_rate", 5),
        limit=args.limit,
    )

    # ---- Phase-1 decoder + metrics ----
    codec = build_codec_from_config(ccfg)
    toggles = dict(ccfg["metrics"])
    metrics = MetricBundle(toggles, asr_model=ccfg["reconstruction"].get("asr_model"),
                           device=args.device)
    from ..data.vbdemand import load_transcripts
    transcripts = load_transcripts(os.path.join(ccfg["data.root"], ccfg["data.txt"][args.split]))

    out_dir = ensure_dir(args.out_dir or os.path.join(ccfg["reconstruction.out_dir"], f"phase2_{task}"))
    wav_dir = ensure_dir(os.path.join(out_dir, "wav")) if args.save_wavs else None

    rows = []
    for utt in ds.utts:
        item = ds.infer_item(utt)
        inp = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in item.items()}
        if "audio_mel" in inp:
            inp["audio_mel"] = inp["audio_mel"].to(dtype)
        n_noisy = int(item["input_ids"].shape[-1])
        with torch.autocast(device_type="cuda", dtype=dtype):
            audio_ids, _ = model.generate(
                inp["input_ids"], attention_mask=inp["attention_mask"],
                audio_mel=inp.get("audio_mel"), modality_mask=inp.get("modality_mask"),
                max_new_tokens=min(1024, 2 * n_noisy + 50),
            )
        # keep only valid CV3 codebook ids
        toks = [t for t in audio_ids if 0 <= t < vocab.audio_vocabsize]
        if not toks:
            LOG.warning("%s produced no valid tokens; skipping", utt)
            continue

        prompt_wav, psr = load_wav(item["noisy_wav" if args.prompt == "noisy" else "clean_wav"])
        dec_wav, dsr = codec.decode(torch.tensor(toks), prompt_wav=prompt_wav, prompt_sr=psr,
                                    prompt_strategy="self")
        if wav_dir:
            from ..utils.audio import save_wav
            save_wav(os.path.join(wav_dir, f"{utt}.wav"), dec_wav, dsr)

        clean_wav, csr = load_wav(item["clean_wav"])
        ref16, est16 = _np16(clean_wav, csr), _np16(dec_wav, dsr)
        ref_emb = codec.extract_spk_emb(clean_wav, csr).numpy()
        est_emb = codec.extract_spk_emb(dec_wav, dsr).numpy()
        m = metrics.score(ref16, est16, ref_text=transcripts.get(utt),
                          ref_emb=ref_emb, est_emb=est_emb)
        m["utt_id"] = utt
        m["n_pred_tokens"] = len(toks)
        rows.append(m)
        LOG.info("%s pred_tokens=%d %s", utt, len(toks),
                 {k: round(v, 3) for k, v in m.items() if isinstance(v, float)})

    agg = aggregate(rows)
    agg["n_utts"] = len(rows)
    result = {"task": task, "ckpt": args.ckpt, "prompt": args.prompt, "aggregate": agg}
    with open(os.path.join(out_dir, "phase2_eval.json"), "w") as fh:
        json.dump({"summary": result, "per_utt": rows}, fh, indent=2)
    LOG.info("AGGREGATE (%d utts): %s", len(rows), {k: round(v, 4) if isinstance(v, float) else v for k, v in agg.items()})
    LOG.info("wrote %s/phase2_eval.json", out_dir)


if __name__ == "__main__":
    main()
