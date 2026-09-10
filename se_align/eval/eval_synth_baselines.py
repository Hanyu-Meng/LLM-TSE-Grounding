"""Baselines on the synthetic LibriSpeech+DNS test set — three reference points:

  * clean    : original clean LibriSpeech vs itself        (true upper bound)
  * noisy    : synthetic noisy signal vs clean             (lower bound)
  * ceiling  : clean -> CV3 tokens -> decode vs clean      (generative ceiling)

Clean reference wav = original LibriSpeech flac (mapped by utt_id; the synth
manifest stores only tokens + noisy wav). Transcripts come from the manifest
'text' field. SECS uses CAMPPlus embeddings extracted on the fly.

    python -m se_align.eval.eval_synth_baselines --config configs/vbdemand_cv3.yaml \
        --tokens-root data_tokens_synth --split test --limit 100 \
        --variants clean noisy ceiling --device cuda
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from typing import Dict, List

import numpy as np
import torch

from ..codec.cosyvoice3_codec import build_codec_from_config
from ..data.store import load_tokens, manifest_to_dict, read_manifest
from ..utils.audio import load_wav, resample
from ..utils.common import ensure_dir, get_logger, set_seed
from ..utils.config import load_config
from .metrics import MetricBundle, aggregate

LOG = get_logger("eval_synth")
LIBRI_TEST = ["test-clean", "test-other", "dev-clean", "dev-other"]


def _np16k(wav: torch.Tensor, sr: int) -> np.ndarray:
    w = resample(wav, sr, 16000) if sr != 16000 else wav
    return w.reshape(-1).cpu().numpy().astype(np.float32)


def _build_flac_map(libri_root: str) -> Dict[str, str]:
    m = {}
    for sub in LIBRI_TEST:
        for f in glob.glob(os.path.join(libri_root, sub, "*", "*", "*.flac")):
            m[os.path.basename(f)[:-5]] = f
    return m


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--tokens-root", default="data_tokens_synth")
    p.add_argument("--split", default="test")
    p.add_argument("--libri-root", default="/data/hshi/datasets/LibriSpeech")
    p.add_argument("--variants", nargs="+", default=["clean", "noisy", "ceiling"])
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="results/synth_baselines.json")
    p.add_argument("--no-ref", action="store_true",
                   help="reference-free (blind set): noisy variant, only DNSMOS/UTMOS/VQScore")
    p.add_argument("--per-utt-dir", default=None,
                   help="if set, append per-utt rows to {dir}/{split}_{variant}_shard{N}.jsonl")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 1986))
    tokens_root = args.tokens_root

    noisy_index = manifest_to_dict(read_manifest(
        os.path.join(tokens_root, args.split, "noisy", "manifest.jsonl")))
    # blind set has no clean manifest: iterate noisy rows, reference-free only
    base = "noisy" if args.no_ref else "clean"
    clean_rows = read_manifest(os.path.join(tokens_root, args.split, base, "manifest.jsonl"))
    if args.limit:
        clean_rows = clean_rows[: args.limit]
    if args.num_shards > 1:
        clean_rows = clean_rows[args.shard::args.num_shards]  # this GPU's slice
    flac = _build_flac_map(args.libri_root)
    transcripts = {r["utt_id"]: r.get("text", "") for r in clean_rows}

    mtog = dict(cfg["metrics"])
    if args.no_ref:
        mtog = {k: (k in ("dnsmos", "utmos", "vqscore")) for k in mtog}
    metrics = MetricBundle(mtog, asr_model=cfg["reconstruction"].get("asr_model"),
                           device=args.device)
    # always build the codec: ceiling needs decode, and every variant needs it
    # for the CAMPPlus speaker embedding (SECS).
    codec = build_codec_from_config(cfg)

    if args.per_utt_dir:
        ensure_dir(args.per_utt_dir)
    summary: Dict[str, dict] = {}
    for variant in args.variants:
        per_utt: List[dict] = []
        pu_fh = None
        done = set()
        if args.per_utt_dir:
            pu_path = os.path.join(args.per_utt_dir, f"{args.split}_{variant}_shard{args.shard}.jsonl")
            # shard-count-agnostic resume: read ALL shard files of this variant
            # (a re-run with a different --num-shards repartitions utts -> dupes).
            import glob as _glob
            for spp in _glob.glob(os.path.join(args.per_utt_dir, f"{args.split}_{variant}_shard*.jsonl")):
                for l in open(spp):
                    try:
                        done.add(json.loads(l)["utt_id"])
                    except Exception:
                        pass
            pu_fh = open(pu_path, "a")  # append (keep prior results)
        t0 = time.time()
        for i, row in enumerate(clean_rows):
            utt = row["utt_id"]
            if utt in done:  # resume: already scored in a previous run
                continue
            fp = flac.get(utt) or row.get("wav_path")  # libri map, else manifest (DNS)
            if not fp:
                continue
            clean_wav, csr = load_wav(fp)
            ref16 = _np16k(clean_wav, csr)

            if variant == "clean":
                est16 = ref16
            elif variant == "noisy":
                nrow = noisy_index.get(utt)
                if nrow is None or not os.path.exists(nrow.get("wav_path", "")):
                    continue
                nw, nsr = load_wav(nrow["wav_path"])
                est16 = _np16k(nw, nsr)
            else:  # ceiling: clean tokens -> decode
                tokens = load_tokens(tokens_root, row)
                dec_wav, dsr = codec.decode(tokens, prompt_wav=clean_wav,
                                            prompt_sr=csr, prompt_strategy="self")
                est16 = _np16k(dec_wav, dsr)

            if args.no_ref:  # blind: ref-free metrics only, skip SECS/WER
                m = metrics.score(est16, est16)
            else:
                ref_emb = codec.extract_spk_emb(clean_wav, csr).numpy() if codec is not None else None
                est_emb = (ref_emb if variant == "clean"
                           else (codec.extract_spk_emb(torch.from_numpy(est16), 16000).numpy()
                                 if codec is not None else None))
                m = metrics.score(ref16, est16, ref_text=transcripts.get(utt),
                                  ref_emb=ref_emb, est_emb=est_emb)
            m["utt_id"] = utt
            per_utt.append(m)
            if pu_fh is not None:
                pu_fh.write(json.dumps(m) + "\n")
                pu_fh.flush()
            if i % 25 == 0:
                LOG.info("[%s][%d/%d] %s", variant, i + 1, len(clean_rows),
                         {k: round(v, 3) for k, v in m.items() if isinstance(v, float)})
        if pu_fh is not None:
            pu_fh.close()
        agg = aggregate(per_utt)
        agg["n_utts"] = len(per_utt)
        agg["seconds"] = round(time.time() - t0, 1)
        summary[variant] = agg
        LOG.info("[%s] %s", variant, {k: round(v, 4) if isinstance(v, float) else v for k, v in agg.items()})

    ensure_dir(os.path.dirname(args.out))
    with open(args.out, "w") as fh:
        json.dump({"dataset": f"synthetic LibriSpeech+DNS {args.split}",
                   "tokens_root": tokens_root, "variants": summary}, fh, indent=2)
    LOG.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
