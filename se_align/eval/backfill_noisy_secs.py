"""Backfill noisy-vs-clean SECS (CAMPPlus cosine) for a split — the full baseline
run skipped it (codec was only built for the ceiling variant). Lightweight: only
extracts speaker embeddings, no other metrics.

    python -m se_align.eval.backfill_noisy_secs --split test
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from ..codec.cosyvoice3_codec import build_codec_from_config
from ..data.store import read_manifest
from ..utils.audio import load_wav
from ..utils.common import get_logger
from ..utils.config import load_config

LOG = get_logger("backfill_secs")
LIBRI_SUBS = ["test-clean", "test-other", "dev-clean", "dev-other"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--config", default="configs/vbdemand_cv3.yaml")
    ap.add_argument("--tokens-root", default="data_tokens_synth")
    ap.add_argument("--libri-root", default="/data/hshi/datasets/LibriSpeech")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    codec = build_codec_from_config(cfg)
    flac = {}
    for sub in LIBRI_SUBS:
        for f in glob.glob(os.path.join(args.libri_root, sub, "*", "*", "*.flac")):
            flac[os.path.basename(f)[:-5]] = f

    nrows = read_manifest(os.path.join(args.tokens_root, args.split, "noisy", "manifest.jsonl"))
    if args.num_shards > 1:
        nrows = nrows[args.shard::args.num_shards]

    out = args.out or f"results/secs_{args.split}_shard{args.shard}.jsonl"
    fh = open(out, "w")
    n = 0
    for r in nrows:
        utt = r["utt_id"]
        cf = flac.get(utt) or r.get("clean_wav")
        nw_p = r.get("wav_path")
        if not cf or not nw_p or not os.path.exists(nw_p):
            continue
        cw, csr = load_wav(cf)
        nw, nsr = load_wav(nw_p)
        ce = codec.extract_spk_emb(cw, csr).numpy().reshape(-1)
        ne = codec.extract_spk_emb(nw, nsr).numpy().reshape(-1)
        cos = float(ce @ ne / (np.linalg.norm(ce) * np.linalg.norm(ne) + 1e-8))
        fh.write(json.dumps({"utt_id": utt, "secs": cos}) + "\n")
        fh.flush()
        n += 1
        if n % 200 == 0:
            LOG.info("[%s sh%d] %d/%d", args.split, args.shard, n, len(nrows))
    fh.close()
    LOG.info("%s shard %d: %d utts -> %s", args.split, args.shard, n, out)


if __name__ == "__main__":
    main()
