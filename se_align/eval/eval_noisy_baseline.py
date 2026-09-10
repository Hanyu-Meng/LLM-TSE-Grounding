"""Unprocessed-noisy baseline: score the raw noisy test signal vs clean reference.

This is the SE *lower bound* (no enhancement). Together with the reconstruction
ceiling (upper bound) and a Phase-2 model's eval, it brackets the result:

    noisy input (this)  <=  Phase-2 model  <=  reconstruction ceiling

Metrics match the other eval paths. SECS uses the CAMPPlus embeddings already
stored at tokenization time (clean vs noisy); the rest run on the 16 kHz wavs.

    python -m se_align.eval.eval_noisy_baseline --config configs/vbdemand_cv3.yaml \
        --split test --limit 100
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ..data.store import load_spk_emb, manifest_to_dict, read_manifest
from ..data.vbdemand import load_transcripts
from ..utils.audio import load_wav, resample
from ..utils.common import ensure_dir, get_logger, set_seed
from ..utils.config import load_config
from .metrics import MetricBundle, aggregate

LOG = get_logger("eval_noisy")


def _np16(path):
    w, sr = load_wav(path)
    w = resample(w, sr, 16000) if sr != 16000 else w
    return w.reshape(-1).numpy().astype(np.float32)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vbdemand_cv3.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 1986))
    root = cfg["tokenize.out_dir"]
    clean_rows = read_manifest(os.path.join(root, args.split, "clean", "manifest.jsonl"))
    noisy_idx = manifest_to_dict(read_manifest(os.path.join(root, args.split, "noisy", "manifest.jsonl")))
    if args.limit:
        clean_rows = clean_rows[: args.limit]

    transcripts = load_transcripts(os.path.join(cfg["data.root"], cfg["data.txt"][args.split]))
    metrics = MetricBundle(dict(cfg["metrics"]), asr_model=cfg["reconstruction"].get("asr_model"),
                           device=args.device)
    out_dir = ensure_dir(args.out_dir or os.path.join(cfg["reconstruction.out_dir"], "noisy_baseline"))

    rows = []
    for i, crow in enumerate(clean_rows):
        utt = crow["utt_id"]
        nrow = noisy_idx.get(utt)
        if nrow is None:
            continue
        ref16 = _np16(crow["wav_path"])      # clean reference
        est16 = _np16(nrow["wav_path"])      # raw noisy (no processing)
        ref_emb = load_spk_emb(root, crow)   # clean CAMPPlus
        est_emb = load_spk_emb(root, nrow)   # noisy CAMPPlus
        m = metrics.score(ref16, est16, ref_text=transcripts.get(utt),
                          ref_emb=ref_emb, est_emb=est_emb)
        m["utt_id"] = utt
        rows.append(m)
        if i % 25 == 0:
            LOG.info("[%d/%d] %s %s", i + 1, len(clean_rows), utt,
                     {k: round(v, 3) for k, v in m.items() if isinstance(v, float)})

    agg = aggregate(rows)
    agg["n_utts"] = len(rows)
    with open(os.path.join(out_dir, "noisy_baseline.json"), "w") as fh:
        json.dump({"aggregate": agg, "per_utt": rows}, fh, indent=2)
    LOG.info("NOISY BASELINE (%d utts): %s", len(rows),
             {k: round(v, 4) if isinstance(v, float) else v for k, v in agg.items()})


if __name__ == "__main__":
    main()
