"""Pick ~2000-utt speaker-balanced eval subsets for main test and dev (model eval).
Baselines are aggregated on the same subset for a paired comparison; the full
baseline (all utts) is kept separately.

    python -m se_align.eval.sample_main_eval --split test --n 2000
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1986)
    ap.add_argument("--tokens-root", default="data_tokens_synth")
    args = ap.parse_args(argv)

    utts = [json.loads(l)["utt_id"]
            for l in open(os.path.join(args.tokens_root, args.split, "noisy", "manifest.jsonl"))]
    by_spk = defaultdict(list)
    for u in utts:
        by_spk[u.split("-")[0]].append(u)

    nspk = len(by_spk)
    per = int(np.ceil(args.n / nspk))  # per-speaker target
    rng = np.random.default_rng(args.seed)
    selected = []
    for spk in sorted(by_spk):
        pool = list(by_spk[spk])
        rng.shuffle(pool)
        selected.extend(pool[:per])
    rng.shuffle(selected)
    selected = sorted(selected[:args.n])  # trim to exactly n

    out = f"results/main_{args.split}_{args.n}.txt"
    os.makedirs("results", exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\n".join(selected) + "\n")

    pc = Counter(u.split("-")[0] for u in selected)
    print(f"{args.split}: selected {len(selected)} -> {out}")
    print(f"  speakers: {len(pc)}/{nspk} | per-speaker min={min(pc.values())} max={max(pc.values())}")


if __name__ == "__main__":
    main()
