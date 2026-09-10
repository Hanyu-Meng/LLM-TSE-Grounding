"""Pick a fixed, paired evaluation subset for the stratified SE test sets.

73 LibriSpeech test speakers x 7 utts each (cross-chapter preferred) = 511 utts,
shared across all 10 stratified sets (SNR x5, DEMAND noise x5) so SNR/noise-type
comparisons are paired. Main dev/test are NOT touched.

    python -m se_align.eval.sample_strat_eval
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np

POOL = "data_tokens_synth/test_snr0/noisy/manifest.jsonl"  # all 10 sets share these utts
LIBRI = "/data/hshi/datasets/LibriSpeech"
OUT = "results/strat_eval_utts.txt"
N_PER_SPK = 7
SEED = 1986


def main() -> None:
    utts = [json.loads(l)["utt_id"] for l in open(POOL)]
    by_spk = defaultdict(list)
    for u in utts:
        by_spk[u.split("-")[0]].append(u)

    rng = np.random.default_rng(SEED)
    selected = []
    for spk in sorted(by_spk):
        by_chap = defaultdict(list)
        for u in by_spk[spk]:
            by_chap[u.split("-")[1]].append(u)
        for c in by_chap:
            rng.shuffle(by_chap[c])
        chap_order = sorted(by_chap)
        rng.shuffle(chap_order)
        pick, ci = [], 0
        while len(pick) < N_PER_SPK and ci < 10000:
            c = chap_order[ci % len(chap_order)]
            if by_chap[c]:
                pick.append(by_chap[c].pop())
            ci += 1
        selected.extend(pick[:N_PER_SPK])

    selected = sorted(set(selected))
    os.makedirs("results", exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(selected) + "\n")

    # ---- report ----
    sub_of = {}
    for sub in ["test-clean", "test-other"]:
        for d in glob.glob(f"{LIBRI}/{sub}/*"):
            if os.path.isdir(d):
                sub_of[os.path.basename(d)] = sub
    csub = Counter(sub_of.get(u.split("-")[0], "?") for u in selected)
    pc = Counter(u.split("-")[0] for u in selected)
    nchap = len({u.rsplit("-", 1)[0] for u in selected})
    print(f"selected: {len(selected)} utts -> {OUT}")
    print(f"speakers: {len(pc)} (pool had {len(by_spk)})")
    print(f"clean/other: {dict(csub)}")
    print(f"per-speaker count: min={min(pc.values())} max={max(pc.values())}")
    print(f"distinct (speaker,chapter) covered: {nchap}")


if __name__ == "__main__":
    main()
