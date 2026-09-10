"""Copy-synthesis reconstruction ceiling on VoiceBank-DEMAND test×clean.

Pipeline per utterance (clean speech only):

    clean wav --encode--> tokens --decode--> 24 kHz wav --resample--> 16 kHz
        --align with clean@16k--> metrics

Two variants quantify speaker drift from using the *noisy* speaker estimate:
  * clean-spk : self-prompt from the clean clip (theoretical upper bound)
  * noisy-spk : same clean self-prompt tokens+mel, but the 192-d CAMPPlus
                speaker embedding is swapped for the one extracted from noisy.

Outputs ``results/reconstruction_ceiling.{json,md}`` — the Phase-2 SE upper bound.

Example
-------
    python -m se_align.eval.run_reconstruction --config configs/vbdemand_cv3.yaml \
        --limit 20 --no-wer            # quick check
    python -m se_align.eval.run_reconstruction --config configs/vbdemand_cv3.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from ..codec.cosyvoice3_codec import build_codec_from_config
from ..data.store import load_spk_emb, load_tokens, manifest_to_dict, read_manifest
from ..data.vbdemand import load_transcripts
from ..utils.audio import load_wav, resample, save_wav
from ..utils.common import ensure_dir, get_logger, set_seed
from ..utils.config import Config, load_config, parse_overrides
from .metrics import MetricBundle, aggregate

LOG = get_logger("reconstruction")


def _to_np16k(wav24: torch.Tensor, sr: int) -> np.ndarray:
    w = resample(wav24, sr, 16000) if sr != 16000 else wav24
    return w.reshape(-1).cpu().numpy().astype(np.float32)


def run(
    cfg: Config,
    variants: List[str],
    limit: Optional[int] = None,
    save_wavs: bool = False,
    use_wer: bool = True,
) -> Dict:
    set_seed(cfg.get("seed", 1986))
    tokens_root = cfg["tokenize.out_dir"]
    out_dir = ensure_dir(cfg["reconstruction.out_dir"])

    clean_manifest = os.path.join(tokens_root, "test", "clean", "manifest.jsonl")
    noisy_manifest = os.path.join(tokens_root, "test", "noisy", "manifest.jsonl")
    clean_rows = read_manifest(clean_manifest)
    if limit is not None:
        clean_rows = clean_rows[:limit]
    noisy_index = manifest_to_dict(read_manifest(noisy_manifest)) if "noisy" in variants else {}

    # WER reference transcripts
    txt_dir = None
    if use_wer and cfg.get("metrics", {}).get("wer"):
        from ..utils.config import Config as _C  # noqa
        sub = cfg["data.txt"]["test"]
        txt_dir = os.path.join(cfg["data.root"], sub) if sub else None
    transcripts = load_transcripts(txt_dir)

    toggles = dict(cfg["metrics"])
    if not use_wer:
        toggles["wer"] = False
    metrics = MetricBundle(
        toggles,
        asr_model=cfg["reconstruction"].get("asr_model"),
        device=cfg["cosyvoice3"].get("device"),
    )
    codec = build_codec_from_config(cfg)
    prompt_strategy = cfg["reconstruction"].get("prompt_strategy", "self")

    summary: Dict[str, dict] = {}
    for variant in variants:
        per_utt: List[dict] = []
        vdir = ensure_dir(os.path.join(out_dir, f"decoded_{prompt_strategy}_{variant}-spk")) if save_wavs else None
        t0 = time.time()
        for i, row in enumerate(clean_rows):
            utt = row["utt_id"]
            tokens = load_tokens(tokens_root, row)
            clean_wav, csr = load_wav(row["wav_path"])  # 48k mono

            spk_emb = None
            if variant == "noisy":
                nrow = noisy_index.get(utt)
                if nrow is None:
                    LOG.warning("no noisy spk_emb for %s; skipping", utt)
                    continue
                spk_emb = torch.from_numpy(load_spk_emb(tokens_root, nrow))

            dec_wav, dsr = codec.decode(
                tokens,
                prompt_wav=clean_wav,
                prompt_sr=csr,
                spk_emb=spk_emb,
                prompt_strategy=prompt_strategy,
            )
            if vdir:
                save_wav(os.path.join(vdir, f"{utt}.wav"), dec_wav, dsr)

            ref16 = _to_np16k(clean_wav, csr)
            est16 = _to_np16k(dec_wav, dsr)
            ref_emb = load_spk_emb(tokens_root, row)  # clean CAMPPlus emb
            est_emb = codec.extract_spk_emb(dec_wav, dsr).numpy()

            m = metrics.score(
                ref16, est16,
                ref_text=transcripts.get(utt),
                ref_emb=ref_emb, est_emb=est_emb,
            )
            m["utt_id"] = utt
            per_utt.append(m)
            if i % 50 == 0:
                LOG.info("[%s][%d/%d] %s %s", variant, i + 1, len(clean_rows), utt,
                         {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items() if k != "utt_id"})

        agg = aggregate(per_utt)
        agg["n_utts"] = len(per_utt)
        agg["seconds"] = round(time.time() - t0, 1)
        summary[f"{variant}-spk"] = agg
        # per-utt dump
        with open(os.path.join(out_dir, f"per_utt_{variant}-spk.jsonl"), "w") as fh:
            for r in per_utt:
                fh.write(json.dumps(r) + "\n")
        LOG.info("[%s-spk] aggregate: %s", variant, {k: round(v, 4) if isinstance(v, float) else v for k, v in agg.items()})

    result = {
        "dataset": "VoiceBank-DEMAND test×clean (copy-synthesis reconstruction)",
        "model": cfg["cosyvoice3"]["model_dir"],
        "prompt_strategy": prompt_strategy,
        "flow_steps": cfg["cosyvoice3"].get("flow_steps"),
        "token_rate_hz": cfg["cosyvoice3"].get("token_rate_hz"),
        "variants": summary,
    }
    _write_outputs(out_dir, result)
    return result


def _write_outputs(out_dir: str, result: Dict) -> None:
    json_path = os.path.join(out_dir, "reconstruction_ceiling.json")
    with open(json_path, "w") as fh:
        json.dump(result, fh, indent=2)

    # markdown table
    metric_order = ["pesq", "stoi", "estoi", "si_sdr", "dnsmos_sig", "dnsmos_bak",
                    "dnsmos_ovl", "dnsmos_p808", "utmos", "wer", "secs",
                    "speechbertscore", "phoneme_sim", "vqscore", "n_utts", "seconds"]
    variants = list(result["variants"].keys())
    lines = [
        "# Reconstruction ceiling — VoiceBank-DEMAND test×clean",
        "",
        f"- **Model**: `{result['model']}`",
        f"- **Prompt strategy**: {result['prompt_strategy']}  |  **Flow steps**: {result['flow_steps']}  |  **Token rate**: {result['token_rate_hz']} Hz",
        "",
        "> 25 Hz semantic tokens => generative reconstruction. PESQ/SI-SDR are "
        "expected to be low even at the ceiling; judge primarily by "
        "DNSMOS/UTMOS + WER + SECS.",
        "",
        "| metric | " + " | ".join(variants) + " |",
        "|---|" + "---|" * len(variants),
    ]
    present = [m for m in metric_order if any(m in result["variants"][v] for v in variants)]
    for m in present:
        cells = []
        for v in variants:
            val = result["variants"][v].get(m)
            cells.append(f"{val:.4f}" if isinstance(val, float) else ("" if val is None else str(val)))
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    md_path = os.path.join(out_dir, "reconstruction_ceiling.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    LOG.info("wrote %s and %s", json_path, md_path)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CV3 reconstruction ceiling on VB-DEMAND")
    p.add_argument("--config", required=True)
    p.add_argument("--variants", nargs="*", default=None, help="subset of {clean,noisy}")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--save-wavs", action="store_true")
    p.add_argument("--no-wer", action="store_true", help="skip WER (no whisper download)")
    p.add_argument("--device", default=None)
    p.add_argument("--set", nargs="*", default=None)
    return p


def main(argv=None) -> None:
    args = build_argparser().parse_args(argv)
    overrides = parse_overrides(args.set)
    if args.device:
        overrides["cosyvoice3.device"] = args.device
    cfg = load_config(args.config, overrides)
    variants = args.variants or cfg["reconstruction"].get("spk_variants", ["clean", "noisy"])
    run(cfg, variants=variants, limit=args.limit, save_wavs=args.save_wavs, use_wer=not args.no_wer)


if __name__ == "__main__":
    main()
