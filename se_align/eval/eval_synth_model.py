"""Evaluate a trained token-LM on the synthetic SE data.

    noisy (tokens or wav) --[SEModel.generate]--> predicted clean CV3 tokens
        --[CV3 decode, prompt=noisy]--> wav --metrics vs clean LibriSpeech--

Clean reference = original LibriSpeech flac (mapped by utt_id). Transcripts come
from the manifest. Supports --utt-list (paired stratified subset), --num-shards
(multi-GPU), and --per-utt-dir (resumable shard aggregation).

    python -m se_align.eval.eval_synth_model --train-config configs/train_synth_t2t.yaml \
        --ckpt exp/synth_t2t_qwen0.5b/model_epoch1_dev2.3879.pt \
        --tokens-root data_tokens_synth --split test --limit 0 --device cuda
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
from ..utils.audio import load_wav, resample
from ..utils.common import ensure_dir, get_logger, set_seed
from ..utils.config import load_config
from .metrics import MetricBundle, aggregate

LOG = get_logger("eval_synth_model")
LIBRI_SUBS = ["test-clean", "test-other", "dev-clean", "dev-other"]


def _np16(wav: torch.Tensor, sr: int) -> np.ndarray:
    w = resample(wav, sr, 16000) if sr != 16000 else wav
    return w.reshape(-1).cpu().numpy().astype(np.float32)


def _flac_map(libri_root: str) -> Dict[str, str]:
    # the recursive glob over LibriSpeech is slow on network FS and identical
    # across the ~hundreds of eval launches in a campaign -> cache to json.
    cache = os.path.join(libri_root, ".flac_map_cache.json")
    if os.path.exists(cache):
        try:
            m = json.load(open(cache))
            if m:
                return m
        except Exception:
            pass
    m = {}
    for sub in LIBRI_SUBS:
        for f in glob.glob(os.path.join(libri_root, sub, "*", "*", "*.flac")):
            m[os.path.basename(f)[:-5]] = f
    try:
        tmp = cache + f".tmp{os.getpid()}"
        json.dump(m, open(tmp, "w"))
        os.replace(tmp, cache)
    except Exception:
        pass
    return m


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-config", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--cv3-config", default="configs/vbdemand_cv3.yaml")
    ap.add_argument("--tokens-root", default="data_tokens_synth")
    ap.add_argument("--split", default="test")
    ap.add_argument("--libri-root", default="/data/hshi/datasets/LibriSpeech")
    ap.add_argument("--utt-list", default=None, help="file of utt_ids to restrict to (paired subset)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--per-utt-dir", default=None)
    ap.add_argument("--out", default="results/synth_model.json")
    ap.add_argument("--tag", default=None, help="label for per-utt file (default: split)")
    ap.add_argument("--no-ref", action="store_true",
                    help="reference-free: no clean/transcript; only DNSMOS/UTMOS/VQScore (blind set)")
    ap.add_argument("--prompt", choices=["noisy", "clean"], default="noisy",
                    help="decode prompt source: noisy (realistic) or clean (oracle speaker)")
    ap.add_argument("--prompt-wav-dir", default=None,
                    help="dir of <utt>.{flac,wav} to use as decode prompt (e.g. E1-enhanced audio); "
                         "overrides --prompt")
    ap.add_argument("--gpu-metrics", action="store_true",
                    help="also run UTMOS + DNSMOS on GPU (default OFF; off = identical to prior CPU runs)")
    ap.add_argument("--ref-cache", default="results/.ref_cache",
                    help="dir caching per-utt clean ref16 + CAMPPlus emb across evals ('' disables)")
    ap.add_argument("--prompt-cache", default="results/.prompt_cache",
                    help="dir caching per-(utt,prompt-src) decode prompt features "
                         "(S3 token + mel + CAMPPlus = ~73%% of decode cost; '' disables)")
    ap.add_argument("--jobs", default=None,
                    help="json list of task dicts overriding {train_config, ckpt, split, tag, "
                         "tokens_root, prompt, prompt_wav_dir, out}; runs them in ONE process "
                         "so CV3 + all metric models load once (saves 60-120s startup/task)")
    ap.add_argument("--gen-batch", type=int, default=1,
                    help="batch N utts through generate_batch (default 1 = exact legacy path; "
                         "N>1 is ~NxLLM-decode speedup but bf16 batched matmuls can flip rare "
                         "argmax ties on degenerate sequences)")
    args = ap.parse_args(argv)

    if args.jobs:
        shared: dict = {}
        jobs = json.load(open(args.jobs))
        for i, jb in enumerate(jobs):
            a = argparse.Namespace(**{**vars(args), **jb})
            LOG.info("=== [job %d/%d] %s ===", i + 1, len(jobs), jb)
            _run_one(a, shared)
        return
    _run_one(args, {})


def _run_one(args, shared: dict) -> None:
    assert args.train_config and args.ckpt, "--train-config/--ckpt required (or via --jobs)"
    if args.ref_cache:
        ensure_dir(args.ref_cache)
    if getattr(args, "prompt_cache", None):
        ensure_dir(args.prompt_cache)
    tcfg = load_config(args.train_config)
    ccfg = load_config(args.cv3_config)
    set_seed(ccfg.get("seed", 1986))
    dtype = torch.bfloat16 if tcfg.get("bf16", True) else torch.float32
    task = tcfg["task_type"]

    from ..train.build import build_model
    from ..train.dataset import SEDataset

    # in --jobs mode reuse the model across tasks with the same (config, ckpt);
    # otherwise drop the old one before building the next (VRAM).
    mkey = (args.train_config, args.ckpt)
    if shared.get("model_key") == mkey:
        model, tokenizer, vocab = shared["model"]
    else:
        if "model" in shared:
            del shared["model"]
            torch.cuda.empty_cache()
        model, tokenizer, vocab = build_model(
            tcfg["llm_path"], task_type=task, whisper_name=tcfg.get("whisper_name", "large-v3"),
            whisper_ds_rate=tcfg.get("whisper_ds_rate", 5), dtype=dtype, device=args.device,
            freeze_encoder=True,
        )
        sd = torch.load(args.ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        LOG.info("loaded %s (missing=%d unexpected=%d)", os.path.basename(args.ckpt), len(missing), len(unexpected))
        model.eval()
        shared["model"], shared["model_key"] = (model, tokenizer, vocab), mkey

    ds = SEDataset(
        args.tokens_root, args.split, vocab, tokenizer, task_type=task,
        cv3_root=tcfg.get("cv3_root", ccfg["cosyvoice3.cv3_root"]),
        whisper_ds_rate=tcfg.get("whisper_ds_rate", 5),
        noisy_wav_root=tcfg.get("noisy_wav_root"),
    )
    utts = list(ds.utts)
    if args.utt_list:
        keep = set(l.strip() for l in open(args.utt_list) if l.strip())
        utts = [u for u in utts if u in keep]
    if args.limit:
        utts = utts[: args.limit]
    if args.num_shards > 1:
        utts = utts[args.shard::args.num_shards]

    # transcripts for WER: read manifest 'text' directly (SEDataset only loads them
    # for *_text tasks, but we need them for WER on every task).
    transcripts = {}
    for l in open(os.path.join(args.tokens_root, args.split, "clean", "manifest.jsonl")):
        r = json.loads(l)
        if r.get("text"):
            transcripts[r["utt_id"]] = r["text"]

    # heavy shared singletons (CV3 flow/vocoder/CAMPPlus, ASR + metric models,
    # LibriSpeech map): built once per process, reused across --jobs tasks.
    flac = shared.get("flac")
    if flac is None:
        flac = shared["flac"] = _flac_map(args.libri_root)
    codec = shared.get("codec")
    if codec is None:
        codec = shared["codec"] = build_codec_from_config(ccfg)
    metrics = shared.get("metrics")
    if metrics is None or shared.get("metrics_noref") != args.no_ref:
        toggles = dict(ccfg["metrics"])
        if args.no_ref:  # blind set: only reference-free metrics
            toggles = {k: (k in ("dnsmos", "utmos", "vqscore")) for k in toggles}
        metrics = MetricBundle(toggles, asr_model=ccfg["reconstruction"].get("asr_model"),
                               device=args.device, gpu_extra=args.gpu_metrics)
        shared["metrics"], shared["metrics_noref"] = metrics, args.no_ref

    tag = args.tag or args.split
    pu_fh, done = None, set()
    if args.per_utt_dir:
        ensure_dir(args.per_utt_dir)
        pp = os.path.join(args.per_utt_dir, f"{tag}_shard{args.shard}.jsonl")
        # resume must be shard-count-agnostic: a re-run with a different --num-shards
        # repartitions utts, so read ALL shard files of this tag (else duplicates).
        for spp in glob.glob(os.path.join(args.per_utt_dir, f"{tag}_shard*.jsonl")):
            for l in open(spp):
                try:
                    done.add(json.loads(l)["utt_id"])
                except Exception:
                    pass
        pu_fh = open(pp, "a")

    per_utt: List[dict] = []
    t0 = time.time()
    is_text = task in ("t2t_text", "audio2token_text", "at2t_text", "e1n_t2t_text")
    pending = [u for u in utts if u not in done]
    gb = max(1, int(getattr(args, "gen_batch", 1) or 1))
    for c0 in range(0, len(pending), gb):
        chunk = pending[c0:c0 + gb]
        items = {u: ds.infer_item(u) for u in chunk}
        gen: dict = {}
        if gb > 1:  # batched LLM decode (see --gen-batch numerics caveat)
            bitems = []
            for u in chunk:
                it = items[u]
                d = {"input_ids": it["input_ids"].squeeze(0)}
                if torch.is_tensor(it.get("audio_mel")):
                    d["audio_mel"] = it["audio_mel"].squeeze(0).to(dtype)
                if torch.is_tensor(it.get("modality_mask")):
                    d["modality_mask"] = it["modality_mask"].squeeze(0)
                bitems.append(d)
            mx = min(1024, 2 * max(items[u]["input_ids"].shape[-1] for u in chunk) + 50)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
                outs = model.generate_batch(
                    bitems, max_new_tokens=mx, text_output=is_text, return_text=is_text)
            gen = dict(zip(chunk, outs))
        yield_rows = _score_chunk(
            args, chunk, items, gen, model, vocab, tokenizer, codec, metrics,
            flac, transcripts, dtype, task, is_text)
        for m in yield_rows:
            per_utt.append(m)
            if pu_fh is not None:
                pu_fh.write(json.dumps(m) + "\n"); pu_fh.flush()
        if (c0 // max(gb, 1)) % max(1, 25 // max(gb, 1)) == 0 and per_utt:
            LOG.info("[%s][%d/%d] %s", tag, c0 + len(chunk), len(pending),
                     {k: round(v, 3) for k, v in per_utt[-1].items() if isinstance(v, float)})
    if pu_fh is not None:
        pu_fh.close()
    agg = aggregate(per_utt); agg["n_utts"] = len(per_utt); agg["seconds"] = round(time.time() - t0, 1)
    ensure_dir(os.path.dirname(args.out))
    json.dump({"ckpt": args.ckpt, "split": args.split, "task": task, "agg": agg},
              open(args.out, "w"), indent=2)
    LOG.info("[%s] %s", tag, {k: round(v, 4) if isinstance(v, float) else v for k, v in agg.items()})


def _score_chunk(args, chunk, items, gen, model, vocab, tokenizer, codec, metrics,
                 flac, transcripts, dtype, task, is_text):
    """Generate (if not pre-generated) + decode + score each utt; yields metric rows."""
    for utt in chunk:
        item = items[utt]
        clean_ref = flac.get(utt) or item.get("clean_wav")  # libri map, else manifest (DNS)
        if not clean_ref and not args.no_ref:
            continue
        if utt in gen:
            audio_ids, text_ids = gen[utt]
        else:
            inp = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in item.items()}
            if "audio_mel" in inp:
                inp["audio_mel"] = inp["audio_mel"].to(dtype)
            n_noisy = int(item["input_ids"].shape[-1])
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
                audio_ids, text_ids = model.generate(
                    inp["input_ids"], attention_mask=inp["attention_mask"],
                    audio_mel=inp.get("audio_mel"), modality_mask=inp.get("modality_mask"),
                    max_new_tokens=min(1024, 2 * n_noisy + 50),
                    # text-output tasks decode both streams (text ends at eot -> pad_t);
                    # audio-only tasks keep the text row pad_t throughout.
                    text_output=is_text, return_text=is_text,
                )
        toks = [t for t in audio_ids if 0 <= t < vocab.audio_vocabsize]
        if not toks:
            continue
        # decode prompt: noisy (realistic) or clean (oracle speaker, isolates token quality)
        if args.prompt_wav_dir:  # use an external prompt (e.g. E1-enhanced audio)
            cand = glob.glob(os.path.join(args.prompt_wav_dir, f"{utt}.*"))
            prompt_src = cand[0] if cand else item["noisy_wav"]
        elif args.prompt == "clean" and clean_ref:
            prompt_src = clean_ref
        else:
            prompt_src = item["noisy_wav"]
        # prompt features (S3(prompt)+CAMPPlus+mel) are ~73% of decode cost and
        # fixed per (utt, prompt source) -> sidecar cache shared across evals.
        prep = None
        pc = None
        if args.prompt_cache:
            import hashlib
            pkey = hashlib.md5(str(prompt_src).encode()).hexdigest()[:10]
            pc = os.path.join(args.prompt_cache, f"{utt}__{pkey}.npz")
            if os.path.exists(pc):
                try:
                    z = np.load(pc)
                    prep = (torch.from_numpy(z["pt"]), torch.from_numpy(z["pf"]),
                            torch.from_numpy(z["emb"]))
                except Exception:
                    prep = None
        if prep is None:
            prompt_wav, psr = load_wav(prompt_src)
            prep = codec.prepare_prompt(prompt_wav, psr, prompt_strategy="self")
            if pc:
                try:
                    tmp = pc + f".{os.getpid()}.tmp.npz"
                    np.savez(tmp, pt=prep[0].cpu().numpy(), pf=prep[1].cpu().numpy(),
                             emb=prep[2].cpu().numpy())
                    os.replace(tmp, pc)
                except Exception:
                    pass
        dec_wav, dsr = codec.decode(torch.tensor(toks), prepared_prompt=prep)
        est16 = _np16(dec_wav, dsr)
        if args.no_ref:  # blind: no clean ref; ref-free metrics ignore ref16
            m = metrics.score(est16, est16)
        else:
            # ref16 + CAMPPlus(clean) are fixed per utt across every eval/tag/
            # prompt -> sidecar npz cache (temp+rename so parallel shards race
            # safely; a torn read just falls back to recompute).
            ref16 = ref_emb = None
            rc = os.path.join(args.ref_cache, f"{utt}.npz") if args.ref_cache else None
            if rc and os.path.exists(rc):
                try:
                    z = np.load(rc)
                    ref16, ref_emb = z["ref16"], z["ref_emb"]
                except Exception:
                    ref16 = ref_emb = None
            if ref16 is None:
                clean_wav, csr = load_wav(clean_ref)
                ref16 = _np16(clean_wav, csr)
                ref_emb = codec.extract_spk_emb(clean_wav, csr).numpy()
                if rc:
                    try:
                        tmp = rc + f".{os.getpid()}.tmp.npz"  # ends .npz: savez keeps name
                        np.savez(tmp, ref16=ref16, ref_emb=ref_emb)
                        os.replace(tmp, rc)
                    except Exception:
                        pass
            est_emb = codec.extract_spk_emb(dec_wav, dsr).numpy()
            m = metrics.score(ref16, est16, ref_text=transcripts.get(utt),
                              ref_emb=ref_emb, est_emb=est_emb)
        # text-WER: the model's directly-generated transcript vs reference (free,
        # no ASR) — for text-output models. Compare against the audio-WER above.
        if is_text and text_ids and transcripts.get(utt):
            import jiwer
            txt = [t for t in text_ids if 0 <= t < vocab.eot]  # drop specials
            pred = tokenizer.decode(txt, skip_special_tokens=True)
            tr = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                                jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
            try:
                m["text_wer"] = float(jiwer.wer(tr(transcripts[utt]), tr(pred)))
            except Exception:
                pass
        m["utt_id"] = utt
        yield m


if __name__ == "__main__":
    main()
