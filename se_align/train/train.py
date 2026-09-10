"""Phase-2 training loop (adapted from EchoMind engine/train.py).

Single-GPU or distributed (DDP / FSDP via torchrun). Supports the three task
types through one config: ``t2t`` | ``audio2token`` | ``audio2token_text``.

Launch (single GPU):
    CUDA_VISIBLE_DEVICES=0 python -m se_align.train.train --config configs/train_se_qwen0.5b.yaml

Launch (DDP, 4 GPUs):
    torchrun --nproc_per_node=4 -m se_align.train.train --config configs/train_se_qwen0.5b.yaml \
        --set distributed=ddp
"""
from __future__ import annotations

import argparse
import functools
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from ..utils.common import ensure_dir, get_logger, set_seed
from ..utils.config import load_config, parse_overrides
from .build import build_model
from .collator import SECollator
from .dataset import SEDataset

LOG = get_logger("se_align.train")


def _is_dist() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


class LengthGroupedBatchSampler(torch.utils.data.Sampler):
    """Shuffled windows sorted by length -> near-uniform-length batches.

    Cuts padding waste (and whisper padding-leakage: batched mels padded to the
    window max attend to near-zero padding) while keeping epoch-level shuffling.
    """

    def __init__(self, lengths, batch_size: int, window: int = 50, seed: int = 0,
                 drop_last: bool = True) -> None:
        self.lengths = list(lengths)
        self.bs = batch_size
        self.win = max(1, window) * batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def __len__(self) -> int:
        n = len(self.lengths)
        return n // self.bs if self.drop_last else math.ceil(n / self.bs)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        idx = torch.randperm(len(self.lengths), generator=g).tolist()
        batches = []
        for w0 in range(0, len(idx), self.win):
            win = sorted(idx[w0:w0 + self.win], key=lambda i: self.lengths[i])
            for b0 in range(0, len(win), self.bs):
                b = win[b0:b0 + self.bs]
                if len(b) == self.bs or not self.drop_last:
                    batches.append(b)
        order = torch.randperm(len(batches), generator=g).tolist()
        return iter(batches[i] for i in order)


def _setup_dist():
    import torch.distributed as dist

    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])


def _wrap_distributed(model, mode: str, local_rank: int):
    if mode == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP

        return DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    if mode == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        layer_cls = type(model.llm.model.layers[0])
        policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={layer_cls})
        return FSDP(model, auto_wrap_policy=policy, device_id=torch.cuda.current_device())
    return model


def _lr_lambda(step, warmup, total):
    if step < warmup:
        return step / max(warmup, 1)
    return max(0.0, (total - step) / max(total - warmup, 1))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, parse_overrides(args.set))
    set_seed(cfg.get("seed", 1986))

    dist_mode = cfg.get("distributed", "none")
    local_rank, rank, world = (0, 0, 1)
    if _is_dist():
        local_rank, rank, world = _setup_dist()
    device = f"cuda:{local_rank}"
    is_main = rank == 0
    dtype = torch.bfloat16 if cfg.get("bf16", True) else torch.float32

    task = cfg["task_type"]
    if is_main:
        LOG.info("task=%s world=%d dist=%s device=%s", task, world, dist_mode, device)

    # ---- model ----
    model, tokenizer, vocab = build_model(
        cfg["llm_path"], task_type=task, whisper_name=cfg.get("whisper_name", "large-v3"),
        whisper_ds_rate=cfg.get("whisper_ds_rate", 5), dtype=dtype, device=device,
        freeze_encoder=cfg.get("freeze_encoder", True),
        whisper_dtype=cfg.get("whisper_dtype", "fp32"),
        grad_ckpt=bool(cfg.get("grad_ckpt", False)),
    )
    # ---- auxiliary evidence trust-region loss (L = CE + beta * L_TR) ----
    if cfg.get("use_trust_region_loss", False):
        from .fsq_neighbors import load_neighbors
        tr_r = int(cfg.get("trust_region_radius", 2))
        nbr_ids, nbr_mask = load_neighbors(tr_r, cfg.get("trust_region_neighbor_cache", ""))
        model.set_trust_region(
            beta=float(cfg.get("trust_region_beta", 0.1)),
            neighbor_ids=nbr_ids.to(device), neighbor_mask=nbr_mask.to(device),
            reliable_only=bool(cfg.get("trust_region_reliable_only", True)),
            reliable_radius=int(cfg.get("trust_region_reliable_radius", 2)),
        )
        if is_main:
            LOG.info("trust-region loss ON: beta=%s r=%d (K=%d) reliable_only=%s r_rel=%d",
                     cfg.get("trust_region_beta", 0.1), tr_r, nbr_ids.shape[1],
                     cfg.get("trust_region_reliable_only", True),
                     cfg.get("trust_region_reliable_radius", 2))

    # Load warmed-up weights from a Stage-1 pretrain checkpoint (e.g. text->token).
    if cfg.get("init_from"):
        sd = torch.load(cfg["init_from"], map_location="cpu")
        miss, unexp = model.load_state_dict(sd, strict=False)
        if is_main:
            LOG.info("init_from %s (missing=%d unexpected=%d)", cfg["init_from"], len(miss), len(unexp))

    if cfg.get("freeze_llm", False):
        for p in model.llm.parameters():
            p.requires_grad = False

    # Warm up only the newly-inserted CV3 audio-token embeddings: freeze everything,
    # train just the embedding table, and mask grads on the original text rows so
    # only rows [audio_shift:] (the new tokens) update. (tied lm_head -> trains the
    # audio output projection too.)
    if cfg.get("train_new_tokens_only", False):
        for p in model.parameters():
            p.requires_grad = False
        emb = model.llm.get_input_embeddings().weight
        emb.requires_grad = True
        shift = vocab.audio_shift
        emb.register_hook(lambda g: torch.cat([torch.zeros_like(g[:shift]), g[shift:]], dim=0))
        if is_main:
            LOG.info("train_new_tokens_only: training audio-token embedding rows [%d:%d]",
                     shift, vocab.total_vocabsize)

    # Optional LoRA on the backbone (cheap extra capacity, less overfitting).
    if cfg.get("use_lora", False):
        from peft import LoraConfig, get_peft_model

        lc = LoraConfig(
            r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            target_modules=cfg.get("lora_targets", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            task_type="CAUSAL_LM",
        )
        model.llm = get_peft_model(model.llm, lc)
        if is_main:
            LOG.info("LoRA enabled (r=%d alpha=%d)", cfg.get("lora_r", 16), cfg.get("lora_alpha", 32))

    # ---- data ----
    if task == "text2token":
        from .pretrain_data import PretrainTextToTokenDataset

        full = PretrainTextToTokenDataset(
            cfg["pretrain_manifest"], cfg["pretrain_tokens_root"], vocab, tokenizer,
            prompt=cfg.get("prompt", ""), limit=cfg.get("limit"),
        )
    elif task == "corrupt2clean_text":
        from .pretrain_data import PretrainCorrupt2CleanDataset

        full = PretrainCorrupt2CleanDataset(
            cfg["pretrain_manifest"], cfg["pretrain_tokens_root"], vocab, tokenizer,
            corruption_ratio=cfg.get("corruption_ratio", 0.25), limit=cfg.get("limit"),
        )
    else:
        full = SEDataset(
            cfg["tokens_root"], cfg.get("train_split", "train"), vocab, tokenizer, task_type=task,
            data_root=cfg.get("data_root"), txt_subdir=cfg.get("txt_subdir"),
            cv3_root=cfg.get("cv3_root"), whisper_ds_rate=cfg.get("whisper_ds_rate", 5),
            limit=cfg.get("limit"), noisy_wav_root=cfg.get("noisy_wav_root"),
        )
    # Hold out a dev set from train for checkpoint selection. Speaker-disjoint if
    # val_speakers is given (utt ids start with the speaker prefix, e.g. p287_001),
    # otherwise a seeded random hold-out of val_size utterances.
    from torch.utils.data import Subset

    val_speakers = cfg.get("val_speakers")
    val_size = cfg.get("val_size") or 0
    dev_split = cfg.get("dev_split")
    ds, val_ds = full, None
    if dev_split and task not in ("text2token", "corrupt2clean_text"):
        # dedicated dev split (e.g. synthetic 'dev' with held-out noise)
        ds = full
        val_ds = SEDataset(
            cfg["tokens_root"], dev_split, vocab, tokenizer, task_type=task,
            data_root=cfg.get("data_root"), txt_subdir=cfg.get("txt_subdir"),
            cv3_root=cfg.get("cv3_root"), whisper_ds_rate=cfg.get("whisper_ds_rate", 5),
            limit=(val_size or None), noisy_wav_root=cfg.get("noisy_wav_root"),
        )
    elif val_speakers and hasattr(full, "utts"):
        vs = set(val_speakers)
        val_idx = [i for i, u in enumerate(full.utts) if u.split("_")[0] in vs]
        tr_idx = [i for i, u in enumerate(full.utts) if u.split("_")[0] not in vs]
        ds, val_ds = Subset(full, tr_idx), Subset(full, val_idx)
    elif val_size and val_size < len(full):
        g = torch.Generator().manual_seed(cfg.get("seed", 1986))
        perm = torch.randperm(len(full), generator=g).tolist()
        ds, val_ds = Subset(full, perm[val_size:]), Subset(full, perm[:val_size])
    if is_main and val_ds is not None:
        LOG.info("train=%d dev=%d (selection by dev loss)", len(ds), len(val_ds))

    sampler = None
    if _is_dist():
        from torch.utils.data import DistributedSampler

        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    nw = cfg.get("num_workers", 4)
    dl_kw = dict(
        num_workers=nw, collate_fn=SECollator(vocab), pin_memory=True,
        persistent_workers=nw > 0, prefetch_factor=(4 if nw > 0 else None),
    )
    batch_sampler = None
    if sampler is None and cfg.get("length_bucketing", True):
        # length-grouped batches: less padding waste + less whisper padding leakage
        try:
            lens = ds.dataset.token_lengths() if hasattr(ds, "dataset") else ds.token_lengths()
            if hasattr(ds, "indices"):  # Subset
                lens = [lens[i] for i in ds.indices]
            batch_sampler = LengthGroupedBatchSampler(
                lens, cfg.get("batch_size", 4), seed=int(cfg.get("seed", 1986)))
            LOG.info("length bucketing ON (%d utts)", len(lens))
        except Exception as e:  # noqa: BLE001 — fall back to plain shuffling
            LOG.warning("length bucketing disabled: %s", e)
    if batch_sampler is not None:
        loader = DataLoader(ds, batch_sampler=batch_sampler, **dl_kw)
    else:
        loader = DataLoader(
            ds, batch_size=cfg.get("batch_size", 4), shuffle=(sampler is None),
            sampler=sampler, drop_last=True, **dl_kw,
        )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=cfg.get("val_batch_size", cfg.get("batch_size", 4)),
            shuffle=False, **dl_kw,
        )

    model = _wrap_distributed(model, dist_mode, local_rank) if _is_dist() else model

    # ---- optim ----
    epochs = cfg.get("num_epochs", 1)
    accum = cfg.get("grad_accum", 1)
    total_steps = max(1, (len(loader) // accum) * epochs)
    warmup = cfg.get("warmup_steps") or int(0.03 * total_steps)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.get("lr", 1e-4), weight_decay=cfg.get("weight_decay", 0.01),
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda s: _lr_lambda(s, warmup, total_steps)
    )
    clip = cfg.get("grad_clip", 1.0)
    ckpt_dir = ensure_dir(cfg.get("ckpt_dir", "exp/se_qwen0.5b"))
    log_every = cfg.get("log_every", 20)
    save_every = cfg.get("save_every", 1000)

    keep_best = cfg.get("keep_best", 5)
    best_ckpts: list = []  # (dev_loss, path), kept sorted ascending

    if is_main:
        ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        LOG.info("trainable=%.1fM steps/epoch=%d total=%d warmup=%d", ntr / 1e6, len(loader), total_steps, warmup)

    # ---- early stopping (dev-loss patience; empirically epoch 2 regresses) ----
    patience = int(cfg.get("patience", 2))          # consecutive bad vals; 0=off
    min_delta = float(cfg.get("min_delta", 0.0))
    val_every_steps = int(cfg.get("val_every_steps", 0))  # mid-epoch val; 0=off
    max_steps = int(cfg.get("max_steps", 0))        # hard gstep cap (smoke); 0=off
    es_state = {"best": float("inf"), "bad": 0, "stop": False}

    def _run_val(tag) -> None:
        """Validate + track best ckpt + update early-stop state (all ranks)."""
        dev_loss = _validate(model, val_loader, device, dtype)
        if _is_dist():
            import torch.distributed as dist
            t = torch.tensor([dev_loss], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            dev_loss = float(t.item())
        if is_main:
            LOG.info("%s DEV loss=%.4f", tag, dev_loss)
            _keep_best(model, ckpt_dir, tag.replace("ep", "", 1), dev_loss,
                       best_ckpts, keep_best)
        if dev_loss < es_state["best"] - min_delta:
            es_state["best"], es_state["bad"] = dev_loss, 0
        else:
            es_state["bad"] += 1
            if patience and es_state["bad"] >= patience:
                es_state["stop"] = True
                if is_main:
                    LOG.info("early stop: %d consecutive non-improving vals "
                             "(best=%.4f)", es_state["bad"], es_state["best"])

    # ---- loop ----
    model.train()
    gstep = 0
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        if batch_sampler is not None:
            batch_sampler.set_epoch(epoch)
        t0 = time.time()
        optim.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            if "audio_mel" in batch:
                batch["audio_mel"] = batch["audio_mel"].to(dtype)
            with torch.autocast(device_type="cuda", dtype=dtype):
                out = model(**batch)
                loss = out["loss"] / accum
            if not torch.isfinite(loss):
                optim.zero_grad(set_to_none=True)
                continue
            loss.backward()
            if (step + 1) % accum == 0:
                if clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                gstep += 1
                if is_main and gstep % log_every == 0:
                    ll = out["layer_loss"]
                    tr = out.get("tr_stats") or {}
                    tr_str = ("" if not tr else
                              " ce=%.4f tr=%.4f mass=%.3f rel=%.2f" % (
                                  tr.get("loss_ce", float("nan")),
                                  tr.get("loss_trust_region", float("nan")),
                                  tr.get("trust_region_mass", float("nan")),
                                  tr.get("reliable_frame_ratio", float("nan"))))
                    LOG.info(
                        "ep%d gstep%d loss=%.4f audio=%.4f text=%.4f%s lr=%.2e %.2fs/it",
                        epoch, gstep, float(out["loss"]), float(ll[0]), float(ll[1]),
                        tr_str, sched.get_last_lr()[0], (time.time() - t0) / max(step + 1, 1),
                    )
                if is_main and save_every and gstep % save_every == 0:
                    _save(model, ckpt_dir, gstep)
                if (val_loader is not None and val_every_steps
                        and gstep % val_every_steps == 0):
                    _run_val(f"ep{epoch}s{gstep}")
                    if es_state["stop"]:
                        break
                if max_steps and gstep >= max_steps:
                    es_state["stop"] = True
                    if is_main:
                        LOG.info("max_steps=%d reached", max_steps)
                    break
        # ---- epoch end: validate + checkpoint selection ----
        if es_state["stop"]:
            break
        if val_loader is not None:
            _run_val(f"ep{epoch}")
            if es_state["stop"]:
                break
        elif is_main:
            _save(model, ckpt_dir, f"epoch{epoch}", keep_last_epochs=cfg.get("keep_last_epochs"))
    if is_main and best_ckpts:
        LOG.info("best %d checkpoints by dev loss: %s", len(best_ckpts),
                 [(round(s, 4), os.path.basename(p)) for s, p in best_ckpts])
    if _is_dist():
        import torch.distributed as dist
        dist.destroy_process_group()


@torch.no_grad()
def _validate(model, loader, device, dtype) -> float:
    """Mean teacher-forced loss over the dev set (cheap, no generation)."""
    m = model.module if hasattr(model, "module") else model
    was_training = m.training
    m.eval()
    tot, n = 0.0, 0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        if "audio_mel" in batch:
            batch["audio_mel"] = batch["audio_mel"].to(dtype)
        with torch.autocast(device_type="cuda", dtype=dtype):
            out = m(**batch)
        if out["loss"] is not None and torch.isfinite(out["loss"]):
            tot += float(out["loss"]); n += 1
    if was_training:
        m.train()
    return tot / max(n, 1)


def _slim_state_dict(m) -> dict:
    """state_dict without the frozen whisper encoder (2.5 GB fp32 per ckpt).

    build_model reconstructs the encoder from the pretrained weights at load
    time and ckpts are applied with strict=False, so dropping ``encoder.*`` is
    safe. ``encoder_projector.*`` (trainable) is kept -- note the prefix check
    is ``encoder.`` with the dot, which does not match ``encoder_projector.``.
    """
    enc = getattr(m, "encoder", None)
    drop_enc = enc is not None and not any(p.requires_grad for p in enc.parameters())
    sd = m.state_dict()
    if drop_enc:
        sd = {k: v for k, v in sd.items() if not k.startswith("encoder.")}
    return {k: v.detach().cpu() for k, v in sd.items()}


def _keep_best(model, ckpt_dir: str, epoch, dev_loss: float, best: list, keep: int) -> None:
    """Save this epoch and keep only the ``keep`` lowest-dev-loss checkpoints."""
    import json

    path = os.path.join(ckpt_dir, f"model_epoch{epoch}_dev{dev_loss:.4f}.pt")
    m = model.module if hasattr(model, "module") else model
    torch.save(_slim_state_dict(m), path)
    best.append((dev_loss, path))
    best.sort(key=lambda x: x[0])  # ascending: best (lowest) first
    for _, p in best[keep:]:
        if os.path.exists(p):
            os.remove(p)
            LOG.info("pruned non-top checkpoint %s", os.path.basename(p))
    del best[keep:]
    with open(os.path.join(ckpt_dir, "best_checkpoints.json"), "w") as fh:
        json.dump([{"dev_loss": s, "ckpt": os.path.basename(p)} for s, p in best], fh, indent=2)
    LOG.info("saved %s | current best: %s", os.path.basename(path),
             [round(s, 4) for s, _ in best])


def _save(model, ckpt_dir: str, tag, keep_last_epochs=None, keep_last_steps=2) -> None:
    m = model.module if hasattr(model, "module") else model
    sd = _slim_state_dict(m)
    path = os.path.join(ckpt_dir, f"model_{tag}.pt")
    torch.save(sd, path)
    LOG.info("saved checkpoint %s (%d tensors)", path, len(sd))
    import glob
    import re

    if keep_last_epochs and str(tag).startswith("epoch"):
        eps = sorted(
            glob.glob(os.path.join(ckpt_dir, "model_epoch*.pt")),
            key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)),
        )
        for old in eps[:-keep_last_epochs]:
            os.remove(old)
            LOG.info("pruned old checkpoint %s", old)
    if keep_last_steps and str(tag).isdigit():
        # save_every intermediates (model_2000.pt ...): never used for model
        # selection, so keep only the newest few instead of growing unbounded.
        steps = sorted(
            (p for p in glob.glob(os.path.join(ckpt_dir, "model_*.pt"))
             if re.fullmatch(r"model_\d+\.pt", os.path.basename(p))),
            key=lambda p: int(re.search(r"model_(\d+)\.pt", p).group(1)),
        )
        for old in steps[:-keep_last_steps]:
            os.remove(old)
            LOG.info("pruned intermediate checkpoint %s", os.path.basename(old))


if __name__ == "__main__":
    main()
