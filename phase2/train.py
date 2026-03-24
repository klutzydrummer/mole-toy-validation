"""
Phase 2 Training Script — HDC (Hierarchical Dynamic Chunking).

Usage:
  python phase2/train.py --config hdc_rulebased   # ALWAYS run first
  python phase2/train.py --config hdc_gate
  python phase2/train.py --config hdc_stride
  python phase2/train.py --config hdc_r2
  python phase2/train.py --config hdc_r8
  python phase2/train.py --config hdc_e2e_isolated  # A5 comparison only
  python phase2/train.py --config hdc_gate --resume

Dual-loss training:
  loss = loss_ntp + lambda_comp * loss_comp
  loss_ntp  = cross_entropy(logits, targets)
  loss_comp = (mean(boundary_probs) - 1/R)^2

  For hdc_rulebased and hdc_stride: lambda_comp = 0 (no learned router to regularize).
  For learned configs: lambda_comp = 0.1 (default).

Logs to checkpoints/<config>.jsonl (same format as Phase 1 + hdc fields).
"""

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2.model import HDCModel
from utils.data import get_dataloader, get_vocab_size, set_dataset, set_tokenizer
from utils.device import configure_sdpa, get_amp_config, get_device
from utils.metrics import ParamCounter, TrainLogger, ce_to_bpc

try:
    from litlogger import LightningLogger as _LightningLogger
    def make_lit_logger(name, teamspace):
        return _LightningLogger(name=name, teamspace=teamspace)
except Exception:
    def make_lit_logger(name, teamspace):
        return None


# Configs that have no learned router — compression ratio is always exactly 1/R
_NO_ROUTER_CONFIGS = {"hdc_rulebased", "hdc_stride", "hdc_upcycle_stride"}


def get_lr_scale(step, warmup_steps=1000, total_steps=50000, min_scale=1e-5/3e-4):
    """Linear warmup + cosine decay. Returns a scale in [min_scale, 1.0].
    Each param group multiplies this by its own peak_lr, so groups with
    different peak LRs (e.g. router) follow the same cosine shape."""
    if step < warmup_steps:
        return min_scale + (1.0 - min_scale) * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_scale + 0.5 * (1.0 - min_scale) * (1 + math.cos(math.pi * min(progress, 1.0)))


def _unwrap(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _lit_log(lit, metrics: dict, step: int = None) -> None:
    """Wrapper around lit.log_metrics that silently drops NaN/inf values.
    Lightning's API serializes metrics to JSON; float('nan') becomes 'NaN'
    which is invalid JSON and raises a 400 error from the Lightning cloud."""
    import math
    clean = {k: v for k, v in metrics.items() if isinstance(v, float) and math.isfinite(v)}
    if not clean:
        return
    try:
        if step is not None:
            lit.log_metrics(clean, step=step)
        else:
            lit.log_metrics(clean)
    except Exception:
        pass  # Don't let logging failures crash training


def boundary_entropy(boundary_probs: torch.Tensor) -> float:
    """Entropy of the boundary probability distribution — higher = more uniform (worse)."""
    # Clamp with 1e-6, NOT 1e-8. Float32 ULP at 1.0 is ~1.19e-7, so
    # 1.0 - 1e-8 == 1.0 exactly in float32 — p=1.0 is not actually clamped,
    # (1-1.0)*log(0) = 0*(-inf) = NaN (IEEE 754). 1e-6 > ULP so it works.
    p = boundary_probs.detach().float().clamp(1e-6, 1.0 - 1e-6)
    ent = -(p * p.log() + (1 - p) * (1 - p).log()).mean().item()
    return ent


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=50, amp_dtype=torch.bfloat16):
    """Returns (val_loss, val_bpc, compression_ratio)."""
    model.eval()
    total_loss  = 0.0
    total_ratio = 0.0
    n_batches   = 0
    for x, y in val_loader:
        if n_batches >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=(device.type == "cuda")):
            logits, bp = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss  += loss.item()
        total_ratio += bp.mean().item()
        n_batches   += 1
    model.train()
    n = max(1, n_batches)
    return total_loss / n, ce_to_bpc(total_loss / n), total_ratio / n


@torch.no_grad()
def evaluate_per_position(model, val_loader, device, boundary_idx_key, max_batches=20, amp_dtype=torch.bfloat16):
    """
    Compute boundary BPC vs. mid-chunk BPC separately.
    Boundary tokens: positions that were selected as concept tokens.
    Mid-chunk tokens: all others.
    Returns (boundary_bpc, midchunk_bpc).
    """
    model.eval()
    boundary_loss_sum = 0.0
    midchunk_loss_sum = 0.0
    boundary_count    = 0
    midchunk_count    = 0

    for x, y in val_loader:
        if boundary_count + midchunk_count >= max_batches * x.numel():
            break
        x, y = x.to(device), y.to(device)
        B, L  = x.shape
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=(device.type == "cuda")):
            logits, bp = model(x)

        # Get boundary mask from the model's last forward pass
        # Re-run to get boundary_idx (not stored on model)
        raw = _unwrap(model)
        h   = raw.embed(x)
        _, _, _, _, bidx = raw.zone_e(h)            # [B, M]

        # Build binary boundary mask [B, L]
        bmask = torch.zeros(B, L, dtype=torch.bool, device=device)
        bmask.scatter_(1, bidx, True)

        # Per-token cross-entropy
        log_probs = F.log_softmax(logits, dim=-1)   # [B, L, V]
        tok_loss  = -log_probs.gather(2, y.unsqueeze(-1)).squeeze(-1)  # [B, L]

        boundary_loss_sum += tok_loss[bmask].sum().item()
        boundary_count    += bmask.sum().item()
        midchunk_loss_sum += tok_loss[~bmask].sum().item()
        midchunk_count    += (~bmask).sum().item()

    model.train()
    b_bpc = ce_to_bpc(boundary_loss_sum / max(1, boundary_count))
    m_bpc = ce_to_bpc(midchunk_loss_sum / max(1, midchunk_count))
    return b_bpc, m_bpc


def train(
    config:         str   = "hdc_rulebased",
    d:              int   = 512,
    n_layers:       int   = 8,
    n_heads:        int   = 8,
    seq_len:        int   = 256,
    batch_size:     int   = 32,
    total_steps:    int   = 50000,
    eval_interval:  int   = 2500,
    log_interval:   int   = 100,
    max_lr:         float = 3e-4,
    router_lr:      float = 1e-4,
    lambda_comp:    float = 0.1,
    n_experts:      int   = 8,
    mol_rank:       int   = 8,
    mol_top_k:      int   = 2,
    mol_ckpt:       str   = "",
    resume:         bool  = False,
    no_compile:     bool  = False,
    ckpt_dir:       str   = "checkpoints",
    teamspace:      str   = "mole-toy-validation-project",
    tokenizer:      str   = "bpe",
    dataset:        str   = "wikitext103",
    seed:           int   = 42,
):
    print(f"\n{'='*60}")
    print(f"Phase 2 Training: config={config}")
    print(f"{'='*60}\n")

    set_dataset(dataset)
    set_tokenizer(tokenizer)
    vocab_size = get_vocab_size()

    # Seed all PRNGs for reproducibility. Must happen before model construction,
    # DataLoader creation, and any random op. Saved in every checkpoint so that
    # resumed runs can document what seed produced their results.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = get_device()
    use_amp, amp_dtype, use_grad_scaler = get_amp_config(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    sdpa_desc = configure_sdpa(device)

    if device.type == "cpu":
        print("WARNING: Running on CPU. This will be very slow.")
    else:
        print(f"AMP dtype: {amp_dtype} | GradScaler: {use_grad_scaler} | SDPA: {sdpa_desc}")

    model = HDCModel(
        config=config, d=d, n_layers=n_layers, n_heads=n_heads,
        vocab_size=vocab_size, seq_len=seq_len,
        n_experts=n_experts, mol_rank=mol_rank, mol_top_k=mol_top_k,
        mol_ckpt=mol_ckpt,
    ).to(device)

    n_params = ParamCounter.count(model)

    # lambda_comp = 0 for configs without a learned router
    effective_lambda = 0.0 if config in _NO_ROUTER_CONFIGS else lambda_comp

    train_loader = get_dataloader("train", seq_len=seq_len, batch_size=batch_size)
    val_loader   = get_dataloader("val",   seq_len=seq_len, batch_size=batch_size)

    # Separate param groups: router (W_q, W_k) gets a lower LR to prevent the
    # gradient explosion that occurs when routing activates after the cold-start
    # phase. Both groups follow the same cosine decay shape via get_lr_scale.
    # Three param groups:
    #   main decay    — 2D+ weight matrices (main network, not router), weight_decay=0.1
    #   main nodecay  — 1D params (biases, RMSNorm scales, log_a recurrence params), wd=0.0
    #   router        — W_q, W_k (2D), lower LR, weight_decay=0.1
    # 1D params include RMSNorm.weight [d], CausalRecurrenceLayer.log_a [d_outer], biases.
    _router = model.zone_e.router
    _router_ids = {id(p) for p in _router.parameters() if p.requires_grad}
    _main_decay   = [p for p in model.parameters()
                     if p.requires_grad and id(p) not in _router_ids and p.ndim >= 2]
    _main_nodecay = [p for p in model.parameters()
                     if p.requires_grad and id(p) not in _router_ids and p.ndim < 2]
    _router_params = [p for p in model.parameters() if p.requires_grad and id(p) in _router_ids]
    _param_groups  = [
        {"params": _main_decay,   "peak_lr": max_lr,   "lr": max_lr,   "weight_decay": 0.1},
        {"params": _main_nodecay, "peak_lr": max_lr,   "lr": max_lr,   "weight_decay": 0.0},
    ]
    if _router_params:
        _param_groups.append(
            {"params": _router_params, "peak_lr": router_lr, "lr": router_lr, "weight_decay": 0.1}
        )
    optimizer = torch.optim.AdamW(
        _param_groups, betas=(0.9, 0.95), fused=(device.type == "cuda"),
    )

    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_prefix  = f"{config}_seed{seed}"
    resume_path  = os.path.join(ckpt_dir, f"{ckpt_prefix}_latest.pt")
    start_step   = 0
    best_val_bpc = float("inf")

    if resume and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}...")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if use_grad_scaler and "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        if "rng_state" in ckpt:
            torch.set_rng_state(ckpt["rng_state"].cpu())
            if torch.cuda.is_available() and "cuda_rng_state" in ckpt:
                torch.cuda.set_rng_state(ckpt["cuda_rng_state"].cpu())
            random.setstate(ckpt["python_rng_state"])
            np.random.set_state(ckpt["numpy_rng_state"])
        start_step   = ckpt["step"] + 1
        best_val_bpc = ckpt.get("best_val_bpc", float("inf"))
        print(f"  Resumed at step {start_step}, best_val_bpc={best_val_bpc:.4f}")

    if device.type == "cuda" and not no_compile:
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile failed ({e}), continuing without it")

    run_name = f"{config}_seed{seed}"
    logger = TrainLogger(ckpt_dir, run_name=run_name)
    lit    = make_lit_logger(name=f"phase2-{config}", teamspace=teamspace)
    if lit is not None:
        lit.log_hyperparams({
            "config": config, "d": d, "n_layers": n_layers, "n_heads": n_heads,
            "seq_len": seq_len, "batch_size": batch_size, "total_steps": total_steps,
            "max_lr": max_lr, "lambda_comp": effective_lambda,
            "R": _unwrap(model).R, "n_params": n_params,
        })
        print("LightningLogger active — metrics streaming to Teamspace")

    model.train()
    train_iter      = iter(train_loader)
    start_time      = time.time()
    log_loss_accum  = 0.0
    log_comp_accum  = 0.0
    log_ent_accum   = 0.0
    log_count       = 0

    print(f"Training steps {start_step} → {total_steps}")
    print(f"AMP: {use_amp} | batch: {batch_size} | seq_len: {seq_len}")
    print(f"R: {_unwrap(model).R} | lambda_comp: {effective_lambda}")
    print(f"Train batches/epoch: {len(train_loader)}\n")

    for step in range(start_step, total_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        lr_scale = get_lr_scale(step, total_steps=total_steps)
        lr = max_lr * lr_scale   # main LR for logging
        for pg in optimizer.param_groups:
            pg["lr"] = pg["peak_lr"] * lr_scale

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits, bp = model(x)
            loss_ntp   = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss_comp  = (bp.mean() - 1.0 / _unwrap(model).R) ** 2
            loss       = loss_ntp + effective_lambda * loss_comp

        if use_grad_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            optimizer.step()

        log_loss_accum += loss_ntp.item()
        log_comp_accum += loss_comp.item()
        log_ent_accum  += boundary_entropy(bp)
        log_count      += 1

        if (step + 1) % log_interval == 0:
            avg_loss = log_loss_accum / log_count
            avg_comp = log_comp_accum / log_count
            avg_ent  = log_ent_accum  / log_count
            comp_ratio = bp.detach().mean().item()

            logger.log_step(step, avg_loss, lr, grad_norm)

            # HDC-specific fields appended to the same JSONL
            with open(os.path.join(ckpt_dir, f"{run_name}.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step": step, "type": "hdc",
                    "loss_comp": avg_comp,
                    "compression_ratio": comp_ratio,
                    "boundary_entropy": avg_ent,
                }) + "\n")

            logger.print_step(step, avg_loss, lr, grad_norm, interval=1)
            print(f"    comp={comp_ratio:.3f}  loss_comp={avg_comp:.5f}  ent={avg_ent:.3f}")

            if lit is not None:
                _lit_log(lit, {
                    "train/loss": avg_loss, "train/bpc": ce_to_bpc(avg_loss),
                    "train/lr": lr, "train/grad_norm": grad_norm,
                    "hdc/compression_ratio": comp_ratio,
                    "hdc/loss_comp": avg_comp,
                    "hdc/boundary_entropy": avg_ent,
                }, step=step)

            log_loss_accum = log_comp_accum = log_ent_accum = log_count = 0

        if (step + 1) % eval_interval == 0:
            val_loss, val_bpc, val_ratio = evaluate(model, val_loader, device, amp_dtype=amp_dtype)
            logger.log_eval(step, val_loss)
            logger.print_eval(step, val_loss, val_bpc)
            print(f"  >>> val compression ratio: {val_ratio:.3f} (target {1/(_unwrap(model).R):.3f})")

            # Per-position loss breakdown
            b_bpc, m_bpc = evaluate_per_position(model, val_loader, device, None, amp_dtype=amp_dtype)
            print(f"  >>> boundary BPC: {b_bpc:.4f}  mid-chunk BPC: {m_bpc:.4f}  "
                  f"delta: {m_bpc - b_bpc:+.4f}")
            with open(os.path.join(ckpt_dir, f"{run_name}.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step": step, "type": "pos_loss",
                    "boundary_bpc": b_bpc, "midchunk_bpc": m_bpc,
                }) + "\n")

            # MoL stats
            mol_stats = _unwrap(model).get_mol_stats()
            if mol_stats:
                avg_bal = sum(s["expert_balance"] for s in mol_stats) / len(mol_stats)
                print(f"  >>> MoL avg balance: {avg_bal:.3f}")
                logger.log_mol_stats(step, mol_stats)
                if lit is not None:
                    _lit_log(lit, {"mol/avg_balance": avg_bal}, step=step)
            _unwrap(model).reset_mol_counts()

            if lit is not None:
                _lit_log(lit, {
                    "val/loss": val_loss, "val/bpc": val_bpc,
                    "hdc/val_compression_ratio": val_ratio,
                    "hdc/boundary_bpc": b_bpc, "hdc/midchunk_bpc": m_bpc,
                }, step=step)

            if val_bpc < best_val_bpc:
                best_val_bpc = val_bpc
                ckpt_path = os.path.join(ckpt_dir, f"{ckpt_prefix}_best.pt")
                torch.save({
                    "step": step, "model_state": _unwrap(model).state_dict(),
                    "val_bpc": val_bpc, "config": config, "n_params": n_params,
                    "seed": seed,
                }, ckpt_path)
                print(f"  >>> New best! Saved to {ckpt_path}")

            ckpt = {
                "step": step,
                "model_state": _unwrap(model).state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_bpc": best_val_bpc,
                "seed": seed,
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
            }
            if use_grad_scaler:
                ckpt["scaler_state"] = scaler.state_dict()
            torch.save(ckpt, resume_path)

    # Final eval
    val_loss, val_bpc, _ = evaluate(model, val_loader, device, max_batches=200, amp_dtype=amp_dtype)
    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Training complete: {config}")
    print(f"  Steps: {start_step} → {total_steps}")
    print(f"  Time:  {elapsed/60:.1f} min")
    print(f"  Final val BPC:  {val_bpc:.4f}")
    print(f"  Best val BPC:   {best_val_bpc:.4f}")
    print(f"  Params: {n_params:,}")
    print(f"{'='*60}\n")

    summary = {
        "config": config, "n_params": n_params,
        "total_steps": total_steps, "final_val_bpc": val_bpc,
        "best_val_bpc": best_val_bpc, "elapsed_seconds": elapsed,
        "d": d, "n_layers": n_layers, "n_heads": n_heads,
        "seq_len": seq_len, "batch_size": batch_size,
        "R": _unwrap(model).R, "lambda_comp": effective_lambda,
        "tokenizer": tokenizer, "dataset": dataset, "vocab_size": vocab_size,
        "seed": seed,
    }
    with open(os.path.join(ckpt_dir, f"{ckpt_prefix}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if lit is not None:
        _lit_log(lit, {"final/best_val_bpc": best_val_bpc, "final/val_bpc": val_bpc})
        lit.finalize()

    return best_val_bpc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 HDC Training")
    parser.add_argument("--config", type=str, default="hdc_rulebased",
                        choices=list(HDCModel.CONFIGS))
    parser.add_argument("--d",              type=int,   default=512)
    parser.add_argument("--n_layers",       type=int,   default=8)
    parser.add_argument("--n_heads",        type=int,   default=8)
    parser.add_argument("--seq_len",        type=int,   default=256)
    parser.add_argument("--tokenizer",      type=str,   default="bpe", choices=["char", "bpe"])
    parser.add_argument("--dataset",        type=str,   default="wikitext103", choices=["wikitext103", "enwik8"])
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--total_steps",    type=int,   default=50000)
    parser.add_argument("--eval_interval",  type=int,   default=2500)
    parser.add_argument("--log_interval",   type=int,   default=100)
    parser.add_argument("--max_lr",         type=float, default=3e-4)
    parser.add_argument("--router_lr",      type=float, default=1e-4,
                        help="LR for router params (W_q, W_k). Lower than main to prevent cold-start explosion.")
    parser.add_argument("--lambda_comp",    type=float, default=0.1,
                        help="Compression regularizer weight. 0 for rulebased/stride.")
    parser.add_argument("--mol_ckpt",       type=str,   default="",
                        help="Path to mol_best.pt for upcycle configs (freeze_inner=True).")
    parser.add_argument("--n_experts",      type=int,   default=8)
    parser.add_argument("--mol_rank",       type=int,   default=8)
    parser.add_argument("--mol_top_k",      type=int,   default=2)
    parser.add_argument("--resume",         action="store_true")
    parser.add_argument("--no_compile",     action="store_true")
    parser.add_argument("--seed",           type=int,   default=42,
                        help="Global random seed for reproducibility (torch, cuda, numpy, random).")
    parser.add_argument("--ckpt_dir",       type=str,   default="checkpoints")
    parser.add_argument("--teamspace",      type=str,   default="mole-toy-validation-project")
    args = parser.parse_args()
    train(**vars(args))
