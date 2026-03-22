"""
Phase 1 Training Script.

Usage:
  python phase1/train.py --config baseline
  python phase1/train.py --config mhc
  python phase1/train.py --config mol
  python phase1/train.py --config compose
  python phase1/train.py --config mol --resume

Logs to checkpoints/<config>.jsonl. Saves best + resume checkpoints.
"""

import argparse
import os
import sys
import time
import math
import json
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1.model import ToyTransformer
from utils.data import get_dataloader, set_dataset, set_tokenizer, get_vocab_size
from utils.metrics import ce_to_bpc, TrainLogger, ParamCounter

# LightningLogger is only available inside Lightning.ai Studios — degrade gracefully elsewhere.
try:
    from litlogger import LightningLogger as _LightningLogger
    def make_lit_logger(name, teamspace):
        return _LightningLogger(name=name, teamspace=teamspace)
except Exception:
    def make_lit_logger(name, teamspace):
        return None


def get_lr(step, warmup_steps=1000, max_lr=3e-4, min_lr=1e-5, total_steps=50000):
    if step < warmup_steps:
        return min_lr + (max_lr - min_lr) * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def _unwrap(model):
    """Get raw model from compiled wrapper."""
    return model._orig_mod if hasattr(model, "_orig_mod") else model


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=50):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for x, y in val_loader:
        if n_batches >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(1, n_batches)


def train(config: str, d: int = 512, n_layers: int = 8, n_heads: int = 8,
          seq_len: int = 256, batch_size: int = 32, total_steps: int = 100000,
          eval_interval: int = 2500, log_interval: int = 100,
          max_lr: float = 3e-4, ckpt_dir: str = "checkpoints",
          mhc_dynamic: bool = False, n_experts: int = 8,
          mol_rank: int = 8, mol_top_k: int = 2, d_ff: int = None,
          resume: bool = False, no_compile: bool = False,
          tokenizer: str = "bpe", dataset: str = "wikitext103",
          teamspace: str = "mole-toy-validation-project"):

    print(f"\n{'='*60}")
    print(f"Phase 1 Training: config={config}")
    print(f"{'='*60}\n")

    set_dataset(dataset)
    set_tokenizer(tokenizer)
    vocab_size = get_vocab_size()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")

    if device.type == "cpu":
        print("WARNING: Running on CPU. This will be very slow.")
        print("Switch to GPU in your Lightning.ai Studio settings.\n")

    # Model
    model = ToyTransformer(
        config=config, d=d, n_layers=n_layers, n_heads=n_heads,
        vocab_size=vocab_size, max_len=seq_len + 64,
        mhc_dynamic=mhc_dynamic, n_experts=n_experts,
        mol_rank=mol_rank, mol_top_k=mol_top_k, d_ff=d_ff,
    ).to(device)

    n_params = ParamCounter.count(model)

    # Data
    train_loader = get_dataloader("train", seq_len=seq_len, batch_size=batch_size)
    val_loader = get_dataloader("val", seq_len=seq_len, batch_size=batch_size)

    # Optimizer — two param groups: weight matrices get decay, 1D params do not.
    # 1D params include biases, RMSNorm scales (nn.Parameter shape [d]), and any
    # other 1D tensors. 2D+ matrices (Linear weights, embeddings) get decay.
    decay_params   = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": 0.1},
         {"params": nodecay_params, "weight_decay": 0.0}],
        lr=max_lr, betas=(0.9, 0.95), fused=(device.type == "cuda"),
    )

    # Resume
    start_step = 0
    best_val_bpc = float("inf")
    os.makedirs(ckpt_dir, exist_ok=True)
    resume_path = os.path.join(ckpt_dir, f"{config}_latest.pt")

    if resume and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}...")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = ckpt["step"] + 1
        best_val_bpc = ckpt.get("best_val_bpc", float("inf"))
        print(f"  Resumed at step {start_step}, best_val_bpc={best_val_bpc:.4f}")

    # Compile (AFTER resume load)
    if device.type == "cuda" and not no_compile:
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile failed ({e}), continuing without it")

    # Local JSONL logger (always active)
    logger = TrainLogger(ckpt_dir, run_name=config)

    # Lightning.ai experiment tracker (active when running in a Studio)
    lit = make_lit_logger(name=f"phase1-{config}", teamspace=teamspace)
    if lit is not None:
        lit.log_hyperparams({
            "config": config, "d": d, "n_layers": n_layers, "n_heads": n_heads,
            "seq_len": seq_len, "batch_size": batch_size, "total_steps": total_steps,
            "max_lr": max_lr, "n_params": n_params,
        })
        print("LightningLogger active — metrics streaming to Teamspace")

    # Training loop
    model.train()
    train_iter = iter(train_loader)
    start_time = time.time()
    log_loss_accum = 0.0
    log_count = 0

    print(f"Training steps {start_step} -> {total_steps}")
    print(f"AMP: {use_amp} | batch: {batch_size} | seq_len: {seq_len}")
    print(f"Train batches/epoch: {len(train_loader)}\n")

    for step in range(start_step, total_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        lr = get_lr(step, total_steps=total_steps, max_lr=max_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=use_amp):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optimizer.step()

        log_loss_accum += loss.item()
        log_count += 1

        if (step + 1) % log_interval == 0:
            avg_loss = log_loss_accum / log_count
            logger.log_step(step, avg_loss, lr, grad_norm)
            logger.print_step(step, avg_loss, lr, grad_norm, interval=1)
            if lit is not None:
                lit.log_metrics({"train/loss": avg_loss, "train/bpc": ce_to_bpc(avg_loss),
                                 "train/lr": lr, "train/grad_norm": grad_norm}, step=step)
            log_loss_accum = 0.0
            log_count = 0

        if (step + 1) % eval_interval == 0:
            val_loss = evaluate(model, val_loader, device)
            val_bpc = logger.log_eval(step, val_loss)
            logger.print_eval(step, val_loss, val_bpc)
            if lit is not None:
                lit.log_metrics({"val/loss": val_loss, "val/bpc": val_bpc}, step=step)

            if config in ("mol", "compose"):
                mol_stats = _unwrap(model).get_mol_stats()
                if mol_stats:
                    avg_bal = sum(s["expert_balance"] for s in mol_stats) / len(mol_stats)
                    print(f"  >>> MoL avg balance: {avg_bal:.3f} (1.0 = perfect)")
                    for s in [mol_stats[0], mol_stats[-1]]:
                        counts = s["expert_counts"]
                        total_c = sum(counts)
                        pcts = [f"{100*c/total_c:.0f}%" for c in counts] if total_c > 0 else []
                        print(f"      layer {s['layer']} experts: {' '.join(pcts)}")
                    logger.log_mol_stats(step, mol_stats)
                    if lit is not None:
                        lit.log_metrics({"mol/avg_balance": avg_bal}, step=step)
                _unwrap(model).reset_mol_counts()

            if val_bpc < best_val_bpc:
                best_val_bpc = val_bpc
                ckpt_path = os.path.join(ckpt_dir, f"{config}_best.pt")
                torch.save({
                    "step": step, "model_state": _unwrap(model).state_dict(),
                    "val_bpc": val_bpc, "config": config, "n_params": n_params,
                }, ckpt_path)
                print(f"  >>> New best! Saved to {ckpt_path}")

            torch.save({
                "step": step,
                "model_state": _unwrap(model).state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_bpc": best_val_bpc,
            }, resume_path)

    # Final eval
    val_loss = evaluate(model, val_loader, device, max_batches=200)
    val_bpc = ce_to_bpc(val_loss)
    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Training complete: {config}")
    print(f"  Steps: {start_step} -> {total_steps}")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"  Final val BPC: {val_bpc:.4f}")
    print(f"  Best val BPC:  {best_val_bpc:.4f}")
    print(f"  Params: {n_params:,}")
    print(f"{'='*60}\n")

    summary = {
        "config": config, "n_params": n_params,
        "total_steps": total_steps, "final_val_bpc": val_bpc,
        "best_val_bpc": best_val_bpc, "elapsed_seconds": elapsed,
        "d": d, "n_layers": n_layers, "n_heads": n_heads,
        "seq_len": seq_len, "batch_size": batch_size,
        "tokenizer": tokenizer, "dataset": dataset, "vocab_size": vocab_size,
    }
    with open(os.path.join(ckpt_dir, f"{config}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if lit is not None:
        lit.log_metrics({"final/best_val_bpc": best_val_bpc, "final/val_bpc": val_bpc})
        lit.finalize()

    return best_val_bpc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 Training")
    parser.add_argument("--config", type=str, default="baseline",
                        choices=["baseline", "baseline_wide", "mhc", "mol", "mol_single", "compose"])
    parser.add_argument("--d", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--tokenizer", type=str, default="bpe", choices=["char", "bpe"])
    parser.add_argument("--dataset",   type=str, default="wikitext103", choices=["wikitext103", "enwik8"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--eval_interval", type=int, default=2500)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--mhc_dynamic", action="store_true")
    parser.add_argument("--n_experts", type=int, default=8)
    parser.add_argument("--mol_rank", type=int, default=8)
    parser.add_argument("--mol_top_k", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=None,
                        help="FFN hidden dim override (default: auto = d*8/3 rounded to 64). "
                             "Use 1600 for baseline_wide to match mol total params.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--teamspace", type=str, default="mole-toy-validation-project",
                        help="Lightning.ai Teamspace name for experiment tracking")
    args = parser.parse_args()

    train(**vars(args))
