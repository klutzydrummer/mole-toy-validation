"""BPC metrics and minimal training logger."""

import math
import time
import json
import os


def ce_to_bpc(loss: float) -> float:
    return loss / math.log(2)


class TrainLogger:
    def __init__(self, log_dir: str = "checkpoints", run_name: str = "run"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"{run_name}.jsonl")
        self._last_time = time.time()
        self._last_step = 0

    def log_step(self, step, loss, lr=0.0):
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"step": step, "loss": loss, "bpc": ce_to_bpc(loss), "lr": lr}) + "\n")

    def log_eval(self, step, val_loss):
        bpc = ce_to_bpc(val_loss)
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"step": step, "val_loss": val_loss, "val_bpc": bpc, "type": "eval"}) + "\n")
        return bpc

    def log_mol_stats(self, step, mol_stats):
        """Persist MoL expert stats to JSONL so the reporter can track them."""
        if not mol_stats:
            return
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"step": step, "mol_stats": mol_stats, "type": "mol"}) + "\n")

    def print_step(self, step, loss, lr, interval=100):
        if step > 0 and step % interval != 0:
            return
        elapsed = time.time() - self._last_time
        steps = step - self._last_step
        sps = steps / elapsed if elapsed > 0 and steps > 0 else 0
        print(f"step {step:>6d} | loss {loss:.4f} | bpc {ce_to_bpc(loss):.4f} | lr {lr:.2e} | {sps:.1f} steps/s")
        self._last_time = time.time()
        self._last_step = step

    def print_eval(self, step, val_loss, val_bpc):
        print(f"  >>> EVAL step {step:>6d} | val_loss {val_loss:.4f} | val_bpc {val_bpc:.4f}")


class ParamCounter:
    @staticmethod
    def count(model, verbose=True):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if verbose:
            print(f"\nModel parameters: {total:,} total, {trainable:,} trainable")
            for name, mod in model.named_children():
                n = sum(p.numel() for p in mod.parameters())
                print(f"  {name:20s}: {n:>10,} ({100*n/total:5.1f}%)")
            print()
        return total
