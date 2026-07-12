"""Training loop for one hybrid variant.

Standard small-LM recipe: AdamW, cosine LR with warmup, gradient accumulation for a bigger effective
batch than 12 GB would otherwise allow, bf16 autocast, and gradient checkpointing on. Every so often
I stop and estimate train/val loss (and val perplexity) over a few batches, and I keep the checkpoint
with the best val loss. W&B is optional — off by default so a quick run doesn't need a login.

    python -m src.train.train --model-config configs/ratio_1_7.yaml --max-steps 2000
    python -m src.train.train --model-config configs/ratio_1_3.yaml --wandb   # log to W&B
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml

from src.data.dataset import get_batch, load_meta, load_split
from src.model.config import ModelConfig
from src.model.lm import HybridLM


@dataclass
class TrainConfig:
    model_config: str = "configs/ratio_1_7.yaml"
    data_dir: str = "data/tinystories"
    # optim
    lr: float = 1e-3
    min_lr: float = 1e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # schedule / length
    max_steps: int = 2000
    warmup_steps: int = 100
    batch_size: int = 8          # micro-batch
    grad_accum: int = 4          # effective batch = batch_size * grad_accum
    block_size: int = 512
    # eval / logging / ckpt
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 20
    ckpt_dir: str = "checkpoints"
    # misc
    seed: int = 1337
    device: str = "cuda"
    grad_checkpointing: bool = True
    wandb: bool = False
    wandb_project: str = "mamba-hybrid-lm"


def load_model_config(path: str) -> ModelConfig:
    return ModelConfig(**yaml.safe_load(Path(path).read_text()))


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    # linear warmup, then cosine down to min_lr, then flat
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    frac = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * frac)) * (cfg.lr - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, splits, cfg: TrainConfig) -> dict:
    # average loss over a handful of batches per split; perplexity = exp(val loss)
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            x, y = get_batch(data, cfg.block_size, cfg.batch_size, cfg.device)
            with torch.autocast(cfg.device, dtype=torch.bfloat16):
                _, loss = model(x, y)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    for f, default in asdict(TrainConfig()).items():
        arg = "--" + f.replace("_", "-")
        if isinstance(default, bool):
            ap.add_argument(arg, action="store_true" if not default else "store_false", dest=f)
        else:
            ap.add_argument(arg, type=type(default), default=default, dest=f)
    cfg = TrainConfig(**vars(ap.parse_args()))

    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # free speed on the matmuls I don't need fp32 for

    mcfg = load_model_config(cfg.model_config)
    splits = {s: load_split(cfg.data_dir, s) for s in ("train", "val")}
    meta = load_meta(cfg.data_dir)
    assert meta["vocab_size"] == mcfg.vocab_size, "tokenizer/model vocab mismatch"

    # keep weights in fp32; autocast does the bf16 compute. Full-bf16 weights would drag the
    # AdamW moments down to bf16 too, which trains worse for only ~110 MB of savings on 54M params.
    model = HybridLM(mcfg).to(cfg.device)
    model.grad_checkpointing = cfg.grad_checkpointing
    print(f"{mcfg.name}: {model.num_params()/1e6:.2f}M params  "
          f"(eff. batch {cfg.batch_size*cfg.grad_accum} x {cfg.block_size} tokens)")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                              weight_decay=cfg.weight_decay, fused=True)

    run = None
    if cfg.wandb:
        import wandb
        run = wandb.init(project=cfg.wandb_project, name=mcfg.name, config={**asdict(cfg), **asdict(mcfg)})

    ckpt_dir = Path(cfg.ckpt_dir) / mcfg.name.replace(":", "_")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for step in range(cfg.max_steps + 1):
        lr = cosine_lr(step, cfg)
        for g in optim.param_groups:
            g["lr"] = lr

        # ---- eval + checkpoint ----
        if step % cfg.eval_interval == 0:
            losses = estimate_loss(model, splits, cfg)
            ppl = math.exp(losses["val"])
            print(f"step {step:5d} | train {losses['train']:.3f} | val {losses['val']:.3f} "
                  f"| ppl {ppl:.1f} | lr {lr:.2e}")
            if run:
                run.log({"val/loss": losses["val"], "val/ppl": ppl, "train/eval_loss": losses["train"]}, step=step)
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({"model": model.state_dict(), "model_config": asdict(mcfg),
                            "step": step, "val_loss": best_val}, ckpt_dir / "best.pt")
        if step == cfg.max_steps:
            break

        # ---- one optimizer step over grad_accum micro-batches ----
        step_start = time.time()
        for micro in range(cfg.grad_accum):
            x, y = get_batch(splits["train"], cfg.block_size, cfg.batch_size, cfg.device)
            with torch.autocast(cfg.device, dtype=torch.bfloat16):
                _, loss = model(x, y)
            (loss / cfg.grad_accum).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()
        optim.zero_grad(set_to_none=True)

        if step % cfg.log_interval == 0:
            if cfg.device == "cuda":
                torch.cuda.synchronize()  # otherwise I'm timing the async launch, not the work
            toks = cfg.batch_size * cfg.grad_accum * cfg.block_size
            tps = toks / (time.time() - step_start)
            print(f"  step {step:5d} | loss {loss.item():.3f} | gnorm {grad_norm:.2f} | {tps:.0f} tok/s")
            if run:
                run.log({"train/loss": loss.item(), "train/grad_norm": grad_norm.item(),
                         "lr": lr, "throughput/tok_per_s": tps}, step=step)

    print(f"done. best val loss {best_val:.3f} (ppl {math.exp(best_val):.1f}) -> {ckpt_dir}/best.pt")
    if run:
        run.finish()


if __name__ == "__main__":
    main()
