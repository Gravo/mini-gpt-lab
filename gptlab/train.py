import argparse
import json
import math
from pathlib import Path

import torch

from .model import GPT, GPTConfig
from .tokenizer import BPETokenizer, CharTokenizer, build_tokenizer


def load_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("Install pyyaml or pass a JSON config.") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model: GPT, train_data: torch.Tensor, val_data: torch.Tensor, cfg: dict, device: str) -> dict[str, float]:
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(cfg["eval_iters"])
        for k in range(cfg["eval_iters"]):
            x, y = get_batch(data, cfg["batch_size"], cfg["block_size"], device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_learning_rate(cfg: dict, step: int) -> float:
    base_lr = cfg["learning_rate"]
    schedule = cfg.get("lr_schedule")
    if not schedule:
        return base_lr

    schedule_type = schedule.get("type", "constant")
    if schedule_type == "constant":
        return base_lr
    if schedule_type != "cosine":
        raise ValueError(f"Unknown lr_schedule type: {schedule_type}")

    warmup_steps = schedule.get("warmup_steps", 0)
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * step / warmup_steps

    decay_steps = schedule.get("decay_steps", cfg["max_steps"])
    min_lr = schedule.get("min_lr", 0.0)
    progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (base_lr - min_lr)


def save_checkpoint(
    path: Path,
    model: GPT,
    model_cfg: GPTConfig,
    tokenizer: CharTokenizer | BPETokenizer,
    metadata: dict,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "config": model_cfg.__dict__,
            "tokenizer": tokenizer.to_state(),
            "chars": getattr(tokenizer, "chars", None),
            "metadata": metadata,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/tiny_shakespeare.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)

    torch.manual_seed(cfg["seed"])
    device = "cuda" if cfg["device"] == "auto" and torch.cuda.is_available() else "cpu"
    if cfg["device"] != "auto":
        device = cfg["device"]

    text = Path(cfg["data_path"]).read_text(encoding="utf-8")
    tokenizer_cfg = cfg.get("tokenizer") or {}
    tokenizer = build_tokenizer(text, tokenizer_cfg)
    split_mode = cfg.get("split", "token")
    if split_mode == "text":
        raw_n = int(0.9 * len(text))
        train_text, val_text = text[:raw_n], text[raw_n:]
        train_cache_path = tokenizer_cfg.get("train_encoded_cache_path")
        val_cache_path = tokenizer_cfg.get("val_encoded_cache_path")
        if train_cache_path and Path(train_cache_path).exists():
            train_data = torch.load(train_cache_path, weights_only=True)
        else:
            train_data = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
            if train_cache_path:
                train_path = Path(train_cache_path)
                train_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(train_data, train_path)
        if val_cache_path and Path(val_cache_path).exists():
            val_data = torch.load(val_cache_path, weights_only=True)
        else:
            val_data = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
            if val_cache_path:
                val_path = Path(val_cache_path)
                val_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(val_data, val_path)
        train_tokens_per_char = len(train_data) / len(train_text)
        val_tokens_per_char = len(val_data) / len(val_text)
    elif split_mode == "token":
        encoded_cache_path = tokenizer_cfg.get("encoded_cache_path")
        if encoded_cache_path and Path(encoded_cache_path).exists():
            ids = torch.load(encoded_cache_path, weights_only=True)
        else:
            ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
            if encoded_cache_path:
                encoded_path = Path(encoded_cache_path)
                encoded_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(ids, encoded_path)
        tokens_per_char = len(ids) / len(text)
        train_tokens_per_char = tokens_per_char
        val_tokens_per_char = tokens_per_char
        n = int(0.9 * len(ids))
        train_data, val_data = ids[:n], ids[n:]
    else:
        raise ValueError(f"Unknown split mode: {split_mode}")

    model_cfg = GPTConfig(**{**cfg["model"], "vocab_size": tokenizer.vocab_size, "block_size": cfg["block_size"]})
    model = GPT(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    best_val = float("inf")
    best_step = -1

    for step in range(cfg["max_steps"] + 1):
        lr = get_learning_rate(cfg, step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if step % cfg["eval_interval"] == 0:
            losses = estimate_loss(model, train_data, val_data, cfg, device)
            is_best = losses["val"] < best_val
            if is_best:
                best_val = losses["val"]
                best_step = step
                save_checkpoint(
                    out_dir / "best_ckpt.pt",
                    model,
                    model_cfg,
                    tokenizer,
                    {
                        "step": step,
                        "best_step": best_step,
                        "best_val": best_val,
                        "train_loss": losses["train"],
                        "val_loss": losses["val"],
                        "learning_rate": lr,
                    },
                )
            row = {
                "step": step,
                **losses,
                "best_val": best_val,
                "best_step": best_step,
                "is_best": is_best,
                "learning_rate": lr,
            }
            row["train_nats_per_char"] = losses["train"] * train_tokens_per_char
            row["val_nats_per_char"] = losses["val"] * val_tokens_per_char
            print(row)
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

        xb, yb = get_batch(train_data, cfg["batch_size"], cfg["block_size"], device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    save_checkpoint(
        out_dir / "ckpt.pt",
        model,
        model_cfg,
        tokenizer,
        {"step": cfg["max_steps"], "best_step": best_step, "best_val": best_val},
    )


if __name__ == "__main__":
    main()
