import argparse
import json
from pathlib import Path

import torch

from .model import GPT, GPTConfig
from .tokenizer import CharTokenizer


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


def save_checkpoint(path: Path, model: GPT, model_cfg: GPTConfig, tokenizer: CharTokenizer, metadata: dict) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "config": model_cfg.__dict__,
            "chars": tokenizer.chars,
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
    tokenizer = CharTokenizer(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(ids))
    train_data, val_data = ids[:n], ids[n:]

    model_cfg = GPTConfig(**{**cfg["model"], "vocab_size": tokenizer.vocab_size, "block_size": cfg["block_size"]})
    model = GPT(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    best_val = float("inf")
    best_step = -1

    for step in range(cfg["max_steps"] + 1):
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
                    },
                )
            row = {"step": step, **losses, "best_val": best_val, "best_step": best_step, "is_best": is_best}
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
