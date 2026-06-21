import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gptlab.model import GPT, GPTConfig
from gptlab.tokenizer import CharTokenizer
from gptlab.train import get_batch


def parse_heads(values: list[str]) -> list[tuple[int, int]]:
    heads = []
    for value in values:
        try:
            layer, head = value.split(":")
            heads.append((int(layer), int(head)))
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Heads must use LAYER:HEAD format, for example 5:4") from exc
    return heads


def make_head_ablation_hook(n_head: int, head_dim: int, head_idx: int):
    def hook(_module, inputs):
        (output,) = inputs
        b, t, c = output.shape
        y = output.view(b, t, n_head, head_dim).clone()
        y[:, :, head_idx, :] = 0
        return (y.view(b, t, c),)

    return hook


@torch.no_grad()
def make_eval_batches(
    val_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str,
    eval_iters: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [get_batch(val_data, batch_size, block_size, device) for _ in range(eval_iters)]


@torch.no_grad()
def estimate_val_loss(model: GPT, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    model.eval()
    losses = torch.zeros(len(batches))
    for i, (x, y) in enumerate(batches):
        _, loss = model(x, y)
        losses[i] = loss.item()
    return losses.mean().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/hongloumeng_80_char_width256/best_ckpt.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/hongloumeng_80.txt"))
    parser.add_argument("--head", action="append", default=[], help="Head to ablate, format LAYER:HEAD. Can be repeated.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    tokenizer = CharTokenizer("".join(ckpt["chars"]))
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])

    text = args.data.read_text(encoding="utf-8")
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(ids))
    val_data = ids[n:]

    cfg = model.config
    batches = make_eval_batches(val_data, args.batch_size, cfg.block_size, device, args.eval_iters)
    baseline = estimate_val_loss(model, batches)
    print(f"baseline val loss: {baseline:.4f}")

    heads = parse_heads(args.head)
    if not heads:
        print("no ablations requested")
        return

    for layer_idx, head_idx in heads:
        if layer_idx < 0 or layer_idx >= cfg.n_layer:
            raise SystemExit(f"Layer {layer_idx} out of range [0, {cfg.n_layer - 1}]")
        if head_idx < 0 or head_idx >= cfg.n_head:
            raise SystemExit(f"Head {head_idx} out of range [0, {cfg.n_head - 1}]")

        torch.manual_seed(args.seed)
        attn = model.blocks[layer_idx].attn
        handle = attn.c_proj.register_forward_pre_hook(make_head_ablation_hook(attn.n_head, attn.head_dim, head_idx))
        ablated = estimate_val_loss(model, batches)
        handle.remove()
        print(f"ablate layer {layer_idx} head {head_idx}: val loss {ablated:.4f}, delta {ablated - baseline:+.4f}")


if __name__ == "__main__":
    main()
