import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gptlab.attention import apply_rope
from gptlab.model import GPT, GPTConfig
from gptlab.tokenizer import CharTokenizer


def clean_char(ch: str) -> str:
    if ch == "\n":
        return "\\n"
    if ch == "\t":
        return "\\t"
    if ch == " ":
        return "space"
    return ch


def attention_weights(block, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    b, t, c = x.size()
    attn = block.attn
    q, k, _ = attn.c_attn(block.ln_1(x)).split(c, dim=2)
    q = q.view(b, t, attn.n_head, attn.head_dim).transpose(1, 2)
    k = k.view(b, t, attn.n_head, attn.head_dim).transpose(1, 2)

    if attn.rope:
        q = apply_rope(q, start_pos=start_pos)
        k = apply_rope(k, start_pos=start_pos)

    scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(attn.head_dim))
    scores = scores.masked_fill(attn.bias[:, :, :t, :t] == 0, float("-inf"))
    return F.softmax(scores, dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/hongloumeng_80_char_width256/best_ckpt.pt"))
    parser.add_argument("--prompt", default="黛玉聽了，忙問")
    parser.add_argument("--target-index", type=int, default=-1)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    tokenizer = CharTokenizer("".join(ckpt["chars"]))
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    missing = sorted({ch for ch in args.prompt if ch not in tokenizer.stoi})
    if missing:
        raise SystemExit(f"Prompt contains out-of-vocabulary characters: {missing}")

    ids = tokenizer.encode(args.prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    _, t = idx.shape
    target = args.target_index if args.target_index >= 0 else t + args.target_index
    if target < 0 or target >= t:
        raise SystemExit(f"--target-index must resolve into [0, {t - 1}]")

    print(f"checkpoint: {args.checkpoint}")
    print(f"prompt length: {t}")
    print(f"target index: {target}, char: {clean_char(args.prompt[target])}")
    print()
    print("indexed prompt:")
    for i, ch in enumerate(args.prompt):
        marker = "<- target" if i == target else ""
        print(f"{i:>3}: {clean_char(ch)} {marker}")
    print()

    x = model.token_emb(idx)
    if model.pos_emb is not None:
        pos = torch.arange(0, t, device=device)
        x = x + model.pos_emb(pos)
    x = model.drop(x)

    for layer_idx, block in enumerate(model.blocks):
        weights = attention_weights(block, x)[0, :, target, :].detach().cpu()
        print(f"layer {layer_idx}")
        for head_idx, row in enumerate(weights):
            entropy = -(row * (row + 1e-12).log()).sum().item()
            effective = math.exp(entropy)
            values, positions = torch.topk(row, min(args.top_n, row.numel()))
            parts = []
            for value, pos in zip(values.tolist(), positions.tolist()):
                parts.append(f"{pos}:{clean_char(args.prompt[pos])}:{value:.3f}")
            print(f"  head {head_idx:>2} effective_tokens={effective:>5.2f} top={', '.join(parts)}")
        x = block(x)
        print()


if __name__ == "__main__":
    main()
