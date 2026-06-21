import argparse
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gptlab.model import GPT, GPTConfig
from gptlab.tokenizer import CharTokenizer


def clean(ch: str) -> str:
    if ch == "\n":
        return "\\n"
    if ch == "\t":
        return "\\t"
    return ch


def mark_diffs(expected: str, generated: str) -> str:
    marks = []
    for exp, got in zip(expected, generated):
        marks.append(" " if exp == got else "^")
    if len(generated) < len(expected):
        marks.extend("^" for _ in range(len(expected) - len(generated)))
    return "".join(marks)


def char_accuracy(expected: str, generated: str) -> float:
    if not expected:
        return 0.0
    matches = sum(1 for exp, got in zip(expected, generated) if exp == got)
    return matches / len(expected)


def first_mismatch(expected: str, generated: str) -> int | None:
    for i, (exp, got) in enumerate(zip(expected, generated)):
        if exp != got:
            return i
    if len(expected) != len(generated):
        return min(len(expected), len(generated))
    return None


def print_wrapped_diff(expected: str, generated: str, width: int) -> None:
    for start in range(0, len(expected), width):
        exp = expected[start : start + width]
        got = generated[start : start + width]
        marks = mark_diffs(exp, got)
        print(f"expected  {start:>4}: {exp}")
        print(f"generated {start:>4}: {got}")
        print(f"diff      {start:>4}: {marks}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/hongloumeng_80_char_width256/best_ckpt.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/hongloumeng_80.txt"))
    parser.add_argument("--offset", type=int, default=None)
    parser.add_argument("--random-offset", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--prefix-len", type=int, default=120)
    parser.add_argument("--target-len", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--width", type=int, default=60)
    args = parser.parse_args()

    text = args.data.read_text(encoding="utf-8")
    total_len = args.prefix_len + args.target_len
    if len(text) <= total_len:
        raise SystemExit("Data file is shorter than prefix-len + target-len")

    if args.random_offset:
        rng = random.Random(args.seed)
        offset = rng.randint(0, len(text) - total_len - 1)
    elif args.offset is not None:
        offset = args.offset
    else:
        offset = 0

    if offset < 0 or offset + total_len > len(text):
        raise SystemExit(f"Offset must allow {total_len} characters inside data length {len(text)}")

    prefix = text[offset : offset + args.prefix_len]
    expected = text[offset + args.prefix_len : offset + total_len]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    tokenizer = CharTokenizer("".join(ckpt["chars"]))
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    missing = sorted({ch for ch in prefix if ch not in tokenizer.stoi})
    if missing:
        raise SystemExit(f"Prefix contains out-of-vocabulary characters: {missing}")

    idx = torch.tensor([tokenizer.encode(prefix)], dtype=torch.long, device=device)
    if args.greedy:
        for _ in range(args.target_len):
            idx_cond = idx[:, -model.config.block_size :]
            logits, _ = model(idx_cond)
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat((idx, next_id), dim=1)
        out = idx
    else:
        out = model.generate(
            idx,
            max_new_tokens=args.target_len,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    generated = tokenizer.decode(out[0].tolist())[len(prefix) : len(prefix) + args.target_len]

    mismatch = first_mismatch(expected, generated)
    print(f"checkpoint: {args.checkpoint}")
    print(f"data: {args.data}")
    print(f"offset: {offset}")
    print(f"prefix_len: {args.prefix_len}")
    print(f"target_len: {args.target_len}")
    print(f"temperature: {args.temperature}")
    print(f"top_k: {args.top_k}")
    print(f"top_p: {args.top_p}")
    print(f"greedy: {args.greedy}")
    print(f"char_accuracy: {char_accuracy(expected, generated):.3f}")
    print(f"first_mismatch: {mismatch if mismatch is not None else 'none'}")
    print()
    print("prefix:")
    print(prefix)
    print()
    print_wrapped_diff(expected, generated, args.width)


if __name__ == "__main__":
    main()
