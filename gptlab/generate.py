import argparse
from pathlib import Path

import torch

from .model import GPT, GPTConfig
from .tokenizer import CharTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/tiny_shakespeare_baseline/ckpt.pt"))
    parser.add_argument("--prompt", default="To be, or not to be")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    tokenizer = CharTokenizer("".join(ckpt["chars"]))
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])

    idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, args.max_new_tokens, args.temperature, use_cache=not args.no_cache)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
