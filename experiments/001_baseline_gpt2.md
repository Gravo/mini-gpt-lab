# 001 Baseline GPT-Style Model

## Goal

Run a nanoGPT-style character model and establish the baseline loss curve, sample quality, and tokens per second.

## Configuration

| Field | Value |
| --- | --- |
| Dataset | Tiny Shakespeare |
| Tokenizer | character |
| Position | learned absolute |
| Norm | LayerNorm |
| MLP | GELU |
| KV cache | off during training, optional during generation |

## Results

| Step | Train loss | Val loss | Tokens/sec | Sample note |
| ---: | ---: | ---: | ---: | --- |
| 0 | | | | |

## Notes

- Use this run as the control for RoPE, RMSNorm, and SwiGLU changes.
- Keep parameter count and training budget fixed when comparing architecture variants.
