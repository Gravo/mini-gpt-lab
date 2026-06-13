# 003 SwiGLU

## Goal

Replace the GELU MLP with SwiGLU and compare loss, speed, and generated samples under the same training budget.

## Results

| Variant | Train loss | Val loss | Tokens/sec | Sample note |
| --- | ---: | ---: | ---: | --- |
| GELU | | | | |
| SwiGLU | | | | |

## Notes

SwiGLU adds multiplicative gating. Keep the rest of the block unchanged when isolating this comparison.
