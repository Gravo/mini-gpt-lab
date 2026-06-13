# 002 RoPE + RMSNorm

## Goal

Compare learned absolute position embeddings against RoPE, and LayerNorm against RMSNorm.

## Hypothesis

RoPE should make position handling more similar to modern decoder-only LLMs. RMSNorm removes mean-centering and may train slightly faster while preserving stability at this scale.

## Results

| Variant | Train loss | Val loss | Tokens/sec | Sample note |
| --- | ---: | ---: | ---: | --- |
| learned + LayerNorm | | | | |
| RoPE + LayerNorm | | | | |
| RoPE + RMSNorm | | | | |
