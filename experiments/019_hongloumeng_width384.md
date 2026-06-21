# 019 Hongloumeng Width 384

## Goal

Scale the current best RoPE baseline by increasing hidden width.

This experiment changes the core capacity setting:

```text
n_embd: 256 -> 384
n_head: 8 -> 12
```

This keeps per-head width stable:

```text
016 head_dim = 256 / 8 = 32
019 head_dim = 384 / 12 = 32
```

Everything else stays close to experiment 016.

## Controlled Setup

Same as 016:

```text
data = cleaned Hongloumeng first 80 chapters
n_layer = 6
block_size = 128
dropout = 0.15
norm = layernorm
mlp = gelu
position = rope
max_steps = 5000
```

Changed:

```text
n_embd = 384
n_head = 12
```

## Why Width Might Help

Increasing hidden width gives every token representation more channels. In this model, that expands:

```text
token embedding width
Q/K/V projections
attention output projection
MLP hidden size
final normalization
lm_head input width
```

Unlike experiment 018, this does not only add MLP capacity. It scales the whole residual stream and
attention stack, so it is a cleaner test of whether the model is capacity-limited.

## Parameter Estimate

Measured with the current Hongloumeng 80-chapter character vocabulary:

```text
vocab_size = 4309
```

Total parameters:

```text
016 width 256:  5,842,176
019 width 384: 12,302,208
delta:         +6,460,032
ratio:          2.11x
```

The number of heads does not directly change parameter count when `n_embd` is fixed. It changes how
the hidden width is split across heads.

## Train

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_char_width384_rope.yaml
```

## Generate

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.generate --checkpoint runs\hongloumeng_80_char_width384_rope\best_ckpt.pt --prompt "黛玉聽了，" --max-new-tokens 300 --temperature 0.7 --top-k 50 --top-p 0.9
```

## Results

| Step | Train Loss | Val Loss | Best Val | Notes |
| ---: | ---: | ---: | ---: | --- |
| 0 | 8.4264 | 8.4281 | 8.4281 | initial |
| 250 | 4.5479 | 4.7853 | 4.7853 | improving |
| 500 | 4.0845 | 4.4678 | 4.4678 | improving |
| 750 | 3.7705 | 4.3451 | 4.3451 | improving |
| 1000 | 3.5661 | 4.2131 | 4.2131 | improving |
| 1250 | 3.4049 | 4.1353 | 4.1353 | improving |
| 1500 | 3.2471 | 4.0998 | 4.0998 | best checkpoint |
| 1750 | 3.0622 | 4.2182 | 4.0998 | overfitting starts |
| 2000 | 2.8801 | 4.1732 | 4.0998 | no new best |
| 2250 | 2.6344 | 4.1481 | 4.0998 | no new best |
| 2500 | 2.5405 | 4.2275 | 4.0998 | overfitting |
| 2750 | 2.3403 | 4.2552 | 4.0998 | overfitting |
| 3000 | 2.1913 | 4.3972 | 4.0998 | overfitting |
| 3250 | 2.0208 | 4.4086 | 4.0998 | overfitting |
| 3500 | 1.8576 | 4.4551 | 4.0998 | overfitting |
| 3750 | 1.7775 | 4.5625 | 4.0998 | overfitting |
| 4000 | 1.6075 | 4.6519 | 4.0998 | overfitting |
| 4250 | 1.4771 | 4.6547 | 4.0998 | overfitting |
| 4500 | 1.3510 | 4.8099 | 4.0998 | overfitting |
| 4750 | 1.2575 | 4.8978 | 4.0998 | overfitting |
| 5000 | 1.1375 | 5.0025 | 4.0998 | final checkpoint much worse than best |

## Compare Against 016

```text
016 RoPE width 256 best validation: 4.0805
019 RoPE width 384 best validation: 4.0998
delta:                              +0.0193
```

Lower is better, so the wider model did not beat the current 016 baseline.

Checkpoint size comparison:

```text
016 best_ckpt.pt: 23,452,563 bytes
019 best_ckpt.pt: 49,292,691 bytes
delta:          +25,840,128 bytes
```

## Sample Notes

Generated from `best_ckpt.pt`, temperature `0.7`, top-k `50`, top-p `0.9`.

Prompt:

```text
黛玉聽了，
```

The sample has plausible Hongloumeng-style dialogue turns and character names. It still repeats
local phrasing around "你們", "我們", "不必", and "說話". The final continuation loses event
continuity.

Prompt:

```text
寶玉笑道：「
```

The sample shows better paragraph rhythm than some smaller runs, with a few scene transitions. It
still drifts into repeated generic social dialogue and weak causal continuity.

Prompt:

```text
賈母笑道：「
```

The sample includes Jia-mu, Bao-yu, Feng-jie, Yuan-yang, and Liu-laolao, but repetition is still
visible:

```text
逛逛逛逛逛
劉姥姥姥姥姥姥姥
```

## Conclusion

Width scaling did help more than experiment 018's SwiGLU change:

```text
018 best val: 4.1145
019 best val: 4.0998
```

But it still did not beat the current best 016:

```text
016 best val: 4.0805
019 best val: 4.0998
```

The wider model fits the training set much more aggressively:

```text
016 final train loss: 2.1368
019 final train loss: 1.1375
```

Validation loss starts worsening after step 1500. This suggests the 12.3M model is not useless, but
the current data size and regularization are not enough to make the extra capacity generalize.

Current best baseline remains:

```text
RoPE + LayerNorm + GELU, width 256
```

The next planned experiment should isolate context length rather than adding more parameters:

```text
020 = 016 + block_size 128 -> 192
```

## Next Context Experiment

After width 384, test block size separately from the current best 016 baseline:

```text
020 = 016 + block_size 128 -> 192
```

With RoPE, this should not add model parameters, but it will increase attention compute.

## Test Scripts

Train 016:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_char_width256_rope.yaml
```

Train 019:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_char_width384_rope.yaml
```

Generate 016:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.generate --checkpoint runs\hongloumeng_80_char_width256_rope\best_ckpt.pt --prompt "黛玉聽了，" --max-new-tokens 300 --temperature 0.7 --top-k 50 --top-p 0.9
```

Generate 019:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.generate --checkpoint runs\hongloumeng_80_char_width384_rope\best_ckpt.pt --prompt "黛玉聽了，" --max-new-tokens 300 --temperature 0.7 --top-k 50 --top-p 0.9
```
