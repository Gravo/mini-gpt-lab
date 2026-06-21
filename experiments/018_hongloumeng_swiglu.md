# 018 Hongloumeng SwiGLU

## Goal

Compare GELU MLP against SwiGLU after the RoPE improvement.

This experiment changes one architectural feature:

```text
mlp: gelu -> swiglu
```

Everything else stays close to experiment 016.

## Controlled Setup

Same as 016:

```text
data = cleaned Hongloumeng first 80 chapters
n_layer = 6
n_head = 8
n_embd = 256
head_dim = 32
block_size = 128
dropout = 0.15
norm = layernorm
position = rope
max_steps = 5000
```

Changed:

```text
mlp = swiglu
```

## Why SwiGLU Might Help

The baseline GELU MLP is:

```text
x -> Linear(256, 1024) -> GELU -> Linear(1024, 256)
```

SwiGLU uses a learned gate:

```text
SwiGLU(x) = Linear_3(SiLU(Linear_1(x)) * Linear_2(x))
```

This gives the MLP a multiplicative interaction path. In modern decoder-only LLMs, gated MLPs are
common because they often improve the feed-forward block's expressiveness.

For this small model, the question is:

```text
Does the extra MLP capacity improve validation loss or generation quality enough to justify the
larger parameter count?
```

## Parameter Estimate

For one GELU MLP:

```text
fc1 weight = 256 * 1024 = 262,144
fc1 bias   = 1,024
fc2 weight = 1024 * 256 = 262,144
fc2 bias   = 256
total      = 525,568
```

For one SwiGLU MLP in this repo:

```text
w1 weight = 256 * 1024 = 262,144
w2 weight = 256 * 1024 = 262,144
w3 weight = 1024 * 256 = 262,144
total     = 786,432
```

Difference per block:

```text
786,432 - 525,568 = 260,864
```

With 6 blocks:

```text
260,864 * 6 = 1,565,184 extra parameters
```

Expected total:

```text
016 RoPE + LayerNorm + GELU:   5,842,176
018 RoPE + LayerNorm + SwiGLU: 7,407,360
```

## Train

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_char_width256_rope_swiglu.yaml
```

## Generate

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.generate --checkpoint runs\hongloumeng_80_char_width256_rope_swiglu\best_ckpt.pt --prompt "黛玉聽了，" --max-new-tokens 300 --temperature 0.7 --top-k 50 --top-p 0.9
```

## Results

| Step | Train Loss | Val Loss | Best Val | Notes |
| ---: | ---: | ---: | ---: | --- |
| 0 | 8.3830 | 8.3902 | 8.3902 | initial |
| 250 | 4.5590 | 4.8125 | 4.8125 | improving |
| 500 | 4.1053 | 4.4490 | 4.4490 | improving |
| 750 | 3.8220 | 4.2958 | 4.2958 | improving |
| 1000 | 3.5937 | 4.1934 | 4.1934 | improving |
| 1250 | 3.4439 | 4.1751 | 4.1751 | improving |
| 1500 | 3.2912 | 4.1145 | 4.1145 | best checkpoint |
| 1750 | 3.1072 | 4.1245 | 4.1145 | no new best |
| 2000 | 2.9550 | 4.1241 | 4.1145 | no new best |
| 2250 | 2.8129 | 4.1947 | 4.1145 | overfitting |
| 2500 | 2.6874 | 4.1593 | 4.1145 | overfitting |
| 2750 | 2.5566 | 4.2217 | 4.1145 | overfitting |
| 3000 | 2.4423 | 4.2460 | 4.1145 | overfitting |
| 3250 | 2.3302 | 4.3408 | 4.1145 | overfitting |
| 3500 | 2.2105 | 4.3131 | 4.1145 | overfitting |
| 3750 | 2.0937 | 4.3593 | 4.1145 | overfitting |
| 4000 | 1.9955 | 4.4155 | 4.1145 | overfitting |
| 4250 | 1.8905 | 4.5228 | 4.1145 | overfitting |
| 4500 | 1.7598 | 4.5926 | 4.1145 | overfitting |
| 4750 | 1.6770 | 4.5674 | 4.1145 | overfitting |
| 5000 | 1.6326 | 4.6033 | 4.1145 | final checkpoint much worse than best |

## Compare Against 016

```text
016 RoPE + LayerNorm + GELU best validation:   4.0805
018 RoPE + LayerNorm + SwiGLU best validation: 4.1145
delta:                                        +0.0340
```

Lower is better, so this SwiGLU run did not beat the current 016 baseline.

Parameter comparison:

```text
016 RoPE + LayerNorm + GELU params:   5,842,176
018 RoPE + LayerNorm + SwiGLU params: 7,407,360
delta:                               +1,565,184
```

Checkpoint size comparison:

```text
016 best_ckpt.pt: 23,452,563 bytes
018 best_ckpt.pt: 29,711,193 bytes
delta:           +6,258,630 bytes
```

## Sample Notes

Generated from `best_ckpt.pt`, temperature `0.7`, top-k `50`, top-p `0.9`.

Prompt:

```text
黛玉聽了，
```

The sample keeps a plausible dialogue rhythm and uses characters such as Bao-yu, Dai-yu, and
Bao-chai. It also repeats local dialogue patterns heavily, especially around "你們", "我們",
"說話", and "也是".

Prompt:

```text
寶玉笑道：「
```

The sample quickly enters repeated quote turns:

```text
寶玉笑道：「寶玉不用叫我取去罷。」寶玉笑道：「我又不用你吃飯。」
```

This is stylistically close but semantically weak. It has more local fluency than exact coherence.

Prompt:

```text
賈母笑道：「
```

The sample produces multiple character turns and even a paragraph break, but repeats "我們" and
"你們" so much that the event thread becomes blurry.

## Conclusion

SwiGLU increased capacity:

```text
5.84M params -> 7.41M params
```

It also drove training loss down much faster:

```text
016 final train loss: 2.1368
018 final train loss: 1.6326
```

But validation loss got worse:

```text
016 best val: 4.0805 at step 2250
018 best val: 4.1145 at step 1500
```

So in this setup, SwiGLU mostly made the model fit the training text faster. It did not improve
held-out prediction or generation quality. This is a useful negative result: adding parameters is
already possible, but it needs either more data, stronger regularization, shorter training, or a
more parameter-balanced SwiGLU hidden size.

Current best baseline remains:

```text
RoPE + LayerNorm + GELU
```

## Test Scripts

Train 016:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_char_width256_rope.yaml
```

Train 018:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_char_width256_rope_swiglu.yaml
```

Generate 016:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.generate --checkpoint runs\hongloumeng_80_char_width256_rope\best_ckpt.pt --prompt "黛玉聽了，" --max-new-tokens 300 --temperature 0.7 --top-k 50 --top-p 0.9
```

Generate 018:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.generate --checkpoint runs\hongloumeng_80_char_width256_rope_swiglu\best_ckpt.pt --prompt "黛玉聽了，" --max-new-tokens 300 --temperature 0.7 --top-k 50 --top-p 0.9
```
