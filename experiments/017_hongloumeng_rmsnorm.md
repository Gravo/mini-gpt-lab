# 017 Hongloumeng RMSNorm

## Goal

Compare LayerNorm against RMSNorm after the RoPE improvement.

This experiment changes one architectural feature:

```text
norm: layernorm -> rmsnorm
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
position = rope
mlp = gelu
max_steps = 5000
```

Changed:

```text
norm = rmsnorm
```

## Why RMSNorm Might Help

LayerNorm normalizes by mean and variance and has both weight and bias.

RMSNorm normalizes by root-mean-square only and has weight but no bias:

```text
x * rsqrt(mean(x^2) + eps) * weight
```

It is common in modern LLM blocks because it is simpler and often trains well at scale.

For this small model, the question is modest:

```text
Does RMSNorm improve validation loss or generation stability after RoPE?
```

## Parameter Estimate

Each LayerNorm has:

```text
weight = 256
bias = 256
total = 512
```

Each RMSNorm has:

```text
weight = 256
total = 256
```

The model has:

```text
2 norms per block * 6 blocks + final norm = 13 norms
```

So RMSNorm removes:

```text
13 * 256 = 3,328 parameters
```

Expected total:

```text
016 RoPE + LayerNorm = 5,842,176
017 RoPE + RMSNorm = 5,838,848
```

## Train

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_char_width256_rope_rmsnorm.yaml
```

## Generate

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.generate --checkpoint runs\hongloumeng_80_char_width256_rope_rmsnorm\best_ckpt.pt --prompt "黛玉聽了，" --max-new-tokens 300 --temperature 0.7 --top-k 50 --top-p 0.9
```

## Results

| Step | Train Loss | Val Loss | Best Val | Notes |
| ---: | ---: | ---: | ---: | --- |
| 0 | 8.4161 | 8.4190 | 8.4190 | initial |
| 250 | 4.5805 | 4.7869 | 4.7869 | improving |
| 500 | 4.2250 | 4.4954 | 4.4954 | improving |
| 750 | 3.9908 | 4.3444 | 4.3444 | improving |
| 1000 | 3.8256 | 4.2166 | 4.2166 | improving |
| 1250 | 3.6436 | 4.2037 | 4.2037 | improving |
| 1500 | 3.5553 | 4.1644 | 4.1644 | improving |
| 1750 | 3.4862 | 4.1729 | 4.1644 | no new best |
| 2000 | 3.3345 | 4.1054 | 4.1054 | improving |
| 2250 | 3.1722 | 4.0863 | 4.0863 | best checkpoint |
| 2500 | 3.1217 | 4.0934 | 4.0863 | no new best |
| 2750 | 2.9883 | 4.1467 | 4.0863 | overfitting starts |
| 3000 | 2.8814 | 4.1637 | 4.0863 | overfitting |
| 3250 | 2.7841 | 4.1425 | 4.0863 | overfitting |
| 3500 | 2.6910 | 4.1798 | 4.0863 | overfitting |
| 3750 | 2.5849 | 4.1862 | 4.0863 | overfitting |
| 4000 | 2.5024 | 4.2320 | 4.0863 | overfitting |
| 4250 | 2.4254 | 4.2426 | 4.0863 | overfitting |
| 4500 | 2.3526 | 4.3092 | 4.0863 | overfitting |
| 4750 | 2.2523 | 4.3065 | 4.0863 | overfitting |
| 5000 | 2.1670 | 4.3398 | 4.0863 | overfitting |

## Compare Against 016

```text
016 RoPE + LayerNorm best validation: 4.0805
017 RoPE + RMSNorm best validation:  4.0863
delta:                              +0.0058
```

Lower is better, so RMSNorm did not beat the previous RoPE + LayerNorm run.

Parameter comparison:

```text
016 RoPE + LayerNorm params: 5,842,176
017 RoPE + RMSNorm params:  5,838,848
delta:                         -3,328
```

Checkpoint size comparison:

```text
016 best_ckpt.pt: 23,452,563 bytes
017 best_ckpt.pt: 23,435,296 bytes
delta:             -17,267 bytes
```

## Sample Notes

Prompt:

```text
黛玉聽了，
```

017 can produce locally plausible dialogue turns, for example question-answer fragments around
Bao-yu, Dai-yu, and sisters. It still falls into repetition such as repeated "妹妹" and recycled
dialogue structure.

Prompt:

```text
寶玉笑道：「
```

The output keeps the Hongloumeng dialogue style, but still loops around "寶玉笑道" and generic
phrases like "你也不知道".

Prompt:

```text
賈母笑道：「
```

The output keeps character-name and quotation rhythm, but also repeats "賈母笑道" and gives
weak long-range event continuity.

## Conclusion

RMSNorm is a useful modern component in many larger LLMs, but it is not automatically better in
this tiny character-level setup. Here it removes 3,328 parameters, but validation loss is slightly
worse and generated text does not show a clear qualitative improvement.

Current best architectural baseline should stay:

```text
RoPE + LayerNorm + GELU
```

The next "modern component" worth testing is probably SwiGLU, using 016 as the baseline again.

## Test Scripts

Train 016:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_char_width256_rope.yaml
```

Train 017:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_char_width256_rope_rmsnorm.yaml
```

Generate 016:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.generate --checkpoint runs\hongloumeng_80_char_width256_rope\best_ckpt.pt --prompt "黛玉聽了，" --max-new-tokens 300 --temperature 0.7 --top-k 50 --top-p 0.9
```

Generate 017:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.generate --checkpoint runs\hongloumeng_80_char_width256_rope_rmsnorm\best_ckpt.pt --prompt "黛玉聽了，" --max-new-tokens 300 --temperature 0.7 --top-k 50 --top-p 0.9
```
