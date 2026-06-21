# 029 Hongloumeng Layers 9 Context 256 Text-Split

## Goal

Extend the current best deeper char model from 192 to 256 context.

This experiment changes 028:

```text
block_size: 192 -> 256
```

Everything else stays the same:

```text
tokenizer = char
split = text
n_layer = 9
n_head = 8
n_embd = 256
dropout = 0.15
norm = layernorm
mlp = gelu
position = rope
max_steps = 2500
```

## Why This Experiment

Current best:

```text
028 layers9 block192: 4.0675
```

Character-level context is still short. A 256-character window may capture more complete dialogue
and scene fragments while keeping parameter count unchanged.

## Parameter Estimate

RoPE means block size does not add model parameters:

```text
028 layers9 block192 params = 8,211,456
029 layers9 block256 params = 8,211,456
```

Attention compute increases:

```text
192 * 192 = 36,864
256 * 256 = 65,536
ratio = 1.78x
```

Compared with block128:

```text
256 * 256 / (128 * 128) = 4.00x
```

## Results

| Step | Train Loss | Val Loss | Val Nats/Char | Best Val | Notes |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 8.3906 | 8.3928 | 8.3928 | 8.3928 | initial |
| 250 | 4.5418 | 4.7672 | 4.7672 | 4.7672 | improving |
| 500 | 4.1232 | 4.4190 | 4.4190 | 4.4190 | improving |
| 750 | 3.8134 | 4.2735 | 4.2735 | 4.2735 | improving |
| 1000 | 3.5840 | 4.1708 | 4.1708 | 4.1708 | improving |
| 1250 | 3.3705 | 4.1003 | 4.1003 | 4.1003 | best checkpoint |
| 1500 | 3.2126 | 4.1106 | 4.1106 | 4.1003 | no new best |
| 1750 | 3.0117 | 4.1118 | 4.1118 | 4.1003 | no new best |
| 2000 | 2.8232 | 4.1464 | 4.1464 | 4.1003 | overfitting |
| 2250 | 2.6491 | 4.1664 | 4.1664 | 4.1003 | overfitting |
| 2500 | 2.4669 | 4.2228 | 4.2228 | 4.1003 | overfitting |

## Compare

```text
027 layers9 block128: 4.0761
028 layers9 block192: 4.0675
029 layers9 block256: 4.1003
```

## Conclusion

Block 256 did not improve the current best:

```text
028 layers9 block192: 4.0675
029 layers9 block256: 4.1003
delta:                +0.0329
```

The best point arrives earlier at step 1250, then validation worsens while train loss keeps falling.
This suggests block256 is too much context for this setup, or it needs stronger regularization /
different batch settings.

Checkpoint size:

```text
best_ckpt.pt = 32,942,842 bytes
```

Sample note:

```text
The sample has longer paragraph flow, but still repeats local tokens such as 寶玉, 你們, and 今兒.
```

Current best remains:

```text
runs/hongloumeng_80_char_width256_rope_layers9_context192_textsplit/best_ckpt.pt
```

## Command

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_char_width256_rope_layers9_context256_textsplit.yaml
```
