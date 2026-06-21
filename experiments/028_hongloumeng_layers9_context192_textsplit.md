# 028 Hongloumeng Layers 9 Context 192 Text-Split

## Goal

Combine the current best depth setting with longer context.

This experiment changes 027:

```text
block_size: 128 -> 192
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

Depth +50% produced the current best validation loss:

```text
027 layers9 block128: 4.0761
```

Longer context was a healthy non-winning direction:

```text
025B layers6 block192: 4.0909
```

This run checks whether the benefits combine.

## Parameter Estimate

RoPE means block size does not add model parameters:

```text
027 layers9 block128 params = 8,211,456
028 layers9 block192 params = 8,211,456
```

Attention compute increases:

```text
128 * 128 = 16,384
192 * 192 = 36,864
ratio = 2.25x
```

## Results

| Step | Train Loss | Val Loss | Val Nats/Char | Best Val | Notes |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 8.3906 | 8.3951 | 8.3951 | 8.3951 | initial |
| 250 | 4.6703 | 4.8893 | 4.8893 | 4.8893 | improving |
| 500 | 4.1338 | 4.4998 | 4.4998 | 4.4998 | improving |
| 750 | 3.8863 | 4.3360 | 4.3360 | 4.3360 | improving |
| 1000 | 3.7249 | 4.2388 | 4.2388 | 4.2388 | improving |
| 1250 | 3.5298 | 4.1679 | 4.1679 | 4.1679 | improving |
| 1500 | 3.3303 | 4.0675 | 4.0675 | 4.0675 | best checkpoint |
| 1750 | 3.1897 | 4.1157 | 4.1157 | 4.0675 | no new best |
| 2000 | 3.0488 | 4.1382 | 4.1382 | 4.0675 | no new best |
| 2250 | 2.8855 | 4.1132 | 4.1132 | 4.0675 | no new best |
| 2500 | 2.7206 | 4.1078 | 4.1078 | 4.0675 | overfitting |

## Compare

```text
025A layers6 block128: 4.0805
025B layers6 block192: 4.0909
027 layers9 block128: 4.0761
028 layers9 block192: 4.0675
```

## Conclusion

The benefits combine:

```text
027 layers9 block128: 4.0761
028 layers9 block192: 4.0675
delta:                -0.0086
```

Compared with the original 016/025A baseline:

```text
025A layers6 block128: 4.0805
028 layers9 block192: 4.0675
delta:                -0.0131
```

This is the current best validation loss. It is still a small improvement, and manual generation
does not show a dramatic jump. Samples still repeat local forms such as "不知道", "什麼", and
"寶玉笑道".

Checkpoint size:

```text
best_ckpt.pt = 32,942,778 bytes
```

Current best by validation loss:

```text
runs/hongloumeng_80_char_width256_rope_layers9_context192_textsplit/best_ckpt.pt
```

## Command

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_char_width256_rope_layers9_context192_textsplit.yaml
```
