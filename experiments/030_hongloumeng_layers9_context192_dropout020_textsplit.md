# 030 Hongloumeng Layers 9 Context 192 Dropout 0.20 Text-Split

## Goal

Test stronger regularization on the current best architecture.

This experiment changes 028:

```text
dropout: 0.15 -> 0.20
```

Everything else stays the same:

```text
tokenizer = char
split = text
n_layer = 9
n_head = 8
n_embd = 256
block_size = 192
norm = layernorm
mlp = gelu
position = rope
max_steps = 2500
```

## Why This Experiment

028 is the current best:

```text
028 layers9 block192 dropout0.15: 4.0675
```

But validation loss worsened after step 1500 while train loss kept dropping. Increasing dropout may
reduce overfitting.

## Parameter Estimate

Dropout does not change parameter count:

```text
028 params = 8,211,456
030 params = 8,211,456
```

## Results

| Step | Train Loss | Val Loss | Val Nats/Char | Best Val | Notes |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 8.3906 | 8.3951 | 8.3951 | 8.3951 | initial |
| 250 | 4.6229 | 4.8469 | 4.8469 | 4.8469 | improving |
| 500 | 4.1360 | 4.5033 | 4.5033 | 4.5033 | improving |
| 750 | 3.9067 | 4.3458 | 4.3458 | 4.3458 | improving |
| 1000 | 3.7609 | 4.2515 | 4.2515 | 4.2515 | improving |
| 1250 | 3.5753 | 4.1812 | 4.1812 | 4.1812 | improving |
| 1500 | 3.3957 | 4.0808 | 4.0808 | 4.0808 | improving |
| 1750 | 3.2640 | 4.1136 | 4.1136 | 4.0808 | no new best |
| 2000 | 3.1523 | 4.1256 | 4.1256 | 4.0808 | no new best |
| 2250 | 3.0061 | 4.0891 | 4.0891 | 4.0808 | close |
| 2500 | 2.8727 | 4.0658 | 4.0658 | 4.0658 | best checkpoint |

## Compare

```text
027 layers9 block128 dropout0.15: 4.0761
028 layers9 block192 dropout0.15: 4.0675
029 layers9 block256 dropout0.15: 4.1003
030 layers9 block192 dropout0.20: 4.0658
```

## Conclusion

Dropout 0.20 slightly improves the metric:

```text
028 dropout0.15: 4.0675
030 dropout0.20: 4.0658
delta:          -0.0016
```

This is the current best validation loss, but the margin is tiny. Manual generation does not show a
clear improvement, and one sample repeated "嘗嘗嘗".

Checkpoint size:

```text
best_ckpt.pt = 32,942,778 bytes
```

Current best by validation loss:

```text
runs/hongloumeng_80_char_width256_rope_layers9_context192_dropout020_textsplit/best_ckpt.pt
```

## Command

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_char_width256_rope_layers9_context192_dropout020_textsplit.yaml
```
