# 027 Hongloumeng Layers 9 Text-Split

## Goal

Increase depth by 50% from the current best 016-style baseline.

This changes:

```text
n_layer: 6 -> 9
```

Everything else stays close to 016:

```text
tokenizer = char
block_size = 128
n_head = 8
n_embd = 256
dropout = 0.15
norm = layernorm
mlp = gelu
position = rope
split = text
```

## Parameter Estimate

Measured with the current Hongloumeng 80-chapter character vocabulary:

```text
6 layers: 5,842,176
9 layers: 8,211,456
delta:   +2,369,280
ratio:    1.41x
```

Depth adds more repeated attention/MLP processing without widening the residual stream.

## Results

| Step | Train Loss | Val Loss | Val Nats/Char | Best Val | Notes |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 8.3950 | 8.3959 | 8.3959 | 8.3959 | initial |
| 250 | 4.7178 | 4.9042 | 4.9042 | 4.9042 | improving |
| 500 | 4.2533 | 4.5761 | 4.5761 | 4.5761 | improving |
| 750 | 4.0171 | 4.3487 | 4.3487 | 4.3487 | improving |
| 1000 | 3.8323 | 4.2838 | 4.2838 | 4.2838 | improving |
| 1250 | 3.7226 | 4.1908 | 4.1908 | 4.1908 | improving |
| 1500 | 3.5464 | 4.1422 | 4.1422 | 4.1422 | improving |
| 1750 | 3.4131 | 4.1689 | 4.1689 | 4.1422 | no new best |
| 2000 | 3.2827 | 4.0761 | 4.0761 | 4.0761 | best checkpoint |
| 2250 | 3.2202 | 4.1290 | 4.1290 | 4.0761 | no new best |
| 2500 | 3.0146 | 4.1431 | 4.1431 | 4.0761 | overfitting |

## Compare

```text
025A char 6-layer block128: 4.0805
027 char 9-layer block128: 4.0761
```

## Conclusion

Depth +50% is the first change to slightly beat the 016/025A validation baseline:

```text
025A 6 layers: 4.0805
027 9 layers: 4.0761
delta:        -0.0044
```

The win is real in the logged metric but very small. It is not large enough to guarantee obvious
manual generation improvement. It does suggest depth scaling is healthier than width384 or BPE6000
for this setup.

Current best by validation loss:

```text
027 char tokenizer, 9 layers, block 128
```

Checkpoint size:

```text
best_ckpt.pt = 32,942,778 bytes
```

Sample notes:

```text
Prompt 黛玉聽了，
```

The sample keeps Hongloumeng-style dialogue rhythm, but still repeats local forms such as "你們",
"說話", and "知道". The loss improvement is too small to guarantee obvious manual sampling gains.

```text
Prompt 寶玉笑道：「
```

The sample has a few better paragraph transitions and named-character turns, but still drifts into
generic dialogue. Depth helps the metric slightly; it does not solve coherence.

## Command

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_char_width256_rope_layers9_textsplit.yaml
```
