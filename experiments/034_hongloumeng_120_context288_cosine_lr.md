# 034 Hongloumeng 120 Context 288 Cosine LR

## Goal

Run a controlled follow-up to 033 by changing only context length.

```text
033 block_size = 192
034 block_size = 288
```

Everything else stays fixed:

```text
data_path = data/hongloumeng_120.txt
tokenizer = char
split = text
n_layer = 9
n_head = 8
n_embd = 256
dropout = 0.20
norm = layernorm
mlp = gelu
position = rope
max_steps = 5000
lr_schedule = warmup + cosine decay
```

## Why This Experiment

Earlier context-length experiments were mixed or negative, especially before the 120-chapter data and
cosine learning-rate schedule. Since 033 made optimization healthier, retest whether the model can
use a longer character context.

## Variable Control

| Experiment | Dataset | Layers | Width | Context | Dropout | LR Schedule | Changed Variable |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 033 | Hongloumeng 120 | 9 | 256 | 192 | 0.20 | cosine | baseline |
| 034 | Hongloumeng 120 | 9 | 256 | 288 | 0.20 | cosine | context length |

## Parameter Note

With RoPE, increasing `block_size` does not add learned position embeddings. The parameter count
should remain the same as 033:

```text
params ~= 8,263,168
```

The compute cost does increase because attention compares more positions:

```text
192 -> 288 tokens = 1.5x sequence length
attention score matrix per head grows by (288 / 192)^2 = 2.25x
```

## Results

| Step | LR | Train Loss | Val Loss | Val Nats/Char | Best Val | Notes |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.000000 | 8.4263 | 8.4160 | 8.4160 | 8.4160 | init |
| 250 | 0.000300 | 4.7768 | 4.7219 | 4.7219 | 4.7219 | faster than 033 |
| 500 | 0.000297 | 4.2144 | 4.2342 | 4.2342 | 4.2342 | slightly behind 033 |
| 750 | 0.000291 | 3.9330 | 4.0081 | 4.0081 | 4.0081 | strong mid-early gain |
| 1000 | 0.000282 | 3.7760 | 3.8734 | 3.8734 | 3.8734 | ahead of 033 |
| 1250 | 0.000269 | 3.5943 | 3.8115 | 3.8115 | 3.8115 | ahead of 033 |
| 1500 | 0.000254 | 3.4625 | 3.7445 | 3.7445 | 3.7445 | ahead of 033 |
| 1750 | 0.000224 | 3.3444 | 3.6236 | 3.6236 | 3.6236 | strong new best |
| 2000 | 0.000222 | 3.2518 | 3.6546 | 3.6546 | 3.6236 | rebound |
| 2250 | 0.000196 | 3.1570 | 3.6489 | 3.6489 | 3.6236 | no new best |
| 2500 | 0.000174 | 3.0605 | 3.6179 | 3.6179 | 3.6179 | new best |
| 2750 | 0.000155 | 3.0276 | 3.5733 | 3.5733 | 3.5733 | beats 033 best |
| 3000 | 0.000130 | 2.9146 | 3.5726 | 3.5726 | 3.5726 | best checkpoint |
| 3250 | 0.000109 | 2.8677 | 3.5905 | 3.5905 | 3.5726 | rebound |
| 3500 | 0.000090 | 2.8223 | 3.5989 | 3.5989 | 3.5726 | no new best |
| 3750 | 0.000073 | 2.7545 | 3.5782 | 3.5782 | 3.5726 | near best |
| 4000 | 0.000058 | 2.7170 | 3.5796 | 3.5796 | 3.5726 | near 033 best |
| 4250 | 0.000046 | 2.6865 | 3.5818 | 3.5818 | 3.5726 | stable |
| 4500 | 0.000038 | 2.6785 | 3.5816 | 3.5816 | 3.5726 | stable |
| 4750 | 0.000032 | 2.6587 | 3.5885 | 3.5885 | 3.5726 | mild rebound |
| 5000 | 0.000030 | 2.6193 | 3.6092 | 3.6092 | 3.5726 | no late collapse |

Best checkpoint:

```text
runs/hongloumeng_120_char_width256_rope_layers9_context288_dropout020_textsplit_cosine/best_ckpt.pt
best val ~= 3.5726 at step 3000
checkpoint size ~= 33.15 MB
```

## Comparison

```text
033 context192 best val ~= 3.5798 at step 4000
034 context288 best val ~= 3.5726 at step 3000
delta ~= -0.0072
```

Context 288 is a small metric win, and it reaches its best point earlier. The gain is real but much
smaller than the LR-schedule gain from 033.

The cost is also real: sequence length increases 1.5x, and the per-head attention score matrix is
2.25x larger. Wall-clock training was visibly slower.

## Sample Notes

Generation does not show a qualitative breakthrough. Samples are still locally plausible dialogue
with unstable scene state.

Prompt:

```text
黛玉聽了，
```

Observation: the model produces smoother short dialogue and paragraph breaks, but still repeats
speaker patterns and loses stable object/person references.

Prompt:

```text
寶玉笑道：「
```

Observation: the sample includes a wider social cast such as 鳳姐 and 鴛鴦, but the scene does not
remain coherent for long.

## Decision Rule

034 beats 033's best validation loss, but only narrowly. For architecture exploration, context 288
is worth keeping in mind. For the next tokenizer comparison, use context 192 first to isolate the
tokenizer variable and avoid mixing "BPE" with "longer context".

Recommended next experiment:

```text
035 = BPE4500 + 120 chapters + cosine LR + context192
```

This cleanly asks whether BPE4500 benefits from the stronger 120-chapter/cosine setup.

## Command

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_120_char_width256_rope_layers9_context288_dropout020_textsplit_cosine.yaml
```
