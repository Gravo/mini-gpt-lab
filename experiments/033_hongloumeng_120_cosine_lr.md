# 033 Hongloumeng 120 Cosine Learning Rate

## Goal

Run a controlled follow-up to 032 by changing only the learning-rate behavior.

```text
032 = fixed learning_rate 0.0003
033 = warmup + cosine decay from 0.0003 to 0.00003
```

Everything else stays fixed:

```text
data_path = data/hongloumeng_120.txt
tokenizer = char
split = text
n_layer = 9
n_head = 8
n_embd = 256
block_size = 192
dropout = 0.20
norm = layernorm
mlp = gelu
position = rope
max_steps = 5000
```

## Why This Experiment

032 improved after 2500 steps but peaked at step 4000:

```text
032 best val ~= 3.6198 at step 4000
032 final val ~= 3.7422 at step 5000
```

That means fixed learning rate kept pushing the training set while validation got worse. A decaying
learning rate may keep the useful extra optimization while reducing late-step damage.

## Variable Control

| Experiment | Dataset | Layers | Width | Context | Dropout | Steps | Learning Rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 032 | Hongloumeng 120 | 9 | 256 | 192 | 0.20 | 5000 | fixed 0.0003 |
| 033 | Hongloumeng 120 | 9 | 256 | 192 | 0.20 | 5000 | cosine 0.0003 -> 0.00003 |

## Schedule

```text
warmup_steps = 200
peak_lr = 0.0003
min_lr = 0.00003
decay_steps = 5000
```

## Results

| Step | LR | Train Loss | Val Loss | Val Nats/Char | Best Val | Notes |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.000000 | 8.4267 | 8.4174 | 8.4174 | 8.4174 | warmup start |
| 250 | 0.000300 | 4.8723 | 4.7444 | 4.7444 | 4.7444 | slower than fixed LR |
| 500 | 0.000297 | 4.2929 | 4.2270 | 4.2270 | 4.2270 | near fixed LR |
| 750 | 0.000291 | 4.0772 | 4.0938 | 4.0938 | 4.0938 | improving |
| 1000 | 0.000282 | 3.8597 | 3.9652 | 3.9652 | 3.9652 | improving |
| 1250 | 0.000269 | 3.7489 | 3.8494 | 3.8494 | 3.8494 | improving |
| 1500 | 0.000254 | 3.6125 | 3.7715 | 3.7715 | 3.7715 | improving |
| 1750 | 0.000224 | 3.4783 | 3.7226 | 3.7226 | 3.7226 | improving |
| 2000 | 0.000222 | 3.4278 | 3.7370 | 3.7370 | 3.7226 | rebound |
| 2250 | 0.000196 | 3.3550 | 3.6906 | 3.6906 | 3.6906 | new best |
| 2500 | 0.000174 | 3.2602 | 3.6657 | 3.6657 | 3.6657 | slightly behind 032 |
| 2750 | 0.000155 | 3.1698 | 3.6545 | 3.6545 | 3.6545 | improving |
| 3000 | 0.000130 | 3.1589 | 3.6610 | 3.6610 | 3.6545 | rebound |
| 3250 | 0.000109 | 3.1089 | 3.6294 | 3.6294 | 3.6294 | beats 032 same step |
| 3500 | 0.000090 | 3.0623 | 3.6187 | 3.6187 | 3.6187 | beats 032 best |
| 3750 | 0.000073 | 3.0268 | 3.6093 | 3.6093 | 3.6093 | new best |
| 4000 | 0.000058 | 2.9769 | 3.5798 | 3.5798 | 3.5798 | best checkpoint |
| 4250 | 0.000046 | 2.9549 | 3.6117 | 3.6117 | 3.5798 | mild rebound |
| 4500 | 0.000038 | 2.9168 | 3.5903 | 3.5903 | 3.5798 | still stable |
| 4750 | 0.000032 | 2.9202 | 3.6077 | 3.6077 | 3.5798 | no new best |
| 5000 | 0.000030 | 2.8742 | 3.6373 | 3.6373 | 3.5798 | better than 032 final |

Best checkpoint:

```text
runs/hongloumeng_120_char_width256_rope_layers9_context192_dropout020_textsplit_cosine/best_ckpt.pt
best val ~= 3.5798 at step 4000
checkpoint size ~= 33.15 MB
```

## Comparison

```text
032 fixed LR best val ~= 3.6198 at step 4000
033 cosine LR best val ~= 3.5798 at step 4000
delta ~= -0.0401

032 fixed LR final val ~= 3.7422 at step 5000
033 cosine LR final val ~= 3.6373 at step 5000
delta ~= -0.1049
```

033 improves the best validation point and also reduces the late-step validation spike. The early
phase is slightly slower because of warmup and decay, but the middle and late phase are healthier.

This is the clearest positive controlled-variable result since moving to 120 chapters.

## Sample Notes

The generated samples are still local-dialogue coherent rather than scene coherent. The metric gain
does not mean the model has solved plot state or long-range consistency.

Prompt:

```text
黛玉聽了，
```

Observation: the sample keeps names and dialogue style active, including 黛玉, 寶玉, 湘雲, and
寶釵. It still drifts into repeated social-dialogue patterns and inconsistent object references.

Prompt:

```text
寶玉笑道：「
```

Observation: the sample has smoother short turns, but it still repeats "這是什麼話" style local
phrasing and does not maintain a stable scene.

## Decision Rule

033 beats 032's best validation loss and avoids most of the late validation spike. Keep warmup +
cosine decay as the default for the next 120-chapter controlled experiments.

Next controlled variable:

```text
034 = keep 033 schedule, increase block_size from 192 to 288
```

This tests whether the 120-chapter corpus can use longer context once optimization is healthier.

## Command

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_120_char_width256_rope_layers9_context192_dropout020_textsplit_cosine.yaml
```
