# 036 Hongloumeng 120 Char 30M Cosine LR

## Goal

Scale the strongest character-tokenizer setup from about 8.26M parameters to about 30M parameters.

This changes model capacity only:

```text
033 = 9 layers, width 256, 8 heads, about 8.26M params
036 = 16 layers, width 384, 12 heads, about 30.12M params
```

Everything else stays fixed:

```text
data_path = data/hongloumeng_120.txt
tokenizer = char
split = text
block_size = 192
dropout = 0.20
norm = layernorm
mlp = gelu
position = rope
max_steps = 5000
lr_schedule = warmup + cosine decay
```

## Why This Experiment

The best small-model route is currently:

```text
033 char context192 cosine best val ~= 3.5798
034 char context288 cosine best val ~= 3.5726
035 BPE4700 context192 cosine best val_nats_per_char ~= 3.6156
```

BPE did not beat char, and context288 produced only a small gain with higher compute cost. The next
question is whether model capacity is now the main bottleneck.

## Variable Control

| Experiment | Tokenizer | Dataset | Layers | Width | Heads | Context | Params | Changed Variable |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 033 | char | Hongloumeng 120 | 9 | 256 | 8 | 192 | 8.26M | baseline |
| 036 | char | Hongloumeng 120 | 16 | 384 | 12 | 192 | 30.12M | capacity |

## Parameter Count

Measured by instantiating the model:

```text
033 params = 8,263,168
036 params = 30,124,416
scale = 3.65x
```

Head dimension remains:

```text
384 / 12 = 32
```

This preserves the same per-head width as the 256/8-head models.

## Expected Tradeoff

The larger model should have enough capacity to learn richer local syntax and reduce repetition, but
it may also overfit the 120-chapter corpus faster. Watch the train/val gap carefully.

Compute will increase substantially because most Transformer block parameters and matmuls scale
roughly with `n_layer * n_embd^2`:

```text
(16 / 9) * (384 / 256)^2 ~= 4.0x block compute
```

## Results

| Step | LR | Train Loss | Val Loss | Val Nats/Char | Best Val | Notes |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.000000 | 8.5021 | 8.4941 | 8.4941 | 8.4941 | init |
| 250 | 0.000300 | 4.6514 | 4.5674 | 4.5674 | 4.5674 | faster than 033 |
| 500 | 0.000297 | 4.1125 | 4.1420 | 4.1420 | 4.1420 | faster than 033 |
| 750 | 0.000291 | 3.8168 | 3.9049 | 3.9049 | 3.9049 | faster than 033 |
| 1000 | 0.000282 | 3.5918 | 3.8042 | 3.8042 | 3.8042 | pilot stopped here |

Pilot checkpoint:

```text
runs/hongloumeng_120_char_width384_rope_layers16_context192_dropout020_textsplit_cosine/best_ckpt.pt
best val ~= 3.8042 at step 1000
checkpoint size ~= 120.63 MB
```

## Pilot Comparison

Same-step comparison:

```text
033 8.26M step 1000 val ~= 3.9652
034 8.26M context288 step 1000 val ~= 3.8734
036 30.12M step 1000 val ~= 3.8042
```

The 30M model learns faster and generalizes better in the early phase. This is a real positive
capacity signal.

However, wall-clock cost is much higher. The pilot was manually stopped after step 1000 because the
full 5000-step run would take substantially longer than the 8M experiments. The current evidence is
enough to say "capacity helps early," but not enough to know the final best validation loss or the
overfitting point.

## Sample Notes

Prompt:

```text
黛玉聽了，
```

Observation: the step-1000 30M sample has smoother local dialogue than an undertrained small model,
but it still drifts and repeats speaker-turn patterns.

Prompt:

```text
寶玉笑道：「
```

Observation: the model can bring in 寶釵, 薛蝌, 薛姨媽 and maintain short dialogue cadence, but scene
state is still unstable.

No qualitative breakthrough yet. This is expected because the checkpoint is still early.

## Decision Rule

The pilot gives a clear early validation improvement, so capacity is a real bottleneck. The next
step is to run a more practical 30M schedule rather than blindly committing to 5000 steps.

Recommended next controlled run:

```text
037 = same 30M model, max_steps 2500, decay_steps 5000, evaluate whether it approaches or beats 3.57
```

Keeping `decay_steps = 5000` preserves the same LR curve through step 2500 while avoiding a very long
full run. If 2500 is still strongly improving, then schedule a longer background run.

## Command

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_120_char_width384_rope_layers16_context192_dropout020_textsplit_cosine.yaml
```
