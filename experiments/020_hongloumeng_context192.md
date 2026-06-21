# 020 Hongloumeng Context 192

## Goal

Test whether longer context helps the current best RoPE baseline.

This experiment changes one training/inference context setting:

```text
block_size: 128 -> 192
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
dropout = 0.15
norm = layernorm
mlp = gelu
position = rope
max_steps = 5000
```

Changed:

```text
block_size = 192
```

## Why Context Might Help

This is a character-level model. A context of 128 characters is often only a few dialogue turns, so
the model may not see enough local scene information to keep speakers, tone, and event thread
stable.

With RoPE, increasing block size does not add learned position embedding parameters. It mainly
increases attention compute:

```text
128 * 128 = 16,384 attention positions
192 * 192 = 36,864 attention positions
ratio     = 2.25x
```

So this experiment asks:

```text
Does giving the same 5.84M-parameter model more context improve held-out prediction?
```

## Parameter Estimate

Measured with the current Hongloumeng 80-chapter character vocabulary:

```text
vocab_size = 4309
016 block 128 params: 5,842,176
020 block 192 params: 5,842,176
delta: 0
```

## Train

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_char_width256_rope_context192.yaml
```

## Generate

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.generate --checkpoint runs\hongloumeng_80_char_width256_rope_context192\best_ckpt.pt --prompt "黛玉聽了，" --max-new-tokens 300 --temperature 0.7 --top-k 50 --top-p 0.9
```

## Results

| Step | Train Loss | Val Loss | Best Val | Notes |
| ---: | ---: | ---: | ---: | --- |
| 0 | 8.4162 | 8.4169 | 8.4169 | initial |
| 250 | 4.5865 | 4.7928 | 4.7928 | improving |
| 500 | 4.1279 | 4.4329 | 4.4329 | improving |
| 750 | 3.8977 | 4.3127 | 4.3127 | improving |
| 1000 | 3.6869 | 4.1904 | 4.1904 | improving |
| 1250 | 3.5684 | 4.1562 | 4.1562 | improving |
| 1500 | 3.4190 | 4.1407 | 4.1407 | improving |
| 1750 | 3.2070 | 4.0909 | 4.0909 | best checkpoint |
| 2000 | 3.1067 | 4.1088 | 4.0909 | no new best |
| 2250 | 2.9878 | 4.1077 | 4.0909 | no new best |
| 2500 | 2.8149 | 4.1023 | 4.0909 | close |
| 2750 | 2.7048 | 4.1672 | 4.0909 | overfitting starts |
| 3000 | 2.5729 | 4.2086 | 4.0909 | overfitting |
| 3250 | 2.4602 | 4.2532 | 4.0909 | overfitting |
| 3500 | 2.3436 | 4.2734 | 4.0909 | overfitting |
| 3750 | 2.2373 | 4.3407 | 4.0909 | overfitting |
| 4000 | 2.1434 | 4.3442 | 4.0909 | overfitting |
| 4250 | 2.0415 | 4.4120 | 4.0909 | overfitting |
| 4500 | 1.9278 | 4.5058 | 4.0909 | overfitting |
| 4750 | 1.8589 | 4.5131 | 4.0909 | overfitting |
| 5000 | 1.7545 | 4.6380 | 4.0909 | final checkpoint much worse than best |

## Compare Against 016

```text
016 RoPE block 128 best validation: 4.0805
020 RoPE block 192 best validation: 4.0909
delta:                              +0.0104
```

Lower is better, so context 192 did not beat 016, but it is much closer than the wider or SwiGLU
runs.

Checkpoint size comparison:

```text
016 best_ckpt.pt: 23,452,563 bytes
020 best_ckpt.pt: 23,452,563 bytes
delta:                   0 bytes
```

## Sample Notes

Generated from `best_ckpt.pt`, temperature `0.7`, top-k `50`, top-p `0.9`.

Prompt:

```text
黛玉聽了，
```

The sample keeps a longer dialogue thread than some earlier runs and uses actions such as "命人去"
and "回頭向寶玉道". It still repeats Bao-yu dialogue turns and loses exact event direction.

Prompt:

```text
寶玉笑道：「
```

The sample is locally fluent and keeps a sustained Bao-yu-centered exchange. It still repeats
generic patterns like "我不知道", "你說", and "什麼話".

Prompt:

```text
賈母笑道：「
```

The sample includes Jia-mu, Yuan-yang, Feng-jie, and Zhou Rui's wife, with a paragraph break and
some scene rhythm. Repetition remains around "不知道", "什麼話", and "說話".

## Conclusion

Context 192 is the best non-winning experiment so far:

```text
016 width 256, block 128: 4.0805
018 SwiGLU:               4.1145
019 width 384:            4.0998
020 block 192:            4.0909
```

Unlike width 384, it does not add parameters. It improves many synchronized validation points over
016 and overfits less violently than the wider model. This suggests context length is a real
bottleneck for the character-level Hongloumeng task, even if block 192 did not set a new best.

Current best remains:

```text
016 RoPE + LayerNorm + GELU, width 256, block 128
```

The next architectural option is depth:

```text
021 = 016 + n_layer 6 -> 8
```

## Test Scripts

Train 016:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_char_width256_rope.yaml
```

Train 020:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_char_width256_rope_context192.yaml
```

Generate 016:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.generate --checkpoint runs\hongloumeng_80_char_width256_rope\best_ckpt.pt --prompt "黛玉聽了，" --max-new-tokens 300 --temperature 0.7 --top-k 50 --top-p 0.9
```

Generate 020:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.generate --checkpoint runs\hongloumeng_80_char_width256_rope_context192\best_ckpt.pt --prompt "黛玉聽了，" --max-new-tokens 300 --temperature 0.7 --top-k 50 --top-p 0.9
```
