# 035 Hongloumeng 120 BPE4700 Cosine LR

## Goal

Run a controlled tokenizer comparison after the 120-chapter and cosine-LR improvements.

```text
033 tokenizer = char
035 tokenizer = BPE4700
```

Everything else stays fixed:

```text
data_path = data/hongloumeng_120.txt
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
lr_schedule = warmup + cosine decay
```

## Why This Experiment

Earlier BPE4500 experiments on the 80-chapter corpus did not beat character tokenization, but the
training setup has changed materially:

```text
data = 80 chapters -> 120 chapters
optimization = fixed LR -> warmup + cosine decay
model = 6 layers -> 9 layers
```

This run retests whether BPE benefits from the stronger setup. Use context192, not context288,
so the only changed variable against 033 is tokenizer.

Important correction:

```text
120-chapter char vocab = 4511
```

Our BPE implementation starts from all single-character pieces. Therefore `vocab_size=4500` is
smaller than the base vocabulary and cannot perform any merges; it degenerates to char tokenization.
Use `vocab_size=4700` instead, which gives roughly the same merge budget as the old 80-chapter
BPE4500 run:

```text
80 chapters:  char vocab 4309 -> BPE4500 gives 191 merges
120 chapters: char vocab 4511 -> BPE4700 gives 189 merges
```

## Variable Control

| Experiment | Tokenizer | Dataset | Layers | Width | Context | Dropout | LR Schedule |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 033 | char | Hongloumeng 120 | 9 | 256 | 192 | 0.20 | cosine |
| 035 | BPE4700 | Hongloumeng 120 | 9 | 256 | 192 | 0.20 | cosine |

## Tokenizer Cache

```text
cache_path = data/tokenizers/hongloumeng_120_bpe4700.json
train_encoded_cache_path = data/tokenizers/hongloumeng_120_bpe4700_train_textsplit_ids.pt
val_encoded_cache_path = data/tokenizers/hongloumeng_120_bpe4700_val_textsplit_ids.pt
```

## Parameter Note

BPE4700 has a small parameter increase over the 120-chapter char model:

```text
delta ~= (4700 - 4511) * 256 = 48,384 params
```

The practical difference is not parameter count. It is sequence compression and token granularity.

## Results

| Step | LR | Train Loss | Val Loss | Val Nats/Char | Best Val Nats/Char | Notes |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.000000 | 8.5043 | 8.5030 | 6.7993 | 6.7993 | init |
| 250 | 0.000300 | 6.1565 | 6.1456 | 4.9142 | 4.9142 | improving |
| 500 | 0.000297 | 5.2675 | 5.3384 | 4.2688 | 4.2688 | improving |
| 750 | 0.000291 | 4.9083 | 5.0934 | 4.0729 | 4.0729 | improving |
| 1000 | 0.000282 | 4.7006 | 4.9036 | 3.9210 | 3.9210 | improving |
| 1250 | 0.000269 | 4.4632 | 4.8057 | 3.8428 | 3.8428 | improving |
| 1500 | 0.000254 | 4.3179 | 4.7314 | 3.7834 | 3.7834 | improving |
| 1750 | 0.000224 | 4.1858 | 4.6463 | 3.7153 | 3.7153 | improving |
| 2000 | 0.000222 | 4.0302 | 4.6304 | 3.7026 | 3.7026 | improving |
| 2250 | 0.000196 | 3.9290 | 4.6241 | 3.6976 | 3.6976 | small gain |
| 2500 | 0.000174 | 3.8168 | 4.5713 | 3.6554 | 3.6554 | improving |
| 2750 | 0.000155 | 3.7173 | 4.5419 | 3.6319 | 3.6319 | improving |
| 3000 | 0.000130 | 3.6340 | 4.5721 | 3.6560 | 3.6319 | rebound |
| 3250 | 0.000109 | 3.5834 | 4.5633 | 3.6490 | 3.6319 | no new best |
| 3500 | 0.000090 | 3.5031 | 4.5671 | 3.6520 | 3.6319 | no new best |
| 3750 | 0.000073 | 3.4514 | 4.5267 | 3.6197 | 3.6197 | new best |
| 4000 | 0.000058 | 3.4051 | 4.5517 | 3.6397 | 3.6197 | rebound |
| 4250 | 0.000046 | 3.3654 | 4.5390 | 3.6295 | 3.6197 | no new best |
| 4500 | 0.000038 | 3.3270 | 4.5293 | 3.6218 | 3.6197 | near best |
| 4750 | 0.000032 | 3.3222 | 4.5572 | 3.6441 | 3.6197 | rebound |
| 5000 | 0.000030 | 3.2890 | 4.5215 | 3.6156 | 3.6156 | best checkpoint |

Best checkpoint:

```text
runs/hongloumeng_120_bpe4700_width256_rope_layers9_context192_dropout020_textsplit_cosine/best_ckpt.pt
best val_nats_per_char ~= 3.6156 at step 5000
checkpoint size ~= 33.35 MB
```

## Token Stats

```text
pieces = 4700
merges = 189
train tokens/char = 0.8142
val tokens/char = 0.7996
```

The tokenizer is now a real BPE tokenizer, not the failed BPE4500 degenerate case. It compresses the
text to about 80% of the character-token length.

## Comparison

```text
033 char context192 best val_nats_per_char ~= 3.5798
034 char context288 best val_nats_per_char ~= 3.5726
035 BPE4700 context192 best val_nats_per_char ~= 3.6156
```

BPE4700 is healthier than the old 80-chapter BPE4500 run, but it still trails character tokenization.
The gap to 033 is:

```text
3.6156 - 3.5798 ~= 0.0358
```

So the stronger 120-chapter/cosine setup helps BPE, but not enough to beat char.

## Sample Notes

Samples are not qualitatively better than the char models. They remain locally fluent but drift in
scene state and speaker relationships.

Prompt:

```text
黛玉聽了，
```

Observation: the continuation has smoother local phrases, but quickly mixes multiple relationship
cues and unstable references.

Prompt:

```text
寶玉笑道：「
```

Observation: the dialogue cadence is plausible, but the model jumps between 寶玉, 黛玉, 秦氏, and
other figures without stable scene grounding.

## Decision Rule

Compare by `val_nats_per_char`, not raw token loss, because BPE and char have different tokens per
character.

```text
033 char context192 best val_nats_per_char ~= 3.5798
035 BPE4500 must beat this to justify switching tokenizer.
```

035 trails char tokenization on the fair nats-per-character metric, and samples do not compensate
qualitatively. Keep char tokenizer as the default for the next experiment.

Recommended next options:

```text
036 = char + 120 chapters + cosine + context192 + mixed corpus, e.g. Rulin Waishi
037 = char + 120 chapters + cosine + context192 + larger capacity
```

Do not test BPE4700 + context288 yet unless the goal is specifically to map BPE behavior; it is not
the best route for quality right now.

## Command

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_120_bpe4700_width256_rope_layers9_context192_dropout020_textsplit_cosine.yaml
```
