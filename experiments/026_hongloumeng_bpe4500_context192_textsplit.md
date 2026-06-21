# 026 Hongloumeng BPE 4500 Context 192 Text-Split

## Goal

Test the most promising BPE setting with longer context and raw-text validation split.

This combines:

```text
BPE vocab_size = 4500
block_size = 192
split = text
```

## Setup

Same backbone as 016:

```text
n_layer = 6
n_head = 8
n_embd = 256
dropout = 0.15
norm = layernorm
mlp = gelu
position = rope
```

Tokenizer/cache:

```text
cache_path = data/tokenizers/hongloumeng_80_bpe4500.json
train_encoded_cache_path = data/tokenizers/hongloumeng_80_bpe4500_train_textsplit_ids.pt
val_encoded_cache_path = data/tokenizers/hongloumeng_80_bpe4500_val_textsplit_ids.pt
```

## Why This Experiment

BPE4500 was closer to the char baselines than BPE6000. Context192 helped both char and BPE6000.
This run checks whether the smaller, less sparse BPE tokenizer benefits more from longer context.

## Results

| Step | Train Loss | Val Loss | Val Nats/Char | Best Val | Notes |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 8.4640 | 8.4642 | 6.9842 | 8.4642 | initial |
| 250 | 5.9593 | 6.1183 | 5.0485 | 6.1183 | improving |
| 500 | 5.1095 | 5.4695 | 4.5132 | 5.4695 | improving |
| 750 | 4.7015 | 5.2329 | 4.3179 | 5.2329 | improving |
| 1000 | 4.4208 | 5.0972 | 4.2059 | 5.0972 | improving |
| 1250 | 4.1305 | 5.0737 | 4.1865 | 5.0737 | improving |
| 1500 | 3.9094 | 5.0160 | 4.1389 | 5.0160 | improving |
| 1750 | 3.6951 | 4.9948 | 4.1214 | 4.9948 | best checkpoint |
| 2000 | 3.4995 | 5.0492 | 4.1664 | 4.9948 | no new best |
| 2250 | 3.2909 | 5.0583 | 4.1738 | 4.9948 | overfitting |
| 2500 | 3.0906 | 5.1396 | 4.2409 | 4.9948 | overfitting |

## BPE Text-Split Token Stats

```text
train tokens/char = 0.8159
val tokens/char = 0.8251
```

The validation text is slightly less compressible than training text under BPE4500.

## Compare

```text
025A char block128 textsplit:       4.0805
025B char block192 textsplit:       4.0909
021 BPE4500 block128 token-split:   4.1059
025C BPE6000 block192 textsplit:    4.1810
026 BPE4500 block192 textsplit:     4.1214
```

## Conclusion

BPE4500 + context192 is much better than BPE6000 + context192 under raw text split:

```text
026 BPE4500 block192: 4.1214
025C BPE6000 block192: 4.1810
```

But it still trails the char tokenizer baselines:

```text
025A char block128: 4.0805
025B char block192: 4.0909
026 BPE4500 block192: 4.1214
```

So smaller BPE is healthier than larger BPE, but current char tokenization remains stronger by loss.

## Command

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_bpe4500_width256_rope_context192_textsplit.yaml
```
