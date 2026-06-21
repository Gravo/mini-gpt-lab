# 024 Hongloumeng BPE 6000 Context 192

## Goal

Test whether BPE 6000 benefits from a longer context window.

This combines two previous directions:

```text
023 = BPE 6000, block 128
020 = char tokenizer, block 192
024 = BPE 6000, block 192
```

Everything else stays close to 016.

## Controlled Setup

Same model backbone:

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

Changed from 023:

```text
block_size = 128 -> 192
```

Tokenizer:

```text
type = bpe
vocab_size = 6000
cache_path = data/tokenizers/hongloumeng_80_bpe6000.json
encoded_cache_path = data/tokenizers/hongloumeng_80_bpe6000_ids.pt
```

## Loss Comparison Note

For BPE runs, token-level loss is not comparable to char-tokenizer loss. The fairer metric is:

```text
val_nats_per_char = token_loss * tokens_per_char
```

This is a reasonable approximation of average negative log-likelihood per original character for a
fixed deterministic tokenizer.

Important caveat:

```text
the current training script splits train/val after tokenization
```

That means char and BPE runs do not have exactly identical raw-text validation boundaries. The
metric is still useful directionally, but a stricter future comparison should split raw text first,
then tokenize train and val separately.

## Parameter Estimate

Same as 023:

```text
params = 6,275,072
```

RoPE means increasing block size does not add model parameters, but attention compute increases:

```text
128 * 128 = 16,384
192 * 192 = 36,864
ratio = 2.25x
```

## Train

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_bpe6000_width256_rope_context192.yaml
```

## Generate

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.generate --checkpoint runs\hongloumeng_80_bpe6000_width256_rope_context192\best_ckpt.pt --prompt "黛玉聽了，" --max-new-tokens 300 --temperature 0.7 --top-k 50 --top-p 0.9
```

## Results

| Step | Train Loss | Val Loss | Val Nats/Char | Best Val | Notes |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 8.7507 | 8.7512 | 5.8707 | 8.7512 | initial |
| 250 | 7.0712 | 7.2469 | 4.8616 | 7.2469 | improving |
| 500 | 6.1888 | 6.6096 | 4.4340 | 6.6096 | improving |
| 750 | 5.6353 | 6.3574 | 4.2648 | 6.3574 | improving |
| 1000 | 5.2443 | 6.2172 | 4.1708 | 6.2172 | improving |
| 1250 | 4.8783 | 6.1359 | 4.1163 | 6.1359 | best checkpoint |
| 1500 | 4.5565 | 6.1979 | 4.1578 | 6.1359 | overfitting starts |
| 1750 | 4.2023 | 6.2102 | 4.1661 | 6.1359 | no new best |
| 2000 | 3.8974 | 6.2527 | 4.1946 | 6.1359 | overfitting |
| 2250 | 3.5824 | 6.3354 | 4.2501 | 6.1359 | overfitting |
| 2500 | 3.2464 | 6.4242 | 4.3096 | 6.1359 | overfitting |
| 2750 | 3.0172 | 6.5799 | 4.4141 | 6.1359 | overfitting |
| 3000 | 2.7473 | 6.7099 | 4.5013 | 6.1359 | overfitting |
| 3250 | 2.4876 | 6.7975 | 4.5601 | 6.1359 | overfitting |
| 3500 | 2.2707 | 6.9350 | 4.6523 | 6.1359 | overfitting |
| 3750 | 2.0679 | 7.0908 | 4.7568 | 6.1359 | overfitting |
| 4000 | 1.8539 | 7.2776 | 4.8821 | 6.1359 | overfitting |
| 4250 | 1.7129 | 7.3077 | 4.9023 | 6.1359 | overfitting |
| 4500 | 1.5444 | 7.4750 | 5.0146 | 6.1359 | overfitting |
| 4750 | 1.4429 | 7.6199 | 5.1118 | 6.1359 | overfitting |
| 5000 | 1.2982 | 7.7408 | 5.1929 | 6.1359 | final checkpoint much worse than best |

## Compare

```text
016 char block128 best nats/char: 4.0805
020 char block192 best nats/char: 4.0909
021 BPE4500 block128 best:       4.1059
023 BPE6000 block128 best:       4.1391
024 BPE6000 block192 best:       4.1163
```

Lower is better. Context 192 helps BPE6000 substantially:

```text
023 BPE6000 block128: 4.1391
024 BPE6000 block192: 4.1163
delta:               -0.0229
```

But the combined run still does not beat the best char-tokenizer runs:

```text
016 char block128: 4.0805
020 char block192: 4.0909
```

## Sample Notes

Generated from `best_ckpt.pt`, temperature `0.7`, top-k `50`, top-p `0.9`.

Prompt:

```text
黛玉聽了，
```

The sample has stronger scene continuity than BPE6000 block128: Dai-yu hears something, reacts,
Bao-yu appears, Xi-ren joins, and the text sustains a longer interaction. Repetition remains around
phrases such as "林黛玉", "怎麼", "走", and "罷".

Prompt:

```text
寶玉笑道：「
```

The sample has paragraph breaks and a more coherent Bao-yu / Xi-ren / Dai-yu movement pattern. It
still loops around motion verbs such as "走" and repeated dialogue formulas.

Prompt:

```text
賈母笑道：「
```

The sample keeps a plausible Jia-mu / Feng-jie / Liu-laolao social scene. Larger BPE pieces and
longer context help the scene feel broader, but repeated chunks such as "大年紀", "劉姥姥道", and
"這樣說" remain visible.

## Conclusion

BPE6000 and context192 are complementary:

```text
023 BPE6000 block128: 4.1391
024 BPE6000 block192: 4.1163
```

So the user's hypothesis was right directionally: longer context helps the larger BPE tokenizer.
However, this combination still trails the char-tokenizer baselines:

```text
016 char block128: 4.0805
020 char block192: 4.0909
024 BPE6000 block192: 4.1163
```

The current best checkpoint remains 016 by validation loss. But for manual generation, 024 may feel
competitive because BPE pieces create more phrase-level and scene-level texture. This is exactly why
loss and human sampling can disagree at small scales.

## Test Scripts

Train 024:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_bpe6000_width256_rope_context192.yaml
```

Generate 024:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.generate --checkpoint runs\hongloumeng_80_bpe6000_width256_rope_context192\best_ckpt.pt --prompt "黛玉聽了，" --max-new-tokens 300 --temperature 0.7 --top-k 50 --top-p 0.9
```
