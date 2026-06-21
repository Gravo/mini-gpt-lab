# 021 Hongloumeng BPE 4500

## Goal

Compare the current best character tokenizer baseline against a small BPE tokenizer.

This experiment changes the tokenization mechanism:

```text
tokenizer: char -> bpe
vocab_size: 4309 -> 4500
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
block_size = 128
dropout = 0.15
norm = layernorm
mlp = gelu
position = rope
max_steps = 5000
```

Changed:

```text
tokenizer = bpe
target vocab_size = 4500
```

## Why BPE Might Help

The character model sees:

```text
寶 玉 笑 道 ： 「 ...
```

BPE can learn repeated units:

```text
寶玉
黛玉
賈母
鳳姐
笑道：“
道：「
```

This is especially relevant for Hongloumeng because names, titles, dialogue markers, and short
social phrases are highly repetitive.

## Tokenizer Stats

Measured on the 80-chapter corpus:

```text
chars = 590,432
char vocab = 4,309
BPE vocab = 4,500
BPE tokens = 482,300
chars per BPE token = 1.2242
tokens per char = 0.8169
```

Sample newly learned BPE pieces:

```text
：“
道：“
。”
了，
寶玉
笑道：“
鳳姐
：「
什麼
賈母
黛玉
道：「
，你
```

## Parameter Estimate

Only the tied token embedding / lm_head grows:

```text
016 char vocab params: 4,309 * 256 = 1,103,104
021 BPE vocab params:  4,500 * 256 = 1,152,000
delta:                                +48,896
```

Total parameters:

```text
016 char tokenizer: 5,842,176
021 BPE tokenizer:  5,891,072
delta:                +48,896
```

## Loss Comparison Note

Token-level validation loss is not directly comparable across tokenizers because one BPE token can
cover multiple characters. For this experiment, compare:

```text
token-level val loss
val_nats_per_char
generation samples
```

The training script records `val_nats_per_char` as:

```text
token_loss * tokens_per_char
```

## Train

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_bpe4500_width256_rope.yaml
```

## Generate

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.generate --checkpoint runs\hongloumeng_80_bpe4500_width256_rope\best_ckpt.pt --prompt "黛玉聽了，" --max-new-tokens 300 --temperature 0.7 --top-k 50 --top-p 0.9
```

## Results

| Step | Train Loss | Val Loss | Val Nats/Char | Best Val | Notes |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 8.4650 | 8.4630 | 6.9131 | 8.4630 | initial |
| 250 | 5.9406 | 6.0999 | 4.9828 | 6.0999 | improving |
| 500 | 5.1839 | 5.5392 | 4.5247 | 5.5392 | improving |
| 750 | 4.8344 | 5.3275 | 4.3518 | 5.3275 | improving |
| 1000 | 4.5903 | 5.1739 | 4.2264 | 5.1739 | improving |
| 1250 | 4.3809 | 5.0823 | 4.1515 | 5.0823 | improving |
| 1500 | 4.1946 | 5.1072 | 4.1718 | 5.0823 | no new best |
| 1750 | 3.9925 | 5.0265 | 4.1059 | 5.0265 | best checkpoint |
| 2000 | 3.8506 | 5.0311 | 4.1097 | 5.0265 | no new best |
| 2250 | 3.6605 | 5.0739 | 4.1446 | 5.0265 | no new best |
| 2500 | 3.5021 | 5.0470 | 4.1227 | 5.0265 | close |
| 2750 | 3.3361 | 5.1435 | 4.2015 | 5.0265 | overfitting |
| 3000 | 3.1924 | 5.1604 | 4.2153 | 5.0265 | overfitting |
| 3250 | 3.0616 | 5.1958 | 4.2442 | 5.0265 | overfitting |
| 3500 | 2.8963 | 5.2009 | 4.2484 | 5.0265 | overfitting |
| 3750 | 2.7550 | 5.3328 | 4.3561 | 5.0265 | overfitting |
| 4000 | 2.6116 | 5.3729 | 4.3889 | 5.0265 | overfitting |
| 4250 | 2.5387 | 5.3826 | 4.3969 | 5.0265 | overfitting |
| 4500 | 2.3654 | 5.4845 | 4.4800 | 5.0265 | overfitting |
| 4750 | 2.2933 | 5.4551 | 4.4561 | 5.0265 | overfitting |
| 5000 | 2.2017 | 5.5795 | 4.5577 | 5.0265 | final checkpoint much worse than best |

## Compare Against 016

```text
016 char tokenizer best validation: 4.0805 token loss
016 char tokenizer best nats/char:  4.0805

021 BPE tokenizer best validation:  5.0265 token loss
021 BPE tokenizer best nats/char:   4.1059
delta nats/char:                   +0.0254
```

Lower is better, so BPE 4500 did not beat 016. It is close to the 019/020 range and better than
the SwiGLU run when compared by character-normalized loss.

Checkpoint size comparison:

```text
016 best_ckpt.pt: 23,452,563 bytes
021 best_ckpt.pt: 23,653,651 bytes
delta:             +201,088 bytes
```

## Sample Notes

Generated from `best_ckpt.pt`, temperature `0.7`, top-k `50`, top-p `0.9`.

Prompt:

```text
黛玉聽了，
```

The sample uses poetry/name-like fragments more readily than the char model:

```text
蕉紅
綠菊
蘅蕪苑
金釧兒
```

This is a plausible effect of BPE learning common multi-character pieces. The downside is that it
also enters loops around these learned pieces, especially repeated `紅玉` and name-like fragments.

Prompt:

```text
寶玉笑道：「
```

The sample has smoother quote turns and paragraph breaks, but still repeats local formulas like
"我不知道", "你說", and "頭上".

Prompt:

```text
賈母笑道：「
```

The sample keeps multiple speakers, including Jia-mu, Xue-yima, Tan-chun, Liu-laolao, and Yuan-yang.
It also repeats BPE-friendly chunks such as "這裏", "頭上", and "娘兒子".

## Conclusion

BPE 4500 is a promising but not yet winning tokenizer change:

```text
016 char best nats/char: 4.0805
020 context192:          4.0909
021 BPE4500:             4.1059
019 width384:            4.0998
018 SwiGLU:              4.1145
```

The important part is parameter efficiency:

```text
016 params: 5,842,176
021 params: 5,891,072
delta:       +48,896
```

For only 0.84% more parameters, BPE gets close to the best char baselines and changes generation in
a visibly meaningful way. It also exposes a new engineering need: tokenizer training should be
cached before trying larger BPE vocabularies such as 6000 or 8000.

Current best remains:

```text
016 char tokenizer, RoPE, width 256, block 128
```

Next tokenizer step:

```text
022 = cache trained BPE tokenizer, then test BPE vocab 6000 or 8000
```

## Test Scripts

Train 016:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_char_width256_rope.yaml
```

Train 021:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_bpe4500_width256_rope.yaml
```

Generate 016:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.generate --checkpoint runs\hongloumeng_80_char_width256_rope\best_ckpt.pt --prompt "黛玉聽了，" --max-new-tokens 300 --temperature 0.7 --top-k 50 --top-p 0.9
```

Generate 021:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.generate --checkpoint runs\hongloumeng_80_bpe4500_width256_rope\best_ckpt.pt --prompt "黛玉聽了，" --max-new-tokens 300 --temperature 0.7 --top-k 50 --top-p 0.9
```
