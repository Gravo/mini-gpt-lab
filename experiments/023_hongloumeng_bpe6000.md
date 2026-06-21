# 023 Hongloumeng BPE 6000

## Goal

Increase the BPE vocabulary from 4500 to 6000 and compare against the best character baseline.

This experiment changes:

```text
BPE vocab_size: 4500 -> 6000
```

Everything else stays close to experiment 016 and 021.

## Controlled Setup

Same model as 016:

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

Tokenizer:

```text
type = bpe
vocab_size = 6000
cache_path = data/tokenizers/hongloumeng_80_bpe6000.json
encoded_cache_path = data/tokenizers/hongloumeng_80_bpe6000_ids.pt
```

## Why 6000

Experiment 021 proved that BPE learns useful Hongloumeng pieces, but 4500 only adds 191 merges over
the character vocabulary. A 6000 vocab gives the tokenizer room to learn about 1691 merge pieces
while staying much smaller than an 8000/12000 tokenizer.

## Parameter Estimate

Compared with 016:

```text
016 char vocab = 4,309
023 BPE vocab  = 6,000
extra vocab    = 1,691
n_embd         = 256
extra params   = 1,691 * 256 = 432,896
```

Expected total:

```text
016 char tokenizer: 5,842,176
023 BPE 6000:       6,275,072
```

## Loss Comparison Note

Token-level validation loss is not directly comparable with char-tokenizer loss. Compare:

```text
val_nats_per_char
generation samples
```

## Train

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_bpe6000_width256_rope.yaml
```

## Generate

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.generate --checkpoint runs\hongloumeng_80_bpe6000_width256_rope\best_ckpt.pt --prompt "黛玉聽了，" --max-new-tokens 300 --temperature 0.7 --top-k 50 --top-p 0.9
```

## Tokenizer Stats

```text
chars = 590,432
BPE vocab = 6,000
BPE tokens = 396,089
chars per BPE token = 1.4907
tokens per char = 0.6708
tokenizer build time = 441.02s
```

Sample newly learned BPE pieces:

```text
寶玉
笑道：“
鳳姐
賈母
黛玉
夫人
襲人
我們
姑娘
寶釵
如今
他們
```

## Results

| Step | Train Loss | Val Loss | Val Nats/Char | Best Val | Notes |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 8.7522 | 8.7486 | 5.8689 | 8.7486 | initial |
| 250 | 7.1839 | 7.3333 | 4.9195 | 7.3333 | improving |
| 500 | 6.4084 | 6.7645 | 4.5379 | 6.7645 | improving |
| 750 | 5.9363 | 6.4925 | 4.3555 | 6.4925 | improving |
| 1000 | 5.5548 | 6.2816 | 4.2140 | 6.2816 | improving |
| 1250 | 5.1953 | 6.2113 | 4.1668 | 6.2113 | improving |
| 1500 | 4.9617 | 6.1902 | 4.1527 | 6.1902 | improving |
| 1750 | 4.6766 | 6.1700 | 4.1391 | 6.1700 | best checkpoint |
| 2000 | 4.3921 | 6.1907 | 4.1530 | 6.1700 | no new best |
| 2250 | 4.1503 | 6.2320 | 4.1807 | 6.1700 | overfitting starts |
| 2500 | 3.8809 | 6.2733 | 4.2084 | 6.1700 | overfitting |
| 2750 | 3.6481 | 6.3593 | 4.2661 | 6.1700 | overfitting |
| 3000 | 3.4224 | 6.4049 | 4.2967 | 6.1700 | overfitting |
| 3250 | 3.2017 | 6.5309 | 4.3813 | 6.1700 | overfitting |
| 3500 | 3.0151 | 6.6518 | 4.4623 | 6.1700 | overfitting |
| 3750 | 2.7670 | 6.7356 | 4.5186 | 6.1700 | overfitting |
| 4000 | 2.6093 | 6.8628 | 4.6039 | 6.1700 | overfitting |
| 4250 | 2.4553 | 6.9307 | 4.6495 | 6.1700 | overfitting |
| 4500 | 2.2533 | 7.0578 | 4.7347 | 6.1700 | overfitting |
| 4750 | 2.1128 | 7.1272 | 4.7812 | 6.1700 | overfitting |
| 5000 | 1.9390 | 7.3143 | 4.9068 | 6.1700 | final checkpoint much worse than best |

## Compare Against 016 and 021

```text
016 char tokenizer best nats/char: 4.0805
021 BPE 4500 best nats/char:      4.1059
023 BPE 6000 best nats/char:      4.1391
```

Lower is better, so BPE 6000 did not improve on BPE 4500. The larger vocabulary compresses the text
more aggressively but makes token prediction sparser for this small model and dataset.

Checkpoint size comparison:

```text
016 best_ckpt.pt: 23,452,563 bytes
021 best_ckpt.pt: 23,653,651 bytes
023 best_ckpt.pt: 25,233,107 bytes
```

## Sample Notes

Generated from `best_ckpt.pt`, temperature `0.7`, top-k `50`, top-p `0.9`.

Prompt:

```text
黛玉聽了，
```

The sample has more multi-character phrase flow and longer conversational runs, but repetition is
heavy around phrases such as "什麼", "瞧瞧", and "才剛剛".

Prompt:

```text
寶玉笑道：「
```

The sample has paragraph transitions and more named characters, including Zhou Rui's wife, Feng-jie,
Liu-laolao, and Ying-er. It also drifts through loosely connected scenes.

Prompt:

```text
賈母笑道：「
```

The sample produces many social dialogue turns, but repeats chunks like "正是呢", "罷", and
"頑笑". Larger BPE pieces seem to make phrase-level repetition more visible.

## Conclusion

BPE 6000 is worse than BPE 4500 in this setup:

```text
021 BPE 4500: 4.1059 nats/char
023 BPE 6000: 4.1391 nats/char
```

The compression is much stronger:

```text
021 tokens per char: 0.8169
023 tokens per char: 0.6708
```

But the larger vocabulary creates more sparse token targets. For a 6M-parameter model trained on
590k characters, this appears to hurt more than the extra compression helps.

Current tokenizer conclusion:

```text
char tokenizer remains best
BPE 4500 is promising
BPE 6000 is too sparse for this setup
```

## Test Scripts

Train 023:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_bpe6000_width256_rope.yaml
```

Generate 023:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.generate --checkpoint runs\hongloumeng_80_bpe6000_width256_rope\best_ckpt.pt --prompt "黛玉聽了，" --max-new-tokens 300 --temperature 0.7 --top-k 50 --top-p 0.9
```
