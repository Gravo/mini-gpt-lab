# 031 Hongloumeng 120 Chapters Layers 9 Context 192 Dropout 0.20

## Goal

Scale the current best architecture from the first 80 chapters to all 120 chapters of Hongloumeng.

This changes only the dataset:

```text
data/hongloumeng_80.txt -> data/hongloumeng_120.txt
```

Architecture stays the same as 030:

```text
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
max_steps = 2500
```

## Dataset Stats

```text
80 chapters chars = 590,432
80 chapters vocab = 4,309

120 chapters chars = 867,755
120 chapters vocab = 4,511
```

This adds about 47% more characters.

## Parameter Estimate

The vocabulary grows by 202 characters:

```text
4511 - 4309 = 202
extra params = 202 * 256 = 51,712
```

Estimated total:

```text
030 80-chapter params ~= 8,211,456
031 120-chapter params ~= 8,263,168
```

## Why This Experiment

Recent architecture changes have produced very small gains:

```text
028 dropout0.15: 4.0675
030 dropout0.20: 4.0658
```

The model also still repeats local phrases during generation. More text may help more than further
small architecture tweaks.

## Results

| Step | Train Loss | Val Loss | Val Nats/Char | Best Val | Notes |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 8.4267 | 8.4174 | 8.4174 | 8.4174 | init |
| 250 | 4.7940 | 4.6756 | 4.6756 | 4.6756 | fast fit |
| 500 | 4.2836 | 4.2240 | 4.2240 | 4.2240 | improving |
| 750 | 4.0640 | 4.0802 | 4.0802 | 4.0802 | improving |
| 1000 | 3.8475 | 3.9600 | 3.9600 | 3.9600 | improving |
| 1250 | 3.7390 | 3.8443 | 3.8443 | 3.8443 | improving |
| 1500 | 3.5950 | 3.7669 | 3.7669 | 3.7669 | improving |
| 1750 | 3.4562 | 3.7158 | 3.7158 | 3.7158 | improving |
| 2000 | 3.3960 | 3.7289 | 3.7289 | 3.7158 | small rebound |
| 2250 | 3.3079 | 3.6863 | 3.6863 | 3.6863 | new best |
| 2500 | 3.2014 | 3.6597 | 3.6597 | 3.6597 | new best at final step |

Best checkpoint:

```text
runs/hongloumeng_120_char_width256_rope_layers9_context192_dropout020_textsplit/best_ckpt.pt
best val ~= 3.6597 at step 2500
checkpoint size ~= 33.15 MB
```

## Compare

```text
030 80 chapters best val: 4.0658
031 120 chapters best val: 3.6597
```

Important caveat:

```text
The validation set is now the last 10% of all 120 chapters, so the raw validation text differs from
the 80-chapter experiments. This measures the 120-chapter training setup, not a perfectly identical
held-out slice.
```

The strongest signal is not the raw number alone. The 031 run is still improving at the final
evaluation step, while several 80-chapter runs had already flattened or started to overfit. More
Hongloumeng text gives this 8.26M-parameter model more useful training room.

## Sample Notes

Prompt:

```text
黛玉聽了，
```

The sample keeps the Hongloumeng dialogue surface style, but still loops around local phrases such
as "這是什麼" and repeated speaker turns. It is better viewed as style modeling than plot-level
continuation.

Prompt:

```text
寶玉笑道：「
```

The sample shifts toward later-book social-scene names and dialogue patterns, including 劉姥姥,
鳳姐, 鴛鴦, and 賈母. This is a useful sign that chapters 81-120 are influencing the distribution.
However, coherence is still local: the model often produces plausible sentence fragments without a
stable scene state.

## Decision

031 is worth extending before mixing another novel. The clean next step is:

```text
032 = same 120-chapter setup, longer training, for example 4000-5000 steps
```

Only after checking whether the 120-chapter run keeps improving should we add Rulin Waishi as a
mixed-corpus experiment. Otherwise we would not know whether any gain came from more Hongloumeng
training, more total text, or cross-book style regularization.

## Future Mixed-Corpus Plan

After this clean 120-chapter run, test a separate mixed ancient vernacular corpus:

```text
032 = Hongloumeng 120 + Rulin Waishi
```

Do not mix corpora into 031, otherwise it is unclear whether any change comes from extra Hongloumeng
chapters or cross-book style transfer.

## Command

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_120_char_width256_rope_layers9_context192_dropout020_textsplit.yaml
```
