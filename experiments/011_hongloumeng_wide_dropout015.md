# 011 Hongloumeng Wide Dropout 0.15

## Goal

Train the current best architecture more carefully with best-checkpoint saving.

Experiment 010 showed that the wider model helped:

```text
6 layers
6 heads
192 hidden
head_dim = 32
```

But validation loss improved before the final step and then worsened. This experiment increases
dropout and trains longer while saving `best_ckpt.pt`.

## Controlled Setup

Same as 010:

```text
data = cleaned Hongloumeng first 80 chapters
block_size = 128
n_layer = 6
n_head = 6
n_embd = 192
norm = layernorm
mlp = gelu
position = learned
learning_rate = 0.0003
```

Changed:

```text
dropout: 0.10 -> 0.15
max_steps: 3000 -> 5000
eval_interval: 300 -> 250
```

## Train

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_char_wide_dropout015.yaml
```

## Generate

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.generate --checkpoint runs\hongloumeng_80_char_wide_dropout015\best_ckpt.pt --prompt "黛玉聽了，" --max-new-tokens 300 --temperature 0.7 --top-k 50 --top-p 0.9
```

## Results

| Step | Train Loss | Val Loss | Best Val | Notes |
| ---: | ---: | ---: | ---: | --- |
| 0 | 8.3783 | 8.3813 | 8.3813 | random initialization |
| 250 | 4.8624 | 5.0735 | 5.0735 | strong early learning |
| 500 | 4.4701 | 4.7933 | 4.7933 | validation improves |
| 750 | 4.2814 | 4.6492 | 4.6492 | steady progress |
| 1000 | 4.1138 | 4.4782 | 4.4782 | below 4.5 |
| 1250 | 3.9951 | 4.4475 | 4.4475 | slower improvement |
| 1500 | 3.8889 | 4.3499 | 4.3499 | good validation region |
| 1750 | 3.7854 | 4.3560 | 4.3499 | slight wobble |
| 2000 | 3.7409 | 4.2864 | 4.2864 | new best |
| 2250 | 3.6220 | 4.2903 | 4.2864 | stable |
| 2500 | 3.5567 | 4.2063 | 4.2063 | strong improvement |
| 2750 | 3.4717 | 4.2220 | 4.2063 | wobble upward |
| 3000 | 3.4132 | 4.2457 | 4.2063 | final would not be best |
| 3250 | 3.3707 | 4.1862 | 4.1862 | best checkpoint |
| 3500 | 3.3126 | 4.2188 | 4.1862 | overfit/noise begins |
| 3750 | 3.2349 | 4.2046 | 4.1862 | close but not best |
| 4000 | 3.2265 | 4.2115 | 4.1862 | train keeps falling |
| 4250 | 3.1344 | 4.2159 | 4.1862 | no val gain |
| 4500 | 3.0993 | 4.2145 | 4.1862 | no val gain |
| 4750 | 3.0236 | 4.2211 | 4.1862 | no val gain |
| 5000 | 3.0028 | 4.2767 | 4.1862 | final checkpoint is worse than best |

## What To Compare Against 010

```text
010 dropout 0.10 best validation: 4.1926
011 dropout 0.15 best validation: TBD
```

Key question: does higher dropout and longer training improve validation while preserving better
sample quality?

## Sample Notes

Generated from `best_ckpt.pt`, temperature `0.7`, top-k `50`, top-p `0.9`.

Prompt `黛玉聽了，`:

```text
黛玉聽了，忙問：“怎麼又不得？”寶釵道：“這個話說的，我說你說話，你倒不是不是，我還說的話。”
```

Prompt `寶玉笑道：「`:

```text
寶玉笑道：「那里有個小的。」黛玉笑道：「這都是這些人家，你只管吃這些，我也有什麼不成？」
```

Prompt `賈母笑道：「`:

```text
賈母笑道：「你們都看著他們的那里去了，他們還有人呢。」鳳姐道：「這裏的姑娘，他們還沒有這樣兒。」
```

The model still repeats generic dialogue, but the wider model plus filtered sampling gives broader
scene and character associations than the early baselines.

## Interim Conclusion

Dropout 0.15 plus longer training gave a small validation improvement:

```text
010 dropout 0.10 best validation: 4.1926
011 dropout 0.15 best validation: 4.1862
```

The bigger win is operational: `best_ckpt.pt` captured step 3250, while the final checkpoint at
step 5000 had worse validation. Future longer runs should always generate from `best_ckpt.pt`.
