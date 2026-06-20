# 009 Hongloumeng Clean 80 Context 256

## Goal

Retest longer context after cleaning Wikisource navigation artifacts.

Experiment 007 tried `block_size=256` on the 40-chapter corpus and exposed navigation noise.
Experiment 008 cleaned the corpus and expanded it to 80 chapters. This experiment asks whether
`block_size=256` helps once the data is cleaner.

## Controlled Setup

Same as 008:

```text
chapters = 80
n_layer = 4
n_head = 4
n_embd = 128
norm = layernorm
mlp = gelu
position = learned
max_steps = 3000
```

Changed:

```text
block_size: 128 -> 256
```

## Train

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_char_ctx256.yaml
```

## Results

| Step | Train Loss | Val Loss | Notes |
| ---: | ---: | ---: | --- |
| 0 | 8.3993 | 8.3997 | random initialization |
| 300 | 4.8433 | 5.0598 | strong early learning |
| 600 | 4.4891 | 4.7731 | similar to 008 |
| 900 | 4.2879 | 4.6280 | validation improves steadily |
| 1200 | 4.1591 | 4.5128 | close to 008 |
| 1500 | 4.0198 | 4.4049 | slightly better than 008 at similar step |
| 1800 | 3.9618 | 4.3650 | good validation region |
| 2100 | 3.8225 | 4.3194 | continued improvement |
| 2400 | 3.7250 | 4.2838 | best validation in this run |
| 2700 | 3.6639 | 4.3429 | wobble upward |
| 3000 | 3.5858 | 4.2902 | near 008 final validation |

## Compare Against 008

```text
008: block_size=128, n_layer=4, n_embd=128
009: block_size=256, n_layer=4, n_embd=128
```

Key question: does longer context improve validation loss or sample continuity on the cleaned
80-chapter corpus?

## Sample Notes

Prompt `寶玉笑道：「`, temperature `0.6`:

```text
寶玉笑道：「你這裡是那裏的人，你還沒有呢？」寶玉聽了，便笑道：「可是這樣的，又是個呆子的
```

Prompt `黛玉聽了，`, temperature `0.6`:

```text
黛玉聽了，忙回說道：“我只管吃茶。”寶玉忙上來。寶玉道：“你們不吃飯，我們姑娘們來。”
```

Prompt `賈母笑道：「`, temperature `0.6`:

```text
賈母笑道：「我們大奶奶奶，你們那裏吃飯，你進去了。」賈母笑道：「你們去罷
```

## Interim Conclusion

Longer context still did not clearly beat the context-128 baseline:

```text
008 context 128 best validation: 4.2905
009 context 256 best validation: 4.2838
```

The difference is small. Samples still repeat patterns such as `姐姐姐姐`, `老太太太`,
and generic dialogue around `不知道`. At this scale, context length is not the main bottleneck.
