# 006 Hongloumeng Data Scale

## Goal

Test whether the same small GPT becomes more controllable when it sees more text.

This experiment keeps the model shape close to `005_hongloumeng_baseline` and changes the
dataset from the first 10 chapters to the first 40 chapters.

## Controlled Setup

Same as the 10-chapter baseline:

```text
n_layer = 4
n_head = 4
n_embd = 128
block_size = 128
norm = layernorm
mlp = gelu
position = learned
```

Changed:

```text
chapters: 10 -> 40
max_steps: 1000 -> 2000
eval_interval: 100 -> 200
```

## Data

```powershell
python scripts/download_hongloumeng.py --chapters 40 --out data/hongloumeng_40.txt
```

## Train

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_40_char.yaml
```

## Generate

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.generate --checkpoint runs\hongloumeng_40_char_baseline\ckpt.pt --prompt "寶玉笑道：「" --max-new-tokens 300 --temperature 0.6
```

## Results

| Step | Train Loss | Val Loss | Notes |
| ---: | ---: | ---: | --- |
| 0 | 8.2608 | 8.2610 | random initialization; larger vocab than 10-chapter run |
| 200 | 5.3490 | 5.3421 | strong early learning |
| 400 | 4.7935 | 4.8201 | already below the 10-chapter final validation region |
| 600 | 4.5620 | 4.6578 | dialogue rhythm improving |
| 800 | 4.4078 | 4.5759 | steady validation improvement |
| 1000 | 4.2252 | 4.4730 | better than 005 despite larger corpus |
| 1200 | 4.1459 | 4.4448 | slower but continued improvement |
| 1400 | 4.0571 | 4.3606 | good validation point |
| 1600 | 3.8970 | 4.3872 | slight wobble |
| 1800 | 3.8549 | 4.3814 | stable region |
| 2000 | 3.7776 | 4.2678 | best validation in this run |

## What To Compare Against 005

- Does validation loss improve compared with the 10-chapter run?
- Do prompts like `寶玉笑道：「`, `黛玉聽了，`, and `賈母笑道：「` diverge more clearly?
- Does the model repeat less?
- Does it keep dialogue rhythm longer?

The target is not semantic understanding yet. The target is stronger local distribution learning
and visibly better prompt conditioning.

## Sample Notes

Prompt `寶玉笑道：「`, temperature `0.6`:

```text
寶玉笑道：「你不知道我今不知道，你怎麼說要緊？」黛玉道：「我原來的話，是你告訴你，我就是你。」
```

Prompt `黛玉聽了，`, temperature `0.6`:

```text
黛玉聽了，便笑道：“你說什麼？”寶玉道：“這麼不是呢？”林黛玉道：“你睡覺了。”
```

Prompt `賈母笑道：「`, temperature `0.6`:

```text
賈母笑道：「好妹妹妹，我們也沒有了。」黛玉笑道：「我是個字，在那裏，只是個字也罷。」
```

Compared with the 10-chapter run, the model keeps dialogue form longer and uses more relevant
character names. It still loops through generic phrases such as `不知道`, `不是`, and `你們`,
so prompt conditioning is better but not yet strong.

## Interim Conclusion

Increasing data from 10 to 40 chapters was the right first move:

```text
005 best validation: about 5.11
006 best validation: 4.2678
```

The same architecture generalizes better with more text. The next controlled experiment should
change context length from `128` to `256` while keeping the 40-chapter dataset.
