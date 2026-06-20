# 005 Hongloumeng Character Baseline

## Goal

Replace Tiny Shakespeare with a Chinese text that is easier to inspect qualitatively.

The first run uses a character-level tokenizer on the first 10 chapters of public-domain
`紅樓夢` text from Wikisource.

## Data

```powershell
python scripts/download_hongloumeng.py --chapters 10 --out data/hongloumeng_10.txt
```

Source:

```text
https://zh.wikisource.org/wiki/紅樓夢
```

Use the original public-domain text, not a modern annotated, translated, or adapted edition.

## Train

```powershell
python -m gptlab.train --config configs/hongloumeng_char.yaml
```

If the default Python does not have torch, use the known CUDA environment:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_char.yaml
```

## Generate

```powershell
python -m gptlab.generate --checkpoint runs/hongloumeng_char_baseline/ckpt.pt --prompt "寶玉" --max-new-tokens 300
```

## Results

| Step | Train Loss | Val Loss | Notes |
| ---: | ---: | ---: | --- |
| 0 | 7.9468 | 7.9466 | random initialization |
| 100 | 5.9991 | 5.9773 | learns frequent characters and punctuation |
| 200 | 5.4007 | 5.5510 | clear early improvement |
| 300 | 5.0135 | 5.3456 | validation still improves |
| 400 | 4.7783 | 5.2142 | local text patterns emerging |
| 500 | 4.5604 | 5.1630 | validation near its best region |
| 600 | 4.3939 | 5.1148 | best validation in this run |
| 700 | 4.2478 | 5.1987 | train keeps falling, val starts wobbling |
| 800 | 4.0882 | 5.1495 | small-data overfit signs |
| 900 | 3.9493 | 5.1242 | no clear validation gain |
| 1000 | 3.7955 | 5.1972 | train improves, val worsens |

## Sample Notes

Prompts to try:

```text
寶玉
黛玉
賈母
只見
說道
```

The Wikisource text is traditional Chinese. A simplified prompt such as `宝玉` or `只见`
is out-of-vocabulary for this character tokenizer and raises `KeyError`.

Look for whether the model starts to learn:

- common character names
- punctuation and dialogue rhythm
- chapter-style narrative phrases
- short local character patterns

Do not overclaim semantic understanding. For this stage, the point is to connect next-character
prediction, loss curves, and visible sample quality.

Sample after 1000 steps, prompt `寶玉`:

```text
寶玉看見寶玉只聽，寶玉未必說：「你的好，倒不知道了，也不得不想著這杯茶來要是個茗煙若不覺呢。」
```

Sample after 1000 steps, prompt `只見`:

```text
只見他吃飯。寶玉笑道：「我不過一個侄兒，我們什麼巧的，不替他們什麼？」
```

These samples have names, dialogue marks, and local phrase rhythm, but not stable long-range
meaning. That is the expected first baseline behavior.

## What This Should Prove

- The training loop works on Chinese text.
- Character-level tokenization is enough for a first Chinese baseline.
- Loss reduction should correspond to more locally plausible Chinese samples.
- The model is learning text distribution patterns, not literary meaning.
