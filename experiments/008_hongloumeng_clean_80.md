# 008 Hongloumeng Clean 80 Chapters

## Goal

Remove Wikisource navigation artifacts and expand the corpus from 40 chapters to 80 chapters.

The previous context-256 experiment exposed source noise such as:

```text
回目录
上一回
下一回
```

This experiment fixes the data pipeline before changing model width or head count.

## Data

```powershell
python scripts/download_hongloumeng.py --chapters 80 --out data/hongloumeng_80.txt
```

The cleaner removes:

- `<ref>...</ref>` annotations
- templates
- chapter navigation link lines
- residual navigation-only lines such as `回目录`, `上一回`, and `下一回`

Generated dataset stats:

```text
characters = 590432
unique characters = 4309
回目录 count = 0
上一回 count = 0
下一回 count = 1
```

The remaining `下一回` is real prose usage (`又下一回棋`), not navigation text.

## Train

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_char.yaml
```

## Generate

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.generate --checkpoint runs\hongloumeng_80_char_baseline\ckpt.pt --prompt "黛玉聽了，" --max-new-tokens 300 --temperature 0.6
```

## Results

| Step | Train Loss | Val Loss | Notes |
| ---: | ---: | ---: | --- |
| 0 | 8.3810 | 8.3848 | random initialization; larger vocab than 40-chapter run |
| 300 | 5.0102 | 5.1397 | strong early learning |
| 600 | 4.6068 | 4.7702 | dialogue and punctuation patterns emerging |
| 900 | 4.3710 | 4.6586 | validation continues improving |
| 1200 | 4.2439 | 4.5277 | stable progress |
| 1500 | 4.1062 | 4.5071 | slower improvement |
| 1800 | 3.9674 | 4.4296 | better local style |
| 2100 | 3.9432 | 4.4390 | slight wobble |
| 2400 | 3.8295 | 4.3430 | good validation point |
| 2700 | 3.8140 | 4.3389 | similar to 2400 |
| 3000 | 3.7392 | 4.2905 | best validation in this run |

## What To Compare

- Does validation loss improve against the 40-chapter baseline?
- Do generated samples stop emitting `回目录`, `上一回`, and `下一回`?
- Does prompt conditioning improve with more character and scene coverage?
- Is `block_size=128` still enough after cleaning, or should context be retested later?

## Sample Notes

Prompt `寶玉笑道：「`, temperature `0.6`:

```text
寶玉笑道：「你見你妹妹妹妹妹，我也不知道了，我還有聽見，自然不知道我心，和我是別人家。」
```

Prompt `黛玉聽了，`, temperature `0.6`:

```text
黛玉聽了，不敢進去，便說道：“這會子可是什麼胭脂花樣！”寶玉道：“他是什麼樣，不知道的，這樣，他就沒見了。”
```

Prompt `賈母笑道：「`, temperature `0.6`:

```text
賈母笑道：「天天天天也不好了。」賈瑞聽了，忙往前去了。鴛鴦道：「我這會子又是那里去的
```

The generated samples no longer show `回目录`, `上一回`, or `下一回` navigation artifacts.
The larger corpus also brings in a broader character set, including 麝月, 鶯兒, 晴雯,
鴛鴦, and 劉姥姥. The model still loops around generic phrases like `不知道` and `你們`,
so data scale helped coverage more than semantic control.

## Interim Conclusion

Cleaning was necessary and successful. Scaling from 40 to 80 chapters did not produce a large
validation-loss jump:

```text
006 clean-ish 40 chapter context 128 best validation: 4.2678
008 cleaned 80 chapter context 128 best validation: 4.2905
```

These numbers are not perfectly comparable because the validation split and vocabulary changed.
Qualitatively, the 80-chapter model has broader character coverage and no navigation artifacts,
but the same small architecture still struggles with repeated generic dialogue.

Next likely experiments:

- retest `block_size=256` on cleaned 80-chapter data
- or increase model width/depth after keeping the cleaned data fixed
