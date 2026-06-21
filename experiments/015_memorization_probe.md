# 015 Memorization Probe

## Goal

Check whether the model is continuing the original training text or mostly generating in the same
style.

This experiment gives the model an exact prefix from `hongloumeng_80.txt`, asks it to generate the
following characters, and compares generation against the real continuation.

## Run

Deterministic-ish low-temperature probe:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe scripts\memorization_probe.py --checkpoint runs\hongloumeng_80_char_width256\best_ckpt.pt --offset 0 --prefix-len 120 --target-len 120 --temperature 0.2 --top-k 20 --top-p 0.8
```

Random offset:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe scripts\memorization_probe.py --checkpoint runs\hongloumeng_80_char_width256\best_ckpt.pt --random-offset --seed 42 --prefix-len 120 --target-len 120 --temperature 0.2 --top-k 20 --top-p 0.8
```

Greedy probe:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe scripts\memorization_probe.py --checkpoint runs\hongloumeng_80_char_width256\best_ckpt.pt --offset 0 --prefix-len 120 --target-len 120 --greedy
```

## Output

The script prints:

```text
expected
generated
diff
```

`^` marks a character mismatch at that position.

## Results

| Probe | Offset | Prefix Len | Target Len | Temperature | Top-k | Top-p | Char Accuracy | First Mismatch | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| start of corpus | 0 | 120 | 120 | 0.2 | 20 | 0.8 | 0.025 | 0 | sampled |
| random seed 42 | 116739 | 120 | 120 | 0.2 | 20 | 0.8 | 0.050 | 0 | sampled |
| start of corpus | 0 | 120 | 120 | greedy | - | - | 0.000 | 0 | max-prob token each step |
| random seed 42 | 116739 | 120 | 120 | greedy | - | - | 0.033 | 0 | max-prob token each step |

## Interpretation

High character accuracy would suggest local memorization. Low character accuracy with plausible
style suggests the model learned distributional style more than exact continuation.

## First Observations

The model did not reliably continue exact source text.

At corpus start, with low-temperature sampling, the expected continuation begins:

```text
塊未用，便棄在此山青埂峰下。誰知此石自經煅煉之後...
```

The generated continuation began:

```text
個，一個月初一個月，一個月，一個月...
```

With greedy decoding, the generated continuation began:

```text
個，也有一個，也有一個不成？」雨村道：「這是。」
```

For a random offset around ceremonial gifts, the expected continuation begins:

```text
表禮二十四端，清錢一百串，是賜與賈母...
```

The generated continuation began:

```text
面設著大小廝，一面又有一個小丫鬟...
```

Greedy decoding also failed to recover the original and entered a high-frequency loop:

```text
面又有一個小丫鬟，一個人皆是賈母的，也有一個...
```

Conclusion:

```text
The model has learned Hongloumeng-like local distribution and style, but it is not reliably
memorizing exact 120-character continuations from 120-character prefixes.
```

This supports the earlier qualitative impression: the model can sound plausible while still
being far from exact source-text recall.
