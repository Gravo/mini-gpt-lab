# 012 Hongloumeng Width 256

## Goal

Test whether increasing model width improves the cleaned Hongloumeng character model.

The previous best model used:

```text
n_layer = 6
n_head = 6
n_embd = 192
head_dim = 32
best validation = 4.1862
```

This experiment keeps depth, context length, data, and dropout fixed, but widens the model:

```text
n_layer = 6
n_head = 8
n_embd = 256
head_dim = 32
```

## Parameter Estimate

With tied token embedding and lm head:

```text
vocab_size = 4309
block_size = 128
n_layer = 6
n_embd = 256
mlp_hidden = 4 * 256 = 1024
```

Expected trainable parameters:

```text
token embedding / tied lm_head = 4309 * 256 = 1,103,104
position embedding = 128 * 256 = 32,768
per block:
  layer norms = 2 * (256 + 256) = 1,024
  attention = (3*256*256 + 3*256) + (256*256 + 256) = 263,168
  mlp = (1024*256 + 1024) + (256*1024 + 256) = 525,568
  block total = 789,760
6 blocks = 4,738,560
final layernorm = 512
total = 5,874,944
```

Approximate fp32 checkpoint size:

```text
5,874,944 * 4 bytes = 23.5 MB
```

## Train

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_char_width256.yaml
```

## Generate

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.generate --checkpoint runs\hongloumeng_80_char_width256\best_ckpt.pt --prompt "黛玉聽了，" --max-new-tokens 300 --temperature 0.7 --top-k 50 --top-p 0.9
```

## Results

| Step | Train Loss | Val Loss | Best Val | Notes |
| ---: | ---: | ---: | ---: | --- |
| 0 | 8.4298 | 8.4320 | 8.4320 | random initialization |
| 250 | 4.8599 | 4.9873 | 4.9873 | strong early learning |
| 500 | 4.4190 | 4.7151 | 4.7151 | validation improves |
| 750 | 4.2003 | 4.5475 | 4.5475 | steady progress |
| 1000 | 4.0162 | 4.3894 | 4.3894 | below 4.4 |
| 1250 | 3.9041 | 4.3076 | 4.3076 | faster than 011 |
| 1500 | 3.7611 | 4.2536 | 4.2536 | good region |
| 1750 | 3.5980 | 4.2285 | 4.2285 | still improving |
| 2000 | 3.5243 | 4.1737 | 4.1737 | new best |
| 2250 | 3.4324 | 4.1655 | 4.1655 | best checkpoint |
| 2500 | 3.3242 | 4.2326 | 4.1655 | overfit/noise begins |
| 2750 | 3.2251 | 4.2035 | 4.1655 | not best |
| 3000 | 3.1529 | 4.1746 | 4.1655 | close, but not best |
| 3250 | 3.0878 | 4.2438 | 4.1655 | validation worsens |
| 3500 | 2.9950 | 4.2804 | 4.1655 | train keeps falling |
| 3750 | 2.9245 | 4.2462 | 4.1655 | no val gain |
| 4000 | 2.8333 | 4.2386 | 4.1655 | no val gain |
| 4250 | 2.7430 | 4.2761 | 4.1655 | overfit |
| 4500 | 2.7033 | 4.2847 | 4.1655 | overfit |
| 4750 | 2.6053 | 4.3579 | 4.1655 | overfit |
| 5000 | 2.5543 | 4.4153 | 4.1655 | final checkpoint much worse than best |

## What To Compare Against 011

```text
011: 6 layers, 192 hidden, 6 heads, 3.52M params, best val 4.1862
012: 6 layers, 256 hidden, 8 heads, about 5.87M params
```

Key question: does more width push validation closer to `4.0` and reduce generic repeated
dialogue?

## Sample Notes

Generated from `best_ckpt.pt`, temperature `0.7`, top-k `50`, top-p `0.9`.

Prompt `黛玉聽了，`:

```text
黛玉聽了，便起身去。寶玉忙道：“好，我說的好，快不是。”
```

This sample later entered a poetry/word-game style loop:

```text
黛玉笑道：“這是‘金玉’，‘杏花香’，‘金文’的字，‘天然’，‘葉’字不得字。”
```

Prompt `寶玉笑道：「`:

```text
寶玉笑道：「你這個小孩子，你就說了。」寶玉笑道：「這是你，你又不能夠了。」
```

Prompt `賈母笑道：「`:

```text
賈母笑道：「你們家去罷。」說著，便帶了劉姥姥來，一齊上來。
```

The wider model has lower validation loss, but it can overfit sharper local patterns. In these
samples, it sometimes loops through quoted-character and poetry-game fragments.

## Interim Conclusion

Widening helped, but the gain is modest and overfitting arrives earlier:

```text
011 width 192 params: 3,521,472
011 width 192 best validation: 4.1862 at step 3250

012 width 256 params: 5,874,944
012 width 256 best validation: 4.1655 at step 2250
```

The parameter count increased by about 67%, while best validation improved by about 0.021.
This suggests width is still useful, but the next improvements may need better regularization,
sampling, or architecture variants rather than only more parameters.
