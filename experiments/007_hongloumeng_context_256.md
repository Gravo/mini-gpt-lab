# 007 Hongloumeng Context 256

## Goal

Test whether a longer context window improves dialogue continuity and reduces short-loop phrasing.

This experiment keeps the 40-chapter dataset and the same small GPT width/depth as experiment 006,
but changes `block_size` from `128` to `256`.

## Controlled Setup

Same as 006:

```text
chapters = 40
n_layer = 4
n_head = 4
n_embd = 128
norm = layernorm
mlp = gelu
position = learned
max_steps = 2000
```

Changed:

```text
block_size: 128 -> 256
```

This means each training example contains 256 characters instead of 128. Attention score matrices
inside each head change from `[128, 128]` to `[256, 256]`, so attention work is roughly 4x larger
for the same batch size.

## Train

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_40_char_ctx256.yaml
```

## Generate

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.generate --checkpoint runs\hongloumeng_40_char_ctx256\ckpt.pt --prompt "黛玉聽了，" --max-new-tokens 300 --temperature 0.6
```

## Results

| Step | Train Loss | Val Loss | Notes |
| ---: | ---: | ---: | --- |
| 0 | 8.2524 | 8.2555 | random initialization |
| 200 | 5.2758 | 5.2154 | strong early learning |
| 400 | 4.7367 | 4.7854 | similar to 006 at this stage |
| 600 | 4.5030 | 4.5961 | validation improves steadily |
| 800 | 4.3011 | 4.5399 | close to 006 |
| 1000 | 4.1353 | 4.4566 | no clear context win yet |
| 1200 | 3.9898 | 4.3640 | close to 006 |
| 1400 | 3.9123 | 4.3436 | validation still improves |
| 1600 | 3.7908 | 4.2790 | best validation in this run |
| 1800 | 3.7030 | 4.3445 | wobble upward |
| 2000 | 3.5671 | 4.3338 | train improves, val does not beat best |

## What To Compare Against 006

- Does validation loss improve with the same data and model width?
- Does generated dialogue keep local context longer?
- Does repetition around `不知道`, `不是`, and `你們` decrease?
- Is the runtime increase worth the quality change?

## Sample Notes

Prompt `寶玉笑道：「`, temperature `0.6`:

```text
寶玉笑道：「你可不住。」寶玉道：「既有些什麼，原來是你們那裏來的。」
```

This sample later drifted into source navigation text:

```text
回　回目录　回
上一回　回　回　　回
上一回目录　下一回
```

Prompt `黛玉聽了，`, temperature `0.6`:

```text
黛玉聽了，黛玉不覺落了一聲。自己倒茶，一時黛玉一個人一聲，又問：“好了？”
```

Prompt `賈母笑道：「`, temperature `0.6`:

```text
賈母笑道：「你是這麼樣的！」說著，誰知是他們的事不是，只顧後悔的，只見你又不知道說
```

## Interim Conclusion

The context-length change did not clearly improve validation loss:

```text
006 context 128 best validation: 4.2678
007 context 256 best validation: 4.2790
```

The longer context is not automatically better at this scale. It also made source-navigation
artifacts more visible in generation, such as `回目录`, `上一回`, and `下一回`.

Before increasing width or context further, the next useful step is to improve data cleaning:
remove chapter navigation lines and then rerun the 40-chapter context-128 baseline or compare
cleaned context 128 vs 256.
