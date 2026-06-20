# 010 Hongloumeng Clean 80 Wider Model

## Goal

Test whether more model capacity helps more than longer context.

This experiment keeps the cleaned 80-chapter corpus and returns to `block_size=128`, but increases
the Transformer depth and width.

## Controlled Setup

Same data and training length as 008:

```text
chapters = 80
block_size = 128
max_steps = 3000
norm = layernorm
mlp = gelu
position = learned
```

Changed:

```text
n_layer: 4 -> 6
n_embd: 128 -> 192
n_head: 4 -> 6
head_dim remains 32
```

Keeping `head_dim=32` means the wider model gets more heads because the hidden width is larger,
not because each head becomes narrower.

## Train

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_char_wide.yaml
```

## Results

| Step | Train Loss | Val Loss | Notes |
| ---: | ---: | ---: | --- |
| 0 | 8.3783 | 8.3813 | random initialization |
| 300 | 4.7322 | 4.9095 | faster early learning than small model |
| 600 | 4.3868 | 4.7040 | clear improvement |
| 900 | 4.1651 | 4.5023 | below small model at same step |
| 1200 | 3.9897 | 4.4200 | stable progress |
| 1500 | 3.8421 | 4.3147 | strong validation point |
| 1800 | 3.7363 | 4.3315 | slight wobble |
| 2100 | 3.5937 | 4.2867 | close to small-model best |
| 2400 | 3.5031 | 4.2187 | better than small model |
| 2700 | 3.3902 | 4.1926 | best validation in this run |
| 3000 | 3.3268 | 4.2775 | validation worsens, possible overfit/noise |

## Compare Against 008

```text
008: 4 layers, 4 heads, 128 hidden
010: 6 layers, 6 heads, 192 hidden
```

Key question: does a larger model reduce repetitive generic dialogue and improve validation loss
without overfitting too quickly?

## Sample Notes

Prompt `寶玉笑道：「`, temperature `0.6`:

```text
寶玉笑道：「噯喲喲喲喲，這是那里的。」寶玉笑道：「我不知道，你倒也不用這些東西。」
```

Prompt `黛玉聽了，`, temperature `0.6`:

```text
黛玉聽了，忙問：“怎麼又來了？”紫鵑笑道：“你仔細的那里跑？”晴雯忙問
```

Prompt `賈母笑道：「`, temperature `0.6`:

```text
賈母笑道：「你們這裏坐著，我的坐坐一坐，睡覺好些。」賈母笑道：「我們就走了
```

The wider model has broader scene and character associations: prompts can bring in 紫鵑,
晴雯, 襲人, 寶釵, 芳官, 秋紋, and 鴛鴦 more naturally. It still repeats words and generic
dialogue, so capacity helped but did not solve decoding quality by itself.

## Interim Conclusion

Increasing model capacity helped more than increasing context:

```text
008 small context 128 best validation: 4.2905
009 small context 256 best validation: 4.2838
010 wider context 128 best validation: 4.1926
```

The validation loss worsened at step 3000, so the best checkpoint would have been around step
2700. The next training-loop improvement should save best validation checkpoints instead of only
the final checkpoint.
