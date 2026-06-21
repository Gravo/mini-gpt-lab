# 016 Hongloumeng RoPE

## Goal

Compare learned absolute position embeddings against RoPE on the current best width-256 setup.

This experiment changes one architectural feature:

```text
position: learned -> rope
```

Everything else stays close to experiment 012.

## Controlled Setup

Same as 012:

```text
data = cleaned Hongloumeng first 80 chapters
n_layer = 6
n_head = 8
n_embd = 256
head_dim = 32
block_size = 128
dropout = 0.15
norm = layernorm
mlp = gelu
max_steps = 5000
```

Changed:

```text
position = rope
```

## Why RoPE Might Help

Learned position embeddings add a learned vector for each absolute position:

```text
x = token_embedding + position_embedding
```

RoPE injects position by rotating Q and K inside attention. This makes relative position
relationships more explicit in the attention score.

For this small model, the question is modest:

```text
Does RoPE improve validation loss or generation stability at this scale?
```

## Parameter Estimate

RoPE removes learned position embedding parameters:

```text
012 learned position total = 5,874,944
position embedding = 128 * 256 = 32,768
016 RoPE total estimate = 5,842,176
```

## Train

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_char_width256_rope.yaml
```

## Generate

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.generate --checkpoint runs\hongloumeng_80_char_width256_rope\best_ckpt.pt --prompt "黛玉聽了，" --max-new-tokens 300 --temperature 0.7 --top-k 50 --top-p 0.9
```

## Results

| Step | Train Loss | Val Loss | Best Val | Notes |
| ---: | ---: | ---: | ---: | --- |
| 0 | 8.4138 | 8.4171 | 8.4171 | random initialization |
| 250 | 4.5724 | 4.7804 | 4.7804 | faster early learning than learned position |
| 500 | 4.2128 | 4.4856 | 4.4856 | validation improves |
| 750 | 3.9795 | 4.3359 | 4.3359 | below 4.4 |
| 1000 | 3.8126 | 4.2071 | 4.2071 | strong region |
| 1250 | 3.6288 | 4.1986 | 4.1986 | steady |
| 1500 | 3.5394 | 4.1589 | 4.1589 | better than 012 best |
| 1750 | 3.4672 | 4.1632 | 4.1589 | slight wobble |
| 2000 | 3.3175 | 4.1001 | 4.1001 | strong improvement |
| 2250 | 3.1540 | 4.0805 | 4.0805 | best checkpoint |
| 2500 | 3.1021 | 4.0872 | 4.0805 | close but not best |
| 2750 | 2.9720 | 4.1408 | 4.0805 | overfit/noise begins |
| 3000 | 2.8644 | 4.1622 | 4.0805 | validation worsens |
| 3250 | 2.7660 | 4.1430 | 4.0805 | no new best |
| 3500 | 2.6707 | 4.1770 | 4.0805 | train keeps falling |
| 3750 | 2.5605 | 4.1946 | 4.0805 | overfit |
| 4000 | 2.4792 | 4.2350 | 4.0805 | overfit |
| 4250 | 2.3982 | 4.2542 | 4.0805 | overfit |
| 4500 | 2.3301 | 4.3079 | 4.0805 | overfit |
| 4750 | 2.2315 | 4.3022 | 4.0805 | overfit |
| 5000 | 2.1368 | 4.3523 | 4.0805 | final checkpoint much worse than best |

## Compare Against 012

```text
012 learned position best validation: 4.1655
016 RoPE best validation: 4.0805
```

## Sample Notes

Generated from `best_ckpt.pt`, temperature `0.7`, top-k `50`, top-p `0.9`.

Prompt `黛玉聽了，`:

```text
黛玉聽了，忙陪笑道：“你說我不知道，我還不知道呢。”
```

Prompt `寶玉笑道：「`:

```text
寶玉笑道：「這是個好的，我竟不能說的。」寶玉道：「我這里有個什麼東西？」
```

Prompt `賈母笑道：「`:

```text
賈母笑道：「你們都別管那些東西，還不知怎麼樣呢？」
```

RoPE improved validation loss substantially, but generation can still enter local loops around
quoted words, `字`, or repeated kinship terms such as `妹妹`. Better average prediction does not
fully solve decoding quality.

## Interim Conclusion

RoPE is clearly useful even at this small scale:

```text
012 learned absolute position: 4.1655
016 RoPE: 4.0805
delta: -0.0850
```

This is a much larger gain than widening from 192 to 256 hidden dimensions. It suggests positional
representation is a real bottleneck for this character-level model.

RoPE also overfits earlier:

```text
best step = 2250
final step = 5000
```

Future architecture comparisons should keep using `best_ckpt.pt` and may need shorter training or
stronger regularization.
