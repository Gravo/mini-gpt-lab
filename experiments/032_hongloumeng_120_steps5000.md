# 032 Hongloumeng 120 Chapters Longer Training

## Goal

Run a controlled follow-up to 031 by changing only the training length.

```text
031 max_steps = 2500
032 max_steps = 5000
```

Everything else stays fixed:

```text
data_path = data/hongloumeng_120.txt
tokenizer = char
split = text
n_layer = 9
n_head = 8
n_embd = 256
block_size = 192
dropout = 0.20
norm = layernorm
mlp = gelu
position = rope
```

## Why This Experiment

031 reached its best validation loss at the final evaluation step:

```text
031 best val ~= 3.6597 at step 2500
```

That means the model had not clearly saturated yet. Before changing architecture or mixing another
book, test whether the same model benefits from more optimization on the larger 120-chapter corpus.

## Variable Control

| Experiment | Dataset | Layers | Width | Context | Dropout | Max Steps | Changed Variable |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 031 | Hongloumeng 120 | 9 | 256 | 192 | 0.20 | 2500 | baseline |
| 032 | Hongloumeng 120 | 9 | 256 | 192 | 0.20 | 5000 | training length |

## Results

| Step | Train Loss | Val Loss | Val Nats/Char | Best Val | Notes |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 8.4267 | 8.4174 | 8.4174 | 8.4174 | init |
| 250 | 4.7940 | 4.6756 | 4.6756 | 4.6756 | matches 031 |
| 500 | 4.2836 | 4.2240 | 4.2240 | 4.2240 | matches 031 |
| 750 | 4.0640 | 4.0802 | 4.0802 | 4.0802 | matches 031 |
| 1000 | 3.8475 | 3.9600 | 3.9600 | 3.9600 | matches 031 |
| 1250 | 3.7390 | 3.8443 | 3.8443 | 3.8443 | matches 031 |
| 1500 | 3.5950 | 3.7669 | 3.7669 | 3.7669 | matches 031 |
| 1750 | 3.4562 | 3.7158 | 3.7158 | 3.7158 | matches 031 |
| 2000 | 3.3960 | 3.7289 | 3.7289 | 3.7158 | rebound |
| 2250 | 3.3079 | 3.6863 | 3.6863 | 3.6863 | new best |
| 2500 | 3.2014 | 3.6597 | 3.6597 | 3.6597 | reproduces 031 best |
| 2750 | 3.0909 | 3.6458 | 3.6458 | 3.6458 | longer training helps |
| 3000 | 3.0494 | 3.6561 | 3.6561 | 3.6458 | rebound |
| 3250 | 2.9690 | 3.6414 | 3.6414 | 3.6414 | new best |
| 3500 | 2.9000 | 3.6356 | 3.6356 | 3.6356 | new best |
| 3750 | 2.8239 | 3.6352 | 3.6352 | 3.6352 | tiny gain |
| 4000 | 2.7379 | 3.6198 | 3.6198 | 3.6198 | best checkpoint |
| 4250 | 2.6720 | 3.6632 | 3.6632 | 3.6198 | overfit signal |
| 4500 | 2.5971 | 3.6570 | 3.6570 | 3.6198 | no new best |
| 4750 | 2.5387 | 3.6894 | 3.6894 | 3.6198 | worse val |
| 5000 | 2.4545 | 3.7422 | 3.7422 | 3.6198 | clear overfit after best |

Best checkpoint:

```text
runs/hongloumeng_120_char_width256_rope_layers9_context192_dropout020_textsplit_steps5000/best_ckpt.pt
best val ~= 3.6198 at step 4000
checkpoint size ~= 33.15 MB
```

## Comparison

```text
031 best val ~= 3.6597 at step 2500
032 best val ~= 3.6198 at step 4000
delta ~= -0.0398
```

Increasing training length is useful, but not all the way to the final step. The useful window is
roughly 2500-4000 steps. After 4000, training loss keeps falling while validation loss gets worse,
which is the first clear overfitting signal on the 120-chapter setup.

## Sample Notes

The 032 best checkpoint still produces mostly local dialogue coherence. It can maintain Hongloumeng
surface style and names, but it does not keep a stable scene state for long.

Prompt:

```text
黛玉聽了，
```

Observation: the continuation is smoother than very early runs, but it still loops through "我說",
"你別", and speaker-turn patterns. This looks like stronger phrase modeling, not plot memory.

Prompt:

```text
寶玉笑道：「
```

Observation: the model produces plausible dialogue turns and paragraph breaks, but also generates
odd local artifacts such as "玉玉" and inconsistent speaker relationships.

## Decision Rule

Validation improves after 2500 and peaks around 4000, then gets worse while train loss keeps
dropping. The next controlled experiment should not simply push more steps at the same learning
rate.

Better next choices:

```text
033 = same 120-chapter setup, max_steps around 4000 with a learning-rate schedule
034 = same 120-chapter setup, block_size 288, only if we want to test longer context
035 = Hongloumeng 120 + Rulin Waishi, only after we decide to test mixed-corpus regularization
```

## Next Candidate Variables

Keep these as separate experiments:

```text
033 = learning-rate schedule, because 032 overfits after step 4000
034 = block_size 288, to test whether longer context helps on 120 chapters
035 = Hongloumeng 120 + Rulin Waishi, to test mixed-corpus regularization
```

## Command

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_120_char_width256_rope_layers9_context192_dropout020_textsplit_steps5000.yaml
```
