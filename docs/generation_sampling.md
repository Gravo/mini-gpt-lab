# Generation Sampling

Generation starts from the model's next-token logits:

```text
logits -> filtering -> softmax -> multinomial sample
```

## Temperature

Temperature changes how sharp the distribution is.

```text
lower temperature: safer, more repetitive
higher temperature: more diverse, more unstable
```

Practical range:

```text
0.4 to 0.8
```

## Top-k

Top-k keeps only the `k` highest-logit tokens before sampling.

Example:

```text
top_k = 50
```

Means: sample only from the 50 most likely next characters.

This helps small models avoid low-probability nonsense tokens.

## Top-p

Top-p, also called nucleus sampling, keeps the smallest set of tokens whose cumulative probability
reaches `p`.

Example:

```text
top_p = 0.9
```

Means: sample from the high-probability nucleus covering about 90% of probability mass.

Unlike top-k, the number of kept tokens changes at every step.

## Suggested Commands

Conservative:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.generate --checkpoint runs\hongloumeng_80_char_wide\ckpt.pt --prompt "黛玉聽了，" --max-new-tokens 300 --temperature 0.5 --top-k 40 --top-p 0.9
```

Balanced:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.generate --checkpoint runs\hongloumeng_80_char_wide\ckpt.pt --prompt "黛玉聽了，" --max-new-tokens 300 --temperature 0.7 --top-k 50 --top-p 0.9
```

More varied:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.generate --checkpoint runs\hongloumeng_80_char_wide\ckpt.pt --prompt "黛玉聽了，" --max-new-tokens 300 --temperature 0.9 --top-k 100 --top-p 0.95
```

## Current Takeaway

Sampling changes decoding behavior, not the learned model. If the model strongly prefers repeated
phrases such as `不知道` or `你們`, top-k/top-p can reduce weird low-probability choices but cannot
fully fix weak long-range semantics.
