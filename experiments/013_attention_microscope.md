# 013 Attention Microscope

## Goal

Inspect what each layer and attention head attends to for a real Hongloumeng prompt.

This does not prove a head has a single human-readable role. It gives a first concrete view of
attention patterns:

```text
for one target character:
  each layer
  each head
  top attended source positions
```

## Run

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe scripts\attention_probe.py --checkpoint runs\hongloumeng_80_char_width256\best_ckpt.pt --prompt "黛玉聽了，忙問"
```

Useful prompts:

```text
黛玉聽了，忙問
寶玉笑道：「你
賈母笑道：「你
```

## How To Read It

The script prints:

```text
layer 0
  head 0 effective_tokens=... top=...
```

`top` entries look like:

```text
position:character:attention_weight
```

`effective_tokens` is `exp(entropy)`. It gives a rough sense of how concentrated a head is:

```text
low value: head focuses on a few positions
high value: head spreads attention across many positions
```

## What This Can Show

Possible patterns:

- recent-token heads that mostly look at nearby characters
- delimiter heads that attend to punctuation or quotes
- name heads that attend to a character name before `道` or `問`
- diffuse heads that spread attention across much of the prefix

Do not overclaim. A head can mix several functions, and attention is not a complete explanation
of model behavior.

## First Observations

Checkpoint:

```text
runs/hongloumeng_80_char_width256/best_ckpt.pt
```

Prompt:

```text
黛玉聽了，忙問
```

Target character:

```text
問
```

Patterns seen:

- shallow layers spread attention across most of the short prefix
- middle and later layers often concentrate on nearby characters such as `忙`, `，`, and the current `問`
- some heads attend back to the name `黛`, suggesting possible name/reference sensitivity

Example:

```text
layer 4 head 0 top=5:忙:0.707, 6:問:0.227, 4:，:0.063
layer 5 head 0 top=0:黛:0.545, 6:問:0.452
```

Prompt:

```text
寶玉笑道：「你
```

Target character:

```text
你
```

Patterns seen:

- many heads attend strongly to the quote marker `「`
- several heads attend to `：` and `道`, which are structural markers for dialogue
- late-layer heads sometimes attend back to `寶`, the name that starts the speaking subject

Example:

```text
layer 5 head 4 top=5:「:0.809, 6:你:0.132
layer 5 head 1 top=4:：:0.722, 6:你:0.171
layer 5 head 3 top=0:寶:0.701, 1:玉:0.094
```

Tentative interpretation:

```text
Some heads appear to track local dialogue structure, punctuation, and nearby speaker/name cues.
```

This is not a final claim about what each head "means." To make stronger claims, the next step
would be head ablation: zero one head at a time and measure loss or generation changes.
