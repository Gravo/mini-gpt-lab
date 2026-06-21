# 014 Head Ablation

## Goal

Move from observing attention patterns to testing whether individual heads affect loss.

`013_attention_microscope` showed that some heads strongly attend to dialogue markers such as
`「`, `：`, `道`, and speaker names. Attention alone is not proof of function, so this experiment
temporarily zeros selected head outputs and measures validation loss change.

## Concept

Ablation means removing or disabling one component to measure what changes.

For an attention head:

```text
normal:
  head output contributes 32 dimensions

ablated:
  that head output is set to zero before heads are concatenated
```

If validation loss rises after ablation, that head was useful for the sampled validation batches.

## Run

Probe heads that looked interesting in experiment 013:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe scripts\head_ablation.py --checkpoint runs\hongloumeng_80_char_width256\best_ckpt.pt --head 5:4 --head 5:3 --head 5:1 --head 5:6 --eval-iters 20
```

The output reports:

```text
baseline val loss
ablated val loss
delta
```

## Results

| Ablation | Val Loss | Delta | Notes |
| --- | ---: | ---: | --- |
| baseline | 4.1955 | +0.0000 | 100 eval batches |
| layer 5 head 4 | 4.2001 | +0.0046 | quote-focused in 013 |
| layer 5 head 3 | 4.1998 | +0.0043 | speaker-name focused in 013 |
| layer 5 head 1 | 4.2002 | +0.0047 | colon-focused in 013 |
| layer 5 head 6 | 4.1978 | +0.0023 | current-token focused in 013 |

## Caveats

- Loss is estimated on sampled validation batches, so small deltas can be noisy.
- A single head can be redundant with other heads.
- A head can matter for specific text patterns even if global validation loss barely changes.
- Stronger evidence would combine loss deltas, targeted prompts, and repeated seeds.

## Interpretation

All four ablations increased validation loss on the same sampled batches.

The deltas are small:

```text
+0.0023 to +0.0047
```

This suggests:

- the inspected heads are useful
- no single inspected head is catastrophic to remove
- dialogue-structure heads may be partly redundant with other heads

The largest deltas in this run came from:

```text
layer 5 head 1: colon-focused in the microscope output
layer 5 head 4: quote-focused in the microscope output
```

This is weak but concrete evidence that the attention patterns observed in 013 correspond to
useful computation, not just decorative attention.
