# 004 QKV Microscope

## Goal

Make Q/K/V and multi-head attention visible as concrete matrix multiplication.

This experiment uses deliberately small dimensions:

```text
batch = 1
seq_len = 4
hidden_dim = 6
n_head = 2
head_dim = 3
vocab_size = 10
```

The key reason for these numbers:

```text
4 = token positions
6 = total hidden width
2 = number of attention heads
3 = width inside each head
10 = vocabulary size used only by lm_head
```

## Run

```powershell
python scripts/qkv_microscope.py
```

If the default Python does not have torch, use the CUDA environment found earlier:

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe scripts\qkv_microscope.py
```

## Expected Shapes

```text
X:                         [1, 4, 6]
Q before split:            [1, 4, 6]
Q/K/V after split:         [1, 2, 4, 3]
attention scores:          [1, 2, 4, 4]
attention output per head: [1, 2, 4, 3]
concat output:             [1, 4, 6]
lm_head logits:            [1, 4, 10]
last-position logits:      [1, 10]
```

## What to Look For

Attention scores are token-to-token:

```text
single head:
Q: [4, 3]
K: [4, 3]

Q @ K.T:
[4, 3] @ [3, 4] -> [4, 4]
```

The `[4, 4]` matrix is not vocab-related. It is:

```text
4 query positions x 4 key positions
```

The attention output per head returns to `head_dim`:

```text
weights: [4, 4]
V:       [4, 3]

weights @ V -> [4, 3]
```

The two heads are merged by concatenating their last dimension:

```text
head0: [4, 3]
head1: [4, 3]

concat -> [4, 6]
```

Only after attention and subsequent Transformer processing does `lm_head` map hidden states to vocab logits:

```text
[1, 4, 6] -> Linear(6 -> 10) -> [1, 4, 10]
```

## Notes

- Q/K/V are activations produced from hidden states.
- Wq/Wk/Wv are learned parameters.
- Heads are parallel relation views inside one attention layer.
- The vocab size affects embedding and lm_head, not attention scores.
- Causal mask prevents each token position from seeing future token positions.
