# Stage 1 Checkpoint: GPT Basics, QKV, and Multi-Head Attention

This note captures the current understanding before starting the QKV microscope experiment.

## Current Mastery

The Stage 1 foundation is mostly in place.

Known clearly:

- tokenizer outputs token ids, not vectors
- embedding lookup converts token ids into hidden-width dense vectors
- `vocab_size`, `context length`, and `hidden_dim` are different dimensions
- GPT pretraining is self-supervised because targets come from shifting text by one token
- attention works in hidden space, not vocab space
- lm_head maps hidden states back to vocab logits
- logits become probabilities only after softmax or sampling logic

Useful standard shape example:

```text
batch = 2
seq_len = 5
hidden_dim = 16
vocab_size = 100

token ids:             [2, 5]
token embeddings:      [2, 5, 16]
Transformer output:    [2, 5, 16]
lm_head logits:        [2, 5, 100]
last-position logits:  [2, 100]
```

## Corrected Mental Model

GPT training:

```text
raw text
-> tokenizer
-> token ids
-> input window and shifted target window
-> embedding lookup
-> position information
-> decoder-only Transformer with causal mask
-> vocab logits
-> cross entropy loss
-> backprop updates all learnable parameters
```

Training targets are constructed automatically:

```text
text:   我 想 吃 苹果
input:  我 想 吃
target: 想 吃 苹果
```

The causal mask is not random. It is a fixed lower-triangular mask:

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

This prevents each position from seeing future answers during training.

## QKV Summary

QKV are not copied vectors. They are activations produced by learned projections:

```text
Q = X @ Wq
K = X @ Wk
V = X @ Wv
```

Where:

```text
X      = hidden states for the current layer
Wq/Wk/Wv = learnable parameters
Q/K/V = per-input activations
```

Role intuition:

```text
Q/K decide which tokens should attend to which tokens.
V provides the content that gets mixed back into each token.
```

The attention relationship matrix is token-to-token:

```text
Q @ K.T -> [seq_len, seq_len]
```

It is not:

```text
[seq_len, vocab_size]
```

The vocab only appears at:

```text
embedding table: [vocab_size, hidden_dim]
lm_head:         [hidden_dim, vocab_size]
```

## Layer vs Head

Layer and head are different concepts.

```text
layer = how many rounds of attention + MLP are stacked
head  = how many parallel attention views exist inside one attention layer
```

Example:

```text
n_layer = 3
n_head = 2
```

Means:

```text
3 Transformer blocks
2 attention heads inside each block
```

Total attention heads across the model:

```text
3 * 2 = 6
```

But they are organized by layer, not all in one place.

## Multi-Head Attention

Given:

```text
seq_len = 4
hidden_dim = 768
n_head = 2
head_dim = 384
```

Input hidden states:

```text
X: [4, 768]
```

After QKV projection and head split:

```text
Q/K/V: [2, 4, 384]
```

Each head independently computes:

```text
scores:  [4, 384] @ [384, 4] -> [4, 4]
weights: softmax(scores) -> [4, 4]
out:     [4, 4] @ [4, 384] -> [4, 384]
```

With 2 heads:

```text
head0 output: [4, 384]
head1 output: [4, 384]
```

Then concat:

```text
[4, 384] + [4, 384] -> [4, 768]
```

Then output projection:

```text
Wo: [768, 768]
output: [4, 768]
```

The heads do not separately map back to vocab. They are intermediate views that merge back into one hidden state.

## Semantic Intuition for Heads

A head can be understood as one learned relation view.

With multiple heads:

```text
head 0 may learn one relation pattern
head 1 may learn another relation pattern
```

For example:

```text
one head may focus on nearby position or formatting patterns
another head may focus on entity, syntax, or dependency patterns
```

This is only an intuition. The model is not told what each head should represent. Training discovers whatever patterns reduce next-token loss.

Better wording:

```text
multi-head attention projects the hidden state into several subspaces,
computes independent token-to-token attention in each subspace,
then concatenates and mixes those views back into one hidden state.
```

## Next Experiment

Next step:

```text
QKV microscope
```

Planned files:

```text
scripts/qkv_microscope.py
experiments/004_qkv_microscope.md
```

First version:

```text
batch = 1
seq_len = 4
hidden_dim = 4
n_head = 1
head_dim = 4
```

Print:

```text
X
Q
K
V
Q @ K.T
causal mask
softmax weights
weights @ V
```

Then change to:

```text
n_head = 2
head_dim = 2
```

And inspect:

```text
scores:  [2, 4, 4]
out:     [2, 4, 2]
concat:  [4, 4]
```

The learning goal:

```text
See attention as concrete matrix multiplication, not only as a metaphor.
```

