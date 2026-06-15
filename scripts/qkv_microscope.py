import math

import torch
import torch.nn.functional as F


def show(name: str, tensor: torch.Tensor) -> None:
    print(f"\n{name} shape = {tuple(tensor.shape)}")
    print(tensor)


def main() -> None:
    torch.set_printoptions(precision=3, sci_mode=False)

    batch = 1
    seq_len = 4
    hidden_dim = 6
    n_head = 2
    head_dim = hidden_dim // n_head
    vocab_size = 10

    # X is already the post-embedding hidden state for 4 token positions.
    x = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ]
        ]
    )

    # Use small deterministic projections so the printed matrices are readable.
    wq = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.5, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.5],
            [0.5, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.5, 0.0, 0.0, 1.0],
        ]
    )
    wk = torch.eye(hidden_dim)
    wv = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 6.0],
        ]
    )

    q_before = x @ wq
    k_before = x @ wk
    v_before = x @ wv

    q = q_before.view(batch, seq_len, n_head, head_dim).transpose(1, 2)
    k = k_before.view(batch, seq_len, n_head, head_dim).transpose(1, 2)
    v = v_before.view(batch, seq_len, n_head, head_dim).transpose(1, 2)

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    masked_scores = scores.masked_fill(~causal_mask.view(1, 1, seq_len, seq_len), float("-inf"))
    weights = F.softmax(masked_scores, dim=-1)
    out_per_head = weights @ v
    concat = out_per_head.transpose(1, 2).contiguous().view(batch, seq_len, hidden_dim)

    lm_head = torch.arange(hidden_dim * vocab_size, dtype=torch.float32).view(hidden_dim, vocab_size) / 100.0
    logits = concat @ lm_head
    last_logits = logits[:, -1, :]

    print("QKV microscope")
    print(f"batch={batch}, seq_len={seq_len}, hidden_dim={hidden_dim}, n_head={n_head}, head_dim={head_dim}")

    show("X", x)
    show("Q before split", q_before)
    show("K before split", k_before)
    show("V before split", v_before)
    show("Q after split", q)
    show("K after split", k)
    show("V after split", v)
    show("attention scores = Q @ K.T / sqrt(head_dim)", scores)
    show("causal mask", causal_mask)
    show("masked scores", masked_scores)
    show("attention weights = softmax(masked scores)", weights)
    show("attention output per head = weights @ V", out_per_head)
    show("concat output", concat)
    show("lm_head logits", logits)
    show("last-position logits", last_logits)


if __name__ == "__main__":
    main()
