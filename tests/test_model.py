import torch

from gptlab.kv_cache import empty_cache
from gptlab.model import GPT, GPTConfig, filter_logits


def test_forward_shape_and_loss():
    model = GPT(GPTConfig(vocab_size=32, block_size=8, n_layer=2, n_head=2, n_embd=16))
    x = torch.randint(0, 32, (4, 8))
    logits, loss = model(x, x)
    assert logits.shape == (4, 8, 32)
    assert loss is not None


def test_rope_rmsnorm_swiglu_forward():
    model = GPT(
        GPTConfig(
            vocab_size=32,
            block_size=8,
            n_layer=2,
            n_head=2,
            n_embd=16,
            norm="rmsnorm",
            mlp="swiglu",
            position="rope",
        )
    )
    x = torch.randint(0, 32, (2, 8))
    logits, _ = model(x)
    assert logits.shape == (2, 8, 32)


def test_kv_cache_matches_full_prefix_shape():
    model = GPT(GPTConfig(vocab_size=32, block_size=8, n_layer=2, n_head=2, n_embd=16, position="rope"))
    model.eval()
    x = torch.randint(0, 32, (1, 5))
    cache = empty_cache(model.config.n_layer)
    with torch.no_grad():
        full_logits, _ = model(x)
        for i in range(x.size(1)):
            cached_logits, _ = model(x[:, i : i + 1], kv_cache=cache, start_pos=i)
    assert cached_logits.shape == full_logits[:, -1:].shape
    torch.testing.assert_close(cached_logits, full_logits[:, -1:], rtol=1e-4, atol=1e-4)


def test_top_k_filter_keeps_only_k_logits():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    filtered = filter_logits(logits, top_k=2)
    assert torch.isneginf(filtered[0, 0])
    assert torch.isneginf(filtered[0, 1])
    assert filtered[0, 2].item() == 3.0
    assert filtered[0, 3].item() == 4.0


def test_generate_with_sampling_filters_shape():
    model = GPT(GPTConfig(vocab_size=32, block_size=8, n_layer=2, n_head=2, n_embd=16))
    x = torch.randint(0, 32, (1, 4))
    out = model.generate(x, max_new_tokens=3, temperature=0.8, top_k=5, top_p=0.9)
    assert out.shape == (1, 7)
