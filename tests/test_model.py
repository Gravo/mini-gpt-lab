import torch

from gptlab.kv_cache import empty_cache
from gptlab.model import GPT, GPTConfig


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
