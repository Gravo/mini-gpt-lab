from gptlab.tokenizer import BPETokenizer, CharTokenizer, build_tokenizer, tokenizer_from_state


def test_char_tokenizer_roundtrip():
    tokenizer = CharTokenizer("寶玉笑道")
    ids = tokenizer.encode("寶玉")
    assert tokenizer.decode(ids) == "寶玉"


def test_bpe_tokenizer_roundtrip():
    text = "寶玉笑道寶玉笑道黛玉笑道"
    tokenizer = BPETokenizer(text, vocab_size=16)
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert len(ids) < len(text)


def test_bpe_tokenizer_state_roundtrip():
    text = "寶玉笑道寶玉笑道黛玉笑道"
    tokenizer = BPETokenizer(text, vocab_size=16)
    restored = tokenizer_from_state(tokenizer.to_state())
    assert restored.encode(text) == tokenizer.encode(text)
    assert restored.decode(restored.encode(text)) == text


def test_build_tokenizer_selects_bpe():
    tokenizer = build_tokenizer("寶玉寶玉", {"type": "bpe", "vocab_size": 8})
    assert isinstance(tokenizer, BPETokenizer)
