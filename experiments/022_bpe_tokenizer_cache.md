# 022 BPE Tokenizer Cache

## Goal

Make BPE experiments reproducible and faster to restart by caching the trained tokenizer state.

Experiment 021 showed that even a small BPE vocabulary has meaningful behavior, but training the
tokenizer itself is slow enough to become part of the experiment. This change adds a config field:

```yaml
tokenizer:
  type: bpe
  vocab_size: 6000
  min_pair_freq: 2
  cache_path: data/tokenizers/hongloumeng_80_bpe6000.json
  encoded_cache_path: data/tokenizers/hongloumeng_80_bpe6000_ids.pt
```

## Behavior

If `cache_path` exists:

```text
load tokenizer state from JSON
```

If `cache_path` does not exist:

```text
train tokenizer from text
write tokenizer state to JSON
```

The checkpoint still stores tokenizer state too, so generation remains self-contained.

For larger BPE vocabularies, loading the tokenizer state is not enough. Encoding the full corpus can
also be slow because the simple BPE encoder applies every merge in order. The training script
therefore also supports `encoded_cache_path`:

If `encoded_cache_path` exists:

```text
load encoded token ids with torch.load
```

If `encoded_cache_path` does not exist:

```text
encode full corpus
write ids with torch.save
```

## Why This Matters

Without caching, every BPE training run must relearn all merges before model training starts. That is
fine for one small probe, but it makes 6000/8000 vocab experiments noisy and slow.

With caching, tokenizer training becomes a one-time data preparation step, and model experiments can
reuse exactly the same pieces, merges, and encoded corpus.

## Verification

The test suite now covers:

```text
char tokenizer roundtrip
BPE tokenizer roundtrip
BPE tokenizer state restore
build_tokenizer selecting BPE
build_tokenizer loading from cache
```

Manual compatibility checks:

```text
old char checkpoint can still generate
new BPE checkpoint can generate
```

## Measured BPE 6000 Cache Cost

Tokenizer state build:

```text
chars = 590,432
BPE vocab = 6,000
BPE tokens = 396,089
chars per token = 1.4907
tokens per char = 0.6708
build time = 441.02s
tokenizer JSON size = 71,280 bytes
```

Encoded corpus cache:

```text
encoded ids = 396,089
encode time = 223.57s
ids cache size = 3,170,043 bytes
```

This means BPE 6000 has about 11 minutes of one-time preprocessing cost with the current simple
implementation. The cache turns that into a normal training restart cost.

## Implementation Note

The current BPE implementation is intentionally simple and transparent. It is good for learning,
but larger vocabularies should eventually use a faster encoder, such as trie-based longest-match
encoding or a mature tokenizer library.

## Next Experiment

Use the cached tokenizer path for:

```text
023 = 021 + vocab_size 4500 -> 6000
```
