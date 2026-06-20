# mini-gpt-lab

From-scratch LLM architecture experiments focused on understanding how small GPT-style models are built, trained, measured, and optimized.

This repository is intentionally small. The goal is not to chase benchmark numbers, but to make the internals visible:

- Transformer block structure
- causal self-attention
- tokenizer choices
- pretraining loop
- loss curves
- KV cache decoding
- RoPE, RMSNorm, and SwiGLU variants
- inference speed tradeoffs
- small-model experiment discipline

## Stage 1 Roadmap

1. Run a nanoGPT-style baseline on Tiny Shakespeare.
2. Document every layer in a GPT block.
3. Add RoPE positional embeddings.
4. Add RMSNorm.
5. Add SwiGLU.
6. Add KV cache for autoregressive generation.
7. Compare loss, speed, and sample quality across variants.

## Repository Layout

```text
configs/       Small experiment configs
data/          Local datasets, ignored by git except for notes
gptlab/        Model, attention, tokenizer, train, generate, KV cache
scripts/       PowerShell entry points
experiments/   Experiment notes and comparison tables
tests/         Unit tests for model behavior
```

## Quick Start

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Train a tiny baseline:

```powershell
.\scripts\train_tiny_shakespeare.ps1
```

Generate from a checkpoint:

```powershell
.\scripts\generate.ps1
```

Run tests:

```powershell
pytest
```

## What This Proves

This project is designed to show working knowledge below the API layer:

- how tokens become vectors
- how attention routes information
- how loss curves reflect training dynamics
- how RoPE changes positional representation
- why RMSNorm and SwiGLU are common in modern LLM blocks
- how KV cache removes repeated work during decoding
- how to evaluate a small model without overclaiming

## Notes

- [GPT principles visual guide](docs/gpt_principles_visual.md): tokenizer, hidden dimensions, Q/K/V, KV cache, vocab logits, and decoder-only self-supervised training.
- [Stage 1 QKV checkpoint](docs/stage_1_qkv_checkpoint.md): current understanding before the QKV microscope experiment.
- [3-month roadmap](docs/three_month_roadmap.md): staged plan from GPT internals to vLLM early exposure and inference systems.
- [Generation sampling](docs/generation_sampling.md): temperature, top-k, and top-p decoding notes.
- [004 QKV microscope](experiments/004_qkv_microscope.md): a small tensor experiment that prints Q/K/V, attention scores, causal mask, per-head output, concat output, and lm_head logits.
- [005 Hongloumeng baseline](experiments/005_hongloumeng_baseline.md): a Chinese character-level baseline using public-domain Wikisource text.
- [006 Hongloumeng data scale](experiments/006_hongloumeng_data_scale.md): same small GPT, more chapters, longer training.
- [007 Hongloumeng context 256](experiments/007_hongloumeng_context_256.md): same small GPT and data, longer context window.
- [008 Hongloumeng clean 80 chapters](experiments/008_hongloumeng_clean_80.md): cleaner Wikisource text and a larger 80-chapter corpus.
- [009 Hongloumeng clean 80 context 256](experiments/009_hongloumeng_clean_80_context_256.md): retest longer context after data cleaning.
- [010 Hongloumeng clean 80 wider model](experiments/010_hongloumeng_clean_80_wide.md): larger depth and width on the cleaned corpus.

## GPT Block Anatomy

The model uses a decoder-only Transformer:

```text
token ids
  -> token embedding
  -> position signal, either learned absolute embeddings or RoPE inside attention
  -> repeated GPT blocks
       -> norm
       -> causal self-attention
       -> residual add
       -> norm
       -> MLP, either GELU or SwiGLU
       -> residual add
  -> final norm
  -> tied language-model head
  -> next-token logits
```

Training uses next-token prediction. For an input sequence `x[0:t]`, the target is `x[1:t+1]`, and the loss is cross entropy over the vocabulary.

## Experiment Method

Each experiment should keep the comparison honest:

- change one architecture feature at a time
- keep parameter scale, dataset, batch size, and training steps close
- record train loss, validation loss, tokens/sec, and a short sample note
- compare generated text qualitatively only after checking the loss curve
- report small-model limits instead of pretending Tiny Shakespeare results generalize directly

## Inference Notes

KV cache is most useful during autoregressive decoding because each new token can reuse previous keys and values. In this lab, cache-based long decoding is paired with RoPE. Learned absolute position embeddings are kept to the fixed training context and use sliding-window generation.

## Related

- `wukong_ai`: closed-loop embodied AI system
- `mini-gpt-lab`: from-scratch LLM architecture experiments
