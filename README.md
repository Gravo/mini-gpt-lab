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
- [011 Hongloumeng wide dropout 0.15](experiments/011_hongloumeng_wide_dropout015.md): longer wide-model training with best-checkpoint saving.
- [012 Hongloumeng width 256](experiments/012_hongloumeng_width256.md): wider 6-layer model with 256 hidden dimensions.
- [013 Attention microscope](experiments/013_attention_microscope.md): inspect per-layer, per-head attention patterns on Hongloumeng prompts.
- [014 Head ablation](experiments/014_head_ablation.md): temporarily zero selected attention heads and measure validation loss changes.
- [015 Memorization probe](experiments/015_memorization_probe.md): compare generated continuations against exact source text continuations.
- [016 Hongloumeng RoPE](experiments/016_hongloumeng_rope.md): compare learned absolute position embeddings with RoPE.
- [017 Hongloumeng RMSNorm](experiments/017_hongloumeng_rmsnorm.md): compare LayerNorm and RMSNorm after RoPE.
- [018 Hongloumeng SwiGLU](experiments/018_hongloumeng_swiglu.md): compare GELU and SwiGLU MLPs after RoPE.
- [019 Hongloumeng width 384](experiments/019_hongloumeng_width384.md): scale the best RoPE baseline from 256 to 384 hidden dimensions.
- [020 Hongloumeng context 192](experiments/020_hongloumeng_context192.md): extend the best RoPE baseline from 128 to 192 character context.
- [021 Hongloumeng BPE 4500](experiments/021_hongloumeng_bpe4500.md): compare character tokenization with a small BPE tokenizer.
- [022 BPE tokenizer cache](experiments/022_bpe_tokenizer_cache.md): cache trained BPE tokenizer state for reproducible larger-vocab experiments.
- [023 Hongloumeng BPE 6000](experiments/023_hongloumeng_bpe6000.md): test a larger BPE vocabulary with the best RoPE baseline.
- [024 Hongloumeng BPE 6000 context 192](experiments/024_hongloumeng_bpe6000_context192.md): combine larger BPE tokenization with longer context.
- [025 Text-split tokenizer comparison](experiments/025_textsplit_tokenizer_comparison.md): compare char and BPE runs after splitting validation on raw text.
- [026 Hongloumeng BPE 4500 context 192 text-split](experiments/026_hongloumeng_bpe4500_context192_textsplit.md): test smaller BPE with longer context under raw-text validation.
- [027 Hongloumeng layers 9 text-split](experiments/027_hongloumeng_layers9_textsplit.md): increase the best char RoPE baseline depth by 50%.
- [028 Hongloumeng layers 9 context 192 text-split](experiments/028_hongloumeng_layers9_context192_textsplit.md): combine the deeper char model with longer context.
- [029 Hongloumeng layers 9 context 256 text-split](experiments/029_hongloumeng_layers9_context256_textsplit.md): extend the deeper char model to a 256-character context.
- [030 Hongloumeng layers 9 context 192 dropout 0.20 text-split](experiments/030_hongloumeng_layers9_context192_dropout020_textsplit.md): test stronger dropout on the current best architecture.
- [031 Hongloumeng 120 chapters](experiments/031_hongloumeng_120_layers9_context192_dropout020_textsplit.md): train the current best architecture on all 120 chapters.
- [032 Hongloumeng 120 longer training](experiments/032_hongloumeng_120_steps5000.md): extend the 120-chapter run while keeping the model fixed.

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
