# 3-Month Roadmap: From mini-gpt-lab to LLM Inference Systems

This roadmap turns the current learning direction into a 12-week plan.

Long-term direction:

```text
LLM inference systems + controllable agent systems
```

Supporting skills:

```text
GPT internals
small-model experiments
KV cache and serving constraints
RAG / data infrastructure
vLLM community contribution
```

The main idea is not to wait until every Transformer detail is mastered before touching industrial systems. Instead, use two tracks:

```text
Track A: mini-gpt-lab
Build the bottom-up mental model: tokenizer, embedding, QKV, loss, KV cache, RoPE, RMSNorm, SwiGLU.

Track B: vLLM early exposure
Build the top-down systems view: serving, prefill/decode, batching, KV cache pressure, OpenAI-compatible APIs.
```

These tracks should reinforce each other:

```text
mini-gpt-lab explains what KV cache is.
vLLM explains why KV cache management becomes a serving problem.

mini-gpt-lab explains autoregressive decoding.
vLLM explains why continuous batching matters.

mini-gpt-lab explains attention shapes.
vLLM explains why long-context serving is memory-bound.
```

## 0. Current Position

Current strengths:

- computer science background
- practical data and business system experience
- audit digitalization and data lake experience
- YOLO-based object detection and image classification experience
- knowledge graph data collection and querying experience
- basic algorithm background, including Dijkstra, greedy methods, and dynamic programming
- growing understanding of decoder-only GPT, next-token prediction, tokenizer, embedding, QKV, and KV cache

Current gap:

- QKV is understood conceptually but needs numerical and code-level grounding
- small-model training has not yet produced a complete loss/speed/sample comparison
- inference infra concepts are still early
- vLLM contribution path needs gradual entry

Best-fit direction:

```text
Primary: LLM inference infrastructure and controllable agent systems
Secondary: algorithmic optimization around inference and retrieval
Long-term reference: training infrastructure concepts, not the first main target
```

## 1. Three-Month Outcome

At the end of 3 months, the goal is to have:

- a working and documented `mini-gpt-lab`
- a QKV microscope experiment
- a Tiny Shakespeare baseline with a recorded loss curve
- at least one architecture comparison, such as RoPE/RMSNorm/SwiGLU
- a KV cache speed comparison
- a basic vLLM usage notebook or notes document
- at least one reproduced vLLM issue, doc gap, or small contribution candidate
- a clear next 3-month path toward vLLM contribution and controllable agents

The standard is not "know everything." The standard is:

```text
Can explain the mechanism.
Can run a minimal experiment.
Can record evidence.
Can state what changed and why it matters.
```

## 2. Weekly Plan

## Weeks 1-2: GPT Mental Model and QKV Microscope

Main question:

```text
How does a token become a hidden vector, and how does attention route information between tokens?
```

Tasks:

- read and revise `docs/gpt_principles_visual.md`
- build `scripts/qkv_microscope.py`
- print `X`, `Q`, `K`, `V`, `Q @ K.T`, causal mask, attention weights, and `weights @ V`
- write `experiments/004_qkv_microscope.md`
- answer concept questions before moving on

Deliverables:

- QKV microscope script
- QKV microscope experiment note
- personal explanation of Q/K/V in 5-8 sentences

Mastery questions:

- Why is tokenizer output not an embedding?
- Why is attention score shape `[seq_len, seq_len]` instead of `[seq_len, vocab_size]`?
- Why are Q/K/V activations, while Wq/Wk/Wv are parameters?
- What does `weights @ V` mean?
- Why does causal mask matter for GPT?

## Weeks 3-4: Training Baseline and Loss Curve

Main question:

```text
How does decoder-only next-token prediction turn raw text into self-supervised training data?
```

Tasks:

- run the Tiny Shakespeare baseline
- record train loss and validation loss
- generate samples from early and later checkpoints if practical
- make sure the training loop, batch construction, and target shift are understood
- update `experiments/001_baseline_gpt2.md`

Deliverables:

- baseline loss table
- sample outputs
- short explanation of overfitting vs underfitting in this small setting

Mastery questions:

- Why is the target sequence shifted by one token?
- Why can a single context window create many training examples?
- What does validation loss measure?
- What would it mean if train loss drops but validation loss does not?

## Weeks 5-6: Modern Block Components

Main question:

```text
How do RoPE, RMSNorm, and SwiGLU change a GPT block compared with a GPT-2-style baseline?
```

Tasks:

- run or inspect baseline with learned position + LayerNorm + GELU
- run or inspect RoPE variant
- run or inspect RMSNorm variant
- run or inspect SwiGLU variant
- record loss, speed, and sample notes
- update `experiments/002_rope_rmsnorm.md` and `experiments/003_swiglu.md`

Deliverables:

- architecture comparison table
- one-paragraph explanation for each component

Mastery questions:

- Where does RoPE inject position information?
- Why does RMSNorm remove mean-centering?
- What does SwiGLU gate?
- Why should only one variable be changed at a time when comparing variants?

## Weeks 7-8: KV Cache and Inference Speed

Main question:

```text
Why is inference slow, and what exactly does KV cache save?
```

Tasks:

- add a KV cache speed experiment if not already present
- compare generation with cache on/off
- inspect cache tensor shapes per layer
- write `experiments/005_kv_cache_speed.md`

Deliverables:

- speed comparison table
- explanation of prefill and decode
- explanation of why Q is not cached

Mastery questions:

- Why does autoregressive generation remain sequential even with KV cache?
- Why do future tokens need old K/V but not old Q?
- What is the difference between training-time full-window parallelism and inference-time token-by-token decoding?
- Why does context length affect inference memory?

## Weeks 9-10: vLLM User-Level Entry

Main question:

```text
What problem does vLLM solve that plain model.generate does not solve well?
```

Tasks:

- read vLLM user docs and contribution docs
- run a small vLLM example if hardware and model size allow
- test OpenAI-compatible server mode if practical
- compare concepts against `mini-gpt-lab`: prefill, decode, KV cache, batching
- write `docs/vllm_entry_notes.md`

Deliverables:

- vLLM entry notes
- command log for at least one successful or attempted run
- list of 3 possible small contribution candidates

Mastery questions:

- What is the difference between prefill and decode?
- Why does serving multiple users require batching?
- Why is KV cache a memory-management problem in production?
- What is PagedAttention trying to improve?

## Weeks 11-12: First Contribution Candidate and Next Roadmap

Main question:

```text
How do I turn learning into a small but real open-source contribution?
```

Tasks:

- scan vLLM issues labeled good first issue, documentation, bug, or tests
- reproduce one small issue or identify one documentation gap
- prepare a local note describing expected behavior, actual behavior, and reproduction steps
- if suitable, open a small PR or prepare a patch
- write the next 3-month roadmap

Deliverables:

- one reproduced issue or contribution note
- one small PR candidate
- next-stage roadmap

Mastery questions:

- Can I explain the issue without guessing?
- Can I reproduce it with a minimal command or script?
- Can I add a test or verification step?
- Is the change small enough for a first PR?

## 3. Question Types for Each Stage

Each stage should be checked with multiple kinds of questions:

Concept questions:

```text
Explain the mechanism in plain language.
```

Shape questions:

```text
Given batch, seq_len, hidden_dim, vocab_size, write tensor shapes.
```

Code questions:

```text
Point to the implementation and explain which line corresponds to the concept.
```

Failure questions:

```text
What breaks if this assumption is false?
```

Interview questions:

```text
Explain it under pressure, briefly and accurately.
```

Experiment questions:

```text
What metric would prove this change helped?
```

## 4. What to Avoid in the First 3 Months

Avoid making these the main target too early:

- training a large model from scratch
- CUDA kernel work
- distributed training infrastructure
- reading many papers without experiments
- rewriting vLLM core scheduler before understanding user-level serving
- building a large agent framework without state, verification, and rollback

Useful but later:

- tensor parallelism
- pipeline parallelism
- FlashAttention internals
- NCCL and distributed serving
- speculative decoding internals
- quantization kernels

## 5. Long-Term Direction After 3 Months

If the first 3 months go well, the next phase should combine:

```text
vLLM contribution
local model serving
RAG over audit/data-lake materials
controllable agent runtime
small-model evaluation
```

Possible 6-12 month identity:

```text
LLM application infrastructure engineer
AI platform engineer
model serving engineer
controllable agent systems engineer
```

The north star:

```text
Build systems that make LLMs reliable, inspectable, cost-aware, and controllable.
```

