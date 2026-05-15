# Grey Liquid Lab — Research Proposal #001
## Mixture of Models (MoM): Domain-Specialized Neural Slices as Independent Expert Systems

**Date:** May 15, 2026  
**Research Track:** Model Compression + Autonomy & Agency  
**Classification:** Research Proposal / Architecture Design  
**Status:** Pre-Experimental (Hypothesis Formation)

---

## Abstract

We propose **Mixture of Models (MoM)** — a distinct architecture from Mixture of Experts (MoE) — where each domain of knowledge is served by a completely independent specialized model rather than by sparse expert layers within a shared model. Starting from Gemma 4's bf16 weights, we describe a methodology for extracting domain-specific sub-models ("slices") from the full 42-layer architecture, fine-tuning each for a specific knowledge domain, and routing queries through a lightweight orchestrator.

Critically, our architecture analysis of Gemma 4 e4b reveals a **secondary research hypothesis**: because 18 of 42 layers use Sliding Window Attention (SWA) — the primary cause of Q2_K quantization failure discovered in Grey Liquid Experiments #6 and #7 — domain slices extracted from **full-attention layers only** may be Q2_K compatible, enabling 80%+ compression on models derived from an architecture that currently cannot survive Q2_K at all.

This connects MoM research directly to Grey Liquid's core quantization discovery.

---

## 1. MoM vs MoE: A Critical Distinction

### 1.1 Mixture of Experts (MoE) — What Already Exists

In MoE architectures (e.g., Mixtral 8x7B), a **single model** contains multiple parallel FFN "expert" networks per layer. A router within the model selects 1–2 experts per token at inference time. The entire model is loaded; only some FFN paths activate.

```
Input → [Shared Attention] → Router → [Expert 1 | Expert 2 | ... | Expert N] → Output
         (all in one model)            (sparse activation within one model)
```

**Key property**: One model file. Experts share attention layers. No knowledge isolation.

### 1.2 Mixture of Models (MoM) — What We Propose

In MoM, **each domain is a completely independent model**. An external orchestrator (router) receives a query, identifies the appropriate domain, and forwards the query to that specialist model. Each model runs independently.

```
Query → [Router Model] → Domain Classification
                              ↓
              ┌───────────────┼───────────────┐
         [Text Model]  [Code Model]  [Math Model]  [Science Model]  ...
              ↓
         Domain Expert Response → [Aggregator/Router] → Final Output
```

**Key properties:**
- Each expert is a **complete, independently deployable model**
- Models can be loaded/unloaded independently (RAM efficiency)
- Each domain can use **different quantization levels** based on task sensitivity
- Domain specialists can be fine-tuned without affecting other domains
- Models are **additive** — new domains = new models, not architecture changes

### 1.3 Why MoM Over MoE for This Use Case

| Property | MoE | MoM |
|----------|-----|-----|
| Single file | ✅ | ❌ (multiple files) |
| RAM at inference | High (full model) | Low (one domain at a time) |
| Domain isolation | ❌ (shared attention) | ✅ (completely separate) |
| Updateable per domain | ❌ | ✅ |
| Different quantization per domain | ❌ | ✅ |
| Deployable on weak hardware | ❌ | ✅ (only load needed domain) |
| Parallelizable across machines | ❌ | ✅ |

For a local-first, privacy-focused deployment (the Grey Liquid philosophy), MoM is superior. A 4GB laptop can run one domain expert at a time. A cluster can run all domains in parallel.

---

## 2. Gemma 4 e4b Architecture Analysis

We analyzed the `gemma4-e4b-bf16.gguf` file (14.02 GB) to establish the slicing baseline.

### 2.1 Core Dimensions

| Parameter | Value |
|-----------|-------|
| Architecture | Gemma 4 |
| Total parameters | ~7.5B |
| Block (layer) count | **42** |
| Embedding length | 2560 |
| Feed-forward length | 10240 |
| **FFN expansion ratio** | **4.0x** (10240/2560) |
| Attention heads | 8 |
| KV heads | 2 (GQA) |
| Context length | 131,072 |
| Sliding window size | 512 |
| **Shared KV (SWA) layers** | **18 of 42** |

### 2.2 The SWA Layer Distribution

Gemma 4 uses **alternating attention**: some layers use full global attention, others use Sliding Window Attention (SWA). Based on metadata:

- **Full attention layers**: 42 − 18 = **24 layers** (key_length=512, global context)
- **SWA layers**: **18 layers** (key_length_swa=256, 512-token window)

This alternating pattern is characteristic of Gemma 4's architecture — local attention for efficiency, global attention for long-range reasoning.

### 2.3 The Already-Sliced Vision Component

The multimodal vision encoder ships as a separate file:

```
gemma4-e4b-bf16.gguf       14.02 GB  ← language backbone (42 layers)
mmproj-e4b-f16.gguf         0.92 GB  ← vision projector (separate domain model)
```

**This is the proof of concept for MoM.** Google already implemented domain slicing for vision. The mmproj is precisely what we propose for all other domains: a specialized sub-network that handles one type of input, communicating with the backbone through a defined interface.

---

## 3. The SWA-Slice Quantization Hypothesis

This is the critical connection between MoM research and Grey Liquid's prior quantization work.

### 3.1 Prior Finding (Experiments #6 & #7)

- **FFN ratio in 3.0x–5.5x danger zone** → Q2_K fails (100% correlation)
- **SWA presence** → amplifies FFN ratio problems → guaranteed failure
- Gemma 4 e4b: FFN ratio = 4.0x (danger zone) + SWA → Q2_K fails

### 3.2 The New Hypothesis

**If we extract only full-attention layers (no SWA) as a domain slice:**

```
Original Gemma 4 e4b:  42 layers = 24 full-attention + 18 SWA  →  Q2_K FAILS
                                              ↓
SWA-free domain slice:  24 full-attention layers only            →  Q2_K ???
```

The SWA-free slice:
- Removes the secondary cause of Q2_K failure (SWA)
- Still has FFN ratio 4.0x (still in danger zone — the primary cause remains)
- But: **with 24 layers instead of 42, error accumulation is halved**
- If error accumulation across layers is what pushes the danger zone model into failure, reducing layers may push it below the failure threshold

**Formal Hypothesis H1:**
> A domain slice extracted from full-attention-only layers of Gemma 4 e4b, after domain fine-tuning, will survive Q2_K quantization at a higher rate than the full Gemma 4 e4b model.

**Formal Hypothesis H2:**
> The effective FFN danger zone threshold is not just a function of the FFN ratio, but also of layer depth. Shallower models (fewer layers) may tolerate FFN ratios within 3.0x–5.5x at Q2_K where deeper models cannot.

These hypotheses, if confirmed, would extend the Grey Liquid quantization model from a 2D predictor (FFN ratio) to a 3D predictor (FFN ratio × layer depth × SWA presence).

---

## 4. Proposed Domain Architecture

### 4.1 Domain Taxonomy

Based on natural knowledge clustering and query type distribution:

| Domain | Expert Model | Input Types | Rationale |
|--------|-------------|-------------|-----------|
| **Vision** | `mmproj-e4b-f16.gguf` | Images | Already exists |
| **Text & Language** | text-expert | Natural language, writing, summarization | Highest query volume |
| **Reasoning & Thinking** | reasoning-expert | Logic, planning, step-by-step analysis | Needs deep context |
| **Code** | code-expert | Programming, debugging, syntax | Domain-specific vocabulary |
| **Mathematics** | math-expert | Calculations, proofs, equations | Precision critical |
| **Science** | science-expert | Physics, chemistry, biology | Factual accuracy critical |
| **History & Knowledge** | history-expert | World knowledge, events, people | Broad recall |
| **Creative** | creative-expert | Stories, poetry, brainstorming | Diversity of output |

8 domains total. Vision is pre-sliced. 7 new language domain models to create.

### 4.2 Layer Allocation Strategy (Three Options)

**Option A: Full Depth, Domain Fine-Tuned (Simplest)**
- Each domain model = full 42-layer Gemma 4 e4b
- Fine-tuned on domain-specific data only
- Storage: 42-layer model × 7 domains + router
- Size: ~14GB × 7 = ~98GB total (before quantization)
- After domain Q2_K (if H1 confirmed): potentially ~3GB × 7 = ~21GB total

**Option B: SWA-Free Slice (Novel)**
- Extract 24 full-attention layers from e4b
- Add new embedding + output head (re-use from original model)
- Fine-tune each slice on domain data
- Hypothesis: Q2_K compatible (no SWA)
- Size: ~8GB × 7 → ~2GB × 7 after Q2_K = ~14GB total

**Option C: LoRA Adapters (Most Practical Near-Term)**
- One base model (e4b or e2b quantized)
- 7 lightweight LoRA adapters (~100-500MB each)
- Router swaps adapter per query
- Total: ~5GB base + ~3GB adapters = ~8GB
- No fine-tuning infrastructure needed beyond LoRA training

### 4.3 Router Architecture

The router is a small classification model:
- Input: user query (text)
- Output: domain label (8 classes)
- Architecture: ~1B parameter model fine-tuned for classification
- OR: keyword/embedding-based routing (no additional model needed)
- Latency target: <50ms routing decision

```
Query: "Write a Python function to sort a list"
Router: P(code)=0.94, P(text)=0.04, P(math)=0.02
→ Route to: code-expert
→ code-expert.generate(query)
→ Response
```

---

## 5. Slicing Methodology

### 5.1 Layer Extraction from GGUF

Using Python with the `gguf` library, layers can be extracted by tensor name pattern:

```python
# Layers in GGUF are named: blk.{N}.{component}.weight
# Full-attention layers: those where N is NOT an SWA layer index
# SWA layers: every K-th layer (determined by shared_kv_layers metadata)

# Extract layers 0-23 (hypothetical full-attention subset)
selected_layers = [f"blk.{i}." for i in range(24)]

# Keep: token_embd.weight, output_norm.weight, output.weight
# Keep: blk.{i}.* for i in selected_layers
# Discard: blk.{i}.* for i in swa_layers
```

The extracted tensors are written to a new GGUF file with updated metadata (block_count, etc.).

**Critical step**: The sliced model needs:
1. Original token embeddings (shared vocabulary)
2. Selected transformer layers
3. A new final layer norm + output projection
4. Updated `block_count` metadata

### 5.2 Fine-Tuning Requirements

Per domain expert (Option B — SWA-free slice):

| Resource | Requirement |
|----------|------------|
| Training data | 50K–500K domain-specific samples |
| GPU VRAM | 24GB+ (for 24-layer model in bf16) |
| Training time | 4–12 hours per domain |
| Framework | transformers + PEFT (LoRA for initial fine-tuning) |

### 5.3 Domain Training Data Sources

| Domain | Primary Sources |
|--------|----------------|
| Code | GitHub Code, The Stack, HumanEval |
| Math | MATH dataset, GSM8K, DeepMind Math |
| Science | arXiv (filtered), PubMed, SciQ |
| Reasoning | ARC, HellaSwag, BIG-Bench reasoning tasks |
| History | Wikipedia filtered, Common Crawl history |
| Text/Language | OpenWebText, C4 (general) |
| Creative | WritingPrompts, BookCorpus |

---

## 6. Connection to Grey Liquid Research Program

### 6.1 This Work Extends Experiment #7

Experiment #7 proved: **SWA presence causes Q2_K failure** (with FFN ratio as primary predictor).

MoM Experiment #8 (proposed) would test: **Does removing SWA layers via slicing restore Q2_K compatibility?**

This would elevate the FFN ratio predictor from a 2D model to a 3D model:

```
Q2_K Compatibility = f(FFN_ratio, depth, SWA_fraction)

Old model (2D):  compatible = (ratio < 3.0) OR (ratio > 5.5)
New model (3D):  compatible = (ratio < 3.0) OR (ratio > 5.5) OR 
                              (depth < threshold AND SWA_fraction = 0)
```

### 6.2 The mmproj Precedent Strengthens the Case

The vision projector (mmproj) proves:
- A domain-specific sub-network can be extracted from a foundation model
- It can be made dramatically smaller (0.92GB vs 14GB = 6.5% of the backbone)
- It maintains useful capability when connected to the backbone

Our proposal extends this pattern to language domains.

### 6.3 Democratization Philosophy

MoM aligns with Grey Liquid's core philosophy: **accessible AI for everyone**.

A user with 4GB RAM can run the math expert for math questions, the code expert for coding, unloading between uses. They never need to load a 14GB+ general model. A student with an old laptop gets specialized expert-level assistance within their hardware constraints.

---

## 7. Proposed Experiments

### Experiment #8: SWA-Free Slice Q2_K Compatibility
**Question:** Does removing SWA layers from Gemma 4 e4b create a Q2_K-compatible slice?
**Method:** Extract 24 full-attention layers, test Q2_K without fine-tuning
**Success metric:** Coherent output on "What is 2+2?" within 10 seconds
**Resources:** Python + gguf library, existing hardware

### Experiment #9: Domain Specialization via LoRA
**Question:** Can LoRA adapters create meaningful domain specialization?
**Method:** Train 3 LoRA adapters (code, math, creative) on e2b-bf16
**Success metric:** Code expert outperforms base on HumanEval; math expert on GSM8K
**Resources:** GPU, domain training data, PEFT library

### Experiment #10: Router Accuracy
**Question:** Can a simple classifier route queries to the correct domain?
**Method:** Build embedding-similarity router using 500 labeled query examples
**Success metric:** >90% routing accuracy on held-out test set
**Resources:** Python, sentence-transformers library

### Experiment #11: MoM vs Single Model Benchmark
**Question:** Does a 3-domain MoM outperform the same parameter-count general model?
**Method:** Compare [code-expert + math-expert + text-expert] vs single e2b on mixed benchmark
**Success metric:** Average score across HumanEval + GSM8K + HellaSwag
**Resources:** Domain-trained models from Experiment #9, benchmark scripts

---

## 8. Resource Requirements

### Near-Term (Experiments #8 and #10 — CPU/RAM only)
- Python + gguf library (installable via pip)
- Existing `gemma4-e4b-bf16.gguf`
- 16GB RAM minimum for tensor manipulation
- No GPU required for slice extraction

### Medium-Term (Experiments #9 and #11 — GPU required)
- GPU with 24GB VRAM (or multiple smaller)
- Domain training datasets (~50K samples each)
- PEFT/transformers Python stack
- ~4-12 hours training per adapter

---

## 9. Publication Plan

1. **Experiment #8 report** — SWA-free slice Q2_K result (fast to run)
2. **MoM Architecture Paper** (this document, expanded post-experiments)
3. **Comparative benchmark paper** — MoM vs MoE vs single model
4. **Deployment guide** — How to build your own MoM system

---

## Appendix A: Gemma 4 e4b Full Architecture Metadata

```
general.architecture = gemma4
general.size_label = 7.5B
gemma4.block_count = 42
gemma4.context_length = 131072
gemma4.embedding_length = 2560
gemma4.feed_forward_length = 10240
gemma4.attention.head_count = 8
gemma4.attention.head_count_kv = 2
gemma4.attention.sliding_window = 512
gemma4.attention.shared_kv_layers = 18
gemma4.attention.key_length = 512        (full attention)
gemma4.attention.key_length_swa = 256    (sliding window)
gemma4.rope.freq_base = 1000000.0
gemma4.rope.freq_base_swa = 10000.0
FFN ratio: 10240 / 2560 = 4.0x          ← DANGER ZONE (3.0x–5.5x)
Full attention layers: 42 - 18 = 24
SWA layers: 18
```

---

## Appendix B: MoM vs MoE vs Standard Architecture Comparison

```
Standard LLM:     [Embed] → [L1] → [L2] → ... → [L42] → [Output]
                   One model, all domains, all queries

MoE (Mixtral):    [Embed] → [Attn + Router → [E1|E2|...|E8]] × 32 → [Output]  
                   One model, sparse expert FFN activation per token

MoM (proposed):   Query → Router → Domain Expert → Response
                   Domain-Text:    [Embed_t] → [L1..L24 fine-tuned: text]    → [Output_t]
                   Domain-Code:    [Embed_c] → [L1..L24 fine-tuned: code]    → [Output_c]
                   Domain-Math:    [Embed_m] → [L1..L24 fine-tuned: math]    → [Output_m]
                   Domain-Science: [Embed_s] → [L1..L24 fine-tuned: science] → [Output_s]
                   Domain-Vision:  [mmproj]  → [backbone]                    ← already exists
```

---

*Research proposal by Grey Liquid Lab, May 15, 2026.*  
*Extends prior work in Grey Liquid Experiments #1–#7 (quantization barrier research).*  
*Subject for independent verification and experimental confirmation.*
