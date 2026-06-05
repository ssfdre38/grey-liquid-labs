# Grey Liquid Lab — Paper #005
## The Hidden Architecture: Physically Distinct Sub-Networks in Gemma 4

**Date:** June 5, 2026  
**Researcher:** ssfdre38  
**Status:** PUBLISHED  
**Track:** Architecture Research / Compression  

---

## Abstract

We document a significant architectural discovery within Google's Gemma 4 e4b (9.6B) model. Our research reveals that Gemma 4 is not a uniform transformer architecture with a configurable attention window; rather, it embeds **two physically distinct and incompatible sub-networks** within a single weight file. These sub-networks are separated not merely by metadata toggles, but by fundamental tensor dimensionality. We demonstrate that the 7 full-attention "global integration" layers are architecturally essential for coherence, and their removal results in total semantic collapse, regardless of quantization precision. This finding provides the physical validation for the **Mixture of Models (MoM)** theory and identifies the specific hardware-level constraints preventing sub-3-bit quantization on this architecture.

---

## 1. Discovery: Physical Tensor Mismatch

During Experiment #8 (De-SWA Patching), we attempted to force llama.cpp to treat all layers of Gemma 4 as "full attention" by overriding GGUF metadata. This attempt failed immediately during the model load phase, revealing a physical discrepancy in the Q/K projection weights.

### 1.1 Dimensionality Comparison

Our analysis of the raw tensor shapes confirms that the SWA layers are physically smaller than the global attention layers:

| Component | Full-Attention (Global) | SWA (Local) | Difference |
|-----------|-------------------------|-------------|------------|
| **Q Weight Shape** | `[2560, 4096]` | `[2560, 2048]` | **50% Smaller** |
| **K Weight Shape** | `[2560, 1024]` | `[2560, 512]` | **50% Smaller** |
| **Head Dimension** | 512 | 256 | 2x scaling |
| **Layer Indices** | 5, 11, 17, 23, 29, 35, 41 | 0-4, 6-10, ... | — |

This mismatch proves that the "Sliding Window" in Gemma 4 is not a software-level mask applied to standard weights, but a hardware-level optimization where the attention heads are physically half-sized.

---

## 2. Functional Dependency: The "Global Integration" Backbone

In Experiment #8b, we tested a "SWA-only" slice of the model (35 local layers) to see if it could operate independently. 

### 2.1 Results of Truncation

| Metric | Full Architecture (42 layers) | SWA-Only Slice (35 layers) |
|--------|------------------------------|----------------------------|
| **Total Layers** | 42 | 35 |
| **Inference Status** | Stable ✅ | Unstable ❌ |
| **Output Coherence** | Perfect | **Total Incoherence** |
| **Sample Response** | "2 + 2 is 4." | "enePA BlCOURendingSelect..." |

### 2.2 The Essential 7

The 7 full-attention layers (placed exactly every 6th layer) act as a **Global Integration Backbone**. These layers are responsible for:
1.  **Context Synthesis:** Aggregating local information from the SWA windows into a global state.
2.  **Semantic Formatting:** The final layer (41) is a full-attention layer, responsible for formatting the final token probabilities.
3.  **Error Correction:** Providing the "averaging effect" that prevents quantization noise from local windows from entering a destructive feedback loop.

---

## 3. Upstream Impact: llama.cpp Bug Fix

Our research into all-SWA models identified a critical crash in the llama.cpp inference engine.

### 3.1 The Null-Buffer Guard (PR #23131)

We discovered that `llm_graph_input_attn_kv_iswa::set_input` failed to check for null buffers when the non-SWA layer count was zero. This caused an immediate `GGML_ASSERT` crash.

**Fix Applied:**
```cpp
// base tensors may not be allocated if there are no non-SWA attention layers
if (self_k_idxs && self_k_idxs->buffer) {
    mctx->get_base()->set_input_k_idxs(self_k_idxs, ubatch);
    ...
}
```
This fix was submitted and merged, enabling the community to experiment with non-standard hybrid architectures for the first time.

---

## 4. Conclusion: Implications for MoM

The physical separation of these layer types validates the core premise of **Mixture of Models (MoM)**. 

Gemma 4 is effectively a **hard-coded ensemble**. The full-attention backbone provides the "intelligence" and "coherence," while the SWA sub-network provides the "processing depth" at half the memory cost. 

### 4.1 Future Research Direction

This architectural split identifies why **Surgical Anchoring** (Paper #004) is so effective. By maintaining high-precision (Q3_K_S) only on the 7 critical backbone layers and the first 23 local layers, we can quantize the "Resilient Zone" (30-40) aggressively without breaking the sematic formatting handled by the final backbone layer.

---

**Lab:** Grey Liquid Labs (https://ssfdre38.xyz/grey-liquid.html)  
**Cite as:** ssfdre38. "The Hidden Architecture: Physically Distinct Sub-Networks in Gemma 4." Grey Liquid Lab Paper #005, June 2026.
