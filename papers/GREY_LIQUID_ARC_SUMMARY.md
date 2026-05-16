# The Sub-3-Bit Research Arc — Grey Liquid Labs

**Author:** ssfdre38 / Grey Liquid Labs  
**Period:** May 13–16, 2026  
**Experiments:** #5 through #8b (complete)

---

## The Question

Can large language models run coherently at less than 3 bits per weight? If not universally, *which architectures can*, and why?

---

## What We Found (Experiments #5–#8b)

**Exp #5 — imatrix Limitation:** The standard importance-weighted quantization pipeline (llama-imatrix) only covers 46% of Gemma 4's tensors, blocking importance-guided sub-3-bit. Pivoted to testing raw Q2_K across architectures.

**Exp #6 — Cross-Architecture Q2_K:** Tested five models. Qwen 2.5-7B and Mistral-Small passed at Q2_K with 80%+ compression. Phi-4, Gemma 4 e2b, and Mistral 7B v0.3 failed. Barrier is *architecture-specific*, not universal.

**Exp #7 — SWA Confirmation:** Mistral 7B v0.3 (uses SWA) fails identically to Gemma 4. Mistral-Small (same family, no SWA) passes. SWA is confirmed as the failure amplifier with 100% correlation.

**Exp #8 — SWA Tensor Shape Discovery:** Attempted to remove SWA via metadata patch on Gemma 4 e4b. Patch applied correctly, Q2_K quantization succeeded — but inference fails at load with a shape mismatch. Root cause: SWA layers in Gemma 4 have physically different Q/K projection dimensions (256 vs 512 head_dim). SWA is not a runtime toggle; it is a distinct sub-architecture baked in during training.

**Exp #8b — SWA-Only Slice + llama.cpp Bug Fix:** Extracted the 35 SWA layers as a standalone GGUF (3.72 GB Q2_K, 4.72 BPW). During testing, discovered an upstream llama.cpp bug: `llm_graph_input_attn_kv_iswa::set_input` and `can_reuse` do not guard against null-buffer tensors when a model has zero non-SWA layers, causing `GGML_ASSERT(buffer) failed` at `ggml-backend.cpp:194`. Fix submitted as [llama.cpp PR #23131](https://github.com/ggml-org/llama.cpp/pull/23131). After fix: model loads and generates at 14.7 t/s but output is fully incoherent regardless of quantization level. The 7 full-attention layers are architecturally required for coherent output — the barrier is architectural, not purely quantization-related.

---

## The Four Key Discoveries

1. **FFN ratio 3.0–5.5x = quantization danger zone.** Models in this range fail Q2_K. Below 3.0x or above 5.5x, they pass. This is mathematically predictable from the config file before quantization.

2. **SWA = failure amplifier.** 100% correlation across all tested models. SWA's local context window prevents the error averaging that makes Q2_K viable in full-attention models.

3. **SWA in Gemma 4 is a physically distinct sub-architecture.** Q/K weights are half the size of full-attention layers — `[2560, 2048]` vs `[2560, 4096]`. This is not a metadata configuration; it cannot be overridden post-training. This extends the barrier from "attention window size" to "weight space incompatibility."

4. **The sub-3-bit barrier is architectural, not quantization-limited.** SWA-only slices remain incoherent at BF16 precision (no quantization). The 7 full-attention layers are essential for global integration — they are the architectural backbone that SWA layers depend on for coherent output.

---

## What's Next

**MoM (Mixture of Models):** Domain-slice Gemma 4 into independent specialist models. Gemma 4's natural architectural boundary (7 full-att + 35 SWA layers with incompatible weight shapes) suggests it already *wants* to be two separate models. Exp#8b sharpens the design: any viable slice must include the full-attention backbone.

---

## Technical Artifacts

| Resource | Location |
|----------|----------|
| Main paper | `papers/GREY_LIQUID_PAPER_001_FFN_RATIO.md` |
| Experiment reports | `papers/GREY_LIQUID_REPORT_00{5-8}.md`, `papers/GREY_LIQUID_REPORT_008B.md` |
| MoM proposal | `papers/GREY_LIQUID_MOM_001.md` |
| Layer extractor tool | `mom_slice_extract.py` |
| GGUF patcher tool | `deswa_patch.py` |
| Research website | `research.html` (grey-liquid.com) |
| llama.cpp upstream fix | [PR #23131](https://github.com/ggml-org/llama.cpp/pull/23131) — null-buffer guard for all-SWA models |
