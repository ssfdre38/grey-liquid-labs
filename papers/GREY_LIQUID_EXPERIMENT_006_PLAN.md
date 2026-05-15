# Grey Liquid Experiment #6: Cross-Architecture Sub-3-Bit Barrier Testing

**Date:** May 14, 2026  
**Researcher:** G4 Turbo Lab  
**Goal:** Determine if the sub-3-bit quantization barrier is Gemma 4-specific or universal across architectures

---

## Background

**Experiment #5** discovered an imatrix tool limitation that blocks importance-weighted sub-3-bit quantization:
- Only 46% tensor coverage (275/601 tensors)
- Missing critical PLE pathway layers (blk.15-34)
- Uniform Q2_K quantization fails on Gemma 4 (model crashes)

**Key Question:** Is this a Gemma 4 architectural limitation, or does Q2_K fail universally?

---

## Hypothesis

**H1:** Sub-3-bit barrier is universal (all architectures fail at Q2_K)  
**H2:** Sub-3-bit barrier is architecture-specific (some architectures succeed)

If H2 is true, we can identify which architectural features enable sub-3-bit quantization.

---

## Test Matrix

| Model | Size | Architecture | bpw @ Q2_K | Expected Result |
|-------|------|--------------|-----------|-----------------|
| **Gemma 4 (e2b)** | 9B | SWA + PLE + Shared KV | ~2.5 | ❌ Known failure (baseline) |
| **Phi-4** | 14B | Mixture of Experts? | ~2.5 | ❓ To test |
| **Qwen 2.5-7B** | 7B | Standard Transformer | ~2.5 | ❓ To test |
| **Mistral-Small** | 24B | Grouped-Query Attention | ~2.5 | ❓ To test |

---

## Methodology

### Phase 1: Download Source Models ✅ (In Progress)

Download BF16/FP16 source weights from HuggingFace:
- ✅ Phi-4: `D:\grey-liquid-models\phi-4\` (~29 GB)
- ✅ Qwen 2.5-7B: `D:\grey-liquid-models\qwen2.5-7b\` (~15 GB)
- ✅ Mistral-Small: `D:\grey-liquid-models\mistral-small\` (~51 GB)

**Status:** Downloads running in parallel (started 4:20 PM)

---

### Phase 2: Convert to GGUF

Use llama.cpp to convert HuggingFace models to GGUF format:

```powershell
# Phi-4
python convert_hf_to_gguf.py D:\grey-liquid-models\phi-4\ `
  --outfile D:\grey-liquid-models\phi-4\phi-4-f16.gguf `
  --outtype f16

# Qwen 2.5-7B
python convert_hf_to_gguf.py D:\grey-liquid-models\qwen2.5-7b\ `
  --outfile D:\grey-liquid-models\qwen2.5-7b\qwen2.5-7b-f16.gguf `
  --outtype f16

# Mistral-Small
python convert_hf_to_gguf.py D:\grey-liquid-models\mistral-small\ `
  --outfile D:\grey-liquid-models\mistral-small\mistral-small-f16.gguf `
  --outtype f16
```

---

### Phase 3: Quantize to Q2_K

Use llama-quantize to create Q2_K versions:

```powershell
# Phi-4
D:\llama.cpp\llama-quantize.exe `
  D:\grey-liquid-models\phi-4\phi-4-f16.gguf `
  D:\grey-liquid-models\phi-4\phi-4-Q2_K.gguf Q2_K

# Qwen 2.5-7B
D:\llama.cpp\llama-quantize.exe `
  D:\grey-liquid-models\qwen2.5-7b\qwen2.5-7b-f16.gguf `
  D:\grey-liquid-models\qwen2.5-7b\qwen2.5-7b-Q2_K.gguf Q2_K

# Mistral-Small
D:\llama.cpp\llama-quantize.exe `
  D:\grey-liquid-models\mistral-small\mistral-small-f16.gguf `
  D:\grey-liquid-models\mistral-small\mistral-small-Q2_K.gguf Q2_K
```

---

### Phase 4: Test Coherence

Load each Q2_K model in Ollama and test with standard prompts:

**Test Prompts:**
1. **Simple:** "What is 2+2?"
2. **Moderate:** "Explain the water cycle in 3 sentences."
3. **Complex:** "Write a Python function that finds prime numbers using the Sieve of Eratosthenes algorithm."

**Success Criteria:**
- ✅ **Pass:** Model loads, responds coherently, answers correctly
- ⚠️ **Degraded:** Model loads but shows quality degradation (nonsense, repetition)
- ❌ **Fail:** Model crashes, hangs, or produces complete garbage

---

## Data Collection

For each model, record:
- **Quantization Success:** Did Q2_K quantization complete?
- **Model Load:** Does Ollama load the Q2_K model?
- **Prompt 1 Response:** Output + coherence rating (1-5)
- **Prompt 2 Response:** Output + coherence rating (1-5)
- **Prompt 3 Response:** Output + coherence rating (1-5)
- **Final bpw:** Actual bits-per-weight achieved
- **File Size:** Q2_K model size in GB

---

## Expected Outcomes

### Scenario A: Universal Barrier (H1 True)
**All models fail at Q2_K** → Sub-3-bit is fundamentally limited by quantization method, not architecture

**Implications:**
- Current quantization algorithms are the bottleneck
- Need new quantization methods (mixed-precision, importance weighting)
- Grey Liquid Micro must target 3.0-3.5 bpw range

---

### Scenario B: Architecture-Specific (H2 True)
**Some models pass, others fail** → Certain architectural features enable sub-3-bit

**Implications:**
- Identify which features matter (attention mechanism, layer structure, etc.)
- Grey Liquid Micro research: Can we retrofit these features into Gemma 4?
- Future model selection: Prioritize architectures with sub-3-bit capability

---

## Timeline

- **Phase 1:** ~1-2 hours (downloads)
- **Phase 2:** ~30 min per model (conversion)
- **Phase 3:** ~15 min per model (quantization)
- **Phase 4:** ~15 min per model (testing)

**Total:** ~4-5 hours for complete experiment

---

## Next Steps After Results

1. Write GREY_LIQUID_REPORT_006.md with findings
2. Update grey-liquid.html website with Experiment #6
3. If H2 (architecture-specific):
   - Research successful architectures in depth
   - Plan follow-up: Gemma 4 modifications for sub-3-bit
4. If H1 (universal):
   - Shift focus to 3.0-3.5 bpw optimization (Q3_K variants)
   - Document "practical floor" for Grey Liquid Micro

---

**Status:** Phase 1 in progress (downloads running)  
**Next Action:** Wait for downloads, then begin Phase 2 conversions
