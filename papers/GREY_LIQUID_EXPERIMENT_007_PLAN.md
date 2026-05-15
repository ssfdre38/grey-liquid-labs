# Grey Liquid Lab — Experiment #7: Sub-3-Bit Architecture Feature Analysis
**Date:** May 14, 2026  
**Status:** 🔬 PLANNED  
**Goal:** Identify which architectural features cause Q2_K failure

---

## Research Question

**What specific architectural features determine Q2_K quantization success or failure?**

From Experiment #6, we know:
- ✅ **Qwen 2.5-7B**: Works at Q2_K (standard transformer)
- ✅ **Mistral-Small**: Works at Q2_K (Grouped-Query Attention)
- ❌ **Phi-4**: Fails at Q2_K (Phi architecture + SWA optimizations)
- ❌ **Gemma 4**: Fails at Q2_K (SWA + PLE + Shared KV)

## Hypotheses

### H1: Sliding Window Attention (SWA) is the culprit
- **Evidence for**: Both failed models (Gemma 4, Phi-4) use SWA or SWA-derived optimizations
- **Evidence against**: Mistral uses sliding window in Mistral 7B (but maybe not Mistral-Small?)
- **Test**: Check if Mistral-Small actually uses SWA; test Llama 3.1 (no SWA) as control

### H2: Shared KV caching propagates quantization errors
- **Evidence for**: Gemma 4 explicitly shares KV across layers
- **Evidence against**: Phi-4 may not use shared KV (need to verify)
- **Test**: Analyze model configs for KV sharing; test Llama (independent KV per layer)

### H3: Grouped-Query Attention (GQA) enables better Q2_K tolerance
- **Evidence for**: Mistral-Small uses 32/8 GQA and works; Qwen uses 32/32 (no GQA) and works
- **Evidence against**: Both GQA and non-GQA models work, so GQA might not matter
- **Test**: Compare attention head ratios across working vs. failed models

### H4: Vocabulary size provides error redundancy
- **Evidence for**: Mistral-Small has 131K vocab (largest) and works best
- **Evidence against**: Qwen has 152K vocab but showed more hallucination than Mistral
- **Test**: Check if embedding layer quantization quality differs

### H5: FFN/Attention weight distribution matters
- **Evidence for**: Phi-4 has unusual FFN size (17920 vs. typical 4x hidden)
- **Evidence against**: Mistral has massive FFN (32768) and works fine
- **Test**: Analyze weight tensor shapes and quantization error distribution

### H6: RoPE base frequency affects position encoding robustness
- **Evidence for**: Mistral uses 100M RoPE base (extreme long-context), Qwen uses 1M, Phi uses 250K
- **Evidence against**: No clear correlation with success/failure
- **Test**: Check if position encoding errors accumulate differently

---

## Methodology

### Phase 1: Architecture Deep Dive (Analysis)
1. **Extract model configurations from GGUF files**
   ```bash
   llama-ls --ctx-size 0 model.gguf | grep -E "attention|rope|kv|vocab"
   ```

2. **Compare architectural features in table**
   - Attention mechanism (standard, GQA, SWA)
   - KV caching (per-layer vs. shared)
   - RoPE configuration
   - FFN size ratios
   - Vocabulary size
   - Layer counts

3. **Identify architectural "red flags"**
   - Features present in ALL failed models
   - Features absent in ALL working models

### Phase 2: Controlled Testing (New Models)
Test edge cases to isolate variables:

1. **Llama 3.1 8B** (control group)
   - Standard transformer, no SWA, no shared KV
   - Expected: Should work at Q2_K (if H1/H2 correct)

2. **Mistral 7B v0.3** (SWA test)
   - Uses sliding window attention (8K window)
   - Expected: Should fail at Q2_K (if H1 correct)

3. **Gemma 2 9B** (predecessor test)
   - Predecessor to Gemma 4, less optimized
   - Expected: Might work if Gemma 4's specific optimizations cause failure

4. **Mixtral 8x7B** (MoE test)
   - Mixture of Experts with GQA
   - Expected: Should work (GQA + standard attention)

### Phase 3: Layer-Selective Quantization (Surgical Approach)
If SWA is the issue, try **hybrid quantization**:

1. **Quantize Phi-4 with mixed precision**
   - Attention layers: Q3_K (safer)
   - FFN layers: Q2_K (aggressive)
   - Output layer: Q4_K (critical)

2. **Quantize Gemma 4 with mixed precision**
   - SWA layers: Q3_K
   - Non-SWA layers: Q2_K
   - Embedding: Q4_K

3. **Test if hybrid approach restores stability**

### Phase 4: Weight Analysis (Technical Deep Dive)
Analyze quantization error distribution:

1. **Compare weight histograms before/after Q2_K**
   - Extract weight tensors from F16 and Q2_K models
   - Plot distribution per layer type (attention vs. FFN)

2. **Calculate per-layer quantization error**
   - Mean Squared Error (MSE) between F16 and Q2_K weights
   - Identify layers with highest error

3. **Correlate error with inference failure**
   - Do failed models have higher error in specific layer types?

---

## Expected Outcomes

### If H1 (SWA) is correct:
- Llama 3.1: ✅ Works at Q2_K
- Mistral 7B: ❌ Fails at Q2_K
- Hybrid Phi-4/Gemma 4: ✅ Works with Q3_K attention layers

### If H2 (Shared KV) is correct:
- Llama 3.1: ✅ Works (independent KV)
- Gemma 2: ✅ Works (if it doesn't use shared KV)
- Architecture configs show KV sharing in failed models only

### If H3 (GQA) is correct:
- Mixtral: ✅ Works (GQA)
- Models with higher KV head ratios work better
- BUT: Qwen works without GQA, so this might be a red herring

### If Multiple Factors:
- Combination of SWA + Shared KV + aggressive optimizations causes failure
- No single "silver bullet" feature

---

## Timeline

### Immediate (Tonight, if autopilot continues):
1. Download Llama 3.1 8B (~16 GB)
2. Quantize to Q2_K
3. Test coherence
4. Compare config with working models

### Short-term (Next session):
1. Download Mistral 7B, Gemma 2, Mixtral
2. Full Q2_K testing matrix
3. Document architectural differences

### Medium-term (Future experiments):
1. Implement hybrid quantization
2. Weight analysis with Python scripts
3. Write academic-style analysis paper

---

## Success Criteria

**Experiment succeeds if we can:**
1. Identify at least ONE architectural feature present in ALL failed models
2. Find at least ONE feature absent in ALL working models
3. Predict Q2_K success/failure for untested models with >80% accuracy

**Bonus success:**
- Successfully stabilize Phi-4 or Gemma 4 with hybrid quantization
- Publish architectural guidelines for sub-3-bit quantization

---

## Resources Needed

### Models to Download (~150 GB total):
- Llama 3.1 8B: ~16 GB (Meta)
- Mistral 7B v0.3: ~14 GB (Mistral AI)
- Gemma 2 9B: ~18 GB (Google)
- Mixtral 8x7B: ~90 GB (Mistral AI, optional - very large)

### Tools:
- `llama.cpp` (already built)
- `llama-ls` (for metadata inspection)
- Python scripts for weight analysis (need to write)

### Disk Space:
- Current usage: ~200 GB (source + GGUF + Q2_K from Exp #6)
- Additional needed: ~300 GB for new models
- Total required: ~500 GB (D:\ has 2.9 TB available, we're good)

---

## Next Steps

1. **Immediate**: Download Llama 3.1 8B as control test
2. **Document**: Extract configs from existing Q2_K models
3. **Analyze**: Create comparison table of architectural features
4. **Test**: Run Llama 3.1 through full Q2_K pipeline
5. **Report**: Update grey-liquid.html with findings

---

## Open Questions

1. Does Mistral-Small actually use sliding window attention?
2. What is Phi-4's exact attention mechanism implementation?
3. Are there other architectural features we haven't considered?
4. Can we mathematically model why SWA amplifies quantization errors?

---

**Experiment #7 will determine the root cause of sub-3-bit architecture sensitivity.**

*Plan created by Ash via Copilot CLI on May 14, 2026.*
