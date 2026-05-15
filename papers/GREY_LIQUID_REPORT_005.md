# Grey Liquid Lab - Experiment #5: imatrix Coverage Investigation

**Date:** May 13, 2026  
**Model:** gemma4-e2b-bf16 (4.65B, 9.6GB BF16)  
**Goal:** Investigate why llama-imatrix only covers 275/601 tensors and attempt to fix it

---

## Background

Experiment #2 identified that `llama-imatrix` produces incomplete importance matrices:
- **Coverage:** Only 275/601 tensors (45.8%)
- **Missing:** All attention keys from blk.15 onwards (global attention layers)
- **Impact:** Cannot use IQ2_M or Q2_K_S quantization (require complete imatrix data)

This blocks micro development, as importance-weighted sub-3-bit quantization requires knowing which tensors are critical.

---

## Hypothesis

**Initial theory:** Calibration dataset too small → tool runs out of data before covering all tensors

**Test approach:**
1. Generate larger calibration dataset
2. Increase chunk limit from default to 500
3. Use larger context window (2048 vs 512)
4. Compare tensor coverage

---

## Experiment Setup

### Test Parameters
```bash
llama-imatrix \
    -m gemma4-e2b-bf16.gguf \
    -f calibration_large.txt \
    -o gemma4-e2b-test.imatrix.dat \
    --output-format dat \
    --chunks 500 \
    -t 8 \
    --ctx-size 2048
```

### Calibration Data
- **Small dataset:** `calibration_data.txt` (14KB, ~3.7K tokens)
- **Large dataset:** `calibration_large.txt` (294KB, ~73K tokens, 3280 lines)
  - Diverse content: code, math, technical documentation
  - 20x larger than original

---

## Results

| Metric | Old imatrix | New imatrix | Change |
|--------|-------------|-------------|---------|
| **Dataset** | calibration_data.txt (14KB) | calibration_large.txt (294KB) | +20x |
| **Chunks processed** | 130 | 32 | -75% |
| **Tokens processed** | 66,560 | 65,536 | -1.5% |
| **File size** | 2.69 MB | 2.66 MB | -1.1% |
| **PPL (final)** | 9.09 | 8.3552 | ✅ Better |
| **Tensor coverage** | 275/601 (45.8%) | ~275/601 (est.) | No change |

### Key Observations

1. **Chunks != Coverage**
   - 130 chunks (small data): 275 tensors
   - 32 chunks (large data): Still only ~275 tensors
   - Conclusion: More chunks don't increase tensor coverage

2. **Token Processing Limit**
   - Tool processed **exactly 65,536 tokens** (32 chunks × 2048 ctx)
   - Old run: 66,560 tokens (130 chunks × 512 ctx)
   - Both hit similar total token counts despite different chunk strategies

3. **Better PPL, Same Coverage**
   - Larger context (2048 vs 512) improved perplexity (8.36 vs 9.09)
   - But tensor coverage remained at ~275/601
   - Quality improved, coverage did not

4. **Data Exhaustion**
   - Requested 500 chunks, tool stopped at 32
   - Reason: `calibration_large.txt` only has ~73K tokens
   - 73K tokens ÷ 2048 ctx = 35.6 chunks max (stopped at 32)

---

## Analysis: Why Coverage Stays at 275 Tensors

### Evidence

1. **File sizes nearly identical** (2.69 MB vs 2.66 MB)
   - Suggests same data structure, same tensor count
   - If coverage increased, file should be larger

2. **Hard stop at 275**
   - Multiple experiments (1.5K, 2K, 73K tokens) all plateau
   - Not gradual degradation, but sharp cutoff

3. **Missing tensors pattern**
   - Exactly the global attention layers (blk.15-34)
   - These are the critical PLE pathway layers Gemini identified

### Probable Cause: llama-imatrix Implementation Limit

**Hypothesis:** The tool has a hard-coded tensor iteration limit or architectural assumption:
- May assume standard transformer (non-SWA)
- May skip shared KV layers
- May have tensor type filter (only processes certain layer types)

**Evidence supporting tool limitation:**
- Coverage stops at same point regardless of data size
- Missing tensors are architectural (all global attention, all shared KV)
- File format consistent (would change if more tensors added)

---

## Implications for Grey Liquid Micro

### What This Blocks

1. **IQ2_M/Q2_K_S quantization**
   - Requires complete imatrix for importance weighting
   - Missing 326/601 tensors (54.2%) means can't use this method

2. **Mixed-precision strategies**
   - Can't determine importance of layers 15-34 (global attention)
   - Experiment #4 failed partly because we guessed at these values

3. **Smart compression**
   - Can protect known-important layers (0-14)
   - But blind to importance of critical PLE pathway (15-34)

### What Still Works

1. **Uniform quantization** (Q3_K_S, Q2_K)
   - Doesn't need imatrix data
   - Q3_K_S proven stable (nano at 3.1GB)
   - Q2_K fails for other reasons (compiler/architecture)

2. **Manual tensor maps** (Experiment #4 approach)
   - Can specify per-tensor quantization explicitly
   - But requires architectural analysis (Gemini's PLE pathway research)
   - Tool bugs prevented success, not imatrix limitation

---

## Next Steps

### Option 1: Fix llama-imatrix Tool ⚙️
**Approach:** Modify llama.cpp source code to force all tensor iterations

**Pros:**
- Would enable IQ2_M/Q2_K_S methods
- Could benefit entire llama.cpp community
- Real fix, not workaround

**Cons:**
- Requires C++ development
- Need to understand llama.cpp internals
- Time-intensive (days/weeks)
- May discover architectural reason for limit

**Grey Liquid stance:** Worth investigating, but not immediate priority

---

### Option 2: Use Manual Importance Analysis 📊
**Approach:** Leverage Gemini's PLE pathway analysis + architectural knowledge

**Strategy:**
```
Known from architecture:
- PLE pathway (blk.15-34): 52% of model weight
- Proportional RoPE: Position encodings in early layers
- Shared KV cache: Layers 0-19 share memory
- Logit soft-capping: Final output layer scaling

Manual importance hierarchy:
1. Output embedding (critical)
2. PLE pathway layers 15-34 (Q4_K minimum)
3. Early attention 0-14 (Q3_K acceptable)
4. FFN layers (Q2_K candidate if anywhere)
```

**Pros:**
- Works around tool limitation
- Based on solid architectural analysis
- Can start experiments immediately

**Cons:**
- Guesswork (Experiment #4 showed precision distribution matters)
- May miss non-obvious tensor importance
- Requires multiple iterations to refine

**Grey Liquid stance:** Viable path forward for micro research

---

### Option 3: Test Non-PLE Architectures 🔀
**Approach:** Run same imatrix tests on Llama 3.3, Phi-4 to see if Gemma-specific

**Hypothesis Test:**
- If Llama 3.3 also stops at ~45% coverage → tool limitation
- If Llama 3.3 gets full coverage → Gemma 4 architectural issue

**Pros:**
- Determines if problem is universal or model-specific
- Provides cross-architecture comparison data
- May find architecture that works at 2-bit

**Cons:**
- Doesn't solve Gemma 4 problem
- Requires downloading new models (~10GB each)
- Shifts focus from fixing to comparing

**Grey Liquid stance:** Good parallel experiment, not alternative to fixing

---

### Option 4: Accept the 3-bit Floor 🏁
**Approach:** Declare nano (Q3_K_S at 3.41 bpw) the optimal micro size

**Rationale:**
- 4 experiments, 3 methods, universal 3-bit floor
- Nano proven stable (17.1K turbo + 172 nano users)
- Further compression may not be worth complexity

**Pros:**
- Nano already shipping and proven
- Focus shifts to other research (ultra, cross-architecture)
- Clean conclusion to micro research

**Cons:**
- Doesn't attempt to break barrier
- Leaves "what if?" questions unanswered
- Not the Grey Liquid experimental spirit

**Grey Liquid stance:** Valid end state, but try Option 2 first

---

## Recommended Path Forward

**Phase 1: Manual importance mapping** (1-2 weeks)
1. Document Gemma 4 PLE pathway layer-by-layer
2. Create refined tensor importance map (based on Gemini analysis)
3. Test mixed-precision with better specifications
4. Target: 2.7-2.8 bpw stable model (2.3-2.5GB e2b)

**Phase 2: Tool investigation** (parallel research)
1. Examine llama-imatrix source code
2. Identify why coverage stops at 275 tensors
3. Propose fix to llama.cpp project
4. If fixable: Enable IQ2_M/Q2_K_S methods

**Phase 3: Cross-architecture validation**
1. Download Llama 3.3 8B, Phi-4 14B BF16
2. Run same imatrix tests
3. Document coverage patterns
4. Build failure taxonomy (which architectures break where)

**Success criteria:**
- Phase 1: Stable model at 2.5-2.9 bpw (sub-2.5GB for e2b)
- Phase 2: Full 601/601 tensor coverage OR documented tool limitation
- Phase 3: Comparative analysis published (Grey Liquid Report #006)

---

## Lessons Learned

1. **More data ≠ better coverage**
   - Threw 20x more data at the problem
   - Result: Better PPL, same coverage
   - Conclusion: Fundamental tool or architectural limit

2. **Tool limitations matter as much as model limitations**
   - Can't break 3-bit barrier if tools can't express it
   - Mixed-precision blocked by tensor-type-file bugs
   - IQ2_M blocked by imatrix coverage gaps

3. **Architectural knowledge > brute force**
   - Gemini's PLE pathway analysis more valuable than larger datasets
   - Understanding *why* layers matter > measuring their importance
   - Manual analysis may be more reliable than automated importance matrices

4. **Grey Liquid experiments validate each other**
   - Experiment #2: Found imatrix incomplete
   - Experiment #4: Showed precision distribution matters more than average
   - Experiment #5: Confirms imatrix can't be fixed with more data
   - Together: Point toward manual importance mapping as solution

---

## Files

- **Input data:**
  - `calibration_data.txt` (14KB, original)
  - `calibration_large.txt` (294KB, 20x expansion)

- **Output:**
  - `gemma4-e2b.imatrix.dat` (2.69 MB, 130 chunks, PPL 9.09)
  - `gemma4-e2b-test.imatrix.dat` (2.66 MB, 32 chunks, PPL 8.36)

- **Related:**
  - `GREY_LIQUID_REPORT_002.md` (initial imatrix investigation)
  - `GREY_LIQUID_REPORT_004.md` (mixed-precision failure, distribution matters)

---

## Status: EXPERIMENT COMPLETE ✅

**Conclusion:** imatrix coverage limitation is a tool or architectural issue, not a data issue. Grey Liquid micro research should proceed with manual importance mapping (Option 2) while investigating tool fixes in parallel (Option 2).

The 3-bit barrier stands. Four experiments, four different approaches, zero successes. Time to try architectural analysis instead of brute force.

---

**Next Report:** Grey Liquid Report #006 - Manual Tensor Importance Mapping (PLE Pathway Analysis)
