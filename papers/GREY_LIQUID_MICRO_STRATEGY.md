# Grey Liquid Lab: The G4 Micro Dual-Front Strategy

**Target:** Sub-2GB stable quantization of Gemma 4 E2B (4.5B parameters)  
**Challenge:** Standard 2-bit quantization fails due to compiler structural mismatch  
**Solution:** Mixed-precision quantization protecting critical architectural features

---

## The Dual-Front Problem

### Front 1: BF16 Source Architecture Blindspots

Gemma 4 introduced features that break standard uniform quantization:

1. **Logit Soft-Capping (30.0)**
   - Protects 128K context attention stability
   - Lives in: `blk.*.layer_output_scale.weight` (scalar per block)
   - Standard 2-bit rounding shears these scale factors → spiral into invalid ranges

2. **Per-Layer Embeddings (PLE) Pathway**
   - Parallel conditioning pathway modulating hidden states after attention
   - Key tensors:
     - `per_layer_token_embd.weight` (4480 MB - **52% of model size!**)
     - `per_layer_model_proj.weight` (26.25 MB)
     - `blk.*.inp_gate.weight` (0.75 MB per block × 35 = 26 MB)
     - `blk.*.proj.weight` (0.75 MB per block × 35 = 26 MB)
   - These matrices are incredibly dense
   - Aggressive 2-bit crunch breaks syntax processing immediately

### Front 2: Compiler Memory Alignment Failure

**Why Q2_K loads but hangs:**
- Standard compilers force weights into rigid **32-element blocks**
- At 2-bit, compiler deforms tensor metadata to fit binary block layout
- Runtime tries to map deformed tensor paths into RAM
- Hits dimension mismatch during dequantization
- Creates infinite loop (not a crash—a structural deadlock)

---

## The G4 Micro Mixed-Precision Pipeline

```
[ Raw Gemma 4 BF16 File (8.66 GB) ]
           │
           ▼
[ Custom Micro Compiler ]
           │
           ├─► Identify Protected Layers ──► Keep at Q3_K/Q4_K (3-4 bit)
           │   • PLE embeddings (per_layer_*)
           │   • Logit caps (layer_output_scale)
           │   • PLE gates (inp_gate, proj)
           │
           └─► Pack Remaining 85% ──► Drop to IQ2_M/IQ2_S (2.5-2.7 bit)
               • FFN matrices (ffn_gate, ffn_up, ffn_down)
               • Attention Q/K/V (attn_q, attn_k, attn_v)
               • Attention output (attn_output)
           │
           ▼
[ Stable G4 Micro Build ]
  • Average: ~2.2-2.4 bits per weight
  • E2B: Sub-2 GB (vs 3.1 GB nano)
  • 31B: 8-9 GB (clears 16GB VRAM line)
  • Loads correctly (no dimension mismatch)
```

---

## Layer Type Breakdown (E2B 4.5B Model)

### Total Tensors: 601

#### Protected Layers (Keep 3-4 bit):
| Layer Type | Count | Size Each | Total | % of Model | Reason |
|-----------|-------|-----------|-------|------------|--------|
| `per_layer_token_embd.weight` | 1 | 4480 MB | 4480 MB | 52% | PLE embeddings - critical density |
| `token_embd.weight` | 1 | 768 MB | 768 MB | 9% | Base embeddings |
| `per_layer_model_proj.weight` | 1 | 26.25 MB | 26 MB | 0.3% | PLE projection |
| `blk.*.inp_gate.weight` | 35 | 0.75 MB | 26 MB | 0.3% | PLE pathway gates |
| `blk.*.proj.weight` | 35 | 0.75 MB | 26 MB | 0.3% | PLE pathway projection |
| `blk.*.layer_output_scale.weight` | 35 | 0.0001 MB | 0.0035 MB | 0% | Logit soft-capping |
| All `*_norm.weight` tensors | ~150 | 0.001-0.006 MB | ~1 MB | 0.01% | Normalization (always F32) |

**Protected subtotal: ~5.3 GB (61% of model)**

#### Aggressive Quantization (2.5-2.7 bit):
| Layer Type | Count | Size Each | Total | % of Model | Can Compress |
|-----------|-------|-----------|-------|------------|-------------|
| `blk.*.ffn_down.weight` | 35 | 18 MB | 630 MB | 7% | Yes - feed-forward |
| `blk.*.ffn_gate.weight` | 35 | 18 MB | 630 MB | 7% | Yes - gating (less critical) |
| `blk.*.ffn_up.weight` | 35 | 18 MB | 630 MB | 7% | Yes - feed-forward |
| `blk.*.attn_q.weight` | 35 | 6 MB | 210 MB | 2.4% | Yes - queries less sensitive |
| `blk.*.attn_output.weight` | 35 | 6 MB | 210 MB | 2.4% | Yes - output projection |
| `blk.*.attn_k.weight` | 35 | 0.75 MB | 26 MB | 0.3% | Moderate - keys need accuracy |
| `blk.*.attn_v.weight` | 35 | 0.75 MB | 26 MB | 0.3% | Moderate - values need accuracy |

**Aggressive subtotal: ~2.4 GB (28% of model)**

---

## First Isolated Test Target

**Start with FFN gates: `blk.*.ffn_gate.weight`**

Why:
1. **Gating functions are resilient** - binary-like decisions (open/close pathways)
2. **High volume** - 35 layers × 18 MB = 630 MB (7% of model)
3. **Easy validation** - if these break, output becomes completely incoherent
4. **Isolated from PLE** - won't cascade into protected pathways

### Test Protocol:

```bash
# Step 1: Create mixed-precision quantization
llama-quantize \
  --imatrix gemma4-e2b-full.imatrix.dat \
  --exclude-weights "per_layer_*,*_embd.weight,*inp_gate*,*proj.weight,*output_scale*" \
  --tensor-type "blk.*.ffn_gate.weight:IQ2_S" \
  gemma4-e2b-bf16.gguf \
  gemma4-e2b-mixed-test1.gguf \
  Q3_K_S

# Step 2: Import to Ollama
ollama create gemma4-micro:test1 -f Modelfile.micro-test1

# Step 3: Run coherence tests
# - Basic math: "What is 127 + 384?"
# - Code gen: "Write a Python function to reverse a string"
# - Logic: "If all dogs are mammals and all mammals breathe, do dogs breathe?"
# - Context: Multi-turn conversation maintaining thread

# Step 4: Compare perplexity
llama-perplexity -m gemma4-e2b-mixed-test1.gguf -f calibration_large.txt
```

If FFN gates at 2.5-bit pass → expand to FFN up/down → then attention outputs → finally Q/K/V.

---

## Expected Results

### Baseline (Q3_K_S - Nano):
- Size: 3.1 GB
- PPL: ~6.5
- Status: ✅ Stable

### Target (Mixed 2.2-2.4 bpw - Micro):
- Size: 1.8-2.0 GB
- PPL: ~8-10 (acceptable degradation)
- Status: 🔬 Testing

### Ultimate Floor (Custom Compiler):
- Size: 1.5-1.8 GB
- Requires: Custom ternary logic + LUT compilation
- Timeline: Post-validation of mixed-precision approach

---

## Implementation Notes

1. **Metadata Patch Required:** Modify GGUF headers to signal mixed-precision to runtime
2. **Block Alignment Override:** Force compiler to use variable-width packing instead of rigid 32-blocks
3. **Validation Gates:** Each layer type must pass coherence tests before proceeding to next
4. **Documentation:** Every failure mode gets documented for white paper

---

**Status:** Waiting for full imatrix generation (287KB calibration data, ~150 chunks)  
**Next:** Run FFN gate isolation test as Experiment #2A  
**Goal:** Prove mixed-precision viability, document path to sub-2GB stable builds

---

*This is not a model quality problem. This is a compiler structural problem. We're fixing the machine, not the weights.*
