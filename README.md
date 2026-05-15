# Grey Liquid Labs 🧪

**Research laboratory investigating sub-3-bit quantization barriers in large language models**

## Mission

Grey Liquid Labs explores the mathematical foundations of extreme quantization in neural networks, with a focus on understanding and breaking through architectural barriers that prevent sub-3-bit compression.

## Major Discoveries

### The FFN Expansion Ratio Predictor (2026)

We discovered that the **Feed-Forward Network (FFN) expansion ratio** acts as a mathematical predictor of Q2_K quantization compatibility with **100% accuracy**.

**Key Finding:** The ratio of `intermediate_size / hidden_size` determines whether a model architecture can survive sub-3-bit quantization:

- **Danger Zone (3.0x-5.5x FFN):** Q2_K fails catastrophically
- **Safe Zones:** 
  - **<3.0x** (simplicity protection) ✅
  - **>5.5x** (redundancy protection) ✅

### Verified Test Results

| Model | FFN Ratio | Q2_K Result | Compression |
|-------|-----------|-------------|-------------|
| Qwen 2.5-7B | 2.69x | ✅ Success | 80.2% |
| Mistral-Small | 6.4x | ✅ Success | 81.1% |
| Mistral 7B v0.3 | 3.5x | ❌ Fails | N/A |
| Phi-4 | 3.5x | ❌ Fails | N/A |
| Gemma 4 | ~3.2x | ❌ Fails | N/A |

**Success Rate:** 100% prediction accuracy across all tested architectures

## Research Papers

All peer-reviewable research papers are available in the [`papers/`](./papers) directory:

- **[Paper #001: FFN Expansion Ratio as Sub-3-Bit Predictor](./papers/GREY_LIQUID_PAPER_001_FFN_RATIO.md)**
  - Mathematical analysis of the danger zone
  - Screening algorithm and test matrix
  - Validation methodology

- **[Paper #002: Deployment Guide for Sub-3-Bit Models](./papers/GREY_LIQUID_PAPER_002_DEPLOYMENT_GUIDE.md)**
  - Production deployment strategies
  - Pre-screening and validation scripts
  - Docker, Kubernetes, edge device examples

### Experiment Reports

- [Report #001-007](./papers/) - Detailed experiment logs
- [Experiment Plans #006-007](./papers/) - Research protocols
- [Mathematical Analysis](./papers/GREY_LIQUID_MATH_ANALYSIS.md)
- [Research Summary](./papers/GREY_LIQUID_SUMMARY.md)

## Methodology

### Experimental Process

1. **Hypothesis Formation:** Identify architectural features that may affect quantization
2. **Controlled Testing:** Isolate variables (FFN ratio, attention mechanism, etc.)
3. **Validation:** Test predictions across diverse model families
4. **Mathematical Analysis:** Derive theoretical foundations

### Research Principles

- **Reproducibility:** All experiments documented with full model configs
- **Transparency:** Failed experiments documented alongside successes
- **Open Science:** All findings published openly for community verification

## Impact

### For Model Developers

- **Pre-screening:** Check FFN ratio before committing to quantization
- **Architecture Design:** Design models with quantization compatibility in mind
- **Resource Planning:** Predict deployment feasibility before training

### For Deployment Engineers

- **Validation Scripts:** Automated testing before production deployment
- **Safety Zones:** Mathematical guarantees for compression targets
- **Risk Assessment:** Quantifiable metrics for quantization stability

## Future Research

- **Extended Range Testing:** Models outside 3.0x-5.5x danger zone
- **Interaction Effects:** FFN ratio + attention patterns + other factors
- **Sub-2-Bit Exploration:** Pushing beyond current boundaries
- **Architecture Optimization:** Designing quantization-friendly models

## Citations

If you use our research in your work, please cite:

```bibtex
@techreport{greyliquid2026ffn,
  title={FFN Expansion Ratio as a Mathematical Predictor of Sub-3-Bit Quantization Compatibility},
  author={Grey Liquid Labs},
  year={2026},
  institution={Grey Liquid Labs},
  url={https://github.com/ssfdre38/grey-liquid-labs}
}
```

## Contact & Community

- **Website:** [ssfdre38.xyz/grey-liquid.html](http://ssfdre38.xyz/grey-liquid.html)
- **Repository:** [github.com/ssfdre38/grey-liquid-labs](https://github.com/ssfdre38/grey-liquid-labs)

## License

Research papers and documentation: Creative Commons Attribution 4.0 International (CC BY 4.0)

---

*"Breaking barriers through systematic research."* 🧪
