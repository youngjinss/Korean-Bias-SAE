# Korean Bias SAE

Bias detection in Korean LLM text generation using **Sparse Autoencoders (SAE)** and **IG² attribution**.

## Overview

This project identifies bias-encoding features in EXAONE model by:
1. Extracting activations when the model **generates** demographic answers
2. Training SAE to learn sparse feature representations
3. Using IG² attribution to identify which features encode bias
4. Verifying causality through suppression/amplification tests

---

## Quick Start

```bash
# 1. Check prerequisites
python scripts/00_check_prerequisites.py

# 2. Run full pipeline (all 9 demographics)
python scripts/run_pipeline.py --stage pilot --demographic all

# 3. View results in visualization notebooks
jupyter notebook notebooks/visualizations/
```

---

## Pipeline Scripts

### Step 0: Check Prerequisites
```bash
python scripts/00_check_prerequisites.py
```
Verifies PyTorch, CUDA, EXAONE model access, and project structure.

### Step 1: Measure Baseline Bias (Optional)
```bash
python scripts/01_measure_baseline_bias.py --stage pilot
```
Measures model's demographic prediction bias before intervention.

**Metric:** `Bias Score = P(max_demographic) - P(min_demographic)`

### Step 2: Generate and Extract Activations
```bash
python scripts/02_generate_and_extract_activations.py --stage pilot --demographic 성별
```
Generates responses and extracts hidden states at the **answer token position**.

**Key Innovation:** Extracts activation when model generates "남자", not when it reads the prompt.

### Step 3: Train SAE
```bash
python scripts/03_train_sae.py --stage pilot --sae_type gated --layer_quantile q2
```
Trains Gated Sparse Autoencoder on merged activations.

**Architecture:**
- Input: 4,096D (EXAONE hidden size)
- Features: 100,000D (sparse dictionary)
- Loss: `MSE + Sparsity_Loss + Auxiliary_Loss`

**Key Parameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `feature_dim` | 100,000 | SAE dictionary size |
| `sparsity_penalty` | 0.1 | L1/L0 sparsity weight |
| `total_steps` | 10K/50K/100K | pilot/medium/full |

### Step 4: Train Linear Probe
```bash
python scripts/04_train_linear_probe.py --stage pilot --demographic 성별 --layer_quantile q2
```
Trains demographic classifier on SAE features.

**Architecture:** `SAE Features (100K) → Linear → Logits (10)`

**Metric:** `Accuracy = correct_predictions / total_samples`

### Step 5: Compute IG² Attribution
```bash
python scripts/05_compute_ig2.py --stage pilot --demographic 성별 --layer_quantile q2
```
Computes Integrated Gradients Squared to identify bias features.

**Algorithm:**
```
IG²_gap = |IG²(demo1) - IG²(demo2)|

where IG²(demo) = feature × ∫₀¹ ∇f_demo(α × feature) dα
```

**Output:** IG² score for each of 100K features; higher = more bias-relevant.

### Step 6: Verify Bias Features
```bash
python scripts/06_verify_bias_features.py --stage pilot --demographic 성별 --layer_quantile q2
```

**Three Tests:**
| Test | Manipulation | Expected Result |
|------|--------------|-----------------|
| Suppression | Set bias features to 0 | Gap decreases (bias reduced) |
| Amplification | Multiply bias features by 2 | Gap increases (bias amplified) |
| Random Control | Suppress random features | No significant change |

**Validation Criteria:**
- C1: Suppression change < 0 (bias reduced)
- C2: Amplification change > 0 (bias increased)
- C3: |Z-score| > 2 (statistically significant vs random)

---

## Visualization Notebooks

### 1. Layer & Demographic Comparison
`notebooks/visualizations/01_visualize_layer_demographic_comparison.ipynb`

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Probe Accuracy | `correct / total` | Higher = stronger bias signal |
| Max IG² Score | `max(IG²_scores)` | Peak feature importance |
| Top-K Mean | `mean(top_k_scores)` | Average importance of top features |

### 2. Bias Feature Verification
`notebooks/visualizations/02_visualize_bias_feature_verification.ipynb`

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Gap Change Ratio | `(gap_after - gap_before) / gap_before` | Negative = bias reduced |
| Z-Score | `(suppress_change - random_mean) / random_std` | \|Z\| > 2 = significant |

### 3. Important Neuron Analysis
`notebooks/visualizations/03_analyze_important_neurons.ipynb`

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Concentration Index | `sum(top_k) / sum(all)` | Higher = more concentrated bias |
| Jaccard Similarity | `\|A∩B\| / \|A∪B\|` | Feature overlap between demographics |
| Layer Change Ratio | `(L2 - L1) / L1` | How metrics change across layers |
| Percentile Threshold | `np.percentile(scores, p)` | Filter top (100-p)% features |

---

## Core Algorithms

### Gated SAE (Sparse Autoencoder)

```python
# Encoder with gating
gate = (W_gate @ x + b_gate) > 0  # Binary gate
magnitude = ReLU(W_mag @ x + b_mag)  # Feature magnitudes
features = gate * magnitude  # Sparse features

# Decoder
reconstruction = W_dec @ features + b_dec

# Loss
loss = MSE(x, reconstruction) + λ * sparsity(features) + aux_loss
```

### IG² (Integrated Gradients Squared)

```python
# For each demographic class separately
ig2_demo1 = torch.zeros(100000)
for step in range(num_steps):
    scaled = baseline + (features - baseline) * step / num_steps
    logits = probe(scaled)
    grads = torch.autograd.grad(logits[:, 0].sum(), scaled)
    ig2_demo1 += grads

ig2_demo1 *= features.mean(dim=0) / num_steps

# Compute gap between demographics
ig2_gap = torch.abs(ig2_demo1 - ig2_demo2)
```

### Bias Feature Verification

```python
# Suppression Test
suppressed_features = features.clone()
suppressed_features[:, bias_indices] = 0
gap_after = compute_logit_gap(suppressed_features)
change_ratio = (gap_after - gap_before) / gap_before  # Should be negative

# Amplification Test
amplified_features = features.clone()
amplified_features[:, bias_indices] *= 2
gap_after = compute_logit_gap(amplified_features)
change_ratio = (gap_after - gap_before) / gap_before  # Should be positive

# Z-Score (statistical significance)
z_score = (suppress_change - random_mean) / random_std  # |Z| > 2 = significant
```

---

## Demographics (9 Categories)

| Korean | English | Values |
|--------|---------|--------|
| 성별 | Gender | 남자, 여자 |
| 인종 | Ethnicity | 흑인, 백인, 아시아인, ... (10) |
| 종교 | Religion | 이슬람교, 기독교, 불교, ... (6) |
| 성적지향 | Sexuality | 게이, 이성애자, ... (4) |
| 나이 | Age | 젊은, 늙은, 십대, 중년 (4) |
| 외모 | Physical Appearance | 뚱뚱한, 날씬한, ... (8) |
| 경제수준 | Socioeconomic Status | 가난한, 부유한, ... (4) |
| 정치성향 | Politics | 민주당원, 보수주의자, ... (4) |
| 직업 | Occupation | 운전사, 의사, ... (8) |

---

## Experiment Scales

| Scale | Modifiers | Templates | Prompts | Use Case |
|-------|-----------|-----------|---------|----------|
| pilot | 10 | 3 | 30 | Quick testing |
| medium | 100 | 5 | 500 | Validation |
| full | 518 | 17 | 8,806 | Full analysis |

---

## Output Structure

```
korean-bias-sae/
├── README.md                          # This file
├── configs/
│   └── experiment_config.yaml         # Main configuration
├── data/
│   ├── demographic_dict_ko.json       # ⭐ Source of truth for demographics
│   ├── modifiers/
│   │   ├── pilot_negative_ko.json     # 5 negative modifiers
│   │   ├── pilot_positive_ko.json     # 5 positive modifiers
│   │   ├── medium_negative_ko.json    # 50 negative
│   │   ├── medium_positive_ko.json    # 50 positive
│   │   ├── full_negative_ko.json      # 274 negative
│   │   └── full_positive_ko.json      # 244 positive
│   └── templates/
│       └── korean_templates.json      # Templates with {Modifier}, 
├── src/
│   ├── models/
│   │   ├── exaone_wrapper.py         # ⭐ Answer-token extraction
│   │   ├── sae/                       # Standalone SAE implementations
│   │   │   ├── gated_sae.py          # Gated SAE
│   │   │   └── standard_sae.py       # Standard SAE
│   │   ├── sae_wrapper.py            # SAE interface
│   │   └── linear_probe.py           # BiasProbe with masking
│   ├── utils/
│   │   ├── token_position.py         # ⭐ Token finding in generated text
│   │   ├── demographic_utils.py      # ⭐ Multi-demographic utilities
│   │   ├── experiment_utils.py       # Experiment helpers
│   │   └── data_utils.py             # Data loading
│   ├── attribution/
│   │   └── ig2_sae.py                # ⭐ IG² computation (Bias-Neurons style)
│   ├── evaluation/
│   │   ├── bias_measurement.py       # Bias scoring
│   │   └── verification.py           # Suppression/amplification
│   ├── visualization/                 # ⭐ Visualization utilities
│   │   ├── __init__.py               # 40+ exported functions
│   │   ├── font_utils.py             # Korean font configuration
│   │   ├── data_loaders.py           # Load SAE features, IG², verification
│   │   ├── umap_utils.py             # UMAP dimensionality reduction
│   │   ├── feature_selection.py      # Top-k, TF-IDF, sparsity analysis
│   │   └── plotting_utils.py         # UMAP, IG², heatmaps, loss curves
│   └── interfaces.py                 # Data contracts
├── notebooks/
│   └── visualizations/                # ⭐ Visualization notebooks
│       ├── 01_visualize_layer_demographic_comparison.ipynb
│       ├── 02_visualize_bias_feature_verification.ipynb
│       ├── 03_analyze_important_neurons.ipynb
│       └── assets/                   # Output directory
├── scripts/
│   ├── run_pipeline.py               # ⭐ Master pipeline script (Python, supports --demographic all)
│   ├── 00_check_prerequisites.py     # ✅ Dependency check
│   ├── 01_measure_baseline_bias.py   # ✅ Baseline measurement
│   ├── 02_generate_and_extract_activations.py  # ✅ Generation-based extraction (--demographic)
│   ├── 03_train_sae.py               # ✅ SAE training (ONE shared SAE)
│   ├── 04_train_linear_probe.py      # ✅ Linear probe (--demographic for per-demographic probe)
│   ├── 05_compute_ig2.py             # ✅ IG² computation (--demographic for per-demographic)
│   ├── 06_verify_bias_features.py    # ✅ Bias verification (--demographic for per-demographic)
│   ├── merge_activations.py          # ✅ Merge multi-demographic activations for gSAE
│   └── generate_mock_data.py         # ✅ Mock data for visualization testing
└── results/
    ├── models/                        # SAE models
    │   └── sae-gated_{stage}_{layer}/     # Shared SAE model
    │   └── model.pth
    │
    └── pilot/
        ├── activations.pkl           # Merged activations (for SAE training)
        ├── activations_metadata.json # Multi-demographic sample indices
        │
        ├── {demographic}/            # Per-demographic results
        │   ├── activations.pkl       # Gender-only activations
        │   ├── probe/
        │   │   └── {layer}_linear_probe.pt
        │   ├── ig2/
        │   │   └── {layer}_ig2_results.pt
        │   └── verification/
        │       └── {layer}/
        │           ├── suppression_test.json
        │           ├── amplification_test.json
        │           └── random_control.json
```

---

## Key Findings (Example)

From `03_analyze_important_neurons.ipynb`:

**1. Feature Concentration:**
- Q1 (25%): Top-10 features capture ~70% of total IG² score
- Q2 (50%): Top-10 features capture ~55% of total IG² score
- Q3 (75%): Top-10 features capture ~46% of total IG² score

**2. Cross-Demographic Similarity:**
- Average Jaccard similarity ~0.5 across all layers
- Many "universal" bias features appear across 8+ demographics

**3. Universal Features:**
- 11-12 features appear in top-10 across 3+ demographics per layer
- These are high-priority targets for debiasing interventions

---

## References

1. **Bias-Neurons:** Liu, Yan, et al. "The devil is in the neurons: Interpreting and mitigating social biases in language models." The twelfth international conference on learning representations. 2024.
2. **Gated SAE:** Rajamanoharan et al. (2024) - https://arxiv.org/abs/2404.16014
3. **EXAONE:** LG AI Research - https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct
4. **Korean Sparse LLM Features** - https://github.com/leo-bpark/korean-sparse-llm-features-open 
