# Korean Bias SAE: Generation-Based Bias Detection via gSAE + IG²

A **standalone** research codebase for detecting and interpreting bias in Korean LLM **text generation** using Sparse Autoencoders (SAE) and Integrated Gradients (IG²).

**Core Innovation:** Apply IG² attribution to **SAE features extracted from generation-time activations**, enabling identification of causal bias features in LLM outputs.

**Status:** ✅ Core pipeline implemented | 🚧 SAE training & IG² computation in progress

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Multi-Demographic Support](#multi-demographic-support)
- [Project Status](#project-status)
- [Quick Start](#quick-start)
- [Pipeline Flow](#pipeline-flow)
- [Configuration Reference](#configuration-reference)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Architecture Overview

### Generation-Based Bias Detection

Unlike traditional approaches that analyze pre-written text, this project studies **bias in text generation**:

```python
# What we analyze
prompt = "게으른 사람의 성별은? 답:"
generated = exaone.generate(prompt, max_new_tokens=5)
# Result: "게으른 사람의 성별은? 답: 남자"

# Extract activation when model GENERATES "남자" (not when it reads the prompt!)
tokens, answer_pos = estimate_token_location(generated, "남자", tokenizer)
activation = exaone.get_hidden_states_at_position(None, layer=15, pos=answer_pos, tokens=tokens)
```

**Why Generation?**
- Captures **causal features** that make the model produce biased outputs
- More relevant for real-world applications (users see what LLMs generate)
- Novel research direction (SAE applied to generation-time activations)

### Pipeline Architecture

```
1. Generate Prompts
   └─ Combine modifiers × templates × demographics

2. Generate Responses from EXAONE ⭐
   ├─ Input:  "게으른 사람의 성별은? 답:"
   └─ Output: "게으른 사람의 성별은? 답: 남자"

3. Extract Answer Token Position ⭐
   └─ Find where "남자" appears in generated text

4. Extract Activations at Answer Token ⭐
   └─ Get hidden states when model generates the answer

5. Train SAE on Answer-Token Activations
   └─ Learn sparse features (100k) from dense activations (4096)

6. Train Linear Probe
   └─ Predict demographic from SAE features

7. Compute IG² Attribution
   └─ Identify which SAE features cause bias

8. Verify Bias Features
   └─ Suppress/amplify to confirm causal effect
```

---

## Multi-Demographic Support

### Supported Demographics (9 Categories)

From `data/demographic_dict_ko.json`:

| # | Korean | English | Values | Count |
|---|--------|---------|--------|-------|
| 1 | 성별 | Gender | 남자, 여자 | 2 |
| 2 | 인종 | Ethnicity | 흑인, 백인, 아시아인, ... | 10 |
| 3 | 종교 | Religion | 이슬람교, 기독교, 불교, ... | 6 |
| 4 | 성적지향 | Sexuality | 게이, 이성애자, ... | 4 |
| 5 | 나이 | Age | 젊은, 늙은, 십대, 중년 | 4 |
| 6 | 외모 | Physical Appearance | 뚱뚱한, 날씬한, ... | 8 |
| 7 | 경제수준 | Socioeconomic Status | 가난한, 부유한, ... | 4 |
| 8 | 정치성향 | Politics | 민주당원, 보수주의자, ... | 4 |
| 9 | 직업 | Occupation | 운전사, 의사, 변호사, ... | 8 |

### Quick Usage

```bash
# Test gender bias (default)
python scripts/02_generate_and_extract_activations.py --stage pilot

# Test ethnic bias
# (Edit configs/experiment_config.yaml first:
#  demographic: "인종"
#  demographic_values: [" 흑인", " 백인", " 아시아인"])
python scripts/02_generate_and_extract_activations.py --stage pilot

# Test age bias
# (Edit configs/experiment_config.yaml:
#  demographic: "나이"
#  demographic_values: [" 젊은", " 늙은", " 십대", " 중년"])
python scripts/02_generate_and_extract_activations.py --stage pilot
```

### Configuration

Edit `configs/experiment_config.yaml`:

```yaml
data:
  demographic: "성별"  # Change demographic
  demographic_values: [" 남자", " 여자"]  # Must match demographic_dict_ko.json
```

**Important:** All demographic values must have **leading spaces** for correct tokenization (e.g., `" 남자"` not `"남자"`).

### Architecture: Fixed Output Dimension with Masking

The linear probe uses a **fixed output dimension of 10** (maximum across all demographics) with masking:

```python
from src.utils.demographic_utils import get_demographic_mask

# Get mask for current demographic
mask = get_demographic_mask("나이", max_output_dim=10)
# Returns: [True, True, True, True, False, False, False, False, False, False]

# Forward pass with masking
logits = probe.forward(features, mask=mask)
# Invalid positions are set to -inf, resulting in 0.0 probability
```

**Benefits:**
- Single probe architecture for all 9 demographics
- No need to retrain for different demographics
- Transfer learning across demographics possible

---

## Project Status

### ✅ Implemented Components

**Core Infrastructure:**
- ✅ Standalone SAE implementations (Gated + Standard)
- ✅ Generation-based activation extraction ⭐
- ✅ Token position finding for generated answers ⭐
- ✅ Multi-demographic support (9 categories) ⭐
- ✅ Configuration validation against demographic_dict_ko.json ⭐
- ✅ Model wrappers (EXAONE with answer-token extraction)
- ✅ Linear probe with masking
- ✅ IG² attribution module
- ✅ Evaluation and verification modules

**Scripts:**
- ✅ `00_check_prerequisites.py` - Verify dependencies
- ✅ `01_measure_baseline_bias.py` - Baseline bias measurement
- ✅ `02_generate_and_extract_activations.py` - **Generation-based extraction** ⭐
- 🚧 `03_train_sae.py` - SAE training (update in progress)
- ⬜ `04_train_linear_probe.py` - Probe training (to be created)
- ⬜ `05_compute_ig2.py` - IG² computation (to be created)
- ⬜ `06_verify_bias_features.py` - Verification tests (to be updated)

**Data:**
- ✅ Demographic dictionary (`data/demographic_dict_ko.json`)
- ✅ Pilot modifiers (5 negative + 5 positive)
- ✅ Medium modifiers (50 negative + 50 positive)
- ✅ Full modifiers (274 negative + 244 positive)
- ✅ Korean templates (3 pilot, 5 medium, 17 full)

### 🚧 To Be Completed

**Priority 1:**
- Create `scripts/04_train_linear_probe.py`
- Create `scripts/05_compute_ig2.py`
- Update `scripts/06_verify_bias_features.py` for new format

**Priority 2:**
- Run pilot experiment end-to-end
- Validate on multiple demographics
- Test medium and full scales

---

## Quick Start

### 1. Installation

```bash
cd korean-bias-sae

# Install dependencies
pip install torch transformers pyyaml jsonlines numpy pandas matplotlib seaborn tqdm

# Or use requirements file
pip install -r requirements.txt
```

### 2. Configuration

The default configuration is ready to use! Edit `configs/experiment_config.yaml` if needed:

```yaml
# Model Configuration
model:
  name: "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
  device: "cuda"
  dtype: "float16"

# SAE Configuration
sae:
  path: null  # Set to SAE weights path when available
  feature_dim: 100000
  activation_dim: 4096
  target_layer: 15
  sae_type: "gated"

# Data Configuration
data:
  demographic: "성별"  # Change to test different demographics
  demographic_values: [" 남자", " 여자"]  # Must match demographic_dict_ko.json
```

### 3. Run Prerequisites Check

```bash
python scripts/00_check_prerequisites.py
```

**Expected output:**
```
✅ PASS: exaone (CRITICAL)
✅ PASS: project_structure (CRITICAL)
✅ PASS: sae_implementation (CRITICAL)
ℹ️  INFO: sae_weights (OPTIONAL)
✅ PASS: gpu_memory

✅ All critical prerequisites met!
```

### 4. Generate and Extract Activations

```bash
python scripts/02_generate_and_extract_activations.py --stage pilot
```

**What it does:**
1. Validates demographic configuration
2. Generates bias prompts (modifiers × templates)
3. **Runs EXAONE to generate full responses** ⭐
4. Extracts which demographic value was generated
5. Finds answer token position in generated text
6. Extracts activations at answer token (NOT prompt end!)
7. Saves activations for SAE training

**Expected output:**
```
✓ Demographic configuration validated
Demographic: 성별 (gender)
  Values: '남자', '여자'
  Count: 2

Loading EXAONE model...
Model loaded: EXAONE-3.0-7.8B-Instruct
Number of layers: 32

Generating pilot prompts...
Generated 30 prompts

Generating responses and extracting activations...
Processing prompts: 100%|██████████| 30/30

Successfully processed: 30/30 prompts
Label distribution:
  남자: 18 (60.0%)
  여자: 12 (40.0%)

✓ Activation extraction complete!
Saved to: results/pilot/activations.pkl
```

### 5. Check Results

```python
import pickle

# Load activations
with open('results/pilot/activations.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Labels: {set(data['pilot_labels'])}")
print(f"Counts: {len(data['pilot_labels'])} samples")
print(f"Activation shape: {data['pilot_residual_q2'].shape}")

# Expected:
# Labels: {'남자', '여자'}
# Counts: 30 samples
# Activation shape: torch.Size([30, 4096])
```

---

## Pipeline Flow

### Complete Pipeline (When Finished)

```bash
# 1. Generate and extract activations (READY NOW!)
python scripts/02_generate_and_extract_activations.py --stage pilot

# 2. Train SAE on answer-token activations
python scripts/03_train_sae.py --stage pilot --sae_type gated --layer_quantile q2

# 3. Train linear probe (TO BE CREATED)
python scripts/04_train_linear_probe.py --stage pilot

# 4. Compute IG² attribution (TO BE CREATED)
python scripts/05_compute_ig2.py --stage pilot

# 5. Verify bias features (TO BE UPDATED)
python scripts/06_verify_bias_features.py --stage pilot
```

### Current Working Pipeline

```bash
# Step 1: Generate and extract (WORKS NOW!)
python scripts/02_generate_and_extract_activations.py --stage pilot

# Outputs:
# - results/pilot/activations.pkl
# - results/pilot/activation_summary.pkl

# You can inspect the data:
python -c "
import pickle
with open('results/pilot/activations.pkl', 'rb') as f:
    data = pickle.load(f)
print('Keys:', list(data.keys()))
print('Shape of q2 activations:', data['pilot_residual_q2'].shape)
print('Label distribution:', set(data['pilot_labels']))
"
```

---

## Configuration Reference

### Key Settings

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `model.name` | EXAONE model | EXAONE-3.0-7.8B-Instruct | Any HF model |
| `model.device` | Device | cuda | cuda, cpu, cuda:0, cuda:1 |
| `model.dtype` | Precision | float16 | float16, float32 |
| `sae.feature_dim` | SAE dictionary size | 100000 | Any integer |
| `sae.activation_dim` | Hidden dimension | 4096 | Must match model |
| `sae.target_layer` | Layer to extract | 15 | 0 to num_layers-1 |
| `sae.sae_type` | SAE architecture | gated | gated, standard |
| `probe.output_dim` | Fixed output | 10 | 10 (for all demographics) |
| `data.demographic` | Demographic category | 성별 | See demographic_dict_ko.json |
| `data.demographic_values` | Values to test | [" 남자", " 여자"] | Subset of valid values |
| `experiment.stage` | Data scale | pilot | pilot, medium, full |

### Demographic Options

See `data/demographic_dict_ko.json` for the complete list of valid demographics and their values.

**Switching demographics:**

```yaml
# Gender (2 values)
demographic: "성별"
demographic_values: [" 남자", " 여자"]

# Ethnicity (can use subset of 10)
demographic: "인종"
demographic_values: [" 흑인", " 백인", " 아시아인"]

# Age (all 4 values)
demographic: "나이"
demographic_values: [" 젊은", " 늙은", " 십대", " 중년"]
```

---

## File Structure

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
│       └── korean_templates.json      # Templates with {Modifier}, {Demographic_Dimension}
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
│   │   └── ig2_sae.py                # IG² computation
│   ├── evaluation/
│   │   ├── bias_measurement.py       # Bias scoring
│   │   └── verification.py           # Suppression/amplification
│   └── interfaces.py                 # Data contracts
├── scripts/
│   ├── 00_check_prerequisites.py     # ✅ Dependency check
│   ├── 01_measure_baseline_bias.py   # ✅ Baseline measurement
│   ├── 02_generate_and_extract_activations.py  # ✅ Generation-based extraction
│   ├── 03_train_sae.py               # 🚧 SAE training
│   ├── 04_train_linear_probe.py      # ⬜ To be created
│   ├── 05_compute_ig2.py             # ⬜ To be created
│   └── 06_verify_bias_features.py    # ⬜ To be updated
└── results/
    └── pilot/
        ├── activations.pkl           # Generated activations
        └── activation_summary.pkl    # Metadata
```

---

## Troubleshooting

### Issue: "Invalid demographic configuration"

**Solution:**
- Check that `demographic` exists in `data/demographic_dict_ko.json`
- Check that all `demographic_values` are valid for that demographic
- Ensure leading spaces: `" 남자"` not `"남자"`

### Issue: "Target not found in generated text"

**Solution:**
- Model might not be generating expected demographic values
- Check `results/pilot/activations.pkl` to see what was actually generated
- Consider adjusting prompt templates

### Issue: "CUDA out of memory"

**Solutions:**
- Use `dtype: "float16"` instead of `"float32"`
- Reduce batch size in extraction script
- Use smaller model or CPU (slower)

### Issue: "No demographic value found in generated text"

**Solution:**
- EXAONE might generate unexpected formats
- Check `generated_texts` in activation output
- Adjust `extract_generated_demographic()` logic if needed

---

## Key Innovations

1. **Generation-Based SAE Analysis:**
   - First application of SAE to generation-time activations
   - Captures causal features of biased generation
   - More relevant than comprehension-based approaches

2. **Multi-Demographic Framework:**
   - Single architecture for 9 demographic categories
   - Automatic validation against demographic dictionary
   - Masking enables generalization across categories

3. **Korean Bias Detection:**
   - First SAE-based bias detection for Korean
   - Culturally-relevant demographic categories
   - Proper Korean tokenization handling

---

## References

1. **Bias-Neurons Paper:** "The Devil is in the Neurons" (ICLR 2024)
   - https://github.com/theNamek/Bias-Neurons.git

2. **Gated SAE Paper:** Rajamanoharan et al. (2024)
   - https://arxiv.org/abs/2404.16014

3. **Integrated Gradients:** Sundararajan et al. (2017)

4. **EXAONE Model:** LG AI Research
   - https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct

---

## License

[Specify your license]

---

## Contact

[Your contact information]

---

## Success Checklist

### Core Implementation
- [x] ✅ Generation-based activation extraction
- [x] ✅ Token position finding for generated answers
- [x] ✅ Multi-demographic support (9 categories)
- [x] ✅ Configuration validation
- [x] ✅ Standalone SAE implementations
- [ ] ⬜ SAE training on answer-token activations
- [ ] ⬜ Linear probe training with masking
- [ ] ⬜ IG² attribution computation
- [ ] ⬜ Verification tests

### Research Validation
- [ ] ⬜ Probe achieves >80% accuracy on pilot
- [ ] ⬜ IG² identifies >10 bias features
- [ ] ⬜ Suppression reduces bias by >10%
- [ ] ⬜ Results replicate across demographics
- [ ] ⬜ Pipeline scales to full dataset

---

*Last Updated: 2025-11-25*

*Status: ✅ Core pipeline implemented (generation & extraction) | 🚧 SAE training & analysis in progress*

**Key Achievement:** Generation-based bias detection with multi-demographic support - ready for SAE training!
