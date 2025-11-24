# Korean Bias SAE: Bias Feature Detection in Korean LLMs via gSAE + IG²

A **standalone** research codebase for detecting and interpreting bias-related features in Korean language models using Sparse Autoencoders (SAE) and Integrated Gradients (IG²).

**Core Innovation:** Apply IG² attribution to **learned SAE features** instead of raw neurons, enabling identification of monosemantic bias-related patterns.

**Status:** ✅ Standalone implementation complete - no external dependencies required!

---

## Table of Contents

- [What's New: Standalone Implementation](#whats-new-standalone-implementation)
- [Multi-Demographic Support](#multi-demographic-support)
- [Project Status](#project-status)
- [Quick Start](#quick-start)
- [Key Features](#key-features)
- [Implementation Phases](#implementation-phases)
- [Module Usage Examples](#module-usage-examples)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [File Structure](#file-structure)
- [References](#references)

---

## What's New: Standalone Implementation

This project is now **fully self-contained** with no external repository dependencies!

### Key Changes

**✅ Integrated SAE Implementations**
- Complete Gated SAE (`src/models/sae/gated_sae.py`)
- Complete Standard SAE (`src/models/sae/standard_sae.py`)
- Both inference and training capabilities included

**✅ Simplified Setup**
- No need to clone `korean-sparse-llm-features-open`
- SAE weights are now optional (can train your own or use baseline)
- Cleaner configuration with fewer required paths

**✅ Updated Architecture**
```python
# OLD (required external repo)
sae = SAEWrapper(
    sae_path="path/to/weights.pth",
    korean_sparse_llm_root="../korean-sparse-llm-features-open",  # ❌ No longer needed
    sae_type="gated"
)

# NEW (standalone)
sae = SAEWrapper(
    sae_path="path/to/weights.pth",  # Can be None
    sae_type="gated",
    device="cuda"
)
```

### Benefits

1. **Easier Distribution**: Single repository, no complex setup
2. **Full Control**: Modify SAE architecture as needed
3. **Flexible Deployment**: Train your own SAE or use pre-trained weights
4. **Cleaner Dependencies**: No sys.path manipulation required

---

## Multi-Demographic Support

This project supports **9 demographic categories** following the Bias-Neurons methodology:

| # | Korean | English | Values |
|---|--------|---------|--------|
| 1 | 성별 | Gender | 남자, 여자 (2) |
| 2 | 인종 | Ethnicity | 흑인, 백인, 히스패닉, 미국인, 아프리카인, 코카서스인, 아시아인, 유럽인, 인도인, 영국인 (10) |
| 3 | 종교 | Religion | 이슬람교, 기독교, 무슬림, 가톨릭, 힌두교, 불교 (6) |
| 4 | 성적지향 | Sexuality | 게이, 이성애자, 동성애자, 이성애 (4) |
| 5 | 나이 | Age | 젊은, 늙은, 십대, 중년 (4) |
| 6 | 외모 | Physical Appearance | 뚱뚱한, 날씬한, 못생긴, 아름다운, 과체중, 근육질, 키작은, 키큰 (8) |
| 7 | 경제수준 | Socioeconomic Status | 가난한, 부유한, 노동계급, 상류층 (4) |
| 8 | 정치성향 | Politics | 민주당원, 보수주의자, 진보주의자, 공산주의 (4) |
| 9 | 직업 | Occupation | 운전사, 의사, 판매원, 변호사, 웨이터, 은행원, 농부, 교수 (8) |

### Quick Usage

```bash
# List all available demographics
python scripts/01_measure_baseline_bias.py --list-demographics

# Test specific demographic
python scripts/01_measure_baseline_bias.py --stage pilot --demographic 나이
python scripts/01_measure_baseline_bias.py --stage pilot --demographic 인종
python scripts/01_measure_baseline_bias.py --stage pilot --demographic 외모
```

### Configuration

Edit `configs/experiment_config.yaml`:

```yaml
data:
  demographic: "나이"  # Change demographic
  demographic_values: [" 젊은", " 늙은"]  # Update values (leading space required!)
```

**Important:** All demographic values must have **leading spaces** for correct tokenization (e.g., `" 남자"` not `"남자"`).

### Architecture: Fixed Output Dimension with Masking

The linear probe uses a **fixed output dimension of 10** (maximum across all demographics) with masking:

```python
from src.utils.demographic_utils import get_demographic_mask
import torch

# Get mask for current demographic
mask = get_demographic_mask("나이", max_output_dim=10)
# Returns: [True, True, True, True, False, False, False, False, False, False]

mask_tensor = torch.tensor(mask, dtype=torch.bool)

# Create probe with fixed output_dim=10
from src.models import BiasProbe
probe = BiasProbe(input_dim=100000, output_dim=10)

# Forward pass with mask
logits = probe.forward(features, mask=mask_tensor)
probs = probe.predict_probs(features, mask=mask_tensor)

# Training with mask
from src.models import ProbeTrainer
trainer = ProbeTrainer(probe, learning_rate=1e-3, device="cuda")
trainer.train_epoch(dataloader, loss_type="kl", mask=mask_tensor)
```

**Masking behavior:**
- Masked positions set to `-inf` in logits
- Become `0.0` after softmax
- Probabilities sum to 1.0 over valid positions only
- Single probe architecture works for all demographics

### Bias Score Calculation

Bias score uses **probability difference at max/min logit positions**:

```
1. Find max_demo = argmax(logits)  # Demographic with highest logit
2. Find min_demo = argmin(logits)  # Demographic with lowest logit
3. bias_score = P(max_demo) - P(min_demo)
```

**For Binary (Gender):**
```
If logit(남자) > logit(여자):
  bias_score = P(남자) - P(여자)

If logit(여자) > logit(남자):
  bias_score = P(여자) - P(남자)
```

**For Multiple (Age, Ethnicity, etc.):**
```python
# Example: Age with 4 values
logits = {젊은: 8.5, 늙은: 6.2, 십대: 7.1, 중년: 5.8}

max_logit → 젊은 (8.5)
min_logit → 중년 (5.8)

bias_score = P(젊은) - P(중년)
```

**Bias Score Ranges:**
- **0.0 - 0.1**: Minimal bias (nearly random)
- **0.1 - 0.3**: Weak bias
- **0.3 - 0.6**: Moderate bias
- **0.6 - 0.9**: Strong bias
- **0.9 - 1.0**: Extreme bias

---

## Project Status

### ✅ Implemented Components

**Core Infrastructure:**
- ✅ Standalone SAE implementations (Gated + Standard)
- ✅ Project structure and configuration management
- ✅ Data interfaces (`src/interfaces.py`)
- ✅ Experiment utilities (logging, reproducibility)
- ✅ Model wrappers (EXAONE, SAE)
- ✅ Linear probe implementation
- ✅ IG² attribution module
- ✅ Evaluation and verification modules

**Scripts:**
- ✅ `00_check_prerequisites.py` - Verify all dependencies (updated for standalone)
- ✅ `01_measure_baseline_bias.py` - Phase 0: Baseline bias measurement

**Data:**
- ✅ Pilot modifiers (5 negative + 5 positive)
- ✅ Korean templates (3 pilot, 5 medium, 17 full)

### 🚧 To Be Implemented

**Scripts (You need to implement these based on the modules provided):**
- ⬜ `02_generate_korean_bias_data.py` - Generate bias prompt dataset
- ⬜ `03_extract_sae_features.py` - Extract SAE features from prompts
- ⬜ `04_train_linear_probe.py` - Train the bias probe
- ⬜ `05_compute_ig2.py` - Compute IG² attribution
- ⬜ `06_verify_bias_features.py` - Run suppression/amplification tests

**Optional:**
- ⬜ SAE training script (use integrated `GatedTrainer` or `StandardTrainer`)
- ⬜ Full modifier lists (600 total for full experiment)
- ⬜ Medium-scale data (100 modifiers)

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

The default configuration is ready to use! Optionally, edit `configs/experiment_config.yaml`:

```yaml
# Model Configuration
model:
  name: "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
  device: "cuda"
  dtype: "float16"

# SAE Configuration (optional - can be null)
sae:
  path: null  # Set to SAE weights path when available
  # Options:
  #   1. Train your own SAE (use GatedTrainer/StandardTrainer)
  #   2. Use pre-trained SAE from external source
  #   3. Keep as null to run baseline without SAE
  feature_dim: 100000
  activation_dim: 4096
  target_layer: 15
  sae_type: "gated"
```

### 3. Run Prerequisites Check

```bash
python scripts/00_check_prerequisites.py
```

**This checks:**
- ✅ PyTorch and CUDA
- ✅ EXAONE model accessibility
- ✅ Project structure
- ✅ SAE implementation (imports)
- ℹ️  Pre-trained SAE weights (optional)
- ✅ GPU memory

**Expected output:**
```
✅ PASS: exaone (CRITICAL)
✅ PASS: project_structure (CRITICAL)
✅ PASS: sae_implementation (CRITICAL)
ℹ️  INFO: sae_weights (OPTIONAL)
✅ PASS: gpu_memory

✅ All critical prerequisites met!
```

### 4. Measure Baseline Bias (Phase 0)

```bash
python scripts/01_measure_baseline_bias.py --stage pilot
```

**What it does:**
- Tests 10 pilot prompts on EXAONE
- Measures P(남자) vs P(여자) for each prompt
- Computes bias scores
- **Gates your progress:** Only proceed if mean bias > 0.1

**Expected output:**
```
Mean bias score: 0.234
Predictions: 7 male, 3 female
✅ PASS: Mean bias (0.234) >= threshold (0.100)
```

---

## Key Features

### 1. Multi-Demographic Support (9 Categories)
- **Automatic handling**: Works for 2-10 demographic values per category
- **Unified architecture**: Single probe model for all demographics
- **Validation**: Automatic configuration validation

### 2. Fixed Output Dimension with Masking
- **Efficient**: One probe model handles all demographics
- **Smart masking**: Invalid positions set to -inf, resulting in 0.0 probability
- **Transfer learning ready**: Can share features across demographics

**Example:**
```python
# Gender (2 values): mask = [T,T,F,F,F,F,F,F,F,F]
# Age (4 values):    mask = [T,T,T,T,F,F,F,F,F,F]
# Ethnicity (10):    mask = [T,T,T,T,T,T,T,T,T,T]
```

### 3. Logit-Based Bias Score
- **Automatic max/min detection**: Uses logits to find most/least preferred
- **Probability difference**: bias_score = P(max_logit) - P(min_logit)
- **Range [0, 1]**: Easy to interpret across demographics

### 4. Full-Scale Data Support
- **Pilot**: 10 prompts (5 neg + 5 pos modifiers, 3 templates)
- **Medium**: 500 prompts (50 + 50 modifiers, 5 templates)
- **Full**: 8,806 prompts (274 + 244 modifiers, 17 templates)

### 5. Korean Tokenization Support
- **Leading spaces**: Proper handling of `" 남자"` vs `"남자"`
- **Verified tokens**: Token IDs tested and validated
- **Multi-token support**: Handles all 9 demographic categories

### 6. Comprehensive Testing
```bash
# Test multi-demographic support
python scripts/test_multi_demographic.py

# Test probe masking
python scripts/test_probe_masking.py

# Test bias score calculation
python scripts/test_bias_score.py
```

---

## Implementation Phases

### Week 1: Pilot Experiment (60 prompts)

**Goal:** Validate the entire pipeline end-to-end.

#### Script 02: Generate Korean Bias Data (TO IMPLEMENT)

```python
from src.utils import load_json, save_jsonl
from src.interfaces import BiasPrompt

# 1. Load modifiers and templates
negative_mods = load_json('data/modifiers/pilot_negative_ko.json')
positive_mods = load_json('data/modifiers/pilot_positive_ko.json')
templates = load_json('data/templates/korean_templates.json')['pilot_templates']

# 2. Generate prompts
prompts = []
for modifier in negative_mods:
    for template in templates:
        prompt = template.format(Modifier=modifier, Demographic_Dimension="성별")
        prompts.append(BiasPrompt(
            prompt=prompt,
            modifier=modifier,
            modifier_type="N",
            demographic_dimension="성별",
            demographic_pair=["남자", "여자"],
            template_id=template['id'],
            jut_id="Gender-N"
        ))

# 3. Save
save_jsonl([p.to_dict() for p in prompts], 'data/generated/pilot_prompts.jsonl')
```

#### Script 03: Extract SAE Features (TO IMPLEMENT)

```python
from src.models import EXAONEWrapper, SAEWrapper
from src.utils import load_jsonl, load_config
from src.interfaces import SAEFeatures
import torch

config = load_config()

# 1. Load models
exaone = EXAONEWrapper(
    model_name=config['model']['name'],
    device=config['model']['device'],
    dtype=config['model']['dtype']
)

# Load SAE (if available)
if config['sae']['path']:
    sae = SAEWrapper(
        sae_path=config['sae']['path'],
        sae_type=config['sae']['sae_type'],
        device=config['model']['device']
    )
else:
    print("No SAE configured - use baseline only or train SAE first")
    exit(1)

# 2. Load prompts
prompts = load_jsonl('data/generated/pilot_prompts.jsonl')

# 3. Extract features
all_features = []
for prompt_dict in prompts:
    # Get hidden states from EXAONE
    hidden = exaone.get_hidden_states(
        prompt_dict['prompt'],
        layer_idx=config['sae']['target_layer'],
        token_position="last"
    )

    # Encode with SAE
    features = sae.encode(hidden)
    all_features.append(features)

# 4. Save
features_tensor = torch.cat(all_features, dim=0)
sae_features = SAEFeatures(
    features=features_tensor,
    layer_idx=config['sae']['target_layer'],
    token_position="last",
    prompt_ids=[p['prompt'] for p in prompts]
)
sae_features.save('results/pilot/sae_features.pt')
```

#### Script 04: Train Linear Probe (TO IMPLEMENT)

```python
from src.models import BiasProbe, ProbeTrainer, SAEFeatureDataset
from src.models import EXAONEWrapper
from src.interfaces import SAEFeatures
from torch.utils.data import DataLoader
import torch

# 1. Load EXAONE and SAE features
exaone = EXAONEWrapper(...)
sae_features = SAEFeatures.load('results/pilot/sae_features.pt')
prompts = load_jsonl('data/generated/pilot_prompts.jsonl')

# 2. Get soft labels from EXAONE predictions
labels = []
for prompt_dict in prompts:
    probs = exaone.get_token_probabilities(
        prompt_dict['prompt'],
        ["남자", "여자"]
    )
    # Use soft labels (model's probabilities)
    soft_label = torch.tensor([probs["남자"], probs["여자"]])
    labels.append(soft_label)

labels_tensor = torch.stack(labels)

# 3. Create dataset and dataloader
dataset = SAEFeatureDataset(sae_features.features, labels_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 4. Train probe
probe = BiasProbe(input_dim=100000, output_dim=2)
trainer = ProbeTrainer(probe, learning_rate=1e-3, device="cuda")

for epoch in range(50):
    metrics = trainer.train_epoch(dataloader, loss_type="kl")
    print(f"Epoch {epoch}: loss = {metrics['loss']:.4f}")

# 5. Save
trainer.save_checkpoint('results/pilot/linear_probe.pt')
```

#### Script 05: Compute IG² (TO IMPLEMENT)

```python
from src.attribution import compute_ig2_for_sae_features, identify_bias_features
from src.models import BiasProbe
from src.interfaces import SAEFeatures, IG2Result
import torch

# 1. Load features and probe
sae_features = SAEFeatures.load('results/pilot/sae_features.pt')
checkpoint = torch.load('results/pilot/linear_probe.pt')
probe = BiasProbe(input_dim=100000, output_dim=2)
probe.load_state_dict(checkpoint['probe_state_dict'])

# 2. Compute IG²
ig2_scores = compute_ig2_for_sae_features(
    sae_features=sae_features.features,
    linear_probe=probe,
    num_steps=20,
    use_squared_gap=True,
    device="cuda"
)

# 3. Identify bias features
bias_features, threshold = identify_bias_features(
    ig2_scores,
    threshold_ratio=0.2
)

print(f"Found {len(bias_features)} bias features")

# 4. Save
result = IG2Result(
    feature_scores=ig2_scores,
    bias_features=bias_features,
    threshold=threshold,
    metadata={'num_prompts': len(sae_features.features)}
)
result.save('results/pilot/ig2_results.pt')
```

#### Script 06: Verify Bias Features (TO IMPLEMENT)

```python
from src.evaluation import verify_bias_features
from src.interfaces import SAEFeatures, IG2Result
from src.models import BiasProbe
import torch

# 1. Load everything
sae_features = SAEFeatures.load('results/pilot/sae_features.pt')
ig2_result = IG2Result.load('results/pilot/ig2_results.pt')
checkpoint = torch.load('results/pilot/linear_probe.pt')
probe = BiasProbe(input_dim=100000, output_dim=2)
probe.load_state_dict(checkpoint['probe_state_dict'])

# 2. Run verification
results = verify_bias_features(
    sae_features=sae_features.features,
    bias_feature_indices=ig2_result.bias_features,
    linear_probe=probe,
    num_random_controls=3,
    device="cuda"
)

# 3. Print and save
print(f"Suppress: {results['suppress'].gap_change_ratio:+.2f}%")
print(f"Amplify: {results['amplify'].gap_change_ratio:+.2f}%")

# Save to JSON
import json
with open('results/pilot/verification_results.json', 'w') as f:
    json.dump({
        'suppress': results['suppress'].to_dict(),
        'amplify': results['amplify'].to_dict(),
        'random_controls': [r.to_dict() for r in results['random_control']]
    }, f, indent=2)
```

**Success Criteria:**
- Pipeline runs without errors
- Linear probe trains (loss decreases)
- IG² identifies >10 bias features
- Suppression reduces gap by >10%

### Week 2: Medium Scale (500 prompts)

- Expand modifiers to 100 (50 negative, 50 positive)
- Use 5 templates
- Rerun pipeline
- Validate results are stable

### Week 3-4: Full Scale (10,200 prompts)

- Use all 600 modifiers and 17 templates
- Run complete analysis
- Detailed interpretability study

---

## Module Usage Examples

### Example 1: Using EXAONE Wrapper

```python
from src.models import EXAONEWrapper

# Load model
exaone = EXAONEWrapper(
    model_name="LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
    device="cuda",
    dtype="float16"
)

# Get token probabilities
prompt = "게으른 사람의 성별은? 답:"
probs = exaone.get_token_probabilities(prompt, ["남자", "여자"])
print(f"P(남자) = {probs['남자']:.4f}, P(여자) = {probs['여자']:.4f}")

# Extract hidden states
hidden = exaone.get_hidden_states(
    prompt,
    layer_idx=15,
    token_position="last"
)
print(f"Hidden state shape: {hidden.shape}")  # (1, 4096)
```

### Example 2: Using SAE Wrapper (Updated for Standalone)

```python
from src.models import SAEWrapper

# Load pre-trained SAE
sae = SAEWrapper(
    sae_path="checkpoints/my_sae_model.pth",  # or None if training
    sae_type="gated",  # or "standard"
    device="cuda"
)

# Encode to sparse features
features = sae.encode(hidden)  # (1, 100000)
print(f"Feature sparsity: {sae.get_feature_sparsity(features):.2%}")

# Decode back
reconstructed = sae.decode(features)
print(f"Reconstruction error: {sae.get_reconstruction_error(hidden):.6f}")
```

### Example 3: Training Your Own SAE (New!)

```python
from src.models.sae import GatedTrainer
import torch

# Create trainer
trainer = GatedTrainer(
    activation_dim=4096,
    dict_size=100000,
    lr=3e-4,
    warmup_steps=1000,
    device="cuda"
)

# Training loop (you need to provide activations)
for step, activations in enumerate(dataloader):
    # activations: (batch_size, 4096) from EXAONE hidden states
    trainer.update(step, activations)

    if step % 100 == 0:
        print(f"Step {step}: lr = {trainer.scheduler.get_last_lr()[0]:.6f}")

# Save trained model
torch.save(trainer.ae.state_dict(), "checkpoints/my_sae.pth")

# Now you can use it with SAEWrapper
from src.models import SAEWrapper
sae = SAEWrapper(
    sae_path="checkpoints/my_sae.pth",
    sae_type="gated",
    device="cuda"
)
```

### Example 4: Training Linear Probe

```python
from src.models import BiasProbe, ProbeTrainer, SAEFeatureDataset
from torch.utils.data import DataLoader
import torch

# Create probe
probe = BiasProbe(input_dim=100000, output_dim=2, hidden_dims=[])

# Prepare data
features = torch.randn(60, 100000)  # Example: 60 samples
labels = torch.randn(60, 2).softmax(dim=-1)  # Soft labels

dataset = SAEFeatureDataset(features, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train
trainer = ProbeTrainer(probe, learning_rate=1e-3, device="cuda")

for epoch in range(10):
    metrics = trainer.train_epoch(dataloader, loss_type="kl")
    print(f"Epoch {epoch}: loss = {metrics['loss']:.4f}")

# Save
trainer.save_checkpoint("linear_probe.pt")
```

### Example 5: Computing IG²

```python
from src.attribution import compute_ig2_for_sae_features, identify_bias_features

# Compute IG²
ig2_scores = compute_ig2_for_sae_features(
    sae_features=features,  # (batch, 100000)
    linear_probe=probe,
    num_steps=20,
    use_squared_gap=True,
    device="cuda"
)

print(f"IG² scores shape: {ig2_scores.shape}")  # (100000,)
print(f"Top score: {ig2_scores.max():.6f}")

# Identify bias features
bias_features, threshold = identify_bias_features(
    ig2_scores,
    threshold_ratio=0.2
)

print(f"Number of bias features: {len(bias_features)}")
print(f"Threshold: {threshold:.6f}")
print(f"Top 5 bias features: {bias_features[:5].tolist()}")
```

### Example 6: Verification

```python
from src.evaluation import verify_bias_features

results = verify_bias_features(
    sae_features=features,
    bias_feature_indices=bias_features,
    linear_probe=probe,
    num_random_controls=3,
    device="cuda"
)

print(f"Suppress gap change: {results['suppress'].gap_change_ratio:+.2f}%")
print(f"Amplify gap change: {results['amplify'].gap_change_ratio:+.2f}%")
print(f"Random gap change: {results['random_control']['mean_change_ratio']:+.2f}%")
```

---

## Configuration Reference

See `configs/experiment_config.yaml` for all configurable parameters.

### Key Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model.name` | EXAONE model identifier | EXAONE-3.0-7.8B-Instruct |
| `model.device` | Device to run on | cuda |
| `model.dtype` | Model precision | float16 |
| `sae.path` | Path to SAE weights (or null) | null |
| `sae.target_layer` | Layer to extract from | 15 |
| `sae.token_position` | Token position | last |
| `sae.sae_type` | SAE type | gated |
| `probe.input_dim` | SAE feature dimension | 100000 |
| `probe.output_dim` | Fixed output dimension | 10 |
| `probe.learning_rate` | Probe learning rate | 1e-3 |
| `probe.epochs` | Training epochs | 50 |
| `ig2.num_steps` | Integration steps | 20 |
| `ig2.threshold_ratio` | Bias feature threshold | 0.2 |
| `data.demographic` | Demographic category | 성별 |
| `data.demographic_values` | Demographic values | [" 남자", " 여자"] |
| `experiment.seed` | Random seed | 42 |
| `experiment.stage` | Experiment stage | pilot |

### SAE Options

You have three options for SAE weights:

**Option 1: Train Your Own**
```yaml
sae:
  path: null  # Will use GatedTrainer/StandardTrainer
```

**Option 2: Use Pre-trained**
```yaml
sae:
  path: "checkpoints/my_sae_model.pth"
```

**Option 3: Skip SAE (Baseline Only)**
```yaml
sae:
  path: null  # Run baseline measurements without SAE features
```

---

## Troubleshooting

### Issue: "SAE weights not found"

**Solution:**
1. Check if path in config is correct
2. Train your own SAE using `GatedTrainer` or `StandardTrainer`
3. Or set `sae.path: null` to skip SAE features

### Issue: "CUDA out of memory"

**Solutions:**
- Use smaller batch size in config
- Use `dtype: "float16"` instead of `"float32"`
- Enable gradient checkpointing
- Use CPU (slower but works)

### Issue: "Linear probe doesn't converge"

**Solutions:**
- Increase learning rate (try 1e-2)
- Add hidden layers: `hidden_dims: [512, 256]`
- Check if features are all zeros: `print((features == 0).float().mean())`
- Reduce L2 regularization: `weight_decay: 0`

### Issue: "No bias detected in baseline"

**Solutions:**
- Try different prompt formats
- Test with more prompts
- Check if model is instruction-tuned (may refuse biased outputs)
- Lower threshold in config

### Issue: "Module import errors"

**Solution:**
```bash
# Ensure you're in the project root
cd korean-bias-sae

# Check project structure
python scripts/00_check_prerequisites.py

# Verify SAE imports
python -c "from src.models.sae import GatedAutoEncoder; print('OK')"
```

---

## File Structure

```
korean-bias-sae/
├── README.md                           # This file (updated with all features)
├── configs/
│   └── experiment_config.yaml          # Main configuration
├── data/
│   ├── demographic_dict_ko.json        # ✅ NEW - All 9 demographic categories
│   ├── modifiers/
│   │   ├── pilot_negative_ko.json      # 5 negative modifiers
│   │   ├── pilot_positive_ko.json      # 5 positive modifiers
│   │   ├── medium_negative_ko.json     # 50 negative modifiers
│   │   ├── medium_positive_ko.json     # 50 positive modifiers
│   │   ├── full_negative_ko.json       # 274 negative modifiers
│   │   └── full_positive_ko.json       # 244 positive modifiers
│   ├── templates/
│   │   └── korean_templates.json       # Shared templates (use {Demographic_Dimension})
│   └── generated/                      # Generated prompts
├── src/
│   ├── __init__.py
│   ├── interfaces.py                   # Data contracts (updated with logits)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── sae/                        # ✅ Standalone SAE
│   │   │   ├── __init__.py
│   │   │   ├── gated_sae.py
│   │   │   └── standard_sae.py
│   │   ├── exaone_wrapper.py           # ✅ UPDATED - get_token_logits()
│   │   ├── sae_wrapper.py              # ✅ UPDATED - standalone
│   │   └── linear_probe.py             # ✅ UPDATED - masking support
│   ├── attribution/
│   │   └── ig2_sae.py
│   ├── evaluation/
│   │   ├── bias_measurement.py         # ✅ UPDATED - logit-based bias score
│   │   └── verification.py
│   └── utils/
│       ├── experiment_utils.py
│       ├── data_utils.py
│       ├── demographic_utils.py        # ✅ NEW - multi-demographic utilities
│       └── prompt_generation.py        # ✅ NEW - prompt generation utilities
├── scripts/
│   ├── 00_check_prerequisites.py       # ✅ UPDATED
│   ├── 01_measure_baseline_bias.py     # ✅ UPDATED - multi-demographic support
│   ├── 02_generate_korean_bias_data.py # TO IMPLEMENT
│   ├── 03_extract_sae_features.py      # TO IMPLEMENT
│   ├── 04_train_linear_probe.py        # TO IMPLEMENT
│   ├── 05_compute_ig2.py               # TO IMPLEMENT
│   ├── 06_verify_bias_features.py      # TO IMPLEMENT
│   ├── test_multi_demographic.py       # ✅ NEW - test demographics
│   ├── test_probe_masking.py           # ✅ NEW - test masking
│   ├── test_bias_score.py              # ✅ NEW - test bias calculation
│   └── translate_bias_neurons_vocab.py # ✅ NEW - vocab translation
├── results/
│   ├── baseline/
│   │   ├── pilot/
│   │   │   ├── 성별/                   # Results by demographic
│   │   │   ├── 나이/
│   │   │   └── 인종/
│   │   └── full/
│   ├── pilot/
│   ├── medium/
│   └── full/
├── tests/
└── requirements.txt
```

---

## Migration from External Repo

If you were previously using `korean-sparse-llm-features-open`:

### Step 1: Update Code

**Old SAEWrapper usage:**
```python
sae = SAEWrapper(
    sae_path="path/to/weights.pth",
    korean_sparse_llm_root="../korean-sparse-llm-features-open",
    sae_type="gated",
    device="cuda"
)
```

**New SAEWrapper usage:**
```python
sae = SAEWrapper(
    sae_path="path/to/weights.pth",
    sae_type="gated",
    device="cuda"
)
```

### Step 2: Update Config

Remove `korean_sparse_llm_root` from `configs/experiment_config.yaml`:

```yaml
# REMOVE this section
paths:
  korean_sparse_llm_root: "../korean-sparse-llm-features-open"

# KEEP this section
paths:
  data_dir: "data/"
  results_dir: "results/"
  checkpoints_dir: "checkpoints/"
```

### Step 3: Verify

```bash
python scripts/00_check_prerequisites.py
```

Expected output:
```
✅ PASS: sae_implementation (CRITICAL)
```

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

### Core Infrastructure
- [x] ✅ Core infrastructure implemented
- [x] ✅ Standalone SAE implementation integrated
- [x] ✅ Model wrappers ready
- [x] ✅ IG² attribution module ready
- [x] ✅ Evaluation modules ready
- [x] ✅ Configuration system ready

### Multi-Demographic Support
- [x] ✅ Demographic dictionary (9 categories)
- [x] ✅ Demographic utility functions
- [x] ✅ Fixed output dimension probe (output_dim=10)
- [x] ✅ Masking implementation
- [x] ✅ Logit-based bias score calculation
- [x] ✅ Multi-demographic validation
- [x] ✅ Test suites (demographics, masking, bias score)

### Data Files
- [x] ✅ Pilot data (5+5 modifiers, 3 templates)
- [x] ✅ Medium data (50+50 modifiers, 5 templates)
- [x] ✅ Full data (274+244 modifiers, 17 templates)
- [x] ✅ Korean templates (shared across demographics)

### Scripts
- [x] ✅ Prerequisites checker (updated)
- [x] ✅ Baseline measurement (multi-demographic support)
- [x] ✅ Vocabulary translation script
- [ ] ⬜ Generate data script (implement using guides above)
- [ ] ⬜ Extract features script (implement using guides above)
- [ ] ⬜ Train probe script (implement using guides above)
- [ ] ⬜ Compute IG² script (implement using guides above)
- [ ] ⬜ Verify features script (implement using guides above)

---

*Last Updated: 2024-11-24*

*Status: ✅ Multi-Demographic Implementation Complete*

**Key Features Implemented:**
- ✅ 9 demographic categories (성별, 인종, 종교, 성적지향, 나이, 외모, 경제수준, 정치성향, 직업)
- ✅ Fixed output dimension with masking (output_dim=10)
- ✅ Logit-based bias score (P(max_logit) - P(min_logit))
- ✅ Full-scale data support (pilot/medium/full stages)
- ✅ Korean tokenization with leading spaces
- ✅ Comprehensive test coverage

*All building blocks provided. Core infrastructure complete. No external dependencies required.*
