# Bias Feature Visualization Suite

Comprehensive visualization tools for analyzing bias-related features in Sparse Autoencoders (SAE) trained on Korean language models.

## Overview

This suite provides 5 visualization notebooks adapted from the [korean-sparse-llm-features-open](../../korean-sparse-llm-features-open/) project, specifically designed to analyze bias detection in the korean-bias-sae project.

### Key Adaptations

| Aspect | korean-sparse-llm-features-open | korean-bias-sae (this suite) |
|--------|--------------------------------|------------------------------|
| **Feature Selection** | TF-IDF by document categories | IG² attribution by demographic bias |
| **Categories** | 8 document topics (art, sports, etc.) | 9 demographic dimensions (gender, ethnicity, religion, etc.) |
| **Data Source** | KEAT dataset (Korean NLI) | BiasPrompt dataset (bias probe prompts) |
| **Visualization Focus** | Topic clustering | Bias feature identification and validation |

## Notebooks

### 1. UMAP Bias Feature Clusters (`01_visualize_bias_feature_clusters.ipynb`)

**Purpose:** Visualize how bias features cluster in 2D latent space using UMAP dimensionality reduction.

**Key Features:**
- 3×3 grid of UMAP scatter plots (one per demographic)
- Highlights top-k bias features per demographic
- Feature frequency histogram
- Overlap analysis between demographics

**Output:**
- `umap_bias_features_{stage}_top{k}.png` - Main UMAP visualization
- `feature_frequency_{stage}_top{k}.png` - Feature frequency histogram

**Expected Insights:**
- Do bias features cluster by demographic dimension?
- Are there shared bias features across demographics?
- Which features are demographic-specific vs. universal?

---

### 2. IG² Attribution Rankings (`02_visualize_ig2_rankings.ipynb`)

**Purpose:** Rank and visualize features by their IG² (Integrated Gradients squared) attribution scores.

**Key Features:**
- 3×3 grid of horizontal bar charts showing top-20 features
- Score distribution histograms per demographic
- Cross-demographic comparison of max scores
- Statistical summary tables

**Output:**
- `ig2_rankings_{stage}_top{k}.png` - Rankings visualization
- `ig2_score_distributions_{stage}.png` - Score distributions
- `ig2_max_scores_comparison_{stage}.png` - Max scores by demographic
- `ig2_statistics_{stage}.csv` - Detailed statistics

**Expected Insights:**
- Which features have highest attribution scores?
- How concentrated are high scores (power-law vs. uniform)?
- Which demographics show strongest bias signals?

---

### 3. Feature Activation Heatmaps (`03_visualize_activation_heatmaps.ipynb`)

**Purpose:** Visualize activation patterns of bias features across prompts.

**Key Features:**
- Heatmap: [prompts × features] showing activation magnitudes
- Sparsity analysis (target: 95-99% zeros)
- K-means clustering of prompts by activation patterns
- Activation frequency analysis

**Output:**
- `activation_heatmap_{stage}_top{k}.png` - Main heatmap
- `clustered_activation_heatmap_{stage}.png` - Clustered version
- `sparsity_analysis_{stage}.png` - Sparsity statistics
- `activation_frequency_{stage}_top{k}.png` - Frequency bar chart

**Expected Insights:**
- Do certain features activate together?
- Are activation patterns consistent across prompt types?
- Is target sparsity achieved?

---

### 4. Verification Effects (`04_visualize_verification_effects.ipynb`)

**Purpose:** Validate that identified features causally contribute to bias predictions.

**Key Features:**
- 3×3 grid comparing: Baseline, Suppress, Amplify, Random Control
- Effect size calculations (% change in logit gap)
- Success criteria evaluation (>5% effect for suppress/amplify, <5% for random)
- Statistical significance testing

**Output:**
- `verification_effects_{stage}.png` - Main comparison
- `effect_sizes_comparison_{stage}.png` - Effect sizes by manipulation type
- `verification_effect_sizes_{stage}.csv` - Detailed statistics

**Expected Results:**
- **Suppress:** Negative effect (reduce bias) >5%
- **Amplify:** Positive effect (increase bias) >5%
- **Random:** Minimal effect (<5%) validates specificity

---

### 5. SAE Training Loss (`05_visualize_sae_training_loss.ipynb`)

**Purpose:** Monitor Gated SAE training dynamics and convergence.

**Key Features:**
- 2×2 grid: Total Loss, Reconstruction Loss, Sparsity Loss, L0 Sparsity
- Smoothed curves with configurable window
- Convergence analysis (coefficient of variation)
- Loss gradient (rate of change)

**Output:**
- `sae_training_loss_{stage}.png` - Loss curves (2×2 grid)
- `sae_training_detailed_{stage}.png` - Smoothed comparison
- `sae_sparsity_analysis_{stage}.png` - Sparsity evolution
- `loss_gradient_{stage}.png` - Training dynamics
- `training_statistics_{stage}.csv` - Detailed metrics

**Expected Behavior:**
- Total loss: Monotonic decrease to stable minimum
- Sparsity (L0): Converge to ~0.05 (5% active features)
- No divergence or sudden spikes

---

## Setup

### 1. Install Dependencies

```bash
cd korean-bias-sae
pip install -r requirements.txt
```

New dependencies added:
- `umap-learn>=0.5.7` - Dimensionality reduction
- `scikit-learn>=1.3.0` - Clustering and preprocessing

### 2. Generate Mock Data (for testing)

```bash
python scripts/generate_mock_data.py
```

This creates test data in `results/mock/`:
- `sae_features.pt` - Mock feature activations [100 × 100,000]
- `ig2_results.pt` - Mock IG² scores for 9 demographics
- `sae_model.pt` - Mock decoder weights [100,000 × 4,096]
- `verification_results.json` - Mock suppress/amplify effects
- `training_logs.csv` - Mock training curves (1000 steps)

### 3. Run Notebooks

```bash
jupyter notebook notebooks/visualizations/
```

Start with `01_visualize_bias_feature_clusters.ipynb` and work through sequentially.

---

## Data Requirements

### For Real Data Visualization

Replace mock data with actual results from the bias detection pipeline:

#### Phase 2: SAE Training
```
results/{stage}/
├── sae_model.pt              # Trained SAE checkpoint
└── training_logs.csv         # Training metrics
```

#### Phase 3: Feature Extraction
```
results/{stage}/
└── sae_features.pt           # Feature activations [N_prompts × 100,000]
```

#### Phase 4: IG² Attribution
```
results/{stage}/
└── ig2_results.pt            # Attribution scores per demographic
```

#### Phase 5: Verification
```
results/{stage}/
└── verification_results.json # Suppress/amplify/random effects
```

---

## Configuration

Each notebook has a configuration cell:

```python
# Paths
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
ASSETS_DIR = PROJECT_ROOT / "notebooks" / "visualizations" / "assets"

# Stage: 'pilot', 'medium', 'full', or 'mock'
STAGE = "mock"

# Visualization parameters
TOP_K = 100  # Number of top features per demographic
```

Change `STAGE` to switch between data stages:
- **pilot:** 30 prompts, quick testing
- **medium:** 500 prompts, intermediate scale
- **full:** 8,806 prompts, full evaluation
- **mock:** Synthetic data for development

---

## Utility Modules

Located in `src/visualization/`:

### `font_utils.py`
- `setup_korean_font()` - Configure matplotlib for Korean text
- `download_and_install_noto_font()` - Download Noto Sans CJK KR

### `data_loaders.py`
- `load_sae_features()` - Load feature activations
- `load_ig2_results()` - Load IG² scores
- `load_verification_results()` - Load verification data
- `load_demographics()` - Load demographic dictionary
- `load_training_logs()` - Load SAE training logs
- `load_sae_decoder_weights()` - Load decoder for UMAP

### `umap_utils.py`
- `apply_umap_to_features()` - Run UMAP dimensionality reduction
- `select_top_features_union()` - Select top-k across demographics
- `compute_feature_overlap()` - Pairwise overlap analysis

### `feature_selection.py`
- `select_top_k_per_demographic()` - Top features per demographic
- `compute_tfidf_weights()` - TF-IDF weighting
- `compute_feature_sparsity()` - Sparsity statistics
- `get_shared_features()` - Features shared across demographics
- `get_unique_features()` - Demographic-specific features

### `plotting_utils.py`
- `plot_umap_clusters()` - UMAP scatter plots
- `plot_ig2_rankings()` - Attribution rankings
- `plot_activation_heatmap()` - Feature activation heatmap
- `plot_verification_effects()` - Suppress/amplify comparison
- `plot_training_loss()` - SAE training curves
- `plot_feature_frequency_histogram()` - Feature frequency

---

## Output Files

All visualizations are saved to `notebooks/visualizations/assets/`:

```
assets/
├── umap_bias_features_{stage}_top{k}.png
├── feature_frequency_{stage}_top{k}.png
├── ig2_rankings_{stage}_top{k}.png
├── ig2_score_distributions_{stage}.png
├── ig2_max_scores_comparison_{stage}.png
├── ig2_statistics_{stage}.csv
├── activation_heatmap_{stage}_top{k}.png
├── clustered_activation_heatmap_{stage}.png
├── sparsity_analysis_{stage}.png
├── activation_frequency_{stage}_top{k}.png
├── verification_effects_{stage}.png
├── effect_sizes_comparison_{stage}.png
├── verification_effect_sizes_{stage}.csv
├── sae_training_loss_{stage}.png
├── sae_training_detailed_{stage}.png
├── sae_sparsity_analysis_{stage}.png
├── loss_gradient_{stage}.png
└── training_statistics_{stage}.csv
```

---

## Troubleshooting

### Korean Text Not Rendering

```python
from src.visualization import download_and_install_noto_font
download_and_install_noto_font(test=True)
```

If still not working, manually set font:
```python
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'AppleGothic'  # macOS
# plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
# plt.rcParams['font.family'] = 'NanumGothic'  # Linux
```

### File Not Found Errors

Check that you're using the correct stage:
```python
STAGE = "mock"  # Use "mock" for testing
```

Real data will be available after running phases 2-5:
```bash
python scripts/03_train_sae.py --stage pilot
python scripts/04_extract_sae_features.py --stage pilot
python scripts/05_compute_ig2.py --stage pilot
python scripts/06_verify_bias_features.py --stage pilot
```

### UMAP Too Slow

Reduce the number of features:
```python
TOP_K = 50  # Instead of 100
```

Or use fewer samples in mock data:
```python
generate_mock_sae_features(n_prompts=50)  # Instead of 100
```

---

## Citation

If you use these visualizations, please cite:

```bibtex
@software{korean_bias_sae_viz,
  title={Bias Feature Visualization Suite for Korean Language Models},
  author={AI Guardians Team},
  year={2024},
  note={Adapted from korean-sparse-llm-features-open}
}
```

---

## License

Same license as the parent korean-bias-sae project.

---

## Contact

For questions or issues with visualizations, please open an issue in the main repository.
