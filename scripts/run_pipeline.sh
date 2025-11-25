#!/bin/bash

# Korean Bias SAE - Complete Pipeline Runner
# This script runs the entire bias detection pipeline end-to-end

set -e  # Exit on error

# Default parameters
STAGE="pilot"
SAE_TYPE="gated"
LAYER_QUANTILE="q2"
NUM_STEPS=20
SKIP_PREREQUISITES=false
SKIP_BASELINE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --sae_type)
            SAE_TYPE="$2"
            shift 2
            ;;
        --layer_quantile)
            LAYER_QUANTILE="$2"
            shift 2
            ;;
        --num_steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --skip-prerequisites)
            SKIP_PREREQUISITES=true
            shift
            ;;
        --skip-baseline)
            SKIP_BASELINE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --stage STAGE              Experiment stage (pilot|medium|full) [default: pilot]"
            echo "  --sae_type TYPE           SAE type (standard|gated) [default: gated]"
            echo "  --layer_quantile QUANTILE Layer quantile (q1|q2|q3) [default: q2]"
            echo "  --num_steps N             Number of IG2 integration steps [default: 20]"
            echo "  --skip-prerequisites      Skip prerequisites check"
            echo "  --skip-baseline           Skip baseline bias measurement"
            echo "  -h, --help                Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --stage pilot --sae_type gated --layer_quantile q2"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo ""
    echo "========================================================================"
    echo -e "${GREEN}STEP $1: $2${NC}"
    echo "========================================================================"
    echo ""
}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Print configuration
echo "========================================================================"
echo "Korean Bias SAE - Pipeline Runner"
echo "========================================================================"
echo ""
log_info "Configuration:"
echo "  Stage:           $STAGE"
echo "  SAE Type:        $SAE_TYPE"
echo "  Layer Quantile:  $LAYER_QUANTILE"
echo "  IG2 Steps:       $NUM_STEPS"
echo "  Project Root:    $PROJECT_ROOT"
echo ""

# Step 0: Prerequisites check (optional)
if [ "$SKIP_PREREQUISITES" = false ]; then
    log_step "0" "Prerequisites Check"
    python scripts/00_check_prerequisites.py
    if [ $? -eq 0 ]; then
        log_success "Prerequisites check passed"
    else
        log_error "Prerequisites check failed"
        exit 1
    fi
else
    log_warning "Skipping prerequisites check (--skip-prerequisites)"
fi

# Step 1: Baseline bias measurement (optional)
if [ "$SKIP_BASELINE" = false ]; then
    log_step "1" "Baseline Bias Measurement"
    python scripts/01_measure_baseline_bias.py --stage "$STAGE"
    if [ $? -eq 0 ]; then
        log_success "Baseline bias measurement complete"
    else
        log_warning "Baseline bias measurement failed (non-critical)"
    fi
else
    log_warning "Skipping baseline measurement (--skip-baseline)"
fi

# Step 2: Generate and extract activations
log_step "2" "Generate Responses and Extract Activations"
python scripts/02_generate_and_extract_activations.py \
    --stage "$STAGE" \
    --layer_quantile "$LAYER_QUANTILE"

if [ $? -ne 0 ]; then
    log_error "Activation extraction failed"
    exit 1
fi
log_success "Activation extraction complete"

# Step 3: Train SAE
log_step "3" "Train Sparse Autoencoder (SAE)"
python scripts/03_train_sae.py \
    --stage "$STAGE" \
    --sae_type "$SAE_TYPE" \
    --layer_quantile "$LAYER_QUANTILE"

if [ $? -ne 0 ]; then
    log_error "SAE training failed"
    exit 1
fi
log_success "SAE training complete"

# Step 4: Train linear probe
log_step "4" "Train Linear Probe"
python scripts/04_train_linear_probe.py \
    --stage "$STAGE" \
    --sae_type "$SAE_TYPE" \
    --layer_quantile "$LAYER_QUANTILE"

if [ $? -ne 0 ]; then
    log_error "Linear probe training failed"
    exit 1
fi
log_success "Linear probe training complete"

# Step 5: Compute IG2 attribution
log_step "5" "Compute IG2 Attribution"
python scripts/05_compute_ig2.py \
    --stage "$STAGE" \
    --sae_type "$SAE_TYPE" \
    --layer_quantile "$LAYER_QUANTILE" \
    --num_steps "$NUM_STEPS"

if [ $? -ne 0 ]; then
    log_error "IG2 computation failed"
    exit 1
fi
log_success "IG2 computation complete"

# Step 6: Verify bias features
log_step "6" "Verify Bias Features"
python scripts/06_verify_bias_features.py \
    --stage "$STAGE" \
    --sae_type "$SAE_TYPE" \
    --layer_quantile "$LAYER_QUANTILE"

if [ $? -ne 0 ]; then
    log_error "Bias feature verification failed"
    exit 1
fi
log_success "Bias feature verification complete"

# Summary
echo ""
echo "========================================================================"
echo -e "${GREEN}PIPELINE COMPLETE!${NC}"
echo "========================================================================"
echo ""
log_info "Results saved to: results/$STAGE/"
echo ""
log_info "Key outputs:"
echo "  - Activations:         results/$STAGE/activations.pkl"
echo "  - SAE Model:           checkpoints/sae-${SAE_TYPE}_${STAGE}_${LAYER_QUANTILE}/model.pth"
echo "  - Linear Probe:        results/$STAGE/probe/linear_probe.pt"
echo "  - IG2 Results:         results/$STAGE/ig2/ig2_results.pt"
echo "  - Verification:        results/$STAGE/verification/"
echo ""
log_success "Pipeline completed successfully!"
echo ""
