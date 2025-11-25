#!/bin/bash

# Quick runner for individual pipeline steps

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default parameters
STAGE="pilot"
SAE_TYPE="gated"
LAYER_QUANTILE="q2"

# Parse step number
if [ $# -lt 1 ]; then
    echo "Usage: $0 STEP [--stage STAGE] [--sae_type TYPE] [--layer_quantile Q]"
    echo ""
    echo "Available steps:"
    echo "  0  - Prerequisites check"
    echo "  1  - Baseline bias measurement"
    echo "  2  - Generate and extract activations"
    echo "  3  - Train SAE"
    echo "  4  - Train linear probe"
    echo "  5  - Compute IG2 attribution"
    echo "  6  - Verify bias features"
    echo ""
    echo "Example:"
    echo "  $0 2 --stage pilot"
    echo "  $0 5 --stage pilot --sae_type gated --layer_quantile q2"
    exit 1
fi

STEP=$1
shift

# Parse additional arguments
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
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run the appropriate step
case $STEP in
    0)
        echo "Running: Prerequisites Check"
        python "$SCRIPT_DIR/00_check_prerequisites.py"
        ;;
    1)
        echo "Running: Baseline Bias Measurement (stage=$STAGE)"
        python "$SCRIPT_DIR/01_measure_baseline_bias.py" --stage "$STAGE"
        ;;
    2)
        echo "Running: Generate and Extract Activations (stage=$STAGE, layer=$LAYER_QUANTILE)"
        python "$SCRIPT_DIR/02_generate_and_extract_activations.py" \
            --stage "$STAGE" \
            --layer_quantile "$LAYER_QUANTILE"
        ;;
    3)
        echo "Running: Train SAE (stage=$STAGE, type=$SAE_TYPE, layer=$LAYER_QUANTILE)"
        python "$SCRIPT_DIR/03_train_sae.py" \
            --stage "$STAGE" \
            --sae_type "$SAE_TYPE" \
            --layer_quantile "$LAYER_QUANTILE"
        ;;
    4)
        echo "Running: Train Linear Probe (stage=$STAGE, type=$SAE_TYPE, layer=$LAYER_QUANTILE)"
        python "$SCRIPT_DIR/04_train_linear_probe.py" \
            --stage "$STAGE" \
            --sae_type "$SAE_TYPE" \
            --layer_quantile "$LAYER_QUANTILE"
        ;;
    5)
        echo "Running: Compute IG2 Attribution (stage=$STAGE, type=$SAE_TYPE, layer=$LAYER_QUANTILE)"
        python "$SCRIPT_DIR/05_compute_ig2.py" \
            --stage "$STAGE" \
            --sae_type "$SAE_TYPE" \
            --layer_quantile "$LAYER_QUANTILE"
        ;;
    6)
        echo "Running: Verify Bias Features (stage=$STAGE, type=$SAE_TYPE, layer=$LAYER_QUANTILE)"
        python "$SCRIPT_DIR/06_verify_bias_features.py" \
            --stage "$STAGE" \
            --sae_type "$SAE_TYPE" \
            --layer_quantile "$LAYER_QUANTILE"
        ;;
    *)
        echo "Invalid step number: $STEP"
        echo "Must be between 0 and 6"
        exit 1
        ;;
esac
