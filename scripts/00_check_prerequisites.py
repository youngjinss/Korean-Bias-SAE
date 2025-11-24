#!/usr/bin/env python3
"""
Script 00: Check Prerequisites

Verifies that all required components are available before starting the experiment:
- EXAONE model accessibility
- Pre-trained gSAE availability
- GPU memory sufficiency
- korean-sparse-llm-features-open codebase functionality

Usage:
    python scripts/00_check_prerequisites.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.utils.experiment_utils import load_config, ExperimentLogger


def check_pytorch_cuda():
    """Check PyTorch and CUDA availability"""
    print("=" * 60)
    print("Checking PyTorch and CUDA...")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Memory: {props.total_memory / 1024**3:.2f} GB")
        print("✅ CUDA is available")
    else:
        print("⚠️  CUDA not available, will use CPU (slower)")

    print()
    return torch.cuda.is_available()


def check_exaone_model(config):
    """Check EXAONE model accessibility"""
    print("=" * 60)
    print("Checking EXAONE Model...")
    print("=" * 60)

    model_name = config['model']['name']
    print(f"Model: {model_name}")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"✅ Tokenizer loaded successfully")
        print(f"   Vocab size: {len(tokenizer)}")

        # Test tokenization
        test_text = "게으른 사람의 성별은? 답:"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"   Test tokenization: '{test_text}'")
        print(f"   Token count: {tokens['input_ids'].shape[1]}")

        # Check if model can be loaded (metadata only, don't load weights)
        print("\nChecking model availability (not loading weights)...")
        try:
            from transformers import AutoConfig
            model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            print(f"✅ Model config accessible")
            print(f"   Hidden size: {model_config.hidden_size}")
            print(f"   Num layers: {model_config.num_hidden_layers}")
            print(f"   Vocab size: {model_config.vocab_size}")

            return True, tokenizer
        except Exception as e:
            print(f"❌ Cannot access model config: {e}")
            return False, None

    except Exception as e:
        print(f"❌ Error loading EXAONE: {e}")
        print("\nTroubleshooting:")
        print("  1. Check internet connection")
        print("  2. Verify Hugging Face access token if required")
        print("  3. Try: huggingface-cli login")
        return False, None


def check_sae_availability(config):
    """Check pre-trained gSAE availability"""
    print("=" * 60)
    print("Checking gSAE Availability...")
    print("=" * 60)

    sae_path = Path(config['sae']['path'])
    print(f"SAE path: {sae_path}")

    if not sae_path.exists():
        print(f"❌ SAE weights not found at: {sae_path}")
        print("\nAction required:")
        print(f"  1. Check if path is correct in config")
        print(f"  2. If SAE doesn't exist, you need to train it first:")
        print(f"     cd {config['paths']['korean_sparse_llm_root']}")
        print(f"     bash x3_train_sae.sh")
        print(f"  3. Or adjust config to use existing SAE")
        return False

    print(f"✅ SAE weights found")

    # Try to load SAE to verify it's valid
    try:
        sae_data = torch.load(sae_path, map_location='cpu')
        print(f"✅ SAE weights loadable")

        # Check dimensions
        if 'encoder.weight' in sae_data:
            encoder_shape = sae_data['encoder.weight'].shape
            print(f"   Encoder shape: {encoder_shape}")
            print(f"   Feature dim: {encoder_shape[0]}")
            print(f"   Activation dim: {encoder_shape[1]}")

            # Verify matches config
            expected_feat = config['sae']['feature_dim']
            expected_act = config['sae']['activation_dim']

            if encoder_shape[0] == expected_feat and encoder_shape[1] == expected_act:
                print(f"✅ SAE dimensions match config")
            else:
                print(f"⚠️  Dimension mismatch!")
                print(f"   Config expects: ({expected_feat}, {expected_act})")
                print(f"   SAE has: {encoder_shape}")
                print(f"   → Update config to match SAE")

        return True

    except Exception as e:
        print(f"❌ Error loading SAE: {e}")
        return False


def check_korean_sparse_llm_repo(config):
    """Check korean-sparse-llm-features-open codebase"""
    print("=" * 60)
    print("Checking korean-sparse-llm-features-open Repo...")
    print("=" * 60)

    repo_path = Path(config['paths']['korean_sparse_llm_root'])
    print(f"Repo path: {repo_path}")

    if not repo_path.exists():
        print(f"❌ Repository not found at: {repo_path}")
        print("\nAction required:")
        print(f"  1. Clone the repository")
        print(f"  2. Update config path if cloned elsewhere")
        return False

    print(f"✅ Repository directory exists")

    # Check for key files
    key_files = [
        'lib/models/gated_sae/GatedSAE.py',
        'lib/models/standard_sae/StandardSAE.py',
    ]

    all_exist = True
    for file in key_files:
        file_path = repo_path / file
        if file_path.exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} not found")
            all_exist = False

    if all_exist:
        print("✅ Key files present")
    else:
        print("⚠️  Some files missing, repo may be incomplete")

    return all_exist


def check_gpu_memory(min_gb=16):
    """Check if GPU has sufficient memory"""
    print("=" * 60)
    print("Checking GPU Memory...")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ℹ️  No GPU available, skipping memory check")
        return True  # Not a failure, just CPU mode

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / 1024**3

        print(f"GPU {i}: {total_gb:.2f} GB")

        if total_gb >= min_gb:
            print(f"  ✅ Sufficient memory (>= {min_gb} GB)")
        else:
            print(f"  ⚠️  Low memory (< {min_gb} GB)")
            print(f"     Recommendations:")
            print(f"     - Use gradient checkpointing")
            print(f"     - Reduce batch size")
            print(f"     - Use CPU (slower but works)")

    print()
    return True


def main():
    """Run all prerequisite checks"""
    print("\n" + "=" * 60)
    print("PREREQUISITE CHECKER")
    print("=" * 60 + "\n")

    # Load config
    try:
        config = load_config("configs/experiment_config.yaml")
        print("✅ Configuration loaded\n")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False

    # Run all checks
    results = {}

    results['pytorch_cuda'] = check_pytorch_cuda()
    results['exaone'], tokenizer = check_exaone_model(config)
    results['sae'] = check_sae_availability(config)
    results['repo'] = check_korean_sparse_llm_repo(config)
    results['gpu_memory'] = check_gpu_memory(min_gb=14)  # Slightly lower threshold

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_pass = True
    critical_checks = ['exaone', 'sae', 'repo']

    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        critical = " (CRITICAL)" if check in critical_checks else ""
        print(f"{status}: {check}{critical}")

        if check in critical_checks and not passed:
            all_pass = False

    print()

    if all_pass:
        print("✅ All critical prerequisites met!")
        print("\nYou can proceed with:")
        print("  python scripts/01_measure_baseline_bias.py")
    else:
        print("❌ Some critical prerequisites failed.")
        print("\nPlease address the issues above before proceeding.")

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
