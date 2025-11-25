"""
Korean Bias SAE - Complete Pipeline Runner (Python Version)

This script runs the entire bias detection pipeline end-to-end.
"""

import sys
import argparse
import subprocess
from pathlib import Path
from typing import List

# Colors for terminal output
class Colors:
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color

def log_info(msg: str):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {msg}")

def log_success(msg: str):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {msg}")

def log_warning(msg: str):
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {msg}")

def log_error(msg: str):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")

def log_step(step_num: int, step_name: str):
    print()
    print("=" * 80)
    print(f"{Colors.GREEN}STEP {step_num}: {step_name}{Colors.NC}")
    print("=" * 80)
    print()

def run_script(script_path: Path, args: List[str]) -> bool:
    """
    Run a Python script with arguments.

    Args:
        script_path: Path to the script
        args: List of command-line arguments

    Returns:
        True if successful, False otherwise
    """
    cmd = [sys.executable, str(script_path)] + args

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        log_error(f"Script failed with exit code {e.returncode}")
        return False
    except Exception as e:
        log_error(f"Unexpected error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Run the complete Korean Bias SAE pipeline"
    )
    parser.add_argument(
        '--stage',
        type=str,
        default='pilot',
        choices=['pilot', 'medium', 'full'],
        help='Experiment stage (default: pilot)'
    )
    parser.add_argument(
        '--sae_type',
        type=str,
        default='gated',
        choices=['standard', 'gated'],
        help='SAE type (default: gated)'
    )
    parser.add_argument(
        '--layer_quantile',
        type=str,
        default='q2',
        choices=['q1', 'q2', 'q3'],
        help='Layer quantile (default: q2)'
    )
    parser.add_argument(
        '--num_steps',
        type=int,
        default=20,
        help='Number of IG2 integration steps (default: 20)'
    )
    parser.add_argument(
        '--skip-prerequisites',
        action='store_true',
        help='Skip prerequisites check'
    )
    parser.add_argument(
        '--skip-baseline',
        action='store_true',
        help='Skip baseline bias measurement'
    )
    parser.add_argument(
        '--start-from',
        type=int,
        default=0,
        help='Start from step N (0-6, default: 0)'
    )

    args = parser.parse_args()

    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    scripts_dir = script_dir

    # Print configuration
    print("=" * 80)
    print("Korean Bias SAE - Pipeline Runner")
    print("=" * 80)
    print()
    log_info("Configuration:")
    print(f"  Stage:           {args.stage}")
    print(f"  SAE Type:        {args.sae_type}")
    print(f"  Layer Quantile:  {args.layer_quantile}")
    print(f"  IG2 Steps:       {args.num_steps}")
    print(f"  Start From:      Step {args.start_from}")
    print(f"  Project Root:    {project_root}")
    print()

    # Define pipeline steps
    pipeline_steps = [
        {
            'num': 0,
            'name': 'Prerequisites Check',
            'script': scripts_dir / '00_check_prerequisites.py',
            'args': [],
            'optional': True,
            'skip_flag': args.skip_prerequisites
        },
        {
            'num': 1,
            'name': 'Baseline Bias Measurement',
            'script': scripts_dir / '01_measure_baseline_bias.py',
            'args': ['--stage', args.stage],
            'optional': True,
            'skip_flag': getattr(args, 'skip_baseline', False)
        },
        {
            'num': 2,
            'name': 'Generate Responses and Extract Activations',
            'script': scripts_dir / '02_generate_and_extract_activations.py',
            'args': ['--stage', args.stage],
            'optional': False,
            'skip_flag': False
        },
        {
            'num': 3,
            'name': 'Train Sparse Autoencoder (SAE)',
            'script': scripts_dir / '03_train_sae.py',
            'args': ['--stage', args.stage, '--sae_type', args.sae_type, '--layer_quantile', args.layer_quantile],
            'optional': False,
            'skip_flag': False
        },
        {
            'num': 4,
            'name': 'Train Linear Probe',
            'script': scripts_dir / '04_train_linear_probe.py',
            'args': ['--stage', args.stage, '--sae_type', args.sae_type, '--layer_quantile', args.layer_quantile],
            'optional': False,
            'skip_flag': False
        },
        {
            'num': 5,
            'name': 'Compute IG2 Attribution',
            'script': scripts_dir / '05_compute_ig2.py',
            'args': ['--stage', args.stage, '--sae_type', args.sae_type, '--layer_quantile', args.layer_quantile, '--num_steps', str(args.num_steps)],
            'optional': False,
            'skip_flag': False
        },
        {
            'num': 6,
            'name': 'Verify Bias Features',
            'script': scripts_dir / '06_verify_bias_features.py',
            'args': ['--stage', args.stage, '--sae_type', args.sae_type, '--layer_quantile', args.layer_quantile],
            'optional': False,
            'skip_flag': False
        }
    ]

    # Run pipeline
    failed_steps = []

    for step in pipeline_steps:
        # Skip if before start-from
        if step['num'] < args.start_from:
            log_info(f"Skipping Step {step['num']} (start-from={args.start_from})")
            continue

        # Skip if flagged
        if step['skip_flag']:
            log_warning(f"Skipping Step {step['num']}: {step['name']}")
            continue

        # Run step
        log_step(step['num'], step['name'])

        success = run_script(step['script'], step['args'])

        if success:
            log_success(f"Step {step['num']} completed")
        else:
            if step['optional']:
                log_warning(f"Step {step['num']} failed (optional, continuing)")
                failed_steps.append((step['num'], step['name'], True))
            else:
                log_error(f"Step {step['num']} failed (critical, stopping)")
                failed_steps.append((step['num'], step['name'], False))
                break

    # Final summary
    print()
    print("=" * 80)
    if not failed_steps or all(optional for _, _, optional in failed_steps):
        print(f"{Colors.GREEN}PIPELINE COMPLETE!{Colors.NC}")
    else:
        print(f"{Colors.RED}PIPELINE FAILED!{Colors.NC}")
    print("=" * 80)
    print()

    if failed_steps:
        log_warning("Failed steps:")
        for num, name, optional in failed_steps:
            status = "optional" if optional else "critical"
            print(f"  - Step {num}: {name} ({status})")
        print()

    log_info(f"Results saved to: results/{args.stage}/")
    print()
    log_info("Key outputs:")
    print(f"  - Activations:         results/{args.stage}/activations.pkl")
    print(f"  - SAE Model:           checkpoints/sae-{args.sae_type}_{args.stage}_{args.layer_quantile}/model.pth")
    print(f"  - Linear Probe:        results/{args.stage}/probe/linear_probe.pt")
    print(f"  - IG2 Results:         results/{args.stage}/ig2/ig2_results.pt")
    print(f"  - Verification:        results/{args.stage}/verification/")
    print()

    if not failed_steps or all(optional for _, _, optional in failed_steps):
        log_success("Pipeline completed successfully!")
        return 0
    else:
        log_error("Pipeline completed with errors")
        return 1

if __name__ == '__main__':
    sys.exit(main())
