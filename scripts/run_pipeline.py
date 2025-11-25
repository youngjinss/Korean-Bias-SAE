"""
Korean Bias SAE - Complete Pipeline Runner (Python Version)

This script runs the entire bias detection pipeline end-to-end.

Supports running for:
- Single demographic: --demographic 성별
- All demographics: --demographic all (generates data for all demographics, then merges)

Usage:
    python scripts/run_pipeline.py --stage pilot --demographic all
    python scripts/run_pipeline.py --stage medium --demographic 성별
"""

import sys
import argparse
import subprocess
from pathlib import Path
from typing import List

# Add project root for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.demographic_utils import get_all_demographics

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
    parser.add_argument(
        '--demographic',
        type=str,
        default=None,
        help='Demographic category to process. Use "all" for all demographics (성별, 인종, etc.)'
    )

    args = parser.parse_args()

    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    scripts_dir = script_dir

    # Determine demographics to process
    if args.demographic == 'all':
        all_demos = get_all_demographics()
        demographics = list(all_demos.keys())
        demographic_mode = 'all'
    elif args.demographic:
        demographics = [args.demographic]
        demographic_mode = 'single'
    else:
        demographics = None  # Use config default
        demographic_mode = 'config'

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
    print(f"  Demographic:     {args.demographic or 'config default'}")
    if demographic_mode == 'all':
        print(f"  Demographics:    {len(demographics)} categories")
    print(f"  Start From:      Step {args.start_from}")
    print(f"  Project Root:    {project_root}")
    print()

    # Build activation extraction args based on demographic mode
    def get_extraction_args(demo=None):
        base_args = ['--stage', args.stage]
        if demo:
            base_args.extend(['--demographic', demo])
        return base_args

    # Run pipeline
    failed_steps = []

    # === STEP 0: Prerequisites Check ===
    if args.start_from <= 0:
        if args.skip_prerequisites:
            log_warning("Skipping Step 0: Prerequisites Check")
        else:
            log_step(0, "Prerequisites Check")
            success = run_script(scripts_dir / '00_check_prerequisites.py', [])
            if success:
                log_success("Step 0 completed")
            else:
                log_warning("Step 0 failed (optional, continuing)")
                failed_steps.append((0, 'Prerequisites Check', True))

    # === STEP 1: Baseline Bias Measurement (optional) ===
    if args.start_from <= 1:
        if getattr(args, 'skip_baseline', False):
            log_warning("Skipping Step 1: Baseline Bias Measurement")
        else:
            log_step(1, "Baseline Bias Measurement")
            baseline_args = ['--stage', args.stage]
            if demographics and demographic_mode == 'single':
                baseline_args.extend(['--demographic', demographics[0]])
            success = run_script(scripts_dir / '01_measure_baseline_bias.py', baseline_args)
            if success:
                log_success("Step 1 completed")
            else:
                log_warning("Step 1 failed (optional, continuing)")
                failed_steps.append((1, 'Baseline Bias Measurement', True))

    # === STEP 2: Generate Responses and Extract Activations ===
    if args.start_from <= 2:
        log_step(2, "Generate Responses and Extract Activations")

        if demographic_mode == 'all':
            # Run for each demographic
            log_info(f"Processing {len(demographics)} demographic categories...")
            for i, demo in enumerate(demographics):
                print(f"\n  [{i+1}/{len(demographics)}] Processing {demo}...")
                extraction_args = get_extraction_args(demo)
                success = run_script(scripts_dir / '02_generate_and_extract_activations.py', extraction_args)
                if not success:
                    log_error(f"Failed to process {demo}")
                    failed_steps.append((2, f'Extract Activations ({demo})', False))
                else:
                    log_success(f"Completed {demo}")

            # Merge activations after all extractions
            if not any(num == 2 for num, _, _ in failed_steps):
                log_step("2.5", "Merge Activations for gSAE Training")
                merge_args = ['--stage', args.stage]
                success = run_script(scripts_dir / 'merge_activations.py', merge_args)
                if success:
                    log_success("Step 2.5 completed - Activations merged")
                else:
                    log_error("Step 2.5 failed - Could not merge activations")
                    failed_steps.append((2.5, 'Merge Activations', False))
        else:
            # Single demographic or config default
            extraction_args = get_extraction_args(demographics[0] if demographics else None)
            success = run_script(scripts_dir / '02_generate_and_extract_activations.py', extraction_args)
            if success:
                log_success("Step 2 completed")
            else:
                log_error("Step 2 failed (critical, stopping)")
                failed_steps.append((2, 'Extract Activations', False))

    # Check if we should continue (no critical failures so far)
    critical_failures = [f for f in failed_steps if len(f) < 3 or not f[2]]
    if critical_failures:
        log_error("Critical failures detected, stopping pipeline")
    else:
        # === STEP 3: Train Sparse Autoencoder (SAE) ===
        # Note: For 'all' mode, merge_activations.py saves to default location (results/{stage}/activations.pkl)
        # so no special path argument needed
        if args.start_from <= 3:
            log_step(3, "Train Sparse Autoencoder (SAE)")
            sae_args = ['--stage', args.stage, '--sae_type', args.sae_type, '--layer_quantile', args.layer_quantile]
            success = run_script(scripts_dir / '03_train_sae.py', sae_args)
            if success:
                log_success("Step 3 completed")
            else:
                log_error("Step 3 failed (critical, stopping)")
                failed_steps.append((3, 'Train SAE', False))
                critical_failures.append((3, 'Train SAE', False))

        # === STEP 4: Train Linear Probe ===
        if args.start_from <= 4 and not critical_failures:
            log_step(4, "Train Linear Probe")
            probe_args = ['--stage', args.stage, '--sae_type', args.sae_type, '--layer_quantile', args.layer_quantile]
            success = run_script(scripts_dir / '04_train_linear_probe.py', probe_args)
            if success:
                log_success("Step 4 completed")
            else:
                log_error("Step 4 failed (critical, stopping)")
                failed_steps.append((4, 'Train Linear Probe', False))
                critical_failures.append((4, 'Train Linear Probe', False))

        # === STEP 5: Compute IG2 Attribution ===
        if args.start_from <= 5 and not critical_failures:
            log_step(5, "Compute IG2 Attribution")
            ig2_args = ['--stage', args.stage, '--sae_type', args.sae_type,
                       '--layer_quantile', args.layer_quantile, '--num_steps', str(args.num_steps)]
            success = run_script(scripts_dir / '05_compute_ig2.py', ig2_args)
            if success:
                log_success("Step 5 completed")
            else:
                log_error("Step 5 failed (critical, stopping)")
                failed_steps.append((5, 'Compute IG2', False))
                critical_failures.append((5, 'Compute IG2', False))

        # === STEP 6: Verify Bias Features ===
        if args.start_from <= 6 and not critical_failures:
            log_step(6, "Verify Bias Features")
            verify_args = ['--stage', args.stage, '--sae_type', args.sae_type, '--layer_quantile', args.layer_quantile]
            success = run_script(scripts_dir / '06_verify_bias_features.py', verify_args)
            if success:
                log_success("Step 6 completed")
            else:
                log_error("Step 6 failed (critical, stopping)")
                failed_steps.append((6, 'Verify Bias Features', False))

    # Final summary
    print()
    print("=" * 80)
    # Check for critical failures (third element is False or not present)
    has_critical = any(len(f) < 3 or not f[2] for f in failed_steps)
    if not failed_steps or not has_critical:
        print(f"{Colors.GREEN}PIPELINE COMPLETE!{Colors.NC}")
    else:
        print(f"{Colors.RED}PIPELINE FAILED!{Colors.NC}")
    print("=" * 80)
    print()

    if failed_steps:
        log_warning("Failed steps:")
        for step_info in failed_steps:
            num, name = step_info[0], step_info[1]
            optional = step_info[2] if len(step_info) > 2 else False
            status = "optional" if optional else "critical"
            print(f"  - Step {num}: {name} ({status})")
        print()

    log_info(f"Results saved to: results/{args.stage}/")
    print()
    log_info("Key outputs:")

    if demographic_mode == 'all':
        print(f"  - Per-demographic:     results/{args.stage}/<demographic>/activations.pkl")
        print(f"  - Merged Activations:  results/{args.stage}/activations.pkl")
        print(f"  - Metadata:            results/{args.stage}/activations_metadata.json")
    else:
        demo_name = demographics[0] if demographics else 'default'
        print(f"  - Activations:         results/{args.stage}/{demo_name}/activations.pkl")

    print(f"  - SAE Model:           checkpoints/sae-{args.sae_type}_{args.stage}_{args.layer_quantile}/model.pth")
    print(f"  - Linear Probe:        results/{args.stage}/probe/linear_probe.pt")
    print(f"  - IG2 Results:         results/{args.stage}/ig2/ig2_results.pt")
    print(f"  - Verification:        results/{args.stage}/verification/")
    print()

    if not failed_steps or not has_critical:
        log_success("Pipeline completed successfully!")
        return 0
    else:
        log_error("Pipeline completed with errors")
        return 1

if __name__ == '__main__':
    sys.exit(main())
