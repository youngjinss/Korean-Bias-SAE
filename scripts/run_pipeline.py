"""
Korean Bias SAE - Complete Pipeline Runner (Python Version)

This script runs the entire bias detection pipeline end-to-end.

Supports running for:
- Single demographic: --demographic 성별
- All demographics: --demographic all (generates data for all demographics, then merges)
- Single layer: --layer_quantile q2
- All layers: --layer_quantile all (trains SAE and probes for q1, q2, q3)

Full experiment matrix:
- Use both --demographic all --layer_quantile all to run full 3×9 matrix
- For comprehensive experiments, consider using run_full_experiment.py instead

Background execution:
- --background: Run pipeline in background with logging
- --status: Check status of background pipeline
- --stop: Stop running background pipeline

Usage:
    python scripts/run_pipeline.py --stage pilot --demographic all
    python scripts/run_pipeline.py --stage medium --demographic 성별
    python scripts/run_pipeline.py --stage pilot --layer_quantile all --demographic 성별
    python scripts/run_pipeline.py --stage pilot --layer_quantile all --demographic all
    python scripts/run_pipeline.py --stage pilot --background  # Run in background
    python scripts/run_pipeline.py --status                    # Check status
    python scripts/run_pipeline.py --stop                      # Stop background run
"""

import sys
import os
import argparse
import subprocess
import signal
import datetime
import time
import json
from pathlib import Path
from typing import List, Optional

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


# Background execution helpers
def get_pid_file() -> Path:
    """Get path to PID file for background process tracking."""
    return PROJECT_ROOT / "logs" / "pipeline.pid"

def get_status_file() -> Path:
    """Get path to status file for background process."""
    return PROJECT_ROOT / "logs" / "pipeline_status.json"

def get_log_file(stage: str) -> Path:
    """Get path to log file for background execution."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "logs" / f"pipeline_{stage}_{timestamp}.log"

def save_status(status: dict):
    """Save pipeline status to JSON file."""
    status_file = get_status_file()
    status_file.parent.mkdir(parents=True, exist_ok=True)
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2, default=str)

def load_status() -> Optional[dict]:
    """Load pipeline status from JSON file."""
    status_file = get_status_file()
    if status_file.exists():
        with open(status_file, 'r') as f:
            return json.load(f)
    return None

def is_process_running(pid: int) -> bool:
    """Check if a process with given PID is running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False

def check_background_status():
    """Check and display status of background pipeline."""
    pid_file = get_pid_file()
    status = load_status()

    if not pid_file.exists():
        log_info("No background pipeline is currently tracked.")
        if status:
            print(f"\nLast run status:")
            print(f"  Stage: {status.get('stage', 'N/A')}")
            print(f"  Started: {status.get('started_at', 'N/A')}")
            print(f"  Ended: {status.get('ended_at', 'N/A')}")
            print(f"  Status: {status.get('status', 'N/A')}")
            if status.get('log_file'):
                print(f"  Log file: {status.get('log_file')}")
        return

    with open(pid_file, 'r') as f:
        pid = int(f.read().strip())

    if is_process_running(pid):
        log_info(f"Background pipeline is RUNNING (PID: {pid})")
        if status:
            print(f"\nCurrent run:")
            print(f"  Stage: {status.get('stage', 'N/A')}")
            print(f"  Demographic: {status.get('demographic', 'N/A')}")
            print(f"  Started: {status.get('started_at', 'N/A')}")
            print(f"  Current step: {status.get('current_step', 'N/A')}")
            if status.get('log_file'):
                print(f"  Log file: {status.get('log_file')}")
                print(f"\nTo follow logs: tail -f {status.get('log_file')}")
    else:
        log_info(f"Background pipeline has STOPPED (was PID: {pid})")
        pid_file.unlink()  # Clean up stale PID file
        if status:
            print(f"\nLast run:")
            print(f"  Stage: {status.get('stage', 'N/A')}")
            print(f"  Status: {status.get('status', 'N/A')}")
            print(f"  Started: {status.get('started_at', 'N/A')}")
            print(f"  Ended: {status.get('ended_at', 'N/A')}")
            if status.get('log_file'):
                print(f"  Log file: {status.get('log_file')}")

def stop_background_pipeline():
    """Stop running background pipeline."""
    pid_file = get_pid_file()

    if not pid_file.exists():
        log_warning("No background pipeline PID file found.")
        return False

    with open(pid_file, 'r') as f:
        pid = int(f.read().strip())

    if not is_process_running(pid):
        log_warning(f"Process {pid} is not running.")
        pid_file.unlink()
        return False

    log_info(f"Stopping background pipeline (PID: {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
        # Wait for process to terminate
        for _ in range(10):
            time.sleep(0.5)
            if not is_process_running(pid):
                break

        if is_process_running(pid):
            log_warning("Process did not terminate, sending SIGKILL...")
            os.kill(pid, signal.SIGKILL)

        pid_file.unlink()

        # Update status
        status = load_status() or {}
        status['status'] = 'stopped'
        status['ended_at'] = datetime.datetime.now().isoformat()
        save_status(status)

        log_success("Background pipeline stopped.")
        return True
    except Exception as e:
        log_error(f"Failed to stop process: {e}")
        return False

def run_in_background(args):
    """Launch pipeline in background with logging."""
    pid_file = get_pid_file()

    # Check if already running
    if pid_file.exists():
        with open(pid_file, 'r') as f:
            old_pid = int(f.read().strip())
        if is_process_running(old_pid):
            log_error(f"Background pipeline already running (PID: {old_pid})")
            log_info("Use --status to check status or --stop to stop it.")
            return False
        else:
            pid_file.unlink()  # Clean up stale PID

    # Prepare log file
    log_file = get_log_file(args.stage)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Build command (remove --background flag)
    cmd = [sys.executable, str(Path(__file__))]
    cmd.extend(['--stage', args.stage])
    cmd.extend(['--sae_type', args.sae_type])
    cmd.extend(['--layer_quantile', args.layer_quantile])
    cmd.extend(['--num_steps', str(args.num_steps)])
    if args.skip_prerequisites:
        cmd.append('--skip-prerequisites')
    if args.skip_baseline:
        cmd.append('--skip-baseline')
    if args.start_from > 0:
        cmd.extend(['--start-from', str(args.start_from)])
    if args.demographic:
        cmd.extend(['--demographic', args.demographic])

    log_info(f"Starting pipeline in background...")
    log_info(f"Log file: {log_file}")

    # Fork and run in background
    with open(log_file, 'w') as log_f:
        process = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True  # Detach from terminal
        )

    # Save PID
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    with open(pid_file, 'w') as f:
        f.write(str(process.pid))

    # Save initial status
    save_status({
        'pid': process.pid,
        'stage': args.stage,
        'demographic': args.demographic or 'config default',
        'started_at': datetime.datetime.now().isoformat(),
        'status': 'running',
        'current_step': 0,
        'log_file': str(log_file),
        'command': ' '.join(cmd)
    })

    log_success(f"Pipeline started in background (PID: {process.pid})")
    print(f"\nTo monitor progress:")
    print(f"  tail -f {log_file}")
    print(f"\nTo check status:")
    print(f"  python {__file__} --status")
    print(f"\nTo stop:")
    print(f"  python {__file__} --stop")

    return True


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
        choices=['q1', 'q2', 'q3', 'all'],
        help='Layer quantile (default: q2). Use "all" to run for all 3 layers.'
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
    # Background execution arguments
    parser.add_argument(
        '--background',
        action='store_true',
        help='Run pipeline in background with logging'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Check status of background pipeline'
    )
    parser.add_argument(
        '--stop',
        action='store_true',
        help='Stop running background pipeline'
    )

    args = parser.parse_args()

    # Handle background control commands first
    if args.status:
        check_background_status()
        return 0

    if args.stop:
        success = stop_background_pipeline()
        return 0 if success else 1

    if args.background:
        success = run_in_background(args)
        return 0 if success else 1

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

    # Determine layers to process
    if args.layer_quantile == 'all':
        layer_quantiles = ['q1', 'q2', 'q3']
        layer_mode = 'all'
    else:
        layer_quantiles = [args.layer_quantile]
        layer_mode = 'single'

    # Print configuration
    print("=" * 80)
    print("Korean Bias SAE - Pipeline Runner")
    print("=" * 80)
    print()
    log_info("Configuration:")
    print(f"  Stage:           {args.stage}")
    print(f"  SAE Type:        {args.sae_type}")
    print(f"  Layer Quantile:  {args.layer_quantile}")
    if layer_mode == 'all':
        print(f"  Layers:          Q1 (25%), Q2 (50%), Q3 (75%)")
    print(f"  IG2 Steps:       {args.num_steps}")
    print(f"  Demographic:     {args.demographic or 'config default'}")
    if demographic_mode == 'all':
        print(f"  Demographics:    {len(demographics)} categories")
    print(f"  Start From:      Step {args.start_from}")
    print(f"  Project Root:    {project_root}")
    print()

    # If running all layers and all demographics, suggest using run_full_experiment.py
    if layer_mode == 'all' and demographic_mode == 'all':
        log_info("Tip: For full experiments, consider using run_full_experiment.py for better progress tracking.")
        print()

    # Build activation extraction args based on demographic mode
    def get_extraction_args(demo=None):
        base_args = ['--stage', args.stage]
        if demo:
            base_args.extend(['--demographic', demo])
        return base_args

    # Helper function to update status during execution
    def update_step_status(step_num: int, step_name: str, status_str: str = 'running'):
        """Update status file with current step (for background monitoring)."""
        status = load_status()
        if status:
            status['current_step'] = step_num
            status['current_step_name'] = step_name
            status['status'] = status_str
            status['last_update'] = datetime.datetime.now().isoformat()
            save_status(status)

    # Run pipeline
    failed_steps = []

    # === STEP 0: Prerequisites Check ===
    if args.start_from <= 0:
        if args.skip_prerequisites:
            log_warning("Skipping Step 0: Prerequisites Check")
        else:
            log_step(0, "Prerequisites Check")
            update_step_status(0, "Prerequisites Check")
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
            update_step_status(1, "Baseline Bias Measurement")
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
        update_step_status(2, "Generate Responses and Extract Activations")

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
        # For layer_mode == 'all', we train separate SAE for each layer
        if args.start_from <= 3:
            log_step(3, "Train Sparse Autoencoder (SAE)")
            update_step_status(3, "Train Sparse Autoencoder (SAE)")

            if layer_mode == 'all':
                # Train SAE for each layer
                log_info(f"Training SAE for {len(layer_quantiles)} layer quantiles...")
                for j, lq in enumerate(layer_quantiles):
                    print(f"\n  [{j+1}/{len(layer_quantiles)}] Training SAE for layer {lq}...")
                    sae_args = ['--stage', args.stage, '--sae_type', args.sae_type, '--layer_quantile', lq]
                    success = run_script(scripts_dir / '03_train_sae.py', sae_args)
                    if not success:
                        log_error(f"Failed to train SAE for layer {lq}")
                        failed_steps.append((3, f'Train SAE ({lq})', False))
                    else:
                        log_success(f"Completed SAE for layer {lq}")

                # Check if all SAEs were trained
                sae_failures = [f for f in failed_steps if f[0] == 3]
                if not sae_failures:
                    log_success("Step 3 completed - All SAEs trained")
                else:
                    log_error(f"Step 3 partially failed - {len(sae_failures)} SAE(s) failed")
                    critical_failures.extend(sae_failures)
            else:
                # Single layer
                sae_args = ['--stage', args.stage, '--sae_type', args.sae_type, '--layer_quantile', layer_quantiles[0]]
                success = run_script(scripts_dir / '03_train_sae.py', sae_args)
                if success:
                    log_success("Step 3 completed")
                else:
                    log_error("Step 3 failed (critical, stopping)")
                    failed_steps.append((3, 'Train SAE', False))
                    critical_failures.append((3, 'Train SAE', False))

        # === STEP 4: Train Linear Probe ===
        # Note: For 'all' mode, we train SEPARATE probes for each demographic
        # This follows the pattern from korean-sparse-llm-features-open where
        # one SAE is trained but multiple independent probes are used for different label types
        # For layer_mode == 'all', we also iterate through layers
        if args.start_from <= 4 and not critical_failures:
            log_step(4, "Train Linear Probe")
            update_step_status(4, "Train Linear Probe")

            # Determine iteration scope
            if layer_mode == 'all' and demographic_mode == 'all':
                # Full matrix: all layers × all demographics
                total = len(layer_quantiles) * len(demographics)
                log_info(f"Training probes for {len(layer_quantiles)} layers × {len(demographics)} demographics = {total} combinations...")
                count = 0
                for lq in layer_quantiles:
                    for demo in demographics:
                        count += 1
                        print(f"\n  [{count}/{total}] Training probe for {lq} × {demo}...")
                        probe_args = ['--stage', args.stage, '--sae_type', args.sae_type,
                                     '--layer_quantile', lq, '--demographic', demo]
                        success = run_script(scripts_dir / '04_train_linear_probe.py', probe_args)
                        if not success:
                            log_error(f"Failed to train probe for {lq} × {demo}")
                            failed_steps.append((4, f'Train Probe ({lq}×{demo})', False))
                        else:
                            log_success(f"Completed probe for {lq} × {demo}")

            elif layer_mode == 'all':
                # All layers, single demographic
                log_info(f"Training probes for {len(layer_quantiles)} layer quantiles...")
                demo = demographics[0] if demographics else None
                for j, lq in enumerate(layer_quantiles):
                    print(f"\n  [{j+1}/{len(layer_quantiles)}] Training probe for layer {lq}...")
                    probe_args = ['--stage', args.stage, '--sae_type', args.sae_type, '--layer_quantile', lq]
                    if demo:
                        probe_args.extend(['--demographic', demo])
                    success = run_script(scripts_dir / '04_train_linear_probe.py', probe_args)
                    if not success:
                        log_error(f"Failed to train probe for layer {lq}")
                        failed_steps.append((4, f'Train Probe ({lq})', False))
                    else:
                        log_success(f"Completed probe for layer {lq}")

            elif demographic_mode == 'all':
                # Single layer, all demographics
                log_info(f"Training probes for {len(demographics)} demographic categories...")
                lq = layer_quantiles[0]
                for i, demo in enumerate(demographics):
                    print(f"\n  [{i+1}/{len(demographics)}] Training probe for {demo}...")
                    probe_args = ['--stage', args.stage, '--sae_type', args.sae_type,
                                 '--layer_quantile', lq, '--demographic', demo]
                    success = run_script(scripts_dir / '04_train_linear_probe.py', probe_args)
                    if not success:
                        log_error(f"Failed to train probe for {demo}")
                        failed_steps.append((4, f'Train Probe ({demo})', False))
                    else:
                        log_success(f"Completed probe for {demo}")
            else:
                # Single layer, single demographic
                probe_args = ['--stage', args.stage, '--sae_type', args.sae_type, '--layer_quantile', layer_quantiles[0]]
                if demographics:
                    probe_args.extend(['--demographic', demographics[0]])
                success = run_script(scripts_dir / '04_train_linear_probe.py', probe_args)
                if success:
                    log_success("Step 4 completed")
                else:
                    log_error("Step 4 failed (critical, stopping)")
                    failed_steps.append((4, 'Train Linear Probe', False))
                    critical_failures.append((4, 'Train Linear Probe', False))

            # Check for failures
            probe_failures = [f for f in failed_steps if f[0] == 4]
            if probe_failures:
                log_error(f"Step 4 partially failed - {len(probe_failures)} probe(s) failed")
                critical_failures.extend(probe_failures)
            elif layer_mode == 'all' or demographic_mode == 'all':
                log_success("Step 4 completed - All probes trained")

        # === STEP 5: Compute IG2 Attribution ===
        # Note: For 'all' mode, we compute IG2 for each demographic using its specific probe
        # For layer_mode == 'all', we also iterate through layers
        if args.start_from <= 5 and not critical_failures:
            log_step(5, "Compute IG2 Attribution")
            update_step_status(5, "Compute IG2 Attribution")

            # Determine iteration scope
            if layer_mode == 'all' and demographic_mode == 'all':
                # Full matrix: all layers × all demographics
                total = len(layer_quantiles) * len(demographics)
                log_info(f"Computing IG2 for {len(layer_quantiles)} layers × {len(demographics)} demographics = {total} combinations...")
                count = 0
                for lq in layer_quantiles:
                    for demo in demographics:
                        count += 1
                        print(f"\n  [{count}/{total}] Computing IG2 for {lq} × {demo}...")
                        ig2_args = ['--stage', args.stage, '--sae_type', args.sae_type,
                                   '--layer_quantile', lq, '--num_steps', str(args.num_steps),
                                   '--demographic', demo]
                        success = run_script(scripts_dir / '05_compute_ig2.py', ig2_args)
                        if not success:
                            log_error(f"Failed to compute IG2 for {lq} × {demo}")
                            failed_steps.append((5, f'Compute IG2 ({lq}×{demo})', False))
                        else:
                            log_success(f"Completed IG2 for {lq} × {demo}")

            elif layer_mode == 'all':
                # All layers, single demographic
                log_info(f"Computing IG2 for {len(layer_quantiles)} layer quantiles...")
                demo = demographics[0] if demographics else None
                for j, lq in enumerate(layer_quantiles):
                    print(f"\n  [{j+1}/{len(layer_quantiles)}] Computing IG2 for layer {lq}...")
                    ig2_args = ['--stage', args.stage, '--sae_type', args.sae_type,
                               '--layer_quantile', lq, '--num_steps', str(args.num_steps)]
                    if demo:
                        ig2_args.extend(['--demographic', demo])
                    success = run_script(scripts_dir / '05_compute_ig2.py', ig2_args)
                    if not success:
                        log_error(f"Failed to compute IG2 for layer {lq}")
                        failed_steps.append((5, f'Compute IG2 ({lq})', False))
                    else:
                        log_success(f"Completed IG2 for layer {lq}")

            elif demographic_mode == 'all':
                # Single layer, all demographics
                log_info(f"Computing IG2 for {len(demographics)} demographic categories...")
                lq = layer_quantiles[0]
                for i, demo in enumerate(demographics):
                    print(f"\n  [{i+1}/{len(demographics)}] Computing IG2 for {demo}...")
                    ig2_args = ['--stage', args.stage, '--sae_type', args.sae_type,
                               '--layer_quantile', lq, '--num_steps', str(args.num_steps),
                               '--demographic', demo]
                    success = run_script(scripts_dir / '05_compute_ig2.py', ig2_args)
                    if not success:
                        log_error(f"Failed to compute IG2 for {demo}")
                        failed_steps.append((5, f'Compute IG2 ({demo})', False))
                    else:
                        log_success(f"Completed IG2 for {demo}")
            else:
                # Single layer, single demographic
                ig2_args = ['--stage', args.stage, '--sae_type', args.sae_type,
                           '--layer_quantile', layer_quantiles[0], '--num_steps', str(args.num_steps)]
                if demographics:
                    ig2_args.extend(['--demographic', demographics[0]])
                success = run_script(scripts_dir / '05_compute_ig2.py', ig2_args)
                if success:
                    log_success("Step 5 completed")
                else:
                    log_error("Step 5 failed (critical, stopping)")
                    failed_steps.append((5, 'Compute IG2', False))
                    critical_failures.append((5, 'Compute IG2', False))

            # Check for failures
            ig2_failures = [f for f in failed_steps if f[0] == 5]
            if ig2_failures:
                log_error(f"Step 5 partially failed - {len(ig2_failures)} IG2 computation(s) failed")
                critical_failures.extend(ig2_failures)
            elif layer_mode == 'all' or demographic_mode == 'all':
                log_success("Step 5 completed - All IG2 computations done")

        # === STEP 6: Verify Bias Features ===
        # Note: For 'all' mode, we verify each demographic using its specific probe and IG2 results
        # For layer_mode == 'all', we also iterate through layers
        if args.start_from <= 6 and not critical_failures:
            log_step(6, "Verify Bias Features")
            update_step_status(6, "Verify Bias Features")

            # Determine iteration scope
            if layer_mode == 'all' and demographic_mode == 'all':
                # Full matrix: all layers × all demographics
                total = len(layer_quantiles) * len(demographics)
                log_info(f"Verifying bias features for {len(layer_quantiles)} layers × {len(demographics)} demographics = {total} combinations...")
                count = 0
                for lq in layer_quantiles:
                    for demo in demographics:
                        count += 1
                        print(f"\n  [{count}/{total}] Verifying {lq} × {demo}...")
                        verify_args = ['--stage', args.stage, '--sae_type', args.sae_type,
                                      '--layer_quantile', lq, '--demographic', demo]
                        success = run_script(scripts_dir / '06_verify_bias_features.py', verify_args)
                        if not success:
                            log_error(f"Failed to verify {lq} × {demo}")
                            failed_steps.append((6, f'Verify ({lq}×{demo})', False))
                        else:
                            log_success(f"Completed verification for {lq} × {demo}")

            elif layer_mode == 'all':
                # All layers, single demographic
                log_info(f"Verifying bias features for {len(layer_quantiles)} layer quantiles...")
                demo = demographics[0] if demographics else None
                for j, lq in enumerate(layer_quantiles):
                    print(f"\n  [{j+1}/{len(layer_quantiles)}] Verifying layer {lq}...")
                    verify_args = ['--stage', args.stage, '--sae_type', args.sae_type, '--layer_quantile', lq]
                    if demo:
                        verify_args.extend(['--demographic', demo])
                    success = run_script(scripts_dir / '06_verify_bias_features.py', verify_args)
                    if not success:
                        log_error(f"Failed to verify layer {lq}")
                        failed_steps.append((6, f'Verify ({lq})', False))
                    else:
                        log_success(f"Completed verification for layer {lq}")

            elif demographic_mode == 'all':
                # Single layer, all demographics
                log_info(f"Verifying bias features for {len(demographics)} demographic categories...")
                lq = layer_quantiles[0]
                for i, demo in enumerate(demographics):
                    print(f"\n  [{i+1}/{len(demographics)}] Verifying {demo}...")
                    verify_args = ['--stage', args.stage, '--sae_type', args.sae_type,
                                  '--layer_quantile', lq, '--demographic', demo]
                    success = run_script(scripts_dir / '06_verify_bias_features.py', verify_args)
                    if not success:
                        log_error(f"Failed to verify {demo}")
                        failed_steps.append((6, f'Verify ({demo})', False))
                    else:
                        log_success(f"Completed verification for {demo}")
            else:
                # Single layer, single demographic
                verify_args = ['--stage', args.stage, '--sae_type', args.sae_type, '--layer_quantile', layer_quantiles[0]]
                if demographics:
                    verify_args.extend(['--demographic', demographics[0]])
                success = run_script(scripts_dir / '06_verify_bias_features.py', verify_args)
                if success:
                    log_success("Step 6 completed")
                else:
                    log_error("Step 6 failed (critical, stopping)")
                    failed_steps.append((6, 'Verify Bias Features', False))

            # Check for failures
            verify_failures = [f for f in failed_steps if f[0] == 6]
            if verify_failures:
                log_error(f"Step 6 partially failed - {len(verify_failures)} verification(s) failed")
            elif layer_mode == 'all' or demographic_mode == 'all':
                log_success("Step 6 completed - All verifications done")

    # Final summary
    print()
    print("=" * 80)
    # Check for critical failures (third element is False or not present)
    has_critical = any(len(f) < 3 or not f[2] for f in failed_steps)
    if not failed_steps or not has_critical:
        print(f"{Colors.GREEN}PIPELINE COMPLETE!{Colors.NC}")
        update_step_status(7, "Complete", "completed")
    else:
        print(f"{Colors.RED}PIPELINE FAILED!{Colors.NC}")
        update_step_status(-1, "Failed", "failed")
    print("=" * 80)
    print()

    # Update final status with end time
    status = load_status()
    if status:
        status['ended_at'] = datetime.datetime.now().isoformat()
        save_status(status)

    # Clean up PID file when running completes
    pid_file = get_pid_file()
    if pid_file.exists():
        pid_file.unlink()

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

    if layer_mode == 'all' and demographic_mode == 'all':
        # Full matrix: all layers × all demographics
        print(f"  - Per-demographic activations: results/{args.stage}/<demographic>/activations.pkl")
        print(f"  - Merged Activations:          results/{args.stage}/activations.pkl")
        print(f"  - Metadata:                    results/{args.stage}/activations_metadata.json")
        print(f"  - SAE Models (per layer):")
        for lq in layer_quantiles:
            print(f"      - {lq}: results/models/sae-{args.sae_type}_{args.stage}_{lq}/model.pth")
        print(f"  - Per-layer × demographic outputs:")
        for lq in layer_quantiles:
            print(f"      {lq}:")
            for demo in demographics:
                print(f"        {demo}:")
                print(f"          - Probe:        results/{args.stage}/{demo}/probe/{lq}_linear_probe.pt")
                print(f"          - IG2:          results/{args.stage}/{demo}/ig2/{lq}_ig2_results.pt")
                print(f"          - Verification: results/{args.stage}/{demo}/verification/{lq}/")
    elif layer_mode == 'all':
        # All layers, single demographic
        demo_name = demographics[0] if demographics else 'default'
        print(f"  - Activations:         results/{args.stage}/{demo_name}/activations.pkl")
        print(f"  - SAE Models (per layer):")
        for lq in layer_quantiles:
            print(f"      - {lq}: results/models/sae-{args.sae_type}_{args.stage}_{lq}/model.pth")
        print(f"  - Per-layer outputs:")
        for lq in layer_quantiles:
            print(f"      {lq}:")
            print(f"        - Probe:        results/{args.stage}/{demo_name}/probe/{lq}_linear_probe.pt")
            print(f"        - IG2:          results/{args.stage}/{demo_name}/ig2/{lq}_ig2_results.pt")
            print(f"        - Verification: results/{args.stage}/{demo_name}/verification/{lq}/")
    elif demographic_mode == 'all':
        # Single layer, all demographics
        lq = layer_quantiles[0]
        print(f"  - Per-demographic activations: results/{args.stage}/<demographic>/activations.pkl")
        print(f"  - Merged Activations:          results/{args.stage}/activations.pkl")
        print(f"  - Metadata:                    results/{args.stage}/activations_metadata.json")
        print(f"  - SAE Model (shared):          results/models/sae-{args.sae_type}_{args.stage}_{lq}/model.pth")
        print(f"  - Per-demographic outputs:")
        for demo in demographics:
            print(f"      {demo}:")
            print(f"        - Probe:        results/{args.stage}/{demo}/probe/linear_probe.pt")
            print(f"        - IG2:          results/{args.stage}/{demo}/ig2/ig2_results.pt")
            print(f"        - Verification: results/{args.stage}/{demo}/verification/")
    else:
        # Single layer, single demographic
        demo_name = demographics[0] if demographics else 'default'
        lq = layer_quantiles[0]
        print(f"  - Activations:         results/{args.stage}/{demo_name}/activations.pkl")
        print(f"  - SAE Model:           results/models/sae-{args.sae_type}_{args.stage}_{lq}/model.pth")
        print(f"  - Linear Probe:        results/{args.stage}/{demo_name}/probe/linear_probe.pt")
        print(f"  - IG2 Results:         results/{args.stage}/{demo_name}/ig2/ig2_results.pt")
        print(f"  - Verification:        results/{args.stage}/{demo_name}/verification/")
    print()

    if not failed_steps or not has_critical:
        log_success("Pipeline completed successfully!")
        return 0
    else:
        log_error("Pipeline completed with errors")
        return 1

if __name__ == '__main__':
    sys.exit(main())
