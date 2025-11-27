"""
Korean Bias SAE - Full Experiment Runner

This script runs the complete experiment across:
- All 3 EXAONE layers: Q1 (25%), Q2 (50%), Q3 (75%)
- All 9 demographic dimensions

This enables comprehensive comparison visualizations in notebook 06.

Usage:
    # Run full experiment (all layers, all demographics)
    python scripts/run_pipeline.py --stage pilot

    # Run for specific layers only
    python scripts/run_pipeline.py --stage pilot --layers q1 q2

    # Run for specific demographics only
    python scripts/run_pipeline.py --stage pilot --demographics 성별 인종

    # Skip activation extraction (if already done)
    python scripts/run_pipeline.py --stage pilot --skip-extraction

    # Run in background
    python scripts/run_pipeline.py --stage pilot --background

Output Structure:
    results/
    ├── {stage}/
    │   ├── {demographic}/
    │   │   ├── activations.pkl          # Per-demographic activations (all 3 layers)
    │   │   ├── probe_{layer}/           # Probe for each layer
    │   │   │   └── linear_probe.pt
    │   │   ├── ig2_{layer}/             # IG2 results for each layer
    │   │   │   └── ig2_results.pt
    │   │   └── verification_{layer}/    # Verification for each layer
    │   └── activations.pkl              # Merged activations (for SAE training)
    └── models/
        ├── sae-gated_{stage}_q1/        # SAE for layer Q1
        ├── sae-gated_{stage}_q2/        # SAE for layer Q2
        └── sae-gated_{stage}_q3/        # SAE for layer Q3
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
from typing import List, Optional, Dict, Tuple
from itertools import product

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
    CYAN = '\033[0;36m'
    MAGENTA = '\033[0;35m'
    NC = '\033[0m'  # No Color
    BOLD = '\033[1m'

def log_info(msg: str):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {msg}")

def log_success(msg: str):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {msg}")

def log_warning(msg: str):
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {msg}")

def log_error(msg: str):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")

def log_step(step_num: str, step_name: str):
    print()
    print("=" * 80)
    print(f"{Colors.GREEN}STEP {step_num}: {step_name}{Colors.NC}")
    print("=" * 80)
    print()

def log_substep(msg: str):
    print(f"{Colors.CYAN}  >>> {msg}{Colors.NC}")


# Layer and demographic configurations
LAYER_QUANTILES = ["q1", "q2", "q3"]
LAYER_LABELS = {
    "q1": "Layer Q1 (25%)",
    "q2": "Layer Q2 (50%)",
    "q3": "Layer Q3 (75%)"
}


# Background execution helpers
def get_pid_file() -> Path:
    return PROJECT_ROOT / "logs" / "full_experiment.pid"

def get_status_file() -> Path:
    return PROJECT_ROOT / "logs" / "full_experiment_status.json"

def get_log_file(stage: str) -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "logs" / f"full_experiment_{stage}_{timestamp}.log"

def save_status(status: dict):
    status_file = get_status_file()
    status_file.parent.mkdir(parents=True, exist_ok=True)
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2, default=str, ensure_ascii=False)

def load_status() -> Optional[dict]:
    status_file = get_status_file()
    if status_file.exists():
        with open(status_file, 'r') as f:
            return json.load(f)
    return None

def is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def run_script(script_path: Path, args: List[str], verbose: bool = True) -> bool:
    """Run a Python script with arguments."""
    cmd = [sys.executable, str(script_path)] + args

    if verbose:
        log_info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        log_error(f"Script failed with exit code {e.returncode}")
        return False
    except Exception as e:
        log_error(f"Unexpected error: {e}")
        return False


# =========================================================================
# Model existence check functions
# =========================================================================

def check_sae_exists(stage: str, sae_type: str, layer_quantile: str) -> bool:
    """Check if SAE model already exists for given configuration."""
    sae_path = PROJECT_ROOT / 'results' / 'models' / f'sae-{sae_type}_{stage}_{layer_quantile}' / 'model.pth'
    return sae_path.exists()


def check_probe_exists(stage: str, demographic: str, layer_quantile: str = None) -> bool:
    """Check if linear probe already exists for given configuration."""
    if layer_quantile:
        # Per-layer probe path (for multi-layer mode) - this is now the primary path
        probe_path = PROJECT_ROOT / 'results' / stage / demographic / 'probe' / f'{layer_quantile}_linear_probe.pt'
        return probe_path.exists()
    # Legacy path (only for single-layer mode without layer_quantile specified)
    probe_path = PROJECT_ROOT / 'results' / stage / demographic / 'probe' / 'linear_probe.pt'
    return probe_path.exists()


def check_ig2_exists(stage: str, demographic: str, layer_quantile: str = None) -> bool:
    """Check if IG2 results already exist for given configuration."""
    if layer_quantile:
        # Per-layer IG2 path (for multi-layer mode) - this is now the primary path
        ig2_path = PROJECT_ROOT / 'results' / stage / demographic / 'ig2' / f'{layer_quantile}_ig2_results.pt'
        return ig2_path.exists()
    # Legacy path (only for single-layer mode without layer_quantile specified)
    ig2_path = PROJECT_ROOT / 'results' / stage / demographic / 'ig2' / 'ig2_results.pt'
    return ig2_path.exists()


def check_verification_exists(stage: str, demographic: str, layer_quantile: str = None) -> bool:
    """Check if verification results already exist for given configuration."""
    if layer_quantile:
        # Per-layer verification path
        verify_dir = PROJECT_ROOT / 'results' / stage / demographic / 'verification' / layer_quantile
    else:
        verify_dir = PROJECT_ROOT / 'results' / stage / demographic / 'verification'
    # Check if directory exists and has content
    if verify_dir.exists():
        return any(verify_dir.iterdir())
    return False


def check_activations_exist(stage: str, demographic: str = None) -> bool:
    """Check if activations already exist for given configuration."""
    if demographic:
        act_path = PROJECT_ROOT / 'results' / stage / demographic / 'activations.pkl'
    else:
        act_path = PROJECT_ROOT / 'results' / stage / 'activations.pkl'
    return act_path.exists()


def run_extraction_for_demographic(
    scripts_dir: Path,
    stage: str,
    demographic: str,
    verbose: bool = True
) -> bool:
    """Run activation extraction for a single demographic."""
    args = ['--stage', stage, '--demographic', demographic]
    return run_script(scripts_dir / '02_generate_and_extract_activations.py', args, verbose)


def run_sae_training(
    scripts_dir: Path,
    stage: str,
    sae_type: str,
    layer_quantile: str,
    verbose: bool = True
) -> bool:
    """Train SAE for a specific layer."""
    args = [
        '--stage', stage,
        '--sae_type', sae_type,
        '--layer_quantile', layer_quantile
    ]
    return run_script(scripts_dir / '03_train_sae.py', args, verbose)


def run_probe_training(
    scripts_dir: Path,
    stage: str,
    sae_type: str,
    layer_quantile: str,
    demographic: str,
    verbose: bool = True
) -> bool:
    """Train linear probe for a specific layer and demographic."""
    args = [
        '--stage', stage,
        '--sae_type', sae_type,
        '--layer_quantile', layer_quantile,
        '--demographic', demographic
    ]
    return run_script(scripts_dir / '04_train_linear_probe.py', args, verbose)


def run_ig2_computation(
    scripts_dir: Path,
    stage: str,
    sae_type: str,
    layer_quantile: str,
    demographic: str,
    num_steps: int = 20,
    verbose: bool = True
) -> bool:
    """Compute IG2 attribution for a specific layer and demographic."""
    args = [
        '--stage', stage,
        '--sae_type', sae_type,
        '--layer_quantile', layer_quantile,
        '--demographic', demographic,
        '--num_steps', str(num_steps)
    ]
    return run_script(scripts_dir / '05_compute_ig2.py', args, verbose)


def run_verification(
    scripts_dir: Path,
    stage: str,
    sae_type: str,
    layer_quantile: str,
    demographic: str,
    verbose: bool = True
) -> bool:
    """Run verification for a specific layer and demographic."""
    args = [
        '--stage', stage,
        '--sae_type', sae_type,
        '--layer_quantile', layer_quantile,
        '--demographic', demographic
    ]
    return run_script(scripts_dir / '06_verify_bias_features.py', args, verbose)


def print_progress_matrix(
    demographics: List[str],
    layers: List[str],
    completed: Dict[Tuple[str, str], List[str]],
    current: Optional[Tuple[str, str, str]] = None
):
    """Print a progress matrix showing completion status."""
    print()
    print(f"{Colors.BOLD}Progress Matrix:{Colors.NC}")
    print("-" * 70)

    # Header
    header = f"{'Demographic':<15} |"
    for lq in layers:
        header += f" {lq:^18} |"
    print(header)
    print("-" * 70)

    # Steps legend
    steps = ['E', 'S', 'P', 'I', 'V']  # Extraction, SAE, Probe, IG2, Verification

    for demo in demographics:
        row = f"{demo:<15} |"
        for lq in layers:
            key = (demo, lq)
            if key in completed:
                status = ''.join([s if s in completed[key] else '.' for s in steps])
            else:
                status = '.....'

            # Highlight current
            if current and current[0] == demo and current[1] == lq:
                status = f"{Colors.YELLOW}{status}{Colors.NC}"
            elif key in completed and len(completed[key]) == 5:
                status = f"{Colors.GREEN}{status}{Colors.NC}"

            row += f" {status:^18} |"
        print(row)

    print("-" * 70)
    print(f"Legend: E=Extraction, S=SAE, P=Probe, I=IG2, V=Verification")
    print(f"        {Colors.GREEN}Green{Colors.NC}=Complete, {Colors.YELLOW}Yellow{Colors.NC}=In Progress, .=Pending")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run full Korean Bias SAE experiment across all layers and demographics"
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
        '--layers',
        nargs='+',
        default=LAYER_QUANTILES,
        choices=LAYER_QUANTILES,
        help='Layer quantiles to process (default: all)'
    )
    parser.add_argument(
        '--demographics',
        nargs='+',
        default=None,
        help='Demographics to process (default: all 9)'
    )
    parser.add_argument(
        '--num_steps',
        type=int,
        default=20,
        help='Number of IG2 integration steps (default: 20)'
    )
    parser.add_argument(
        '--skip-extraction',
        action='store_true',
        help='Skip activation extraction (use existing)'
    )
    parser.add_argument(
        '--skip-sae',
        action='store_true',
        help='Skip SAE training (use existing)'
    )
    parser.add_argument(
        '--skip-verification',
        action='store_true',
        help='Skip verification step'
    )
    parser.add_argument(
        '--start-from-layer',
        type=str,
        default=None,
        choices=LAYER_QUANTILES,
        help='Start from specific layer'
    )
    parser.add_argument(
        '--start-from-demo',
        type=str,
        default=None,
        help='Start from specific demographic'
    )
    parser.add_argument(
        '--background',
        action='store_true',
        help='Run in background with logging'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Check background experiment status'
    )
    parser.add_argument(
        '--stop',
        action='store_true',
        help='Stop background experiment'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Handle status/stop commands
    if args.status:
        status = load_status()
        pid_file = get_pid_file()

        if pid_file.exists():
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())

            if is_process_running(pid):
                log_info(f"Full experiment is RUNNING (PID: {pid})")
                if status:
                    print(f"\nCurrent progress:")
                    print(f"  Stage: {status.get('stage', 'N/A')}")
                    print(f"  Current layer: {status.get('current_layer', 'N/A')}")
                    print(f"  Current demographic: {status.get('current_demographic', 'N/A')}")
                    print(f"  Current step: {status.get('current_step', 'N/A')}")
                    print(f"  Started: {status.get('started_at', 'N/A')}")
                    print(f"\nCompleted: {status.get('completed_count', 0)}/{status.get('total_count', 0)}")
                    if status.get('log_file'):
                        print(f"\nLog file: {status.get('log_file')}")
            else:
                log_info("Background experiment has STOPPED")
                pid_file.unlink()
        else:
            log_info("No background experiment is running")
            if status:
                print(f"\nLast run: {status.get('status', 'N/A')}")
        return 0

    if args.stop:
        pid_file = get_pid_file()
        if pid_file.exists():
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            if is_process_running(pid):
                log_info(f"Stopping experiment (PID: {pid})...")
                os.kill(pid, signal.SIGTERM)
                time.sleep(2)
                if is_process_running(pid):
                    os.kill(pid, signal.SIGKILL)
                pid_file.unlink()
                log_success("Experiment stopped")
            else:
                log_warning("Process not running")
                pid_file.unlink()
        else:
            log_warning("No background experiment to stop")
        return 0

    # Background execution
    if args.background:
        pid_file = get_pid_file()
        if pid_file.exists():
            with open(pid_file, 'r') as f:
                old_pid = int(f.read().strip())
            if is_process_running(old_pid):
                log_error(f"Experiment already running (PID: {old_pid})")
                return 1
            pid_file.unlink()

        log_file = get_log_file(args.stage)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Build command without --background
        cmd = [sys.executable, str(Path(__file__))]
        cmd.extend(['--stage', args.stage])
        cmd.extend(['--sae_type', args.sae_type])
        cmd.extend(['--layers'] + args.layers)
        if args.demographics:
            cmd.extend(['--demographics'] + args.demographics)
        cmd.extend(['--num_steps', str(args.num_steps)])
        if args.skip_extraction:
            cmd.append('--skip-extraction')
        if args.skip_sae:
            cmd.append('--skip-sae')
        if args.skip_verification:
            cmd.append('--skip-verification')

        log_info(f"Starting experiment in background...")
        log_info(f"Log file: {log_file}")

        with open(log_file, 'w') as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                start_new_session=True
            )

        pid_file.parent.mkdir(parents=True, exist_ok=True)
        with open(pid_file, 'w') as f:
            f.write(str(process.pid))

        log_success(f"Experiment started (PID: {process.pid})")
        print(f"\nTo monitor: tail -f {log_file}")
        print(f"To check status: python {__file__} --status")
        print(f"To stop: python {__file__} --stop")
        return 0

    # Get demographics
    all_demos = get_all_demographics()
    if args.demographics:
        demographics = args.demographics
    else:
        demographics = list(all_demos.keys())

    layers = args.layers

    # Calculate total experiments
    total_experiments = len(demographics) * len(layers)

    # Print configuration
    print()
    print("=" * 80)
    print(f"{Colors.BOLD}Korean Bias SAE - Full Experiment Runner{Colors.NC}")
    print("=" * 80)
    print()
    log_info("Configuration:")
    print(f"  Stage:           {args.stage}")
    print(f"  SAE Type:        {args.sae_type}")
    print(f"  Layers:          {', '.join([LAYER_LABELS[l] for l in layers])}")
    print(f"  Demographics:    {len(demographics)} categories")
    for d in demographics:
        print(f"                   - {d}")
    print(f"  Total experiments: {total_experiments}")
    print(f"  IG2 Steps:       {args.num_steps}")
    print()

    scripts_dir = Path(__file__).parent
    results_dir = PROJECT_ROOT / "results"

    # Track progress
    completed: Dict[Tuple[str, str], List[str]] = {}
    failed: List[Tuple[str, str, str]] = []

    # Initialize status
    save_status({
        'stage': args.stage,
        'layers': layers,
        'demographics': demographics,
        'started_at': datetime.datetime.now().isoformat(),
        'status': 'running',
        'total_count': total_experiments,
        'completed_count': 0
    })

    # Determine starting point
    start_layer_idx = 0
    start_demo_idx = 0
    if args.start_from_layer:
        start_layer_idx = layers.index(args.start_from_layer)
    if args.start_from_demo:
        start_demo_idx = demographics.index(args.start_from_demo)

    # =========================================================================
    # PHASE 1: Extract activations for all demographics
    # =========================================================================
    if not args.skip_extraction:
        log_step("1", "Extract Activations for All Demographics")

        for i, demo in enumerate(demographics):
            if i < start_demo_idx and start_layer_idx == 0:
                log_info(f"Skipping {demo} (before start point)")
                continue

            log_substep(f"[{i+1}/{len(demographics)}] Extracting activations for {demo}...")

            save_status({
                **load_status(),
                'current_step': 'extraction',
                'current_demographic': demo,
                'current_layer': 'all'
            })

            success = run_extraction_for_demographic(scripts_dir, args.stage, demo, args.verbose)

            if success:
                log_success(f"Completed extraction for {demo}")
                for lq in layers:
                    key = (demo, lq)
                    if key not in completed:
                        completed[key] = []
                    completed[key].append('E')
            else:
                log_error(f"Failed extraction for {demo}")
                for lq in layers:
                    failed.append((demo, lq, 'extraction'))

        # Merge activations for SAE training
        log_substep("Merging activations...")
        merge_args = ['--stage', args.stage]
        run_script(scripts_dir / 'merge_activations.py', merge_args, args.verbose)
    else:
        log_warning("Skipping activation extraction (--skip-extraction)")
        # Mark all as extracted
        for demo in demographics:
            for lq in layers:
                key = (demo, lq)
                if key not in completed:
                    completed[key] = []
                completed[key].append('E')

    # =========================================================================
    # PHASE 2: Train SAE for each layer
    # =========================================================================
    if not args.skip_sae:
        log_step("2", "Train SAE for Each Layer")

        for i, lq in enumerate(layers):
            if i < start_layer_idx:
                log_info(f"Skipping {LAYER_LABELS[lq]} (before start point)")
                continue

            # Check if SAE already exists
            if check_sae_exists(args.stage, args.sae_type, lq):
                log_warning(f"[{i+1}/{len(layers)}] SAE for {LAYER_LABELS[lq]} already exists, skipping...")
                for demo in demographics:
                    key = (demo, lq)
                    if key not in completed:
                        completed[key] = []
                    if 'S' not in completed[key]:
                        completed[key].append('S')
                continue

            log_substep(f"[{i+1}/{len(layers)}] Training SAE for {LAYER_LABELS[lq]}...")

            save_status({
                **load_status(),
                'current_step': 'sae_training',
                'current_layer': lq,
                'current_demographic': 'all'
            })

            success = run_sae_training(scripts_dir, args.stage, args.sae_type, lq, args.verbose)

            if success:
                log_success(f"Completed SAE training for {LAYER_LABELS[lq]}")
                for demo in demographics:
                    key = (demo, lq)
                    if key not in completed:
                        completed[key] = []
                    if 'S' not in completed[key]:
                        completed[key].append('S')
            else:
                log_error(f"Failed SAE training for {LAYER_LABELS[lq]}")
                for demo in demographics:
                    failed.append((demo, lq, 'sae'))
    else:
        log_warning("Skipping SAE training (--skip-sae)")
        for demo in demographics:
            for lq in layers:
                key = (demo, lq)
                if key not in completed:
                    completed[key] = []
                if 'S' not in completed[key]:
                    completed[key].append('S')

    # =========================================================================
    # PHASE 3: Train probes, compute IG2, and verify for each (layer, demographic)
    # =========================================================================
    log_step("3", "Train Probes, Compute IG2, and Verify")

    experiment_count = 0
    for lq_idx, lq in enumerate(layers):
        for demo_idx, demo in enumerate(demographics):
            experiment_count += 1

            # Skip if before start point
            if lq_idx < start_layer_idx:
                continue
            if lq_idx == start_layer_idx and demo_idx < start_demo_idx:
                continue

            key = (demo, lq)
            print()
            print(f"{Colors.MAGENTA}{'='*60}{Colors.NC}")
            print(f"{Colors.MAGENTA}Experiment {experiment_count}/{total_experiments}: {demo} + {LAYER_LABELS[lq]}{Colors.NC}")
            print(f"{Colors.MAGENTA}{'='*60}{Colors.NC}")

            # Initialize completed entry if not exists
            if key not in completed:
                completed[key] = []

            # 3a. Train linear probe
            probe_exists = check_probe_exists(args.stage, demo, lq)
            if probe_exists:
                log_warning(f"Probe for {demo} @ {lq} already exists, skipping...")
                if 'P' not in completed[key]:
                    completed[key].append('P')
            else:
                log_substep(f"Training probe for {demo} @ {lq}...")
                save_status({
                    **load_status(),
                    'current_step': 'probe_training',
                    'current_layer': lq,
                    'current_demographic': demo,
                    'completed_count': experiment_count - 1
                })

                success = run_probe_training(scripts_dir, args.stage, args.sae_type, lq, demo, args.verbose)

                if success:
                    log_success(f"Completed probe for {demo} @ {lq}")
                    completed[key].append('P')
                else:
                    log_error(f"Failed probe for {demo} @ {lq}")
                    failed.append((demo, lq, 'probe'))
                    continue  # Skip IG2 and verification if probe failed

            # 3b. Compute IG2
            ig2_exists = check_ig2_exists(args.stage, demo, lq)
            if ig2_exists:
                log_warning(f"IG2 for {demo} @ {lq} already exists, skipping...")
                if 'I' not in completed[key]:
                    completed[key].append('I')
            else:
                log_substep(f"Computing IG2 for {demo} @ {lq}...")
                save_status({
                    **load_status(),
                    'current_step': 'ig2_computation',
                    'current_layer': lq,
                    'current_demographic': demo
                })

                success = run_ig2_computation(
                    scripts_dir, args.stage, args.sae_type, lq, demo,
                    args.num_steps, args.verbose
                )

                if success:
                    log_success(f"Completed IG2 for {demo} @ {lq}")
                    completed[key].append('I')
                else:
                    log_error(f"Failed IG2 for {demo} @ {lq}")
                    failed.append((demo, lq, 'ig2'))
                    continue  # Skip verification if IG2 failed

            # 3c. Verification
            if not args.skip_verification:
                verification_exists = check_verification_exists(args.stage, demo, lq)
                if verification_exists:
                    log_warning(f"Verification for {demo} @ {lq} already exists, skipping...")
                    if 'V' not in completed[key]:
                        completed[key].append('V')
                else:
                    log_substep(f"Verifying {demo} @ {lq}...")
                    save_status({
                        **load_status(),
                        'current_step': 'verification',
                        'current_layer': lq,
                        'current_demographic': demo
                    })

                    success = run_verification(
                        scripts_dir, args.stage, args.sae_type, lq, demo, args.verbose
                    )

                    if success:
                        log_success(f"Completed verification for {demo} @ {lq}")
                        completed[key].append('V')
                    else:
                        log_error(f"Failed verification for {demo} @ {lq}")
                        failed.append((demo, lq, 'verification'))

            # Print progress matrix
            print_progress_matrix(demographics, layers, completed, (demo, lq, 'done'))

    # =========================================================================
    # Final Summary
    # =========================================================================
    print()
    print("=" * 80)
    print(f"{Colors.BOLD}EXPERIMENT COMPLETE{Colors.NC}")
    print("=" * 80)
    print()

    # Final progress matrix
    print_progress_matrix(demographics, layers, completed)

    # Count successes
    full_success = sum(1 for k, v in completed.items() if len(v) >= 4)  # E, S, P, I (V optional)

    print(f"\nSummary:")
    print(f"  Total experiments:    {total_experiments}")
    print(f"  Fully completed:      {full_success}")
    print(f"  Failed steps:         {len(failed)}")

    if failed:
        print(f"\n{Colors.RED}Failed experiments:{Colors.NC}")
        for demo, lq, step in failed:
            print(f"  - {demo} @ {lq}: {step}")

    # Update final status
    save_status({
        **load_status(),
        'status': 'completed' if not failed else 'completed_with_errors',
        'ended_at': datetime.datetime.now().isoformat(),
        'completed_count': full_success,
        'failed_count': len(failed)
    })

    # Clean up PID file
    pid_file = get_pid_file()
    if pid_file.exists():
        pid_file.unlink()

    print()
    log_info("Key outputs:")
    print(f"  Results directory: {results_dir / args.stage}/")
    print(f"  SAE models:        {results_dir / 'models'}/sae-{args.sae_type}_{args.stage}_*/")
    print()
    print(f"Next step: Run visualization notebook")
    print(f"  jupyter notebook notebooks/visualizations/06_visualize_layer_demographic_comparison.ipynb")
    print()

    if failed:
        log_warning("Experiment completed with some failures")
        return 1
    else:
        log_success("All experiments completed successfully!")
        return 0


if __name__ == '__main__':
    sys.exit(main())
