"""
Experiment utilities: logging, configuration, reproducibility.
"""

import logging
import random
import numpy as np
import torch
import yaml
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def load_config(config_path: str = "configs/experiment_config.yaml") -> Dict[str, Any]:
    """
    Load experiment configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Make CuDNN deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ExperimentLogger:
    """
    Experiment logger with file and console output, checkpointing support.
    """

    def __init__(self, experiment_name: str, output_dir: str, log_level: str = "INFO"):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Name of experiment
            output_dir: Directory for outputs
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        log_file = self.output_dir / f'{experiment_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(experiment_name)
        self.logger.info(f"Experiment '{experiment_name}' initialized")
        self.logger.info(f"Output directory: {self.output_dir}")

    def log(self, message: str, level: str = "INFO"):
        """Log a message"""
        log_func = getattr(self.logger, level.lower())
        log_func(message)

    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)

    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)

    def log_config(self, config: Dict[str, Any]):
        """
        Save configuration to JSON file.

        Args:
            config: Configuration dictionary
        """
        config_path = self.output_dir / 'config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Configuration saved to {config_path}")

    def save_checkpoint(self, name: str, data: Any):
        """
        Save checkpoint data.

        Args:
            name: Checkpoint name
            data: Data to save (will use torch.save)
        """
        checkpoint_path = self.output_dir / f'{name}.pt'
        torch.save(data, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, name: str) -> Any:
        """
        Load checkpoint data.

        Args:
            name: Checkpoint name

        Returns:
            Loaded data
        """
        checkpoint_path = self.output_dir / f'{name}.pt'
        if not checkpoint_path.exists():
            self.logger.error(f"Checkpoint not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        data = torch.load(checkpoint_path)
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return data

    def save_results(self, name: str, data: Dict[str, Any]):
        """
        Save results to JSON file.

        Args:
            name: Results file name
            data: Results dictionary
        """
        results_path = self.output_dir / f'{name}.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Results saved: {results_path}")

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        Log metrics (can be extended to support tensorboard/wandb).

        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step/epoch number
        """
        step_str = f" (step {step})" if step is not None else ""
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Metrics{step_str}: {metrics_str}")
