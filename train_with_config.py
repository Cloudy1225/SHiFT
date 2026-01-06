"""
train_with_config.py

Training script that loads configuration from YAML file
"""

import argparse
import yaml
from pathlib import Path
from model import HierarchicalTAGLearner
from utils import set_random_seed, get_cur_time
import logging
import os
from datetime import datetime

# Remove old logging handlers
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def config_to_args(config):
    """Convert config dict to argparse Namespace"""
    from argparse import Namespace
    return Namespace(**config)


def setup_logging(dataset_name):
    """
    Setup logging to both console and file.
    Log file is named with timestamp and dataset name.
    Example: logs/cora_20251202_103000.log
    """

    # Create log directory
    os.makedirs("logs", exist_ok=True)

    # Timestamp for log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construct log file path
    log_file = f"logs/{dataset_name}_{timestamp}.log"

    # Create handlers: file + console
    file_handler = logging.FileHandler(log_file, mode='w')
    stdout_handler = logging.StreamHandler()

    # Log message format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)

    # Clear old handlers (safety)
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    # Add new handlers
    logging.root.addHandler(file_handler)
    logging.root.addHandler(stdout_handler)
    logging.root.setLevel(logging.INFO)

    return log_file


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train with config file')
    # parser.add_argument('--config', type=str, required=True,
    #                     help='Path to configuration YAML file')
    parser.add_argument('--config', type=str, default="./configs/cora.yaml",
                        help='Path to configuration YAML file')
    parser.add_argument('--override', type=str, nargs='+',
                        help='Override config parameters (e.g., --override lr=0.01 device=cuda:1)')

    cmd_args = parser.parse_args()

    # Load YAML configuration file
    config = load_config(cmd_args.config)

    # Override parameters from command line
    if cmd_args.override:
        for override in cmd_args.override:
            key, value = override.split('=')
            # Try to convert to appropriate type
            try:
                value = eval(value)
            except:
                pass
            config[key] = value

    # Convert to args
    args = config_to_args(config)

    # Setup logging based on dataset name
    dataset_name = config.get("dataset", "unknown")
    log_path = setup_logging(dataset_name)
    logger.info(f"Log file: {log_path}")

    # Set random seed
    set_random_seed(args.random_seed)

    # Print config info
    logger.info("=" * 60)
    logger.info("Training with Configuration File")
    logger.info("=" * 60)
    logger.info(f"Config file: {cmd_args.config}")
    logger.info(f"Start time: {get_cur_time()}")
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    # Initialize and train
    learner = HierarchicalTAGLearner(args)

    # Phase 1: Warmup
    logger.info("\n" + "=" * 60)
    logger.info("Phase 1: Warmup Pretraining")
    logger.info("=" * 60)
    learner.warmup_pretrain()
    learner.evaluate(finetune=args.finetune)  # Evaluation after warmup

    # Phase 2: Initial taxonomy construction
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2: Initial Taxonomy Construction")
    logger.info("=" * 60)
    learner.build_initial_taxonomy()

    # Phase 3: Hierarchy-aware training
    logger.info("\n" + "=" * 60)
    logger.info("Phase 3: Hierarchy-Aware Training")
    logger.info("=" * 60)
    learner.hierarchy_aware_training()

    # Phase 4: Final evaluation
    logger.info("\n" + "=" * 60)
    logger.info("Phase 4: Evaluation")
    logger.info("=" * 60)
    learner.evaluate(finetune=args.finetune)

    logger.info("=" * 60)
    logger.info(f"End time: {get_cur_time()}")
    logger.info("Training completed successfully!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
