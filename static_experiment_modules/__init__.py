"""
Static Experiment Modules

This package contains all the modular components for running static noise
quantum walk experiments.

Modules:
- experiment_config: Configuration management
- experiment_orchestrator: Main experiment coordination
- worker_functions: Multiprocessing workers
- data_manager: Data processing and archiving
- experiment_logging: Logging utilities
- system_monitor: System monitoring and resource management
- background_executor: Background process execution
"""

from .experiment_config import ExperimentConfig, create_default_config
from .experiment_orchestrator import run_static_experiment

__version__ = "1.0.0"
__all__ = [
    "ExperimentConfig",
    "create_default_config", 
    "run_static_experiment"
]
