"""
Configuration management for static noise quantum walk experiments.

This module handles all configuration parameters, environment variable overrides,
and provides a centralized configuration object.
"""

import os
import math
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Union, Tuple, Optional


@dataclass
class ExperimentConfig:
    """Configuration class for static noise quantum walk experiments."""
    
    # Core experiment parameters
    N: int = 20000  # System size
    samples: int = 20  # Samples per deviation
    theta: float = field(default_factory=lambda: math.pi/3)  # Base theta parameter
    
    # Deviation values for static noise experiments
    devs: List[Union[float, Tuple[float, float]]] = field(default_factory=lambda: [
        (0, 0),        # No noise
        (0, 0.2),      # Small noise range
        (0, 0.6),      # Medium noise range  
        (0, 0.8),      # Medium noise range  
        (0, 1),        # Medium noise range  
    ])
    
    # Computed parameters
    steps: Optional[int] = None  # Will be computed as N//4 if not provided
    
    # Execution mode switches
    enable_plotting: bool = True
    use_loglog_plot: bool = True
    plot_final_probdist: bool = True
    save_figures: bool = True
    
    # Archive settings
    create_tar_archive: bool = False
    use_multiprocess_archiving: bool = True
    max_archive_processes: Optional[int] = None
    exclude_samples_from_archive: bool = True
    
    # Computation control
    calculate_samples_only: bool = False
    skip_sample_computation: bool = True
    
    # Background execution
    run_in_background: bool = False
    background_log_file: str = "static_experiment_multiprocessing.log"
    background_pid_file: str = "static_experiment_mp.pid"
    
    # Multiprocessing configuration
    max_processes: Optional[int] = None
    use_multiprocess_mean_prob: bool = True
    max_mean_prob_processes: Optional[int] = None
    
    # Timeout configuration
    base_timeout_per_sample: int = 30
    timeout_scale_factor: Optional[float] = None
    process_timeout: Optional[int] = None
    mean_prob_timeout_multiplier: float = 2.0
    mean_prob_timeout: Optional[int] = None
    
    # Directory configuration
    process_log_dir: str = "process_logs"
    
    def __post_init__(self):
        """Post-initialization to compute derived parameters and apply environment overrides."""
        # Compute steps if not provided
        if self.steps is None:
            self.steps = self.N // 4
            
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        # Compute multiprocessing parameters
        self._compute_multiprocessing_params()
        
        # Compute timeout parameters
        self._compute_timeout_params()
        
        # Validate configuration
        self._validate_config()
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        env_mappings = {
            'ENABLE_PLOTTING': ('enable_plotting', 'bool'),
            'CREATE_TAR_ARCHIVE': ('create_tar_archive', 'bool'),
            'USE_MULTIPROCESS_ARCHIVING': ('use_multiprocess_archiving', 'bool'),
            'MAX_ARCHIVE_PROCESSES': ('max_archive_processes', 'int'),
            'EXCLUDE_SAMPLES_FROM_ARCHIVE': ('exclude_samples_from_archive', 'bool'),
            'USE_MULTIPROCESS_MEAN_PROB': ('use_multiprocess_mean_prob', 'bool'),
            'MAX_MEAN_PROB_PROCESSES': ('max_mean_prob_processes', 'int'),
            'CALCULATE_SAMPLES_ONLY': ('calculate_samples_only', 'bool'),
            'SKIP_SAMPLE_COMPUTATION': ('skip_sample_computation', 'bool'),
            'RUN_IN_BACKGROUND': ('run_in_background', 'bool'),
            'FORCE_N_VALUE': ('N', 'int'),
        }
        
        for env_var, (attr_name, value_type) in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                try:
                    if value_type == 'bool':
                        setattr(self, attr_name, env_value.lower() == 'true')
                    elif value_type == 'int':
                        setattr(self, attr_name, int(env_value))
                except ValueError:
                    pass  # Ignore invalid values
        
        # Special handling for N value change
        if os.environ.get('FORCE_N_VALUE') and hasattr(self, '_n_changed'):
            self.steps = self.N // 4  # Recalculate steps
            
        # Check if background execution has been disabled externally
        if os.environ.get('RUN_IN_BACKGROUND') == 'False':
            self.run_in_background = False
    
    def _compute_multiprocessing_params(self):
        """Compute multiprocessing parameters."""
        cpu_count = mp.cpu_count()
        
        if self.max_processes is None:
            if self.N > 10000:
                # Use fewer processes for very large problems
                self.max_processes = min(len(self.devs), max(1, cpu_count // 2))
            else:
                self.max_processes = min(len(self.devs), cpu_count)
        
        if self.max_mean_prob_processes is None:
            self.max_mean_prob_processes = min(5, self.max_processes)
    
    def _compute_timeout_params(self):
        """Compute timeout parameters."""
        if self.timeout_scale_factor is None:
            self.timeout_scale_factor = (self.N * self.steps) / 1000000
        
        if self.process_timeout is None:
            self.process_timeout = max(3600, int(self.base_timeout_per_sample * self.samples * self.timeout_scale_factor))
        
        if self.mean_prob_timeout is None:
            self.mean_prob_timeout = max(7200, int(self.process_timeout * self.mean_prob_timeout_multiplier))
    
    def _validate_config(self):
        """Validate configuration for conflicts."""
        if self.calculate_samples_only and self.skip_sample_computation:
            raise ValueError("Cannot set both calculate_samples_only=True and skip_sample_computation=True")
    
    @property
    def initial_state_kwargs(self) -> dict:
        """Get initial state configuration."""
        return {"nodes": [self.N // 2]}
    
    @property
    def execution_mode(self) -> str:
        """Get human-readable execution mode."""
        if self.calculate_samples_only:
            return "Samples Only"
        elif self.skip_sample_computation:
            return "Analysis Only"
        else:
            return "Full Pipeline"
    
    def print_config_summary(self):
        """Print configuration summary."""
        print("=== EXECUTION CONFIGURATION ===")
        print(f"Execution mode: {self.execution_mode}")
        print(f"System size (N): {self.N}")
        print(f"Time steps: {self.steps}")
        print(f"Samples per deviation: {self.samples}")
        print(f"Number of deviations: {len(self.devs)}")
        print(f"Theta parameter: {self.theta:.4f}")
        print(f"Sample computation: {'Enabled' if not self.skip_sample_computation else 'Disabled'}")
        print(f"Analysis phase: {'Enabled' if not self.calculate_samples_only else 'Disabled'}")
        print(f"Multiprocessing: {self.max_processes} max processes")
        print(f"Mean probability multiprocessing: {'Enabled' if self.use_multiprocess_mean_prob else 'Disabled'}")
        if self.use_multiprocess_mean_prob:
            print(f"Max mean probability processes: {self.max_mean_prob_processes}")
        print(f"Plotting: {'Enabled' if self.enable_plotting else 'Disabled'}")
        print(f"Archiving: {'Enabled' if self.create_tar_archive else 'Disabled'}")
        if self.create_tar_archive:
            print(f"Multiprocess archiving: {'Enabled' if self.use_multiprocess_archiving else 'Disabled'}")
            if self.use_multiprocess_archiving:
                archive_processes = self.max_archive_processes or "auto-detect"
                print(f"Max archive processes: {archive_processes}")
            print(f"Exclude samples from archive: {'Yes' if self.exclude_samples_from_archive else 'No'}")
        print(f"Background execution: {'Enabled' if self.run_in_background else 'Disabled'}")
        print("=" * 40)
    
    def get_resource_estimates(self) -> dict:
        """Get computational resource estimates."""
        total_qw_simulations = len(self.devs) * self.samples
        estimated_time_per_sim = (self.N * self.steps) / 1000000  # rough estimate in minutes
        total_estimated_time = total_qw_simulations * estimated_time_per_sim
        
        # Memory estimation for streaming approach
        estimated_memory_per_process_mb = (self.N * 16 * 3) / (1024 * 1024)  # ~3 states max
        total_estimated_memory_mb = estimated_memory_per_process_mb * self.max_processes
        
        return {
            'total_qw_simulations': total_qw_simulations,
            'estimated_time_minutes': total_estimated_time,
            'memory_per_process_mb': estimated_memory_per_process_mb,
            'total_memory_mb': total_estimated_memory_mb,
            'process_timeout_seconds': self.process_timeout,
            'mean_prob_timeout_seconds': self.mean_prob_timeout
        }


def create_default_config(**kwargs) -> ExperimentConfig:
    """Create a default configuration with optional overrides."""
    return ExperimentConfig(**kwargs)
