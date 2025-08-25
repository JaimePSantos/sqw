"""
Cluster Deployment Module

This module provides decorators and utilities for deploying quantum walk experiments
to cluster environments with automatic dependency management, virtual environment setup,
and result bundling.
"""

from .cluster_deployment import (
    cluster_deploy,
    cluster_experiment
)
from .config import (
    setup_cluster_environment,
    bundle_results,
    ClusterConfig
)

__version__ = "1.0.0"
__author__ = "Quantum Walk Project"

__all__ = [
    "cluster_deploy",
    "cluster_experiment",
    "setup_cluster_environment", 
    "bundle_results",
    "ClusterConfig"
]
