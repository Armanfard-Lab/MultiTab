"""
Data module for multitab project.

This module contains dataset implementations and utilities for loading and processing
tabular data for multi-task learning.
"""

from .dataset import H5Dataset, create_h5_dataset_from_config
from .dataset_utils import (
    create_stratified_splits,
    create_h5_from_dataframe,
    validate_h5_dataset,
    inspect_h5_dataset,
    analyze_class_distributions
)

__all__ = [
    'H5Dataset',
    'create_h5_dataset_from_config',
    'create_stratified_splits',
    'create_h5_from_dataframe',
    'validate_h5_dataset',
    'inspect_h5_dataset',
    'analyze_class_distributions'
]
