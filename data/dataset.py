"""
Modular H5 Dataset Class for Multi-Task Tabular Learning

This module provides a unified dataset class that can handle any preprocessed H5 dataset
based on configuration specifications. The class assumes H5 datasets are preprocessed with
separated categorical and continuous features (both optional, but at least one required). 
The class handles:
- H5 file loading with separate categorical/continuous features
- Task-specific target handling
- Field dimension computation
"""

import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class H5Dataset(Dataset):
    """
    A modular dataset class for loading preprocessed H5 data based on configuration.
    
    Assumes H5 datasets have the structure:
    /split/
        features_categorical  # Categorical features (optional)
        features_numerical    # Continuous features (optional)
        task1                 # Task targets
        task2                 # Task targets
        ...
    
    Note: At least one of features_categorical or features_numerical must be present.
    """
    
    def __init__(
        self, 
        h5_path: str, 
        split: str,
        tasks: List[str],
        task_types: Dict[str, str],
        seperate_ft_types: bool = False
    ):
        """
        Initialize the H5Dataset.
        
        Args:
            h5_path: Path to the H5 file
            split: Dataset split ('train', 'val', 'test')
            tasks: List of task names
            task_types: Dictionary mapping task names to types ('binary', 'classification', 'regression')
            seperate_ft_types: Whether to separate categorical and continuous features
        """
        self.h5_path = h5_path
        self.split = split
        self.tasks = tasks
        self.task_types = task_types
        self.seperate_ft_types = seperate_ft_types
        
        # Initialize data storage
        self.features = None
        self.targets = None
        self.field_dims = None
        self.length = 0
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load data from H5 file with separated categorical and continuous features."""
        try:
            with h5py.File(self.h5_path, 'r') as file:
                split_group = file[self.split]
                
                # Load categorical features (optional)
                if 'features_categorical' in split_group:
                    cat_features = torch.from_numpy(split_group['features_categorical'][:]).float()
                else:
                    cat_features = None
                
                # Load continuous features (optional)
                if 'features_numerical' in split_group:
                    num_features = torch.from_numpy(split_group['features_numerical'][:]).float()
                else:
                    num_features = None
                
                # Ensure at least one feature type is present
                if cat_features is None and num_features is None:
                    raise ValueError(f"No feature data found in H5 file for split '{self.split}'. "
                                   f"Expected 'features_categorical' and/or 'features_numerical'.")
                
                # Handle empty tensors for missing feature types
                if cat_features is None:
                    cat_features = torch.empty(num_features.shape[0], 0)
                if num_features is None:
                    num_features = torch.empty(cat_features.shape[0], 0)
                
                # Organize features based on seperate_ft_types
                if self.seperate_ft_types:
                    self.features = {
                        'categorical': cat_features,
                        'continuous': num_features
                    }
                else:
                    self.features = torch.cat([cat_features, num_features], dim=1)
                
                # Load targets
                targets = {}
                for task in self.tasks:
                    if task in split_group:
                        targets[task] = torch.from_numpy(split_group[task][:]).float()
                    else:
                        logger.warning(f"Task '{task}' not found in H5 file for split '{self.split}'")
                
                self.targets = targets
                self.length = cat_features.shape[0] if cat_features.numel() > 0 else num_features.shape[0]
                
                # Compute field dimensions
                self._compute_field_dims(cat_features, num_features)
                
        except Exception as e:
            logger.error(f"Error loading H5 dataset from {self.h5_path}: {str(e)}")
            raise
    
    def _compute_field_dims(self, cat_features: torch.Tensor, num_features: torch.Tensor):
        """Compute field dimensions for categorical and numerical features."""
        field_dims = []
        
        # Categorical field dimensions (max value + 1)
        if cat_features.numel() > 0:
            cat_dims = torch.max(cat_features, dim=0).values + 1
            field_dims.extend(cat_dims.long().tolist())
        
        # Numerical field dimensions (always 1)
        if num_features.numel() > 0:
            num_dims = torch.ones(num_features.shape[1], dtype=torch.long)
            field_dims.extend(num_dims.tolist())
        
        self.field_dims = torch.tensor(field_dims, dtype=torch.long)
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Get a single sample from the dataset."""
        # Get features
        if self.seperate_ft_types and isinstance(self.features, dict):
            features = {
                'categorical': self.features['categorical'][idx].long(),
                'continuous': self.features['continuous'][idx].float()
            }
        else:
            features = self.features[idx].float()
        
        # Get targets
        targets = {}
        for task in self.tasks:
            if task in self.targets:
                targets[task] = self.targets[task][idx]
        
        return {
            'features': features,
            'targets': targets
        }


def create_h5_dataset_from_config(config, split: str) -> H5Dataset:
    """
    Create an H5Dataset instance from configuration.
    
    Args:
        config: Configuration object containing dataset specifications
        split: Dataset split ('train', 'val', 'test')
    
    Returns:
        H5Dataset instance configured according to the provided config
    """
    # Construct H5 path
    data_path = config.data_root + config.data.path
    
    h5_filename = 'train_val_test.h5'
    
    h5_path = data_path + h5_filename
    
    return H5Dataset(
        h5_path=h5_path,
        split=split,
        tasks=config.data.tasks,
        task_types=config.data.task_type,
        seperate_ft_types=config.data.seperate_ft_types
    )
