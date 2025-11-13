"""
Utility functions for working with the modular H5 dataset system.

This module provides helper functions for:
- Creating H5 datasets from pandas DataFrames with separate categorical/continuous features
- Validating H5 dataset structure (assumes separated features format)
- Inspecting H5 datasets
- Stratified sampling for multi-task learning
"""

import h5py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def create_stratified_splits(
    df: pd.DataFrame,
    tasks: List[str],
    task_types: Dict[str, str],
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/val/test splits for multi-task learning.
    
    Args:
        df: Input DataFrame
        tasks: List of task column names
        task_types: Dict mapping task names to types ('binary', 'classification', 'regression')
        train_split: Proportion for training set
        val_split: Proportion for validation set
        test_split: Proportion for test set
        random_seed: Random seed for reproducible splits
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    np.random.seed(random_seed)
    
    # Validate splits
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
    
    # Find classification tasks for stratification
    stratify_tasks = [task for task in tasks if task_types.get(task, 'binary') in ['binary', 'classification']]
    
    if not stratify_tasks:
        # No classification tasks, use random sampling
        logger.info("No classification tasks found, using random sampling")
        n_samples = len(df)
        indices = np.random.permutation(n_samples)
        
        train_end = int(train_split * n_samples)
        val_end = train_end + int(val_split * n_samples)
        
        train_df = df.iloc[indices[:train_end]]
        val_df = df.iloc[indices[train_end:val_end]]
        test_df = df.iloc[indices[val_end:]]
        
        return train_df, val_df, test_df
    
    # Create composite stratification column for multi-task stratification
    if len(stratify_tasks) == 1:
        stratify_col = df[stratify_tasks[0]]
    else:
        # For multiple classification tasks, create a composite key
        # Handle potential missing values by filling with a default value
        composite_parts = []
        for task in stratify_tasks:
            task_vals = df[task].fillna(-999)  # Use -999 as missing value indicator
            composite_parts.append(task_vals.astype(str))
        
        stratify_col = composite_parts[0]
        for part in composite_parts[1:]:
            stratify_col = stratify_col + "_" + part
    
    try:
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df,
            test_size=val_split + test_split,
            stratify=stratify_col,
            random_state=random_seed
        )
        
        # Second split: val vs test from temp_df
        if val_split > 0:
            # Calculate relative proportions for val/test split
            relative_test_size = test_split / (val_split + test_split)
            
            # Create stratification column for the remaining data
            if len(stratify_tasks) == 1:
                temp_stratify_col = temp_df[stratify_tasks[0]]
            else:
                temp_composite_parts = []
                for task in stratify_tasks:
                    task_vals = temp_df[task].fillna(-999)
                    temp_composite_parts.append(task_vals.astype(str))
                
                temp_stratify_col = temp_composite_parts[0]
                for part in temp_composite_parts[1:]:
                    temp_stratify_col = temp_stratify_col + "_" + part
            
            val_df, test_df = train_test_split(
                temp_df,
                test_size=relative_test_size,
                stratify=temp_stratify_col,
                random_state=random_seed
            )
        else:
            val_df = pd.DataFrame()
            test_df = temp_df
            
    except ValueError as e:
        # Fallback to random sampling if stratification fails
        logger.warning(f"Stratification failed ({str(e)}), falling back to random sampling")
        n_samples = len(df)
        indices = np.random.permutation(n_samples)
        
        train_end = int(train_split * n_samples)
        val_end = train_end + int(val_split * n_samples)
        
        train_df = df.iloc[indices[:train_end]]
        val_df = df.iloc[indices[train_end:val_end]]
        test_df = df.iloc[indices[val_end:]]
    
    logger.info(f"Created stratified splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Log class distributions for verification
    for task in stratify_tasks:
        logger.info(f"Task '{task}' distribution:")
        logger.info(f"  Train: {train_df[task].value_counts().to_dict()}")
        logger.info(f"  Val: {val_df[task].value_counts().to_dict()}")
        logger.info(f"  Test: {test_df[task].value_counts().to_dict()}")
    
    return train_df, val_df, test_df


def create_h5_from_dataframe(
    df: pd.DataFrame,
    output_path: str,
    tasks: List[str],
    task_types: Dict[str, str],
    categorical_cols: Optional[List[str]] = None,
    continuous_cols: Optional[List[str]] = None,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42,
    use_stratified: bool = True
):
    """
    Create an H5 dataset from a pandas DataFrame with stratified sampling.
    
    Args:
        df: Input DataFrame
        output_path: Path for output H5 file
        tasks: List of task column names
        task_types: Dict mapping task names to types ('binary', 'classification', 'regression')
        categorical_cols: List of categorical feature column names
        continuous_cols: List of continuous feature column names
        train_split: Proportion for training set
        val_split: Proportion for validation set
        test_split: Proportion for test set
        random_seed: Random seed for reproducible splits
        use_stratified: Whether to use stratified sampling for classification tasks
    """
    np.random.seed(random_seed)
    
    # Validate splits
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
    
    # Separate features and targets
    feature_cols = [col for col in df.columns if col not in tasks]
    
    if categorical_cols is None and continuous_cols is None:
        # Auto-detect categorical vs continuous
        categorical_cols = []
        continuous_cols = []
        
        for col in feature_cols:
            if df[col].dtype in ['object', 'category'] or df[col].nunique() < 50:
                categorical_cols.append(col)
            else:
                continuous_cols.append(col)
    elif categorical_cols is None:
        categorical_cols = [col for col in feature_cols if col not in continuous_cols]
    elif continuous_cols is None:
        continuous_cols = [col for col in feature_cols if col not in categorical_cols]
    
    # Ensure at least one feature type is specified
    if not categorical_cols and not continuous_cols:
        raise ValueError("At least one of categorical_cols or continuous_cols must be non-empty")
    
    # Create stratified splits
    if use_stratified:
        train_data, val_data, test_data = create_stratified_splits(
            df, tasks, task_types, train_split, val_split, test_split, random_seed
        )
    else:
        # Create random splits (legacy behavior)
        n_samples = len(df)
        indices = np.random.permutation(n_samples)
        
        train_end = int(train_split * n_samples)
        val_end = train_end + int(val_split * n_samples)
        
        train_data = df.iloc[indices[:train_end]]
        val_data = df.iloc[indices[train_end:val_end]]
        test_data = df.iloc[indices[val_end:]]
    
    # Create H5 file
    with h5py.File(output_path, 'w') as f:
        for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
            if len(split_data) == 0:  # Skip empty splits (e.g., when val_split = 0)
                continue
                
            split_group = f.create_group(split_name)
            
            # Store categorical features
            if categorical_cols:
                cat_data = split_data[categorical_cols].values.astype(np.int32)
                split_group.create_dataset('features_categorical', data=cat_data)
            
            # Store continuous features
            if continuous_cols:
                cont_data = split_data[continuous_cols].values.astype(np.float32)
                split_group.create_dataset('features_numerical', data=cont_data)
            
            # Store tasks
            for task in tasks:
                task_data = split_data[task].values.astype(np.float32)
                split_group.create_dataset(task, data=task_data)
    
    logger.info(f"Created H5 dataset at {output_path}")
    logger.info(f"Categorical features: {len(categorical_cols)}")
    logger.info(f"Continuous features: {len(continuous_cols)}")
    logger.info(f"Tasks: {len(tasks)}")
    logger.info(f"Train samples: {len(train_data)}")
    logger.info(f"Val samples: {len(val_data)}")
    logger.info(f"Test samples: {len(test_data)}")
    if use_stratified:
        logger.info("Used stratified sampling for classification tasks")


def validate_h5_dataset(h5_path: str, expected_tasks: List[str]) -> Dict[str, any]:
    """
    Validate the structure and contents of an H5 dataset.
    Assumes the dataset has separated categorical and continuous features.
    
    Args:
        h5_path: Path to H5 file
        expected_tasks: Expected task names
    
    Returns:
        Dictionary with validation results and dataset info
    """
    info = {
        'valid': True,
        'errors': [],
        'splits': [],
        'tasks_found': [],
        'num_samples': {},
        'feature_shapes': {}
    }
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # Check splits
            for split in ['train', 'val', 'test']:
                if split in f:
                    info['splits'].append(split)
                    split_group = f[split]
                    
                    # Check that at least one feature type exists
                    has_categorical = 'features_categorical' in split_group
                    has_numerical = 'features_numerical' in split_group
                    
                    if not has_categorical and not has_numerical:
                        info['errors'].append(f"No feature data found in split '{split}'. "
                                            f"Expected 'features_categorical' and/or 'features_numerical'")
                        info['valid'] = False
                        continue
                    
                    # Check sample counts and shapes
                    if has_categorical:
                        info['num_samples'][split] = len(split_group['features_categorical'])
                        info['feature_shapes'][f'{split}_categorical'] = split_group['features_categorical'].shape
                    
                    if has_numerical:
                        num_samples = len(split_group['features_numerical'])
                        info['feature_shapes'][f'{split}_numerical'] = split_group['features_numerical'].shape
                        
                        # Ensure consistent sample counts between feature types
                        if has_categorical and num_samples != info['num_samples'][split]:
                            info['errors'].append(f"Sample count mismatch in split '{split}': "
                                                f"categorical={info['num_samples'][split]}, numerical={num_samples}")
                            info['valid'] = False
                        elif not has_categorical:
                            info['num_samples'][split] = num_samples
                    
                    # Check tasks
                    for task in expected_tasks:
                        if task in split_group:
                            if task not in info['tasks_found']:
                                info['tasks_found'].append(task)
                            # Verify task sample count matches
                            if len(split_group[task]) != info['num_samples'][split]:
                                info['errors'].append(f"Task '{task}' sample count mismatch in split '{split}'")
                                info['valid'] = False
                        else:
                            info['errors'].append(f"Task '{task}' not found in split '{split}'")
                            info['valid'] = False
            
            # Check if all expected splits are present
            if len(info['splits']) == 0:
                info['errors'].append("No valid splits found")
                info['valid'] = False
            
            # Check if all tasks are found
            missing_tasks = set(expected_tasks) - set(info['tasks_found'])
            if missing_tasks:
                info['errors'].append(f"Missing tasks: {list(missing_tasks)}")
                info['valid'] = False
                
    except Exception as e:
        info['valid'] = False
        info['errors'].append(f"Error reading H5 file: {str(e)}")
    
    return info


def migrate_legacy_dataset(
    dataset_name: str,
    config_path: str,
    output_h5_path: str,
    data_root: str = '/storage/dsinod/datasets/'
):
    """
    Migrate a legacy dataset to H5 format.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'cdc_diabetes')
        config_path: Path to the dataset config file
        output_h5_path: Path for output H5 file
        data_root: Root directory for data files
    """
    # This is a placeholder for migration logic
    # Implementation would depend on the specific legacy dataset format
    raise NotImplementedError("Legacy dataset migration not yet implemented")


def inspect_h5_dataset(h5_path: str) -> None:
    """
    Print detailed information about an H5 dataset structure.
    
    Args:
        h5_path: Path to H5 file
    """
    def print_structure(name, obj):
        print(name)
        if isinstance(obj, h5py.Group):
            print(f"  Group with {len(obj)} items")
        elif isinstance(obj, h5py.Dataset):
            print(f"  Dataset: shape={obj.shape}, dtype={obj.dtype}")
    
    try:
        with h5py.File(h5_path, 'r') as f:
            print(f"H5 Dataset Structure: {h5_path}")
            print("=" * 50)
            f.visititems(print_structure)
    except Exception as e:
        print(f"Error inspecting H5 file: {str(e)}")


def analyze_class_distributions(h5_path: str, tasks: List[str]) -> Dict[str, Dict[str, Dict]]:
    """
    Analyze class distributions across splits for classification tasks.
    
    Args:
        h5_path: Path to H5 file
        tasks: List of task names to analyze
    
    Returns:
        Dictionary with distribution analysis results
    """
    distributions = {}
    
    try:
        with h5py.File(h5_path, 'r') as f:
            for task in tasks:
                distributions[task] = {}
                
                for split in ['train', 'val', 'test']:
                    if split in f and task in f[split]:
                        task_data = f[split][task][:]
                        unique_vals, counts = np.unique(task_data, return_counts=True)
                        
                        # Calculate proportions
                        total = len(task_data)
                        proportions = counts / total
                        
                        distributions[task][split] = {
                            'counts': dict(zip(unique_vals.tolist(), counts.tolist())),
                            'proportions': dict(zip(unique_vals.tolist(), proportions.tolist())),
                            'total_samples': total
                        }
        
        # Calculate distribution similarity across splits
        for task in tasks:
            if len(distributions[task]) > 1:
                splits = list(distributions[task].keys())
                logger.info(f"\nTask '{task}' distribution analysis:")
                
                for split in splits:
                    dist = distributions[task][split]
                    logger.info(f"  {split}: {dist['counts']} (total: {dist['total_samples']})")
                
    except Exception as e:
        logger.error(f"Error analyzing distributions: {str(e)}")
    
    return distributions


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="H5 Dataset Utilities")
    parser.add_argument("--inspect", type=str, help="Path to H5 file to inspect")
    parser.add_argument("--validate", type=str, help="Path to H5 file to validate")
    parser.add_argument("--tasks", nargs="+", help="Expected tasks for validation")
    
    args = parser.parse_args()
    
    if args.inspect:
        inspect_h5_dataset(args.inspect)
    
    if args.validate:
        if not args.tasks:
            print("Error: --tasks required for validation")
        else:
            result = validate_h5_dataset(args.validate, args.tasks)
            print("Validation Result:")
            print(f"Valid: {result['valid']}")
            if result['errors']:
                print("Errors:")
                for error in result['errors']:
                    print(f"  - {error}")
            print(f"Info: {result}")
