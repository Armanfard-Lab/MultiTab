#!/usr/bin/env python3
"""
WandB Hyperparameter Sweep Script for Multi-Task Tabular Learning

Before running this script, please update the following:
1. Line 35: Set --data_root to your dataset directory path
2. Line 63: Set "program" path to the absolute path of this script
3. Line 100-101: Set your W&B project name and username/entity
4. Line 142: Set your W&B project name

Usage:
    python wandb_sweep.py --dataset <dataset_name> --model <model_name>
"""

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from config import create_config, create_data_loaders
from models.wrappers import MultitaskModel, SingletaskModel
import argparse
import wandb
from easydict import EasyDict as edict
import os
import yaml
from pytorch_lightning.callbacks import EarlyStopping

torch.set_float32_matmul_precision('high')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser()

# environment configurations
parser.add_argument('--model', default='mtt', type=str, help='name of the model')
parser.add_argument('--data_root', default='/path/to/your/datasets/', type=str, help='root directory of the data')
parser.add_argument('--dataset', default='acs_income', type=str, help='name of the dataset')
parser.add_argument('--seed', default=42, type=int, help='random seed for pytorch')
parser.add_argument('--patience', default=5, type=int, help='patience for early stopping')
parser.add_argument('--method', default='grid', type=str, help='sweep method')
parser.add_argument('--sweep_id', default='none', type=str, help='sweep id')
parser.add_argument('--run_name', default='auto', type=str, help='override the name of the run')

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def build_sweep_config(args):
    parameters = {}

    # copy general args
    for k,v in args.items():
        parameters[k] = {'value': v}

    # copy model args
    dataset = args['dataset']
    model_config = load_config(f'configs/{dataset}/sweep.yaml')
    for k,v in model_config[args['model']].items():
        parameters[k] = v
    
    sweep_config = {
        "name": f'{dataset}_{args["model"]}',
        "program": '/path/to/your/multitab/wandb_sweep.py',  # Update this path
        "method": args['method'],
        "metric": {"goal": "maximize", "name": "best_val_metric_mean"},
        "parameters": parameters,
    }

    return sweep_config

def create_config(args):
    cfg = edict()
    
    # copy general and model args
    for k,v in args.items():
        cfg[k] = v
    
    # copy data config
    data_config = load_config(f'configs/{cfg.dataset}/dataset.yaml')
    for k,v in data_config.items():
        cfg[k] = v
        
    if cfg.model.name == 'tabt':
        cfg.data.seperate_ft_types = True
    else:
        cfg.data.seperate_ft_types = False
    
    if cfg.data.name == 'synthetic_v2':
        cfg.data.feature_dims = {f'{i}':1 for i in range(cfg.data.num_features)}
        cfg.data.tasks = [f'task_{i}' for i in range(cfg.data.num_tasks)]
        cfg.data.task_out_dim = {task: 1 for task in cfg.data.tasks}
        cfg.data.task_type = {task: 'regression' for task in cfg.data.tasks}
    
    return cfg
    
def main():
    # Initialize wandb
    wandb.init()
    logger = WandbLogger(
        project='your_project_name',  # Update with your W&B project name
        entity='your_wandb_username',  # Update with your W&B username/entity
        reinit=True,
        settings=wandb.Settings(start_method="fork")
    )

    # Create the sweep configuration
    config = create_config(wandb.config)

    # Set seeds
    seed_everything(config.seed)

    # Create data loaders
    train_loader, validation_loader, _ = create_data_loaders(config)

    # Initialize the model
    model = MultitaskModel(config) if config.model.type == 'mt' else SingletaskModel(config)

    early_stop = EarlyStopping(
        monitor="val_metric_mean",
        patience=config.patience,
        mode="max"
    )

    # Train and test the model
    trainer = Trainer(
        max_epochs=config.training.epochs,
        logger=logger,
        enable_checkpointing=False,
        callbacks=[early_stop],
        accelerator='gpu',
        deterministic=True)
    trainer.fit(model, train_loader, validation_loader)

    # Explicitly finish the W&B run
    wandb.finish()

if __name__ == "__main__":
    args = parser.parse_args()
    sweep_configuration = build_sweep_config(vars(args))
    
    if vars(args)['sweep_id'] == 'none':
        sweep_id = wandb.sweep(sweep=sweep_configuration, project='your_project_name')  # Update with your W&B project name
    else:
        sweep_id = vars(args)['sweep_id']

    wandb.agent(sweep_id, function=main)