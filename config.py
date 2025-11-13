import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
import numpy as np
import yaml
from easydict import EasyDict as edict
from torch.utils.data import DataLoader, random_split
from torchmetrics import MetricCollection, R2Score
from torchmetrics.classification import Accuracy, F1Score, AUROC
from torchmetrics.regression import MeanAbsoluteError, ExplainedVariance, MeanSquaredError, R2Score


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_config(args):
    cfg = edict()
    
    # copy args
    for k,v in args.items():
        cfg[k] = v

    # copy model config
    model_config = load_config(f'configs/{cfg.dataset}/{cfg.model}.yaml')
    for k,v in model_config.items():
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
    
def load_dataset(config):
    # Check if dataset has H5 format specified in config
    if hasattr(config.data, 'format') and config.data.format == 'h5':
        from data.dataset import create_h5_dataset_from_config
        # Create datasets for train/val/test splits
        train_set = create_h5_dataset_from_config(config, 'train')
        test_set = create_h5_dataset_from_config(config, 'test')
        val_set = create_h5_dataset_from_config(config, 'val')
        
        # Set feature dimensions and number of features
        if hasattr(train_set, 'field_dims') and train_set.field_dims is not None:
            config.data.feature_dims = {f'{i}': int(train_set.field_dims[i]) for i in range(len(train_set.field_dims))}
            config.data.num_features = len(train_set.field_dims)
        
        return {'train': train_set, 'val': val_set, 'test': test_set}
    
    # Legacy dataset loading for backward compatibility
    if config.data.name == 'synthetic_v2':
        from data.synthetic import SyntheticDataset
        train_set = SyntheticDataset(tasks=config.data.tasks,
                                input_dim=config.data.num_features,
                                correlation=config.data.correlation,
                                num_samples=config.data.num_samples,
                                polynomial_degrees=config.data.poly_degrees,
                                noise_levels=config.data.noise_levels)
        test_set = SyntheticDataset(tasks=config.data.tasks,
                                input_dim=config.data.num_features,
                                correlation=config.data.correlation,
                                num_samples=config.data.num_samples,
                                polynomial_degrees=config.data.poly_degrees,
                                noise_levels=config.data.noise_levels)
        # copy all attributes from train_set to test_set
        test_set.__dict__.update(train_set.__dict__)
        # increase noise levels for test set
        test_set.noise_levels = [noise * 1 for noise in config.data.noise_levels]

        return {'train': train_set, 'val': test_set, 'test': test_set}

    elif config.data.name == 'aliexpress':
        from data.aliexpress import AliExpressDataset
        data_path = config.data_root + config.data.path
        h5_path = data_path + 'train_val_test.h5'
        train_set = AliExpressDataset(h5_path, split='train', seperate_ft_types=config.data.seperate_ft_types)
        val_set = AliExpressDataset(h5_path, split='val', seperate_ft_types=config.data.seperate_ft_types)
        test_set = AliExpressDataset(h5_path, split='test', seperate_ft_types=config.data.seperate_ft_types)
        config.data.feature_dims = {f'{i}': int(train_set.field_dims[i]) for i in range(len(train_set.field_dims))}
        config.data.num_features = len(train_set.field_dims)
        return {'train': train_set, 'val': val_set, 'test': test_set}
    else:
        raise NotImplementedError

def create_data_loaders(config, return_splits=False): 
    # Load the dataset
    dataset = load_dataset(config)

    if isinstance(dataset, dict):
        # Use pre-made splits
        train_dataset, val_dataset, test_dataset = dataset['train'], dataset['val'], dataset['test']
    else:
        # Calculate the number of samples for train and test splits
        train_size = int(config.data.split * len(dataset))
        test_size = len(dataset) - train_size
        
        # Split the dataset
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size], torch.Generator().manual_seed(42))
    
        # Always use validation - split the test set (val set) from the train set
        # val_size = int(0.1 * len(train_dataset))
        # train_size = len(train_dataset) - val_size
        # train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], torch.Generator().manual_seed(42))
        val_dataset = test_dataset
        val_dataset.dataset.noise_levels = [noise * 2 for noise in config.data.noise_levels] # Increase noise for validation set

    if return_splits:
        if isinstance(dataset, dict):
            return dataset
        else:
            return dataset, train_dataset.indices, val_dataset.indices, test_dataset.indices
    
    # Create DataLoader objects for train and test sets
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, num_workers=4, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, num_workers=4, pin_memory=True, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, num_workers=4, pin_memory=True, shuffle=False)
    return train_loader, val_loader, test_loader

def create_model(config):
    mt_tasks = config.data['tasks'].copy()
    mt_task_out_dim = config.data['task_out_dim'].copy()

    if config.model.name == 'mtt':
        from models.mtt import MTT
        return MTT(tasks=mt_tasks,
                   task_out_dim=mt_task_out_dim,
                   feature_dims=config.data['feature_dims'],
                   tower_hid_dims=config.model['tower_hid_dims'],
                   n_blocks=config.model['n_blocks'],
                   embed_dim=config.model['embed_dim'],
                   ff_hid_dim=config.model['ff_hid_dim'],
                   n_heads=config.model['n_heads'],
                   ff_dropout=config.training['ff_dropout'],
                   att_dropout=config.training['att_dropout'],
                   mask_mode_if=config.model['mask_mode_if'],
                   att_type=config.model['att_type'],
                   multi_token=config.model['multi_token'],
                   rope=config.model['rope'])
    
    elif config.model.name == 'st_mlp':
        from models.mlp import ST_MLP
        return nn.ModuleDict({task: ST_MLP(feature_dims=config.data['feature_dims'],
                                           output_dim=config.data['task_out_dim'][task],
                                           hidden_dims=config.model['hidden_dims'],
                                           embed_dim=config.model['embed_dim'],
                                           dropout=config.training['dropout']) for task in config.data['tasks']})
    
    elif config.model.name == 'mt_mlp':
        from models.mlp import MT_MLP
        return MT_MLP(tasks=mt_tasks,
                      task_out_dim=mt_task_out_dim,
                      feature_dims=config.data['feature_dims'],
                      shared_hidden_dims=config.model['shared_hidden_dims'],
                      task_hidden_dims=config.model['task_hidden_dims'],
                      embed_dim=config.model['embed_dim'],
                      dropout=config.training['dropout'])
    
    elif config.model.name == 'mmoe':
        from models.mmoe import MMoE
        return MMoE(tasks=mt_tasks,
                    output_dims=mt_task_out_dim,
                    feature_dims=config.data['feature_dims'],
                    expert_hidden_dims=config.model['expert_hidden_dims'],
                    tower_hidden_dims=config.model['tower_hidden_dims'],
                    num_experts=config.model['num_experts'],
                    embed_dim=config.model['embed_dim'],
                    dropout=config.training['dropout'])

    elif config.model.name == 'ple':
        from models.ple import PLE
        return PLE(tasks=mt_tasks,
                   task_output_dims=mt_task_out_dim,
                   feature_dims=config.data['feature_dims'],
                   num_layers=config.model['num_layers'],
                   num_shared_experts=config.model['num_shared_experts'],
                   num_task_experts=config.model['num_task_experts'],
                   expert_hidden_dims=config.model['expert_hidden_dims'],
                   tower_hidden_dims=config.model['tower_hidden_dims'],
                   embed_dim=config.model['embed_dim'],
                   dropout=config.training['dropout']) 

    elif config.model.name == 'stem':
        from models.stem import STEM
        return STEM(tasks=mt_tasks,
                    task_output_dims=mt_task_out_dim,
                    num_layers=config.model['num_layers'],
                    num_shared_experts=config.model['num_shared_experts'],
                    num_task_experts=config.model['num_task_experts'],
                    expert_hidden_dims=config.model['expert_hidden_dims'],
                    tower_hidden_dims=config.model['tower_hidden_dims'],
                    embed_dim=config.model['embed_dim'],
                    feature_dims=config.data['feature_dims'],
                    dropout=config.training['dropout']) 
    
    elif config.model.name == 'tabt':
        from models.tab_transformer import TabTransformer
        return nn.ModuleDict({task: TabTransformer(task_out_dim=config.data['task_out_dim'][task],
                                                   n_blocks=config.model['n_blocks'],
                                                   tower_hidden_dims=config.model['tower_hidden_dims'],
                                                   feature_dims=config.data['feature_dims'],
                                                   embed_dim=config.model['embed_dim'],
                                                   ff_hid_dim=config.model['ff_hid_dim'],
                                                   n_heads=config.model['n_heads'],
                                                   ff_dropout=config.training['ff_dropout'],
                                                   att_dropout=config.training['att_dropout']) for task in config.data['tasks']})

    elif config.model.name == 'ftt':
        from models.ft_transformer import FTT
        return nn.ModuleDict({task: FTT(task_out_dim=config.data['task_out_dim'][task],
                                        n_blocks=config.model['n_blocks'],
                                        tower_hidden_dims=config.model['tower_hidden_dims'],
                                        feature_dims=config.data['feature_dims'],
                                        embed_dim=config.model['embed_dim'],
                                        ff_hid_dim=config.model['ff_hid_dim'],
                                        n_heads=config.model['n_heads'],
                                        ff_dropout=config.training['ff_dropout'],
                                        att_dropout=config.training['att_dropout']) for task in config.data['tasks']}) 

    elif config.model.name == 'saint':
        from models.saint import SAINT
        return nn.ModuleDict({task: SAINT(task_out_dim=config.data['task_out_dim'][task],
                                          n_blocks=config.model['n_blocks'],
                                          tower_hidden_dims=config.model['tower_hidden_dims'],
                                          feature_dims=config.data['feature_dims'],
                                          embed_dim=config.model['embed_dim'],
                                          ff_hid_dim=config.model['ff_hid_dim'],
                                          n_heads=config.model['n_heads'],
                                          ff_dropout=config.training['ff_dropout'],
                                          att_dropout=config.training['att_dropout'],
                                          rope=config.model['rope']) for task in config.data['tasks']})
    else:
        raise NotImplementedError
    
def get_optimizer(config, model):
    if config.model.type == 'mt':
        if config.training.optimizer == 'adam':
            optimizer = Adam(model.parameters(), config.training.lr, weight_decay=config.training.weight_decay)
        elif config.training.optimizer == 'adamw':
            optimizer = AdamW(model.parameters(), config.training.lr, weight_decay=config.training.weight_decay)
        elif config.training.optimizer == 'sgd':
            optimizer = SGD(model.parameters(), config.training.lr)
        else:
            raise NotImplementedError
        return optimizer  
    elif config.model.type == 'st':
        if config.training.optimizer == 'adam':
            optimizer = [Adam(model[task].parameters(), config.training.lr, weight_decay=config.training.weight_decay) for task in config.data.tasks]
        elif config.training.optimizer == 'adamw':
            optimizer = [AdamW(model[task].parameters(), config.training.lr, weight_decay=config.training.weight_decay) for task in config.data.tasks]
        elif config.training.optimizer == 'sgd':
            optimizer = [SGD(model[task].parameters(), config.training.lr) for task in config.data.tasks]
        else:
            raise NotImplementedError
        return optimizer
    else:
        raise NotImplementedError
    
def get_metrics(config, prefix):
    metrics = {}
    for task in config.data.tasks:
        if config.data.task_type[task] == 'binary':
            task_metrics = MetricCollection([
                Accuracy(task='binary'),
                F1Score(task='binary'),
                AUROC(task='binary')
            ], prefix=f'{prefix}_{task}_')
        elif config.data.task_type[task] == 'classification':
            task_metrics = MetricCollection([
                Accuracy(task='multiclass', num_classes=config.data.task_out_dim[task]),
                F1Score(task='multiclass', num_classes=config.data.task_out_dim[task]),
                AUROC(task='multiclass', num_classes=config.data.task_out_dim[task])
            ], prefix=f'{prefix}_{task}_')
        elif config.data.task_type[task] == 'regression':
            task_metrics = MetricCollection([
                MeanAbsoluteError(),
                ExplainedVariance(),
                R2Score(),
                MeanSquaredError()
            ], prefix=f'{prefix}_{task}_')
        elif config.data.task_type[task] == 'resampled_denoising' or config.data.task_type[task] == 'masked_denoising':
            task_metrics = MetricCollection([
                MeanSquaredError(),
                MeanAbsoluteError(),
            ], prefix=f'{prefix}_{task}_')
        else:
            raise NotImplementedError
        metrics[task] = task_metrics
    metrics = nn.ModuleDict(metrics)
    return metrics      

def model_fit(pred, gt, task_type):
    if task_type == 'classification':
        loss = F.cross_entropy(pred, gt.long())
    elif task_type == 'binary':
        loss = F.binary_cross_entropy(F.sigmoid(pred).squeeze(), gt)
    elif task_type == 'regression':
        loss = F.mse_loss(pred.squeeze(), gt)
    elif task_type == 'resampled_denoising' or task_type == 'masked_denoising':
        loss = F.mse_loss(pred, gt)
    else:
        raise NotImplementedError
    return loss

def create_run_name(config):
    data = config.data.short_name
    lr = config.training.lr
    
    if config.model.name == 'mtt':
        b = config.model.n_blocks
        h = config.model.n_heads
        e = config.model.embed_dim
        hd = config.model.ff_hid_dim
        thd = config.model.tower_hid_dims
        fdrop = config.training.ff_dropout
        adrop = config.training.att_dropout
        mask_if = config.model.mask_mode_if
        mtk = 'mtk' if config.model.multi_token else 'stk'
        att = config.model.att_type
        rope = 'rope' if config.model.rope else ''
        name = f'{data}_mtt_b{b}_h{h}_e{e}_hd{hd}_thd{thd}_lr{lr}_fdrop{fdrop}_adrop{adrop}_att_{att}_maskif_{mask_if}_{mtk}_{rope}'
        return name
    
    elif config.model.name == 'st_mlp':
        hd = config.model.hidden_dims
        drop = config.training.dropout
        name = f'{data}_stmlp_hd{hd}_lr{lr}_drop{drop}'
        return name
    
    elif config.model.name == 'mt_mlp':
        shd = config.model.shared_hidden_dims
        thd = config.model.task_hidden_dims
        drop = config.training.dropout
        name = f'{data}_mtmlp_shd{shd}_thd{thd}_lr{lr}_drop{drop}'
        return name
    
    elif config.model.name == 'mmoe':
        ne = config.model.num_experts
        ehd = config.model.expert_hidden_dims
        thd = config.model.tower_hidden_dims
        drop = config.training.dropout
        name = f'{data}_mmoe_ne{ne}_ehd{ehd}_thd{thd}_lr{lr}_drop{drop}'
        return name
    
    elif config.model.name == 'ple':
        nl = config.model.num_layers
        nse = config.model.num_shared_experts
        nte = config.model.num_task_experts
        ehd = config.model.expert_hidden_dims
        thd = config.model.tower_hidden_dims
        drop = config.training.dropout
        name = f'{data}_ple_nl{nl}_nse{nse}_nte{nte}_ehd{ehd}_thd{thd}_lr{lr}_drop{drop}'
        return name
    
    elif config.model.name == 'stem':
        nl = config.model.num_layers
        nse = config.model.num_shared_experts
        nte = config.model.num_task_experts
        ehd = config.model.expert_hidden_dims
        thd = config.model.tower_hidden_dims
        e = config.model.embed_dim
        drop = config.training.dropout
        name = f'{data}_stem_nl{nl}_nse{nse}_nte{nte}_e{e}_ehd{ehd}_thd{thd}_lr{lr}_drop{drop}'
        return name
    
    elif config.model.name == 'tabt':
        b = config.model.n_blocks
        h = config.model.n_heads
        e = config.model.embed_dim
        hd = config.model.ff_hid_dim
        thd = config.model.tower_hidden_dims
        fdrop = config.training.ff_dropout
        adrop = config.training.att_dropout
        name = f'{data}_tabt_b{b}_h{h}_e{e}_hd{hd}_thd{thd}_lr{lr}_fdrop{fdrop}_adrop{adrop}'
        return name
    
    elif config.model.name == 'ftt':
        b = config.model.n_blocks
        h = config.model.n_heads
        e = config.model.embed_dim
        hd = config.model.ff_hid_dim
        thd = config.model.tower_hidden_dims
        fdrop = config.training.ff_dropout
        adrop = config.training.att_dropout
        name = f'{data}_ftt_b{b}_h{h}_e{e}_hd{hd}_thd{thd}_lr{lr}_fdrop{fdrop}_adrop{adrop}'
        return name
    
    elif config.model.name == 'saint':
        b = config.model.n_blocks
        h = config.model.n_heads
        e = config.model.embed_dim
        hd = config.model.ff_hid_dim
        thd = config.model.tower_hidden_dims
        fdrop = config.training.ff_dropout
        adrop = config.training.att_dropout
        rope = 'rope' if config.model.rope else ''
        name = f'{data}_saint_b{b}_h{h}_e{e}_hd{hd}_thd{thd}_lr{lr}_fdrop{fdrop}_adrop{adrop}_{rope}'
        return name

    else:
        raise NotImplementedError