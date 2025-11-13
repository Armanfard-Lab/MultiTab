import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from config import create_config, create_data_loaders, create_run_name
import argparse
import os
from models.wrappers import MultitaskModel, SingletaskModel

torch.set_float32_matmul_precision('high')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser()

# environment configurations
parser.add_argument('--data_root', required=True, type=str, help='root directory of the data')
parser.add_argument('--model', default='mtt', type=str, help='name of the model')
parser.add_argument('--dataset', default='acs_income', type=str, help='name of the dataset')
parser.add_argument('--seed', default=42, type=int, help='random seed for pytorch')
parser.add_argument('--patience', default=5, type=int, help='patience for early stopping')
parser.add_argument('--run_name', default='auto', type=str, help='override the name of the run')
parser.add_argument('--cuda', default=0, type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7], help='gpu index')

if __name__ == "__main__":
    # Create the configuration
    args = parser.parse_args()
    config = create_config(vars(args))

    # Set seeds
    seed_everything(config.seed, workers=True)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config)
    print(f'Loaded dataset: {config.data.name} | # train samples: {len(train_loader)*config.training.batch_size}')

    # Initialize the model
    model = MultitaskModel(config) if config.model.type == 'mt' else SingletaskModel(config)

    # Initialize CSV logger
    run_name = create_run_name(config) if config.run_name == 'auto' else config.run_name
    logger = CSVLogger(
        save_dir='logs',
        name=run_name,
        version=f'seed_{config.seed}'
    )
    
    # Log hyperparameters
    logger.log_hyperparams(config)

    early_stop = EarlyStopping(
        monitor='val_metric_mean',
        patience=config.patience,
        mode='max',
        verbose=True
    )

    checkpoint = ModelCheckpoint(
        monitor='val_metric_mean',
        dirpath='/storage/dsinod/cold/multi_tab',
        filename=f'{run_name}_seed_{config.seed}',
        save_top_k=1,
        mode='max',
        save_weights_only=True
    )

    # Train and test the model
    trainer = Trainer(
        max_epochs=config.training.epochs,
        logger=logger, 
        accelerator='gpu', 
        devices=[config.cuda],
        callbacks=[early_stop, checkpoint] if config.patience > 0 else None, 
        deterministic=True)
    
    trainer.fit(model, train_loader, val_loader)
    ckpt_path = checkpoint.best_model_path
    trainer.test(model, test_loader, ckpt_path)