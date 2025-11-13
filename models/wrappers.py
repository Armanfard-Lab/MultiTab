import pytorch_lightning as pl
import torch
from config import create_model, get_metrics, get_optimizer, model_fit

class MultitaskModel(pl.LightningModule):
    def __init__(self, config):
        super(MultitaskModel, self).__init__()
        self.config = config
        self.tasks = self.config.data.tasks
        self.model = create_model(config)
        self.val_metrics = get_metrics(config, prefix='val')
        self.test_metrics = get_metrics(config, prefix='test')
        self.best_metric_mean = float('-inf')

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch['features'], batch['targets']
        out = self(x)
        task_losses = {task: model_fit(out[task], y[task], self.config.data.task_type[task]) for task in self.tasks}
        loss = sum(task_losses.values())
        self.log('train_total_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        for task in self.tasks:
            self.log(f'train_{task}_loss', task_losses[task], on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['features'], batch['targets']
        out = self(x)
        task_losses = {task: model_fit(out[task], y[task], self.config.data.task_type[task]) for task in self.tasks}
        loss = sum(task_losses.values())
        self.log('val_total_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        for task in self.tasks:
            self.log(f'val_{task}_loss', task_losses[task], on_step=False, on_epoch=True, sync_dist=True)
            if self.config.data.task_type[task] == 'classification':
                self.val_metrics[task].update(out[task], y[task].long())
            elif self.config.data.task_type[task] == 'binary':
                self.val_metrics[task].update(torch.sigmoid(out[task]).squeeze(), y[task])
            else:
                self.val_metrics[task].update(out[task].squeeze(), y[task])
        return loss
    
    def on_validation_epoch_end(self):
        best_metric_mean = []
        for task in self.tasks:
            output = self.val_metrics[task].compute()
            if self.config.data.task_type[task] == 'binary':
                best_metric_mean.append(output[f'val_{task}_BinaryAUROC'])
            elif self.config.data.task_type[task] == 'classification':
                best_metric_mean.append(output[f'val_{task}_MulticlassAUROC'])
            else:
                best_metric_mean.append(output[f'val_{task}_ExplainedVariance'])
            self.log_dict(output, sync_dist=True)
            self.val_metrics[task].reset()
        self.log(f'val_metric_mean', torch.mean(torch.tensor(best_metric_mean)), sync_dist=True)

        # Get val_metric_mean from callback_metrics
        current_val_metric_mean = self.trainer.callback_metrics.get(f'val_metric_mean')
        if current_val_metric_mean is not None:
            current_val_metric_mean = current_val_metric_mean.item()  # convert from tensor
            if current_val_metric_mean > self.best_metric_mean:
                self.best_metric_mean = current_val_metric_mean

            self.log("best_val_metric_mean", self.best_metric_mean, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch['features'], batch['targets']
        out = self(x)
        task_losses = {task: model_fit(out[task], y[task], self.config.data.task_type[task]) for task in self.tasks}
        loss = sum(task_losses.values())
        self.log('test_total_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        for task in self.tasks:
            self.log(f'test_{task}_loss', task_losses[task], on_step=False, on_epoch=True, sync_dist=True)
            if self.config.data.task_type[task] == 'classification':
                self.test_metrics[task].update(out[task], y[task].long())
            elif self.config.data.task_type[task] == 'binary':
                self.test_metrics[task].update(torch.sigmoid(out[task]).squeeze(), y[task])
            else:
                self.test_metrics[task].update(out[task].squeeze(), y[task])
        return loss
    
    def on_test_epoch_end(self):
        for task in self.tasks:
            output = self.test_metrics[task].compute()
            self.log_dict(output, sync_dist=True)
            self.test_metrics[task].reset()

    def configure_optimizers(self):
        optimizer = get_optimizer(self.config, self.model)
        return [optimizer]

    def load_model_weights(self, state_dict):
        self.model.load_state_dict(state_dict)

class SingletaskModel(pl.LightningModule):
    def __init__(self, config):
        super(SingletaskModel, self).__init__()
        self.config = config
        self.tasks = self.config.data.tasks
        self.model = create_model(config)
        self.val_metrics = get_metrics(config, prefix='val')
        self.test_metrics = get_metrics(config, prefix='test')
        self.best_mean_metric = float('-inf')
        self.automatic_optimization = False

    def forward(self, x):
        out = {}
        for task in self.config.data.tasks:
            out[task] = self.model[task](x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch['features'], batch['targets']
        out = self(x)
        task_losses = {task: model_fit(out[task], y[task], self.config.data.task_type[task]) for task in self.tasks}
        optimizers = self.optimizers()
        for i, task in enumerate(self.tasks):
            optimizers[i].zero_grad()
            self.manual_backward(task_losses[task])
            optimizers[i].step()
        loss = sum(task_losses.values())
        self.log('train_total_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        for task in self.tasks:
            self.log(f'train_{task}_loss', task_losses[task], on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['features'], batch['targets']
        out = self(x)
        task_losses = {task: model_fit(out[task], y[task], self.config.data.task_type[task]) for task in self.tasks}
        loss = sum(task_losses.values())
        self.log('val_total_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        for task in self.tasks:
            self.log(f'val_{task}_loss', task_losses[task], on_step=False, on_epoch=True, sync_dist=True)
            if self.config.data.task_type[task] == 'classification':
                self.val_metrics[task].update(out[task], y[task].long())
            elif self.config.data.task_type[task] == 'binary':
                self.val_metrics[task].update(torch.sigmoid(out[task]).squeeze(), y[task])
            else:
                self.val_metrics[task].update(out[task].squeeze(), y[task])
        return loss
    
    def on_validation_epoch_end(self):
        best_metric = []
        for task in self.tasks:
            output = self.val_metrics[task].compute()
            if self.config.data.task_type[task] == 'binary':
                best_metric.append(output[f'val_{task}_BinaryAUROC'])
            elif self.config.data.task_type[task] == 'classification':
                best_metric.append(output[f'val_{task}_MulticlassAUROC'])
            else:
                best_metric.append(output[f'val_{task}_ExplainedVariance'])
            self.log_dict(output, sync_dist=True)
            self.val_metrics[task].reset()
        self.log(f'val_metric_mean', torch.mean(torch.tensor(best_metric)), sync_dist=True)

        # Get val_metric_mean from callback_metrics
        current_val_metric_mean = self.trainer.callback_metrics.get(f'val_metric_mean')
        if current_val_metric_mean is not None:
            current_val_metric_mean = current_val_metric_mean.item()  # convert from tensor
            if current_val_metric_mean > self.best_mean_metric:
                self.best_mean_metric = current_val_metric_mean

            self.log("best_val_metric_mean", self.best_mean_metric, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch['features'], batch['targets']
        out = self(x)
        task_losses = {task: model_fit(out[task], y[task], self.config.data.task_type[task]) for task in self.tasks}
        loss = sum(task_losses.values())
        self.log('test_total_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        for task in self.tasks:
            self.log(f'test_{task}_loss', task_losses[task], on_step=False, on_epoch=True, sync_dist=True)
            if self.config.data.task_type[task] == 'classification':
                self.test_metrics[task].update(out[task], y[task].long())
            elif self.config.data.task_type[task] == 'binary':
                self.test_metrics[task].update(torch.sigmoid(out[task]).squeeze(), y[task])
            else:
                self.test_metrics[task].update(out[task].squeeze(), y[task])
        return loss

    def on_test_epoch_end(self):
        for task in self.tasks:
            output = self.test_metrics[task].compute()
            self.log_dict(output, sync_dist=True)
            self.test_metrics[task].reset()

    def configure_optimizers(self):
        optimizer = get_optimizer(self.config, self.model)
        return optimizer
    
    def load_model_weights(self, state_dicts):
        for task in self.config.data.tasks:
            self.model[task].load_state_dict(state_dicts[task])