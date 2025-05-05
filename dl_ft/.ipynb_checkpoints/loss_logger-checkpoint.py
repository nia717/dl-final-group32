import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

class LossLogger(Callback):
    def __init__(self, log_frequency=100, output_dir="./loss_logs"):
        super().__init__()
        self.log_frequency = log_frequency
        self.output_dir = output_dir
        self.train_losses = []
        self.val_losses = []
        self.global_steps = []
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if hasattr(outputs, 'get') and outputs.get('loss') is not None:
            loss = outputs['loss'].detach().cpu().numpy().item()
            if trainer.global_step % self.log_frequency == 0:
                self.train_losses.append(loss)
                self.global_steps.append(trainer.global_step)
                self._save_losses()
                self._plot_losses()
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if hasattr(outputs, 'get') and outputs.get('loss') is not None:
            loss = outputs['loss'].detach().cpu().numpy().item()
            if len(self.val_losses) < len(self.train_losses):
                self.val_losses.append(loss)
                self._save_losses()
                self._plot_losses()
    
    def _save_losses(self):
        os.makedirs(self.output_dir, exist_ok=True)
        loss_data = {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'global_steps': self.global_steps
        }
        with open(os.path.join(self.output_dir, 'loss_history.json'), 'w') as f:
            json.dump(loss_data, f)
    
    def _plot_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.global_steps, self.train_losses, label='Training Loss')
        if self.val_losses:
            # 只绘制与训练损失数量相同的验证损失
            val_steps = self.global_steps[:len(self.val_losses)]
            plt.plot(val_steps, self.val_losses, label='Validation Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'loss_curve.png'))
        plt.close()