import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax
from tqdm.std import tqdm


class PlModule(pl.LightningModule):
    
    def __init__(
            self,  
            lr=0.001, 
            weight_decay=0.01
        ):
        super(PlModule, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.train_step_losses = []
        self.train_step_accuracies = []
        self.train_epoch_losses = []
        self.valid_epoch_losses = []
        self.train_epoch_accuracies = []
        self.valid_epoch_accuracies = []
        
    def forward(self, x):
        pass
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay, 
            betas=(0.95, 0.999)
        )
    
    def step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y.long())
        preds = torch.argmax(logits, dim=1)
        torch.cuda.empty_cache()
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self.step(batch)
        acc = (preds == y).detach().float()
        self.train_step_losses.append(loss.detach().cpu().item())
        self.train_step_accuracies.append(np.mean(acc.cpu().numpy()))
        torch.cuda.empty_cache()
        return {'loss': loss, 'acc': acc, 'log': {'train_loss': loss.detach()}}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        train_acc = torch.stack([x['acc'] for x in outputs]).mean()
        self.train_epoch_losses.append(avg_loss.detach().cpu().item())
        self.train_epoch_accuracies.append(train_acc.detach().cpu().item())

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self.step(batch)
        acc = (preds == y).detach().float()
        torch.cuda.empty_cache()
        return {'loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        val_acc = torch.stack([x['acc'] for x in outputs]).mean()
        out = {'val_loss': avg_loss, 'val_acc': val_acc}
        self.valid_epoch_losses.append(avg_loss.detach().cpu().item())
        self.valid_epoch_accuracies.append(val_acc.detach().cpu().item())
        return {**out, 'log': out}
    
    def plot(self):
        plt.figure(figsize=(15, 6))

        plt.subplot(2, 2, 1)
        plt.plot(range(len(self.train_epoch_losses)), self.train_epoch_losses, label="train")
        plt.plot(range(len(self.valid_epoch_losses)), self.valid_epoch_losses, label="valid")
        plt.title("Loss (per epoch)")
        plt.legend(loc="upper right")
        plt.grid()

        plt.subplot(2, 2, 2)
        plt.plot(range(len(self.train_epoch_accuracies)), self.train_epoch_accuracies, label="train")
        plt.plot(range(len(self.valid_epoch_accuracies)), self.valid_epoch_accuracies, label="valid")
        plt.title("Accuracy")
        plt.legend(loc="upper right")
        plt.grid()

        plt.subplot(2, 2, 3)
        plt.plot(range(len(self.train_step_losses)), self.train_step_losses, label="train")
        plt.title("Training loss (per step)")
        plt.grid()

        plt.subplot(2, 2, 4)
        plt.plot(range(len(self.train_step_accuracies)), self.train_step_accuracies, label="valid")
        plt.title("Training accuracy (per step)")
        plt.grid()

        plt.tight_layout()
        plt.show()
        
    def predict_proba(self, test_dataloader):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        self.eval()
        self.to(self.device)
        # print("Device: ", device)
        y_probs = []
        with torch.no_grad():
            for _, batch in enumerate(tqdm(test_dataloader, leave=False), 0):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                outputs = self(inputs)
                y_probs.extend(outputs.detach().cpu().numpy())
        return softmax(np.vstack(y_probs), axis=1)

    def predict(self, test_dataloader):
        y_prob = self.predict_proba(test_dataloader)
        y_pred = np.argmax(y_prob, axis=1)
        return y_pred