import torch
import torch.nn as nn
import tqdm.auto as tqdm
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.special import softmax


class Engine(pl.LightningModule):
    
    def __init__(self, model):
        super(Engine, self).__init__()
        self.model = model
        self.train_losses = []
        self.valid_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []
    
    def forward(self, inputs):
        logits = self.model(inputs)
        return logits

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), 
            lr=1e-4, 
            betas=(0.95, 0.999)
        )
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y.long())
        acc = (logits.argmax(-1) == y).float()
        return {'loss': loss, 'acc': acc.detach(), 'log': {'train_loss': loss}}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        train_acc = torch.stack([x['acc'] for x in outputs]).mean()
        self.train_losses.append(avg_loss.detach().cpu().item())
        self.train_accuracies.append(train_acc.detach().cpu().item())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y.long())
        acc = (logits.argmax(-1) == y).float()
        return {'loss': loss, 'acc': acc.detach()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        val_acc = torch.stack([x['acc'] for x in outputs]).mean()
        out = {'val_loss': avg_loss, 'val_acc': val_acc}
        self.valid_losses.append(avg_loss.detach().cpu().item())
        self.valid_accuracies.append(val_acc.detach().cpu().item())
        return {**out, 'log': out}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y.long())
        acc = (logits.argmax(-1) == y).float()
        return {'test_loss': loss, 'test_acc': acc.detach()}

    def predict_proba(self, test_dataloader):
        y_probs = []
        with torch.no_grad():
            for _, batch in enumerate(test_dataloader, 0):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                # Forward pass with inputs
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                # Store outputs
                y_probs.extend(predicted.cpu())
        return softmax(np.vstack(y_probs), axis=1)

    def score(self, test_dataloader):
        correct, total = 0, 0
        with torch.no_grad():
            # Iterate over the test data and generate predictions
            for _, data in enumerate(test_dataloader, 0):
                # Get inputs
                inputs, targets = data
                # Generate outputs
                outputs = self(inputs)
                # Set total and correct
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            # Print accuracy
            print('Accuracy: %d %%' % (100 * correct / total))
        return correct / total

    def plot(self):
        plt.figure(figsize=(15, 6))
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.train_losses)), self.train_losses, label="train")
        plt.plot(range(len(self.valid_losses)), self.valid_losses, label="valid")
        plt.title("Loss")
        plt.legend(loc="upper right")
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.train_accuracies)), self.train_accuracies, label="train")
        plt.plot(range(len(self.valid_accuracies)), self.valid_accuracies, label="valid")
        plt.title("Accuracy")
        plt.legend(loc="upper right")
        plt.grid()
        plt.tight_layout()
        plt.show()