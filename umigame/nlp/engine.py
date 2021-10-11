import re
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import dirname
from pathlib import Path
from sklearn import metrics
from sklearn.model_selection import train_test_split
from scipy.special import softmax


import os
import sys
CURRENT_DIR = os.getcwd()
sys.path.append(CURRENT_DIR)
MODULE_PATH = Path(dirname(__file__))
from umigame.nlp import labelling
from umigame.nlp.preprocess import Preprocessor
from umigame.datasets import fetch_crypto
from umigame.nlp.datasets import SimpleTextDataset


class Engine(pl.LightningModule):
    
    def __init__(
        self, 
        symbol="BTC-USD", 
        dataset="news", 
        label_type="ft", 
        vocab_size=10000, 
        embedding_dim=300, 
        batch_size=256, 
        num_workers=0, 
        lr=1e-4, 
        weight_decay=0.1
    ):
        super(Engine, self).__init__()
        self.symbol = symbol
        self.dataset = dataset
        self.label_type = label_type
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.weight_decay = weight_decay

        self.train_losses = []
        self.valid_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []
    
    def set_model(self, model):
        self.model = model

    def forward(self, inputs):
        return self.model(inputs)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay, 
            betas=(0.95, 0.999)
        )
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y.long())
        acc = (logits.argmax(-1) == y).detach().float()
        torch.cuda.empty_cache()
        return {'loss': loss, 'acc': acc, 'log': {'train_loss': loss.detach()}}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        train_acc = torch.stack([x['acc'] for x in outputs]).mean()
        self.train_losses.append(avg_loss.detach().cpu().item())
        self.train_accuracies.append(train_acc.detach().cpu().item())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y.long())
        acc = (logits.argmax(-1) == y).detach().float()
        torch.cuda.empty_cache()
        return {'loss': loss, 'acc': acc}

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
        acc = (logits.argmax(-1) == y).detach().float()
        torch.cuda.empty_cache()
        return {'test_loss': loss, 'test_acc': acc}

    def predict_proba(self, test_dl=None):
        if test_dl is None:
            test_dl = self.test_dataloader()
        y_probs = []
        with torch.no_grad():
            for _, batch in enumerate(test_dl, 0):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                outputs = self(inputs)
                y_probs.extend(outputs.cpu())
        return softmax(np.vstack(y_probs), axis=1)

    def predict(self, test_dl=None):
        if test_dl is None:
            test_dl = self.test_dataloader()
        y_prob = self.predict_proba(test_dl)
        y_pred = np.argmax(y_prob, axis=1)
        return y_pred

    def score(self, scoring="acc"):
        y_true = self.test_ds[:][1].numpy()
        y_prob = self.predict_proba(self.test_dataloader())
        y_pred = np.argmax(y_prob, axis=1)
        if scoring == "acc":
            return metrics.accuracy_score(y_true, y_pred)
        elif scoring == "f1":
            return metrics.f1_score(y_true, y_pred)
        elif scoring == "mcc":
            return metrics.matthews_corrcoef(y_true, y_pred)

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

    @staticmethod
    def download_data(dataset, symbol, label_type):
        # Load news dataset
        file_path = os.path.join(MODULE_PATH, "..", "datasets", "text", f"{dataset}.csv")
        news_df = pd.read_csv(file_path)
        news_df["text"] = news_df["text"].apply(lambda x: x.lower())
        # news_df = self._remain_rows_contain_keywords(news_df, "text", ["btc", "blockchain"])
        news_df.set_index("date", inplace=True, drop=True)
        news_df.index = pd.to_datetime(news_df.index)

        # Download price dataset using Yahoo API
        price_df = fetch_crypto([symbol])[symbol]
        price_df.index = pd.to_datetime(price_df.index)

        # Label generation for supervised learning
        if label_type == "ft":
            price_df["label"] = labelling.fixed_time_horizon(price_df, column='close', lookahead=1)
        elif label_type == "tb":
            price_df["label"] = labelling.triple_barrier(price_df, column='close')
        price_df = price_df.dropna()

        # Combine textual dataset and price dataset
        full_df = news_df.join(price_df)

        return full_df

    def prepare_data(self):
        full_df = self.download_data(self.dataset, self.symbol, self.label_type)
        train_size = len(full_df.loc[:"2017"])
        valid_size = len(full_df.loc["2017":"2018"])
        test_size = len(full_df.loc["2018":])
        print(
            f'TRAIN: {train_size}\n'
            f'VALID: {valid_size}\n'
            f'TEST: {test_size}\n'
        )

        # Tokenise the sentence in each document
        # text = [re.split('\W+', doc.lower()) for doc in full_df["text"]]
        text = full_df["text"]
        target = full_df["label"]
        max_length = max([len(doc) for doc in text])

        X_train, X_valid, y_train, y_valid = train_test_split(
            text[:-test_size], target[:-test_size], 
            test_size=0.2, 
            stratify=target[:-test_size]
        )

        processor = Preprocessor(vocab_size=self.vocab_size)
        processor.build(full_df["text"])
        # text = processor.encode(text, max_length=max_length, padding=True)
        X_train = processor.encode(X_train, max_length=max_length, padding=True)
        X_valid = processor.encode(X_valid, max_length=max_length, padding=True)
        X_test = processor.encode(text[:-test_size], max_length=max_length, padding=True)

        self.train_ds = SimpleTextDataset(
            # text[:train_size], target[:train_size]
            X_train, y_train
        )
        self.valid_ds = SimpleTextDataset(
            # text[train_size:train_size+valid_size], 
            # target[train_size:train_size+valid_size]
            X_valid, y_valid
        )
        self.test_ds = SimpleTextDataset(
            # text[-test_size:], 
            # target[-test_size:]
            X_test, 
            target[:-test_size]
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            drop_last=True, 
            num_workers=self.num_workers, 
            pin_memory=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.valid_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            drop_last=True, 
            num_workers=self.num_workers, 
            pin_memory=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.test_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            drop_last=False, 
            num_workers=self.num_workers, 
            pin_memory=True
        )
