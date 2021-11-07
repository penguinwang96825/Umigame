import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn import metrics
from abc import abstractmethod
from tqdm.auto import tqdm
from scipy.special import softmax
from collections import defaultdict
logging.basicConfig(format='%(asctime)s -  %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


import os
import sys
CURRENT_DIR = os.getcwd()
sys.path.append(CURRENT_DIR)
from umigame.nlp.attack import FGM, PGD


class Trainer(nn.Module):

    def __init__(self):
        super().__init__()
        self.history = defaultdict(list)

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def compile(self, loss_fn, optimiser):
        self.loss_fn = loss_fn
        self.optimiser = optimiser
    
    def fit(self, train_dataloader, valid_dataloader=None, max_epoch=10, gpu=True):
        device = "cuda" if gpu else "cpu"
        gpu_available = 'True' if torch.cuda.is_available() else 'False'
        gpu_used = 'True' if gpu else 'False'
        self.to(device)
        logger.info(f'GPU available: {gpu_available}, used: {gpu_used}')
        for epoch in range(1, max_epoch+1):
            try: 
                self.train()
                pbar = tqdm(train_dataloader, leave=False)
                losses = 0
                for step, (x, y) in enumerate(pbar):
                    pbar.set_description(f'Epoch {epoch}')
                    x, y = x.to(device), y.to(device)
                    pred = self(x)
                    loss = self.loss_fn(pred, y)
                    losses += loss.item()
                    self.optimiser.zero_grad()
                    loss.backward()
                    self.optimiser.step()
                    pbar.set_postfix({'train_loss':loss.item()})
                avg_loss = losses / len(train_dataloader.dataset)
                self.history['train_loss'].append(avg_loss)
                
                if valid_dataloader is not None:
                    avg_loss, accuracy = self.evaluate(valid_dataloader, gpu=gpu)
                    self.history['val_loss'].append(avg_loss)
            except KeyboardInterrupt:
                break

    def evaluate(self, valid_dataloader, gpu=True):
        device = "cuda" if gpu else "cpu"
        self.to(device)
        with torch.no_grad():
            self.eval()
            losses, correct = 0, 0
            y_hats, targets = [], []
            for x, y in valid_dataloader:
                x, y = x.to(device), y.to(device)
                pred = self(x)
                loss = self.loss_fn(pred, y)
                losses += loss.item()
                y_hat = torch.max(pred, 1)[1]
                y_hats += y_hat.tolist()
                targets += y.tolist()
                correct += (y_hat == y).sum().item()
        avg_loss = losses / len(valid_dataloader.dataset)
        accuracy = metrics.accuracy_score(targets, y_hats)
        return avg_loss, accuracy
    
    def predict_proba(self, test_dataloader, gpu=True):
        device = "cuda" if gpu else "cpu"
        self.eval()
        self.to(device)
        y_probs = []
        with torch.no_grad():
            for _, batch in enumerate(tqdm(test_dataloader, leave=False), 0):
                inputs, targets = batch
                inputs = inputs.to(device)
                outputs = self(inputs)
                y_probs.extend(outputs.detach().cpu().numpy())
        return softmax(np.vstack(y_probs), axis=1)
    
    def predict(self, test_dataloader, gpu=True):
        y_prob = self.predict_proba(test_dataloader, gpu=gpu)
        y_pred = np.argmax(y_prob, axis=1)
        return y_pred
    
    def plot(self):
        plt.figure(figsize=(15, 6))
        plt.plot(range(len(self.history['train_loss'])), self.history['train_loss'], label="train")
        plt.plot(range(len(self.history['val_loss'])), self.history['val_loss'], label="valid")
        plt.title("Loss (per epoch)")
        plt.legend(loc="upper right")
        plt.grid()
        plt.show()


class TrainerAT(Trainer):
    
    def fit(self, train_dataloader, valid_dataloader=None, max_epoch=10, attack=None, flooding=None, gpu=True):
        device = "cuda" if gpu else "cpu"
        gpu_available = 'True' if torch.cuda.is_available() else 'False'
        gpu_used = 'True' if gpu else 'False'
        self.to(device)
        logger.info(f'GPU available: {gpu_available}, used: {gpu_used}')
        fgm = FGM(self)
        pgd = PGD(self)
        for epoch in range(1, max_epoch+1):
            try: 
                self.train()
                pbar = tqdm(train_dataloader, leave=False)
                losses, fgm_losses, pgd_losses = 0, 0, 0
                for step, (x, y) in enumerate(pbar):
                    pbar.set_description(f'Epoch {epoch}')
                    x, y = x.to(device), y.to(device)
                    
                    # Normal training
                    pred = self(x)
                    loss = self.loss_fn(pred, y)
                    # ICML 2020: https://arxiv.org/pdf/2002.08709.pdf
                    if flooding is not None:
                        loss = (loss-flooding).abs() + flooding
                    losses += loss.item()
                    self.optimiser.zero_grad()
                    loss.backward()
                    
                    # Adversarial training (FGM)
                    if attack == 'fgm':
                        fgm.attack(emb_name='embed')
                        pred_adv = self(x)
                        loss_adv = self.loss_fn(pred_adv, y)
                        fgm_losses += loss_adv
                        loss_adv.backward()
                        fgm.restore(emb_name='embed')
                        
                    # Adversarial training (PGD)
                    pgd.backup_grad()
                    if attack == 'pgd':
                        K = 3
                        for t in range(K):
                            pgd.attack(is_first_attack=(t==0), emb_name='embed')
                            if t != K-1:
                                self.zero_grad()
                            else:
                                pgd.restore_grad()
                            pred_adv = self(x)
                            loss_adv = self.loss_fn(pred_adv, y)
                            pgd_losses += loss_adv
                            loss_adv.backward()
                        pgd.restore(emb_name='embed')
                    
                    self.optimiser.step()
                    pbar.set_postfix({'train_loss':loss.item()})
                avg_loss = losses / len(train_dataloader.dataset)
                avg_fgm_loss = fgm_losses / len(train_dataloader.dataset)
                avg_pgd_loss = pgd_losses / len(train_dataloader.dataset)
                self.history['train_loss'].append(avg_loss)
                self.history['train_fgm_loss'].append(avg_fgm_loss)
                self.history['train_pgd_loss'].append(avg_pgd_loss)
                
                if valid_dataloader is not None:
                    avg_loss, accuracy = self.evaluate(valid_dataloader, gpu=gpu)
                    self.history['val_loss'].append(avg_loss)
            except KeyboardInterrupt:
                break



def aggregate_proba_by_date(y_proba, date_index):
    df = pd.DataFrame({
        'y_prob_0':np.array(y_proba)[:, 0], 
        'y_prob_1':np.array(y_proba)[:, 1]
    }, index=date_index)
    df_agg = df.groupby(df.index).agg({
        "y_prob_0":"mean", "y_prob_1":"mean"
    })
    y_prob_agg = df_agg[["y_prob_0", "y_prob_1"]].values
    y_pred_agg = np.argmax(y_prob_agg, axis=1)
    return y_prob_agg, y_pred_agg

def aggregate_label_by_date(y_true, date_index):
    df = pd.DataFrame({
        'label':np.array(y_true)
    }, index=date_index)
    df_agg = df.groupby(df.index).agg({
        "label":"max"
    })
    y_true_agg = df_agg['label'].values
    return y_true_agg