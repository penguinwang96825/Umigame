import os
import torch
import gensim
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from os.path import dirname
from pathlib import Path
from sklearn import metrics


import os
import sys
CURRENT_DIR = os.getcwd()
sys.path.append(CURRENT_DIR)
from umigame.nlp import utils
from umigame.nlp.engine import PlModule
from umigame.nlp.attention import SelfAttention
from umigame.nlp.utils import get_weights, get_dataloaders, get_dataframes


MODULE_PATH = Path(dirname(__file__))
os.system("rm -r -f lightning_logs")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
pd.options.mode.chained_assignment = None
utils.seed_everything()


DATASET = "twitter"
LABELING = 'tb'
WORD_EMBEDDING = 'word2vec'
EPOCH = 20
MAXLEN = 16
BATCH_SIZE = 256


class TextAttentionCNN(PlModule):
    
    def __init__(
            self, 
            weights, 
            freeze=False, 
            embedding_dim=300, 
            num_layers=1, 
            trainable=False, 
            channel_dim=64, 
            kernel_list=[3, 4, 5], 
            dropout=0.1, 
            num_classes=2, 
            *args, 
            **kwargs
        ):
        super(TextAttentionCNN, self).__init__(*args, **kwargs)
        self.embed = nn.Embedding.from_pretrained(weights, freeze=freeze)
        self.relu = nn.ReLU6(inplace=True)
        
        self.attn_layers = nn.ModuleList()
        for _ in range(num_layers):
            attn = SelfAttention(dimensions=embedding_dim, trainable=trainable)
            self.attn_layers.append(attn)
            
        self.convs = nn.ModuleList([nn.Conv2d(1, channel_dim, (w, embedding_dim)) for w in kernel_list])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_list)*channel_dim, num_classes)
        
    def forward(self, x):
        emb_x = self.embed(x)
        for attn_layer in self.attn_layers:
            skip_x = self.relu(emb_x)
            emb_x, _ = attn_layer(emb_x)
            emb_x = self.relu(emb_x)
            emb_x = skip_x + emb_x
        emb_x = emb_x.unsqueeze(1)
        con_x = [conv(emb_x) for conv in self.convs]
        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]
        fc_x = torch.cat(pool_x, dim=1)
        fc_x = fc_x.squeeze(-1)
        fc_x = self.dropout(fc_x)
        logit = self.fc(fc_x)
        return logit


def main():
    #########################################
    ### Load pre-trained Embedding Matrix ###
    #########################################

    if WORD_EMBEDDING == 'word2vec':
        print('Loading pre-trained word2vec model...')
        word2vec_path = r'F:\kaggle\embedding\processed\GoogleNews-vectors-negative300.bin'
        w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    elif WORD_EMBEDDING == 'glove':
        print('Loading pre-trained glove model...')
        glove_path = r'F:\kaggle\embedding\processed\glove.840B.300d.txt'
        w2v = gensim.models.KeyedVectors.load_word2vec_format(glove_path, binary=False)

    weights = get_weights(dataset=DATASET, w2v=w2v)

    ##################
    ### Train Code ###
    ##################

    train_dataloader, test_dataloader = get_dataloaders(dataset=DATASET, labeling=LABELING, maxlen=MAXLEN, batch_size=BATCH_SIZE)
    text_clf = TextAttentionCNN(
        torch.FloatTensor(weights), 
        freeze=False, 
        embedding_dim=300, 
        num_layers=4, 
        trainable=False, 
        channel_dim=64, 
        kernel_list=[3, 4, 5], 
        dropout=0.25, 
        lr=3e-3, 
        weight_decay=0.0
    )
    trainer = pl.Trainer(
        gpus=(1 if torch.cuda.is_available() else None), 
        deterministic=True, 
        max_epochs=EPOCH, 
        precision=(16 if torch.cuda.is_available() else 32), 
        num_sanity_val_steps=0, 
        fast_dev_run=False
    )
    trainer.fit(text_clf, train_dataloaders=train_dataloader)

    ##########################
    ### Evaluate the Model ###
    ##########################

    train_df, test_df = get_dataframes(dataset=DATASET)
    y_train, y_test = train_df[LABELING], test_df[LABELING]

    y_test_proba = text_clf.predict_proba(test_dataloader)
    y_test_pred = text_clf.predict(test_dataloader)

    acc = metrics.accuracy_score(y_test, y_test_pred)
    f1 = metrics.f1_score(y_test, y_test_pred, average="weighted")
    p = metrics.precision_score(y_test, y_test_pred, average="weighted")
    r = metrics.recall_score(y_test, y_test_pred, average="weighted")
    print(
        f'\nBefore Aggregate: \n'
        f"\tAcc: {acc:.4f}\n"
        f"\tP: {p:.4f}\n"
        f"\tR: {r:.4f}\n"
        f"\tF1: {f1:.4f}"
    )

    test_df.loc[:, "y_prob_0"] = np.array(y_test_proba)[:, 0]
    test_df.loc[:, "y_prob_1"] = np.array(y_test_proba)[:, 1]
    test_df.loc[:, "label"] = y_test
    test_tfal_df = test_df.groupby(test_df.index).agg({
        "y_prob_0":"mean", "y_prob_1":"mean", "label":"max", 
        "open":"max", "high":"max", "low":"max", "close":"max"
    })
    y_prob_tfal = test_tfal_df[["y_prob_0", "y_prob_1"]].values
    y_pred_tfal = np.argmax(y_prob_tfal, axis=1)
    y_true_tfal = test_tfal_df["label"].values.ravel()
    acc = metrics.accuracy_score(y_true_tfal, y_pred_tfal)
    f1 = metrics.f1_score(y_true_tfal, y_pred_tfal, average="weighted")
    p = metrics.precision_score(y_true_tfal, y_pred_tfal, average="weighted")
    r = metrics.recall_score(y_true_tfal, y_pred_tfal, average="weighted")
    print(
        f'\nAfter Aggregate: \n'
        f"\tAcc: {acc:.4f}\n"
        f"\tP: {p:.4f}\n"
        f"\tR: {r:.4f}\n"
        f"\tF1: {f1:.4f}"
    )
    print(metrics.confusion_matrix(y_true_tfal, y_pred_tfal))
    text_clf.plot()


if __name__ == '__main__':
    main()