import os
import talib
import torch
import gensim
import pandas as pd
import numpy as np
import torch.nn as nn
import vectorbt as vbt
import matplotlib.pyplot as plt
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
from umigame.utils import crossover, crossunder
from umigame.nlp.utils import get_weights, get_dataloaders, get_dataframes


MODULE_PATH = Path(dirname(__file__))
os.system("rm -r -f lightning_logs")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
pd.options.mode.chained_assignment = None
utils.seed_everything()


DATASET = "news"
LABELING = 'tb'
WORD_EMBEDDING = 'word2vec'
EPOCH = 20
MAXLEN = 16
BATCH_SIZE = 256


class TextLR(PlModule):
    
    def __init__(
            self, 
            weights, 
            freeze=False, 
            embedding_dim=300, 
            dropout=0.1, 
            num_classes=2, 
            *args, 
            **kwargs
        ):
        super(TextLR, self).__init__(*args, **kwargs)
        self.embed = nn.EmbeddingBag.from_pretrained(weights, mode='mean', freeze=freeze)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x):
        emb_x = self.embed(x)
        emb_x = self.dropout(emb_x)
        logit = self.fc(emb_x)
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
    text_clf = TextLR(
        torch.FloatTensor(weights), 
        freeze=False, 
        embedding_dim=300, 
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
    test_tfal_df['trend'] = test_tfal_df['y_prob_1'].apply(lambda x: 1 if x >= 0.5 else 0)
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

    ###################
    ### Backtesting ###
    ###################

    def get_entry_and_exit(trigger):
        if trigger == 'sma':
            fast_ma = talib.SMA(test_tfal_df['close'], timeperiod=10)
            slow_ma = talib.SMA(test_tfal_df['close'], timeperiod=20)
            entries = crossover(fast_ma, slow_ma)
            exits = crossunder(fast_ma, slow_ma)
        
        elif trigger == 'wma':
            fast_ma = talib.WMA(test_tfal_df['close'], timeperiod=10)
            slow_ma = talib.WMA(test_tfal_df['close'], timeperiod=20)
            entries = crossover(fast_ma, slow_ma)
            exits = crossunder(fast_ma, slow_ma)

        elif trigger == 'ema':
            fast_ma = talib.EMA(test_tfal_df['close'], timeperiod=10)
            slow_ma = talib.EMA(test_tfal_df['close'], timeperiod=20)
            entries = crossover(fast_ma, slow_ma)
            exits = crossunder(fast_ma, slow_ma)

        return entries, exits

    plt.figure(figsize=(15, 3))
    perf = pd.DataFrame()
    for trigger in ['sma', 'ema', 'wma']:
        entries, exits = get_entry_and_exit(trigger)

        pf_kwargs = dict(
            size=np.inf, 
            fees=0.001, 
            slippage=0.001, 
            freq='1D', 
            init_cash=10000, 
            sl_stop=0.5, 
            tp_stop=1.5
        )
        pf_trigger_without_filter = vbt.Portfolio.from_signals(test_tfal_df['close'], entries, exits, **pf_kwargs)
        trigger_without_filter_returns = (pf_trigger_without_filter.daily_returns() + 1).cumprod()
        benchmark_returns = (pf_trigger_without_filter.benchmark_returns() + 1).cumprod()

        new_entries = entries & (test_tfal_df['trend']==0)
        new_exits = exits
        pf_trigger_with_filter = vbt.Portfolio.from_signals(test_tfal_df['close'], new_entries, new_exits, **pf_kwargs)
        trigger_with_filter_returns = (pf_trigger_with_filter.daily_returns() + 1).cumprod()

        perf = pd.concat([
            perf, 
            pd.DataFrame(pf_trigger_without_filter.stats(), columns=[f'{trigger}']), 
            pd.DataFrame(pf_trigger_with_filter.stats(), columns=[f'{trigger}-with-filter'])
        ], axis=1)

        plt.plot(trigger_without_filter_returns, label=f'{trigger}')
        plt.plot(trigger_with_filter_returns, label=f'{trigger} with filter')

    plt.plot(benchmark_returns, label='buy and hold')
    plt.legend()
    plt.grid()
    plt.show()

    print(perf)


if __name__ == '__main__':
    main()