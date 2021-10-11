import os
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from os.path import dirname
from pathlib import Path
from sklearn import metrics


import os
import sys
CURRENT_DIR = os.getcwd()
sys.path.append(CURRENT_DIR)
from umigame.nlp import utils
from umigame.nlp import labelling
from umigame.nlp.engine import Engine
from umigame.nlp.datasets import NewsWVDataset, NewsBertDataset
from umigame.nlp.nnets import TextLR, TextBERT, TextLSTM, TextCNN, TextAttention


MODULE_PATH = Path(dirname(__file__))
os.system("rm -r -f lightning_logs")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def test1():
    DATASET = "twitter"
    VOCAB_SIZE = 10000
    LABEL_TYPE = "tb"
    EMBED_DIM = 100
    BATCH_SIZE = 256
    WORKERS_NUM = 4
    EPOCH = 10
    LR = 1e-4
    WEIGHT_DECAY = 0.0

    # model = TextLR(
    #     vocab_size=VOCAB_SIZE, 
    #     embedding_dim=EMBED_DIM
    # )
    model = TextAttention(
        vocab_size=VOCAB_SIZE, 
        embedding_dim=EMBED_DIM, 
        trainable=False
    )
    # model = TextCNN(
    #     vocab_size=VOCAB_SIZE, 
    #     embedding_dim=EMBED_DIM, 
    #     channel_dim=64, 
    #     kernel_list=[3, 4, 5], 
    #     dropout=0.1
    # )
    # model = TextLSTM(
    #     vocab_size=VOCAB_SIZE, 
    #     embedding_dim=EMBED_DIM, 
    #     hidden_dim=16, 
    #     num_layers=2, 
    #     bidirectional=True
    # )
    engine = Engine(
        dataset=DATASET, 
        label_type=LABEL_TYPE, 
        vocab_size=VOCAB_SIZE, 
        embedding_dim=EMBED_DIM, 
        batch_size=BATCH_SIZE, 
        num_workers=WORKERS_NUM, 
        lr=LR, 
        weight_decay=WEIGHT_DECAY
    )
    engine.set_model(model)
    trainer = pl.Trainer(
        gpus=(1 if torch.cuda.is_available() else None), 
        deterministic=True, 
        max_epochs=EPOCH, 
        precision=(16 if torch.cuda.is_available() else 32), 
        num_sanity_val_steps=0, 
        fast_dev_run=False
    )
    trainer.fit(engine)
    print("F1: ", engine.score(scoring="f1"))
    print("ACC: ", engine.score(scoring="acc"))
    print(metrics.classification_report(engine.test_ds[:][1].numpy(), engine.predict()))
    engine.plot()


def main():
    train_ds = NewsBertDataset(state="train")
    valid_ds = NewsBertDataset(state="valid")
    test_ds = NewsBertDataset(state="test")

    train_dl = train_ds.create_dataloader(batch_size=128, shuffle=True, num_workers=4, drop_last=True)
    valid_dl = valid_ds.create_dataloader(batch_size=128, shuffle=False, num_workers=4, drop_last=True)
    test_dl = test_ds.create_dataloader(batch_size=128, shuffle=False, num_workers=4, drop_last=False)

    model_engine = Engine(
        TextBERT(
            dropout=0.2, 
            freeze=False
        )
    )
    trainer = pl.Trainer(
        gpus=(1 if torch.cuda.is_available() else None), 
        deterministic=True, 
        max_epochs=10, 
        precision=(16 if torch.cuda.is_available() else 32), 
        num_sanity_val_steps=0, 
        fast_dev_run=False
    )
    trainer.fit(model_engine, train_dl, valid_dl)
    y_test_proba = model_engine.predict_proba(test_dl)
    model_engine.plot()
    y_true, y_pred = test_ds[:][1].numpy(), np.argmax(y_test_proba, axis=1)
    acc = model_engine.score(test_dl)
    f1 = metrics.f1_score(y_true, y_pred, average="weighted")
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    print(f"F1: {f1:.4f} ACC: {acc:.4f} MCC: {mcc:.4f}")
    print(metrics.classification_report(y_true, y_pred))


if __name__ == "__main__":
    test1()