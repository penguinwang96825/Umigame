import os
import torch
import transformers
import torch.nn as nn
import pandas as pd
import pytorch_lightning as pl
from os.path import dirname
from pathlib import Path
from collections import Counter


import os
import sys
CURRENT_DIR = os.getcwd()
sys.path.append(CURRENT_DIR)
from umigame.nlp import labelling
from umigame.nlp import engine
from umigame.nlp.nnets import TextBERT, TextLSTM, TextCNN
from umigame.datasets import fetch_crypto


MODULE_PATH = Path(dirname(__file__))
os.system("rm -r -f lightning_logs")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class NewsDataset(torch.utils.data.Dataset):

    def __init__(self, text, label, max_len=16):
        self.text = text
        self.target = label
        self.tokeniser = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        text = " ".join(text.split())

        inputs = self.tokeniser.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )

        ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
        targets = torch.tensor(self.target[item], dtype=torch.float)

        return ids, targets

    def create_dataloader(self, batch_size, num_workers=0, shuffle=False, drop_last=False):
        return torch.utils.data.DataLoader(
            dataset=self, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            drop_last=drop_last, 
            num_workers=num_workers, 
            pin_memory=True
        )


def main():
    file_path = os.path.join(MODULE_PATH, "..", "datasets", "text", "news.csv")
    news_df = pd.read_csv(file_path)
    news_df.set_index("date", inplace=True, drop=True)
    news_df.index = pd.to_datetime(news_df.index)
    price_df = fetch_crypto(["BTC-USD"])["BTC-USD"]
    price_df.index = pd.to_datetime(price_df.index)
    price_df["label"] = labelling.triple_barrier(price_df, column='close')
    price_df = price_df.dropna()
    full_df = news_df.join(price_df)
    train_df = full_df.loc[:"2017"]
    valid_df = full_df.loc["2017":"2018"]
    test_df = full_df.loc["2018":]
    print(Counter(train_df.label))
    print(Counter(test_df.label))

    train_ds = NewsDataset(train_df["text"], train_df["label"])
    valid_ds = NewsDataset(valid_df["text"], valid_df["label"])
    test_ds = NewsDataset(test_df["text"], test_df["label"])

    train_dl = train_ds.create_dataloader(batch_size=128, shuffle=True, num_workers=4, drop_last=True)
    valid_dl = valid_ds.create_dataloader(batch_size=128, shuffle=False, num_workers=4, drop_last=True)
    test_dl = test_ds.create_dataloader(batch_size=128, shuffle=False, num_workers=4, drop_last=False)

    # model_engine = engine.Engine(BertNewsModel())
    model_engine = engine.Engine(TextCNN(vocab_size=30522, embedding_dim=200, channel_dim=64, kernel_list=[3, 4, 5], dropout=0.1, num_classes=2))
    # model_engine = engine.Engine(TextLSTM(vocab_size=30522, embedding_dim=100, hidden_dim=32, num_layers=2))
    trainer = pl.Trainer(
        gpus=(1 if torch.cuda.is_available() else None), 
        deterministic=True, 
        max_epochs=40, 
        precision=(16 if torch.cuda.is_available() else 32), 
        num_sanity_val_steps=0, 
        fast_dev_run=False
    )
    trainer.fit(model_engine, train_dl, valid_dl)
    y_test_proba = model_engine.predict_proba(test_dl)
    accuracy = model_engine.score(test_dl)
    model_engine.plot()


if __name__ == "__main__":
    main()