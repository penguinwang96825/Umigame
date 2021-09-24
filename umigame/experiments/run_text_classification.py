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


class BertNewsModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(
            "bert-base-uncased", return_dict=False
        )
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 2)
        self.freeze_bert()

    def forward(self, ids):
        _, o_2 = self.bert(ids)
        b_o = self.bert_drop(o_2)
        output = self.out(b_o)
        return output

    def freeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False


class LogisticRegression(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.1, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            bidirectional=bidirectional, 
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim*2, 2)
        self.act = nn.Sigmoid()
        
    def forward(self, text):
        # embedded: [batch, seq_len, emb_dim]
        embedded = self.embedding(text)
        # packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True)
        # hidden: [batch, num_layers * num_directions, hidden_dim]
        # cell: [batch, num_layers * num_directions, hidden_dim]
        packed_output, (hidden, cell) = self.lstm(embedded)
        # hidden: [batch, hidden_dim * num_directions]
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # outputs: [batch, num_classes]
        outputs = self.fc(hidden)
        return outputs


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
    model_engine = engine.Engine(LogisticRegression(vocab_size=30522, embedding_dim=100, hidden_dim=32, num_layers=2))
    trainer = pl.Trainer(
        gpus=(1 if torch.cuda.is_available() else None), 
        deterministic=True, 
        max_epochs=20, 
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