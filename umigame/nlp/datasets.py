import re
import gensim
import torch
import transformers
import pandas as pd
import numpy as np
from os.path import dirname
from pathlib import Path
from tqdm.auto import tqdm
from collections import Counter


import os
import sys
CURRENT_DIR = os.getcwd()
sys.path.append(CURRENT_DIR)
MODULE_PATH = Path(dirname(__file__))
from umigame.nlp import labelling
from umigame.nlp.preprocess import Preprocessor
from umigame.datasets import fetch_crypto


class SimpleTextDataset(torch.utils.data.Dataset):

    def __init__(self, texts, targets):
        self.texts = texts
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        text = torch.tensor(self.texts[item], dtype=torch.long)
        target = torch.tensor(self.targets[item], dtype=torch.float)
        return text, target


class NewsBertDataset(torch.utils.data.Dataset):
    """
    Vocab size: 30522
    """
    def __init__(self, state="train", max_length=16):
        file_path = os.path.join(MODULE_PATH, "..", "datasets", "text", "news.csv")
        news_df = pd.read_csv(file_path)
        news_df["text"] = news_df["text"].apply(lambda x: x.lower())
        # news_df = self._remain_rows_contain_keywords(news_df, "text", ["btc", "blockchain"])
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

        if state == "train":
            self.text = train_df["text"]
            self.target = train_df["label"]
        elif state == "valid":
            self.text = valid_df["text"]
            self.target = valid_df["label"]
        elif state == "test":
            self.text = test_df["text"]
            self.target = test_df["label"]

        self.tokeniser = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.max_length = max_length

    def __len__(self):
        return len(self.target)

    def __getitem__(self, item):
        text = str(self.text[item])
        text = " ".join(text.split())

        inputs = self.tokeniser.encode_plus(
            text,
            None,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True
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


class NewsWVDataset(torch.utils.data.Dataset):

    def __init__(self, vocab_size, state="train", label_type='ft', embedding_type="word2vec", embed_dim=300):
        file_path = os.path.join(MODULE_PATH, "..", "datasets", "text", "news.csv")
        news_df = pd.read_csv(file_path)
        news_df["text"] = news_df["text"].apply(lambda x: x.lower())
        # news_df = self._remain_rows_contain_keywords(news_df, "text", ["btc", "blockchain"])
        news_df.set_index("date", inplace=True, drop=True)
        news_df.index = pd.to_datetime(news_df.index)
        price_df = fetch_crypto(["BTC-USD"])["BTC-USD"]
        price_df.index = pd.to_datetime(price_df.index)
        if label_type == "ft":
            price_df["label"] = labelling.fixed_time_horizon(price_df, column='close', lookahead=1)
        elif label_type == "tb":
            price_df["label"] = labelling.triple_barrier(price_df, column='close')
        price_df = price_df.dropna()
        full_df = news_df.join(price_df)
        train_size = len(full_df.loc[:"2017"])
        valid_size = len(full_df.loc["2017":"2018"])
        test_size = len(full_df.loc["2018":])
        print(
            f'TRAIN: {train_size}\n'
            f'VALID: {valid_size}\n'
            f'TEST: {test_size}\n'
        )

        # Tokenise the sentence in each document
        self.text = [re.split('\W+', doc.lower()) for doc in full_df["text"]]
        self.target = full_df["label"]
        max_length = max([len(doc) for doc in self.text])

        processor = Preprocessor(vocab_size=vocab_size)
        processor.build(full_df["text"])
        self.text = processor.encode(full_df["text"], max_length=max_length, padding=True)
        self.word2idx = processor.word2idx
        self.n_vocab = len(self.word2idx)

        if embedding_type == 'word2vec':
            print("Getting word2vec embedding matrix...")
            self.get_word2vec()
        elif embedding_type == 'glove':
            print("Getting glove embedding matrix...")
            self.get_glove(embed_dim)

        if state == "train":
            self.text = self.text[:train_size]
            self.target = self.target[:train_size]
            print("TRAIN LABEL: ", Counter(self.target))
        elif state == "valid":
            self.text = self.text[train_size:train_size+valid_size]
            self.target = self.target[train_size:train_size+valid_size]
            print("VALID LABEL: ", Counter(self.target))
        elif state == "test":
            self.text = self.text[-test_size:]
            self.target = self.target[-test_size:]
            print("TEST LABEL: ", Counter(self.target))

    def __len__(self):
        return len(self.target)

    def __getitem__(self, item):
        texts = torch.tensor(self.text[item], dtype=torch.long)
        targets = torch.tensor(self.target[item], dtype=torch.float)

        return texts, targets

    def _remain_rows_contain_keywords(self, df, col, keywords):
        mask = df[col].apply(lambda x: any(item for item in keywords if item in x.split()))
        df = df[mask]
        return df

    def get_word2vec(self):
        word2vec_path = r"F:\kaggle\embedding\processed\GoogleNews-vectors-negative300.bin"
        w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        tmp = []
        for word, index in self.word2idx.items():
            try:
                tmp.append(w2v.get_vector(word))
            except:
                pass
        mean = np.mean(np.array(tmp))
        std = np.std(np.array(tmp))
        vocab_size = self.n_vocab
        embed_size = w2v.vectors.shape[1]
        weights = np.random.normal(mean, std , [vocab_size, embed_size])
        oov = []
        for word, index in tqdm(self.word2idx.items()):
            try:
                weights[index, :] = w2v.get_vector(word)
            except:
                oov.append(word)
        self.weights = weights
        print(f"OOF ratio: {len(oov)/weights.shape[0]:.4f}")

    def get_glove(self, embed_dim):
        if embed_dim == 25:
            glove_path = r"F:\kaggle\embedding\processed\glove.twitter.27B.25d.txt"
        elif embed_dim == 50:
            glove_path = r"F:\kaggle\embedding\processed\glove.twitter.27B.50d.txt"
        elif embed_dim == 100:
            glove_path = r"F:\kaggle\embedding\processed\glove.twitter.27B.100d.txt"
        elif embed_dim == 200:
            glove_path = r"F:\kaggle\embedding\processed\glove.twitter.27B.200d.txt"
        elif embed_dim == 300:
            glove_path = r"F:\kaggle\embedding\processed\glove.840B.300d.txt"
        glove = gensim.models.KeyedVectors.load_word2vec_format(glove_path, binary=False)
        tmp = []
        for word, index in tqdm(self.word2idx.items()):
            try:
                tmp.append(glove.get_vector(word))
            except:
                pass
        mean = np.mean(np.array(tmp))
        std = np.std(np.array(tmp))
        vocab_size = self.n_vocab
        embed_size = glove.vectors.shape[1]
        weights = np.random.normal(mean, std , [vocab_size, embed_size])
        oov = []
        for word, index in self.word2idx.items():
            try:
                weights[index, :] = glove.get_vector(word)
            except:
                oov.append(word)
        self.weights = weights
        print(f"OOF ratio: {len(oov)/weights.shape[0]:.4f}")

    def create_dataloader(self, batch_size, num_workers=0, shuffle=False, drop_last=False):
        return torch.utils.data.DataLoader(
            dataset=self, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            drop_last=drop_last, 
            num_workers=num_workers, 
            pin_memory=True
        )