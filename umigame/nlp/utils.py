import os
import torch
import random
import numpy as np
import pandas as pd
from itertools import starmap, repeat, chain, islice
from torch.utils.data import TensorDataset, DataLoader


import os
import sys
CURRENT_DIR = os.getcwd()
sys.path.append(CURRENT_DIR)
from umigame.nlp import utils
from umigame.datasets import fetch_crypto
from umigame.datasets.text import load_dataset
from umigame.nlp.labelling import fixed_time_horizon, triple_barrier
from umigame.nlp.preprocess import build_vocab_from_iterator
from umigame.nlp.preprocess import build_stoi_from_vocab
from umigame.nlp.preprocess import pad_sequences
from umigame.nlp.preprocess import numericalise_tokens_from_iterator
from umigame.nlp.preprocess import tokeniser
from umigame.nlp.preprocess import yield_tokens


def get_dataframes(dataset='news'):
    text_df = load_dataset(dataset)
    text_df = text_df.set_index('date')
    text_df.index = pd.to_datetime(text_df.index)
    price_df = fetch_crypto(tickers=['BTC-USD'], start='2014-09-17')['BTC-USD']
    price_df['ft'] = fixed_time_horizon(price_df, lookahead=30)
    price_df['tb'] = triple_barrier(price_df, max_period=30)
    text_price_df = text_df.join(price_df[['ft', 'tb', 'open', 'high', 'low', 'close']])

    train_df = text_price_df.loc[:'2018']
    test_df = text_price_df.loc['2018':]
    
    return train_df, test_df

def get_dataloaders(dataset='news', labeling='tb', maxlen=16, batch_size=256):
    text_df = load_dataset(dataset)
    text_df = text_df.set_index('date')
    text_df.index = pd.to_datetime(text_df.index)
    price_df = fetch_crypto(tickers=['BTC-USD'], start='2014-09-17')['BTC-USD']
    price_df['ft'] = fixed_time_horizon(price_df, lookahead=30)
    price_df['tb'] = triple_barrier(price_df, max_period=30)
    text_price_df = text_df.join(price_df[['ft', 'tb']])

    train_df = text_price_df.loc[:'2018']
    test_df = text_price_df.loc['2018':]
    
    vocab = build_vocab_from_iterator(
        yield_tokens(text_price_df['clean_text'], tokeniser), 
        min_freq=1, 
        specials=['[PAD]', '[UNK]', '[EMOJIS]', '[MENTION]', '[HASHTAG]', '[URL]', '[NUMBERS]', '[SMILEYS]']
    )
    
    train_ids = numericalise_tokens_from_iterator(yield_tokens(train_df['clean_text'], tokeniser), vocab)
    train_ids = pad_sequences(train_ids, maxlen=maxlen)
    test_ids = numericalise_tokens_from_iterator(yield_tokens(test_df['clean_text'], tokeniser), vocab)
    test_ids = pad_sequences(test_ids, maxlen=maxlen)
    
    y_train = train_df[labeling]
    y_test = test_df[labeling]
    
    train_vectors = torch.tensor(train_ids)
    test_vectors = torch.tensor(test_ids)
    train_dataset = TensorDataset(train_vectors, torch.LongTensor(y_train))
    test_dataset = TensorDataset(test_vectors, torch.LongTensor(y_test))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)
    
    return train_dataloader, test_dataloader


def get_weights(dataset='news', w2v=None):
    text_df = load_dataset(dataset)
    text_df = text_df.set_index('date')
    text_df.index = pd.to_datetime(text_df.index)
    price_df = fetch_crypto(tickers=['BTC-USD'], start='2014-09-17')['BTC-USD']
    price_df['ft'] = fixed_time_horizon(price_df, lookahead=30)
    price_df['tb'] = triple_barrier(price_df, max_period=30)
    text_price_df = text_df.join(price_df[['ft', 'tb']])
    
    vocab = build_vocab_from_iterator(
        yield_tokens(text_price_df['clean_text'], tokeniser), 
        min_freq=1, 
        specials=['[PAD]', '[UNK]', '[EMOJIS]', '[MENTION]', '[HASHTAG]', '[URL]', '[NUMBERS]', '[SMILEYS]']
    )

    word2idx = build_stoi_from_vocab(vocab)
    tmp = []
    for word, index in word2idx.items():
        try:
            tmp.append(w2v.get_vector(word))
        except:
            pass
    mean = np.mean(np.array(tmp))
    std = np.std(np.array(tmp))
    vocab_size = len(word2idx)
    embed_size = w2v.vectors.shape[1]
    weights = np.random.normal(mean, std , [vocab_size, embed_size])
    oov = []
    for word, index in word2idx.items():
        try:
            weights[index, :] = w2v.get_vector(word)
        except:
            oov.append(word)
            
    return weights


def seed_everything(seed=914):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def padding(seq, size, mode):
    """
    Parameters
    ----------
    seq: np.array
        The sequence to be padded.
    mode: str
        Select padding mode among {"zero", "repeat"}.
        
    Returns
    -------
    seq: np.ndarray
    """
    if mode == "zero":
        seq = np.array(trimmer(seq, size, filler=0))
    elif mode == "repeat":
        seq = np.array(repeat_padding(seq, size))
    return seq


def repeat_padding(seq, size):
    """
    Parameters
    ----------
    src: list or np.array
    trg: list or np.array
    
    Returns
    -------
    list

    References
    ----------
    1. https://stackoverflow.com/a/60972703
    """
    src = seq
    trg = [0] * size
    data = [src, trg]
    m = len(max(data, key=len))
    r = list(starmap(np.resize, ((e, m) for e in data)))
    return r[0][:size]


def trimmer(seq, size, filler=0):
    """
    Parameters
    ----------
    seq: np.array
        The sequence to be padded.
    size: int
        The size of the output sequence.
    filler: float or int
        Pads with a constant value.
        
    Returns
    -------
    list

    References
    ----------
    1. https://stackoverflow.com/a/30475648
    """
    return list(islice(chain(seq, repeat(filler)), size))
