import re
from collections import Counter
from itertools import chain


class Preprocessor(object):

    def __init__(self, vocab_size=None, mode="basic"):
        super().__init__()
        self.vocab_size = vocab_size
        self.mode = mode
    
    def encode(self, documents, max_length, padding=True):
        ids = []
        for i, doc in enumerate(self.tokenise(documents)):
            token_ids = []
            for j, token in enumerate(doc):
                token_ids.append(self.word2idx.get(token, 1))
            ids.append(token_ids)
            if padding:
                ids[i] = ids[i] + [0]*(max_length - len(ids[i]))
        return ids

    def decode(self, ids):
        tokens = []
        for i, doc in enumerate(ids):
            token_ids = []
            for j, id_ in enumerate(doc):
                token_ids.append(self.idx2word.get(id_))
            tokens.append(token_ids)
        return tokens

    def tokenise(self, documents):
        if self.mode == "basic":
            return [re.split('\W+', doc.lower()) for doc in documents]

    def build(self, documents):
        self.tokenised_documents = self.tokenise(documents)
        self.vocab = Counter(chain(*[doc for doc in self.tokenised_documents]))
        if self.vocab_size is not None:
            self.vocab_size = min(len(self.vocab), self.vocab_size)
            self.vocab = self.vocab.most_common(self.vocab_size)
            self.vocab = [v for v, c in self.vocab]
        else:
            self.vocab = self.vocab.keys()
        self.word2idx = {'<PAD>':0, '<UNK>':1}
        for i, doc in enumerate(self.tokenised_documents):
            for j, token in enumerate(doc):
                if (self.word2idx.get(token) is None) and (token in self.vocab):
                    self.word2idx[token] = len(self.word2idx)
        self.idx2word = {idx:word for word, idx in self.word2idx.items()}
