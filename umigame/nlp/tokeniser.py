import json
from tqdm.auto import tqdm
from collections import Counter


class Tokeniser(object):
    
    def __init__(
            self, 
            char_level=False, 
            num_tokens=None, 
            pad_token='<PAD>', 
            oov_token='<UNK>', 
            token2idx=None
        ):
        self.char_level = char_level
        self.separator = '' if self.char_level else ' '
        # <PAD> + <UNK> tokens
        if num_tokens: 
            num_tokens -= 2
        self.num_tokens = num_tokens
        self.oov_token = oov_token
        if not token2idx:
            token2idx = {pad_token: 0, oov_token: 1}
        self.token2idx = token2idx
        self.idx2token = {v: k for k, v in self.token2idx.items()}

    def __len__(self):
        return len(self.token2idx)

    def __str__(self):
        return f"<Tokenizer(num_tokens={len(self)})>"

    def fit_on_texts(self, texts):
        if self.char_level:
            all_tokens = [token for text in texts for token in text]
        if not self.char_level:
            all_tokens = [token for text in texts for token in text.split(' ')]
        counts = Counter(all_tokens).most_common(self.num_tokens)
        self.min_token_freq = counts[-1][1]
        for token, count in tqdm(counts, prefix="VOCAB"):
            index = len(self)
            self.token2idx[token] = index
            self.idx2token[index] = token
        return self

    def texts_to_sequences(self, texts):
        sequences = []
        for text in tqdm(texts, prefix="TEXT2SEQ"):
            if not self.char_level:
                text = text.split(' ')
            sequence = []
            for token in text:
                sequence.append(self.token2idx.get(
                    token, self.token2idx[self.oov_token]))
            sequences.append(sequence)
        return sequences

    def sequences_to_texts(self, sequences):
        texts = []
        for sequence in tqdm(sequences, prefix="SEQ2TEXT"):
            text = []
            for index in sequence:
                text.append(self.idx2token.get(index, self.oov_token))
            texts.append(self.separator.join([token for token in text]))
        return texts

    def save(self, fp):
        with open(fp, 'w') as fp:
            contents = {
                'char_level': self.char_level,
                'oov_token': self.oov_token,
                'token2idx': self.token2idx
            }
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, fp):
        with open(fp, 'r') as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)