import re
import json
import pandas as pd
from tqdm.auto import tqdm
from collections import Counter


class Vocabulary(object):
    PAD_token = 0   # Used for padding short sentences
    UNK_token = 1   # Unknown word
    BOS_token = 2   # Begin-of-sentence token
    EOS_token = 3   # End-of-sentence token
    
    """
    Examples
    --------
    >>> corpus = [
    ...     'Acting CoC Hsu More crypto regulation is needed', 
    ...     'Argo Blockchain's Texas mining facility could cost up to $2B', 
    ...     'New study reveals which US cities lead crypto hires in 2021'
    ... ]
    >>> vocab = Vocabulary()
    >>> for doc in corpus:
    ...     vocab.add_sentence(doc)
    """

    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            self.PAD_token: "[PAD]", self.UNK_token: "[UNK]", self.BOS_token: "[BOS]", self.EOS_token: "[EOS]"
        }
        self.num_words = 4
        self.num_sentences = 0
        self.longest_sentence = 0

    def __str__(self):
        return f"<Vocabulary(num_vocabs={len(self.num_words)})>"

    def add_word(self, word):
        if word not in self.word2index:
            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # Word exists; increase word count
            self.word2count[word] += 1
            
    def add_sentence(self, sentence):
        sentence_len = 0
        for word in tokenise(sentence):
            sentence_len += 1
            self.add_word(word)
        if sentence_len > self.longest_sentence:
            # This is the longest sentence
            self.longest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1

    def index_to_word(self, index):
        return self.index2word[index]

    def word_to_index(self, word):
        return self.word2index.get(word, self.UNK_token)


class Patterns:
    
    URL_PATTERN_STR = r"""(?i)((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info
                      |int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|
                      bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|
                      cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|
                      gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|
                      la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|
                      nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|
                      sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|
                      uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]
                      *?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’])|(?:(?<!@)
                      [a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name
                      |post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn
                      |bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg
                      |eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id
                      |ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|
                      md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|
                      ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|
                      sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|
                      za|zm|zw)\b/?(?!@)))"""
    URL_PATTERN = re.compile(URL_PATTERN_STR, re.IGNORECASE)
    HASHTAG_PATTERN = re.compile(r'#\w*')
    MENTION_PATTERN = re.compile(r'@\w*')
    RESERVED_WORDS_PATTERN = re.compile(r'\b(?<![@#])(RT|FAV)\b')

    try:
        # UCS-4
        EMOJIS_PATTERN = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
    except re.error:
        # UCS-2
        EMOJIS_PATTERN = re.compile(u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])')

    SMILEYS_PATTERN = re.compile(r"(\s?:X|:|;|=)(?:-)?(?:\)+|\(|O|D|P|S|\\|\/\s){1,}", re.IGNORECASE)
    NUMBERS_PATTERN = re.compile(r"(^|\s)(-?\d+([.,]?\d+)*)")


class Tokeniser(object):
    
    def __init__(self, 
                 char_level=False, 
                 num_tokens=None, 
                 pad_token='[PAD]', 
                 oov_token='[UNK]', 
                 token2index=None
                ):
        self.char_level = char_level
        self.separator = '' if self.char_level else ' '
        # <PAD> + <UNK> tokens
        if num_tokens: num_tokens -= 2
        self.num_tokens = num_tokens
        self.oov_token = oov_token
        if not token2index:
            token2index = {pad_token: 0, oov_token: 1}
        self.token2index = token2index
        self.index2token = {v: k for k, v in self.token2index.items()}

    def __len__(self):
        return len(self.token2index)

    def __str__(self):
        return f"<Tokeniser(num_tokens={len(self)})>"

    def fit_on_texts(self, texts):
        if self.char_level:
            all_tokens = [token for text in texts for token in text]
        if not self.char_level:
            all_tokens = [token for text in texts for token in tokenise(text)]
        counts = Counter(all_tokens).most_common(self.num_tokens)
        self.min_token_freq = counts[-1][1]
        for token, count in tqdm(counts, leave=False):
            index = len(self)
            self.token2index[token] = index
            self.index2token[index] = token
        return self

    def texts_to_sequences(self, texts):
        """
        texts: List[str]
        """
        sequences = []
        for text in tqdm(texts, leave=False):
            if not self.char_level:
                text = tokenise(text)
            sequence = []
            for token in text:
                sequence.append(self.token2index.get(token, 1))
            sequences.append(sequence)
        return sequences

    def sequences_to_texts(self, sequences):
        texts = []
        for sequence in tqdm(sequences, leave=False):
            text = []
            for index in sequence:
                text.append(self.index2token.get(index, self.oov_token))
            texts.append(self.separator.join([token for token in text]))
        return texts

    def save(self, fp):
        with open(fp, 'w') as fp:
            contents = {
                'char_level': self.char_level,
                'oov_token': self.oov_token,
                'token2index': self.token2index
            }
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, fp):
        with open(fp, 'r') as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)


def tokenise(text, errors="strict", lower=False):

    def to_unicode(text, encoding='utf8', errors='strict'):
        if isinstance(text, str):
            return text
        return str(text, encoding, errors=errors)

    PAT_ALPHABETIC = re.compile('(((?![\d])\w)+)', re.UNICODE)
    text = to_unicode(text, errors=errors)
    if lower:
        text = text.lower()
    for match in PAT_ALPHABETIC.finditer(text):
        yield match.group()


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won(\'|\’)t", "will not", phrase)
    phrase = re.sub(r"can(\'|\’)t", "can not", phrase)

    # general
    phrase = re.sub(r"n(\'|\’)t", " not", phrase)
    phrase = re.sub(r"(\'|\’)re", " are", phrase)
    phrase = re.sub(r"(\'|\’)s", " is", phrase)
    phrase = re.sub(r"(\'|\’)d", " would", phrase)
    phrase = re.sub(r"(\'|\’)ll", " will", phrase)
    phrase = re.sub(r"(\'|\’)t", " not", phrase)
    phrase = re.sub(r"(\'|\’)ve", " have", phrase)
    phrase = re.sub(r"(\'|\’)m", " am", phrase)
    return phrase


def clean_news(text, lower=True):
    text = text.lower() if lower else text
    text = re.sub('', "'", text)
    text = decontracted(text)
    text = re.sub(Patterns.URL_PATTERN, '', text)
    text = re.sub(Patterns.NUMBERS_PATTERN, '', text)
    text = re.sub(' +', ' ', text)
    return text


def main():
    vocab = Vocabulary()
    df = pd.read_parquet('./cointelegraph.parquet.gzip')
    corpus = df['title'].map(clean_news).tolist()
    for doc in corpus:
        vocab.add_sentence(doc)

    tokeniser = Tokeniser(token2index=vocab.word2index)
    print(tokeniser.texts_to_sequences(['I love natural language processing']))


if __name__ == '__main__':
    main()