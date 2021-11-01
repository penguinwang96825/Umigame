import re
import numpy as np
from collections import Counter
from collections import Counter, OrderedDict


TWITTER_SPECIALS = {'EMOJIS':'[EMOJIS]', 'MENTION':'[MENTION]', 'HASHTAG':'[HASHTAG]', 'URL':'[URL]', 'NUMBERS':'[NUMBERS]'}
KEYWORDS = [
    'btc', 'bitcoin', 'eth', 'ethereum', 'blockchain', 
    'crypto', 'cryptocurrency', 'cryptocurrencies', 'exchange', 'token', 
    'ico', 'wallet', 'coin', 'mining', 'hashing', 
    'altcoin', 'market', 'satoshi', 'dogecoin', 'binance', 
    'proof', 'doge', 'digital', 'cryptography', 'fork', 
    'litecoin', 'ltc', 'peercoin', 'dash', 'monero', 
    'mine', 'virtual', 'coinbase', 'coins', 'verge'
]


def tokeniser(text):
    return text.split()


def yield_tokens(data_iter, tokeniser):
    for text in data_iter:
        yield tokeniser(text)


def build_vocab_from_iterator(iterator, min_freq=1, specials=[], special_first=True):
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)

    if specials is not None:
        for tok in specials:
            del counter[tok]

    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[0])
    sorted_by_freq_tuples.sort(key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)

    if specials is not None:
        if special_first:
            specials = specials[::-1]
        for symbol in specials:
            ordered_dict.update({symbol: min_freq})
            ordered_dict.move_to_end(symbol, last=not special_first)

    tokens = []
    for token, freq in ordered_dict.items():
        if freq >= min_freq:
            tokens.append(token)
            
    return tokens


def build_stoi_from_vocab(vocab):
    stoi = {s:i for i, s in enumerate(vocab)}
    return stoi
    

def build_itos_from_vocab(vocab):
    itos = {i:s for i, s in enumerate(vocab)}
    return itos


def numericalise_tokens_from_iterator(iterator, vocab):
    stoi = build_stoi_from_vocab(vocab)
    ids = [list(map(lambda x: stoi.get(x, 1), text)) for text in iterator]
    return ids


def pad_sequences(sequences, maxlen=64, value=0, truncating='pre', padding='post'):
    if padding == 'post':
        if truncating == 'post':
            pad_ = lambda seq, maxlen : seq[0:maxlen] if len(seq) > maxlen else seq + [value] * (maxlen-len(seq))
        elif truncating == 'pre':
            pad_ = lambda seq, maxlen : seq[-maxlen:] if len(seq) > maxlen else seq + [value] * (maxlen-len(seq))
    elif padding == 'pre':
        if truncating == 'post':
            pad_ = lambda seq, maxlen : seq[0:maxlen] if len(seq) > maxlen else [value] * (maxlen-len(seq)) + seq
        elif truncating == 'pre':
            pad_ = lambda seq, maxlen : seq[-maxlen:] if len(seq) > maxlen else [value] * (maxlen-len(seq)) + seq
    padded_sequences = [pad_(seq, maxlen) for seq in sequences]
    return np.stack(padded_sequences)


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


def text_to_word_sequence(text, lower=False, source='twitter'):
    tokens = tokenise(text, lower=lower)
    tokens = list(tokens)
    if source == 'twitter':
        specials = {
            'EMOJIS':'[EMOJIS]', 'MENTION':'[MENTION]', 'HASHTAG':'[HASHTAG]', 
            'URL':'[URL]', 'NUMBERS':'[NUMBERS]', 'SMILEYS':'[SMILEYS]'
        }
        tokens = [specials.get(token) if token in specials.keys() else token for token in tokens]
    elif source == 'news':
        specials = {
            'URL':'[URL]', 'NUMBERS':'[NUMBERS]'
        }
        tokens = [specials.get(token) if token in specials.keys() else token for token in tokens]
    return tokens


def clean_tweet(text):
    text = text.lower()
    text = decontracted(text)
    text = re.sub(r'&amp;', 'and', text)
    text = re.sub(r'=&gt', 'geotagged', text)
    text = re.sub(Patterns.EMOJIS_PATTERN, '[EMOJIS]', text)
    text = re.sub(Patterns.SMILEYS_PATTERN, '[SMILEYS]', text)
    text = re.sub(Patterns.MENTION_PATTERN, '[MENTION]', text)
    text = re.sub(Patterns.HASHTAG_PATTERN, '[HASHTAG]', text)
    text = re.sub(Patterns.URL_PATTERN, '[URL]', text)
    text = re.sub(Patterns.NUMBERS_PATTERN, '[NUMBERS]', text)
    text = re.sub(' +', ' ', text)
    return text


def clean_news(text):
    text = text.lower()
    text = re.sub('', "'", text)
    text = decontracted(text)
    text = re.sub(Patterns.URL_PATTERN, '[URL]', text)
    text = re.sub(Patterns.NUMBERS_PATTERN, '[NUMBERS]', text)
    text = re.sub(' +', ' ', text)
    return text


def keywords_in_tokens(tokens, keywords=None):
    for token in tokens:
        if token in keywords:
            return True
    return False
