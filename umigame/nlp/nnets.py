import math
import torch
import transformers
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from .attention import SelfAttention


class TextLR(nn.Module):

    def __init__(self, vocab_size=20000, embedding_dim=300, num_classes=2, dropout=0.1):
        super(TextLR, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.drop = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.embed(x)
        x = torch.mean(x, dim=1)
        x = self.classifier(self.drop(x))
        return x


class TextAttention(nn.Module):

    def __init__(self, vocab_size=20000, embedding_dim=300, num_classes=2, dropout=0.1, trainable=True):
        super(TextAttention, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.drop = nn.Dropout(p=dropout)
        self.attn = SelfAttention(dimensions=embedding_dim, trainable=trainable)
        self.classifier = nn.Sequential(OrderedDict([
            ('hidden', nn.Linear(embedding_dim, embedding_dim//2)), 
            ('bn', nn.BatchNorm1d(embedding_dim//2)), 
            # ('dropout', nn.Dropout(p=dropout)), 
            ('nonlinearity', nn.LeakyReLU(inplace=True)), 
            ('output', nn.Linear(embedding_dim//2, num_classes))
        ]))

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.attn(x)
        x = torch.mean(x, dim=1)
        x = self.classifier(x)
        return x


class TextBERT(nn.Module):

    def __init__(self, num_classes=2, dropout=0.3, freeze=True):
        super(TextBERT).__init__()
        self.bert = transformers.BertModel.from_pretrained(
            "bert-base-uncased", return_dict=False
        )
        self.bert_drop = nn.Dropout(dropout)
        self.out = nn.Linear(768, num_classes)
        if freeze:
            self.freeze_bert()

    def forward(self, ids):
        _, o_2 = self.bert(ids)
        b_o = self.bert_drop(o_2)
        output = self.out(b_o)
        return output

    def freeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False


class TextCNN(nn.Module):
    
    def __init__(
            self, 
            vocab_size, 
            embedding_dim, 
            channel_dim, 
            kernel_list, 
            dropout, 
            num_classes=2
        ):
        super(TextCNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, channel_dim, (w, embedding_dim)) for w in kernel_list])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_list)*channel_dim, num_classes)
        
    def forward(self, x):
        emb_x = self.embed(x)
        emb_x = emb_x.unsqueeze(1)
        con_x = [conv(emb_x) for conv in self.convs]
        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]
        fc_x = torch.cat(pool_x, dim=1)
        fc_x = fc_x.squeeze(-1)
        fc_x = self.dropout(fc_x)
        logit = self.fc(fc_x)
        return logit


class TextLSTM(nn.Module):

    def __init__(
            self, 
            vocab_size, 
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            dropout=0.1,  
            bidirectional=True
        ):
        super(TextLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
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
        embedded = self.embed(text)
        # hidden: [batch, num_layers * num_directions, hidden_dim]
        # cell: [batch, num_layers * num_directions, hidden_dim]
        packed_output, (hidden, cell) = self.lstm(embedded)
        # hidden: [batch, hidden_dim * num_directions]
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # outputs: [batch, num_classes]
        outputs = self.fc(hidden)
        return outputs


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class StatsPool(nn.Module):

    def __init__(self, floor=1e-10, bessel=False):
        super(StatsPool, self).__init__()
        self.floor = floor
        self.bessel = bessel

    def forward(self, x):
        means = torch.mean(x, dim=1)
        _, t, _ = x.shape
        if self.bessel:
            t = t - 1
        residuals = x - means.unsqueeze(1)
        numerator = torch.sum(residuals**2, dim=1)
        stds = torch.sqrt(torch.clamp(numerator, min=self.floor)/t)
        x = torch.cat([means, stds], dim=1)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Initialiser(object):
    """
    References
    ----------
    1. https://github.com/3ammor/Weights-Initializer-pytorch
    """
    @staticmethod
    def initialise(model, initialization, **kwargs):

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                initialization(m.weight.data, **kwargs)
                try:
                    initialization(m.bias.data)
                except:
                    pass

            elif isinstance(m, nn.Linear):
                initialization(m.weight.data, **kwargs)
                try:
                    initialization(m.bias.data)
                except:
                    pass

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0)

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0)

        model.apply(weights_init)