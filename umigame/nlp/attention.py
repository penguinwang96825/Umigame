import torch
import torch.nn as nn


class NonparametricSelfAttention(nn.Module):
    """
    Examples
    --------
    >>> context = torch.Tensor([
            [
                [0.6, 0.2, 0.8], 
                [0.2, 0.3, 0.1], 
                [0.9, 0.1, 0.8], 
                [0.4, 0.1, 0.4], 
                [0.4, 0.1, 0.6]
            ]
        ])
    >>> context_, attention_weights = NonparametricSelfAttention(3)(context)
    >>> print("Input: ", sentences.shape)
    Input:  torch.Size([1, 5, 3])
    >>> print("Output: ", context_.shape)
    Output:  torch.Size([1, 5, 3])
    """
    def __init__(self, dimensions):
        super(NonparametricSelfAttention, self).__init__()
        self.dimensions = dimensions
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, context, return_weights=True):
        """
        context: [sequence_length, embedding_dimension]
        """
        attention_scores  = torch.bmm(context, context.transpose(1, 2))
        attention_weights = self.softmax(attention_scores )
        context_ = torch.bmm(attention_weights, context)
        if return_weights:
            return context_ , attention_weights
        return context_ 

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ParametricSelfAttention(nn.Module):
    """
    Examples
    --------
    >>> context = torch.Tensor([
            [
                [0.6, 0.2, 0.8], 
                [0.2, 0.3, 0.1], 
                [0.9, 0.1, 0.8], 
                [0.4, 0.1, 0.4], 
                [0.4, 0.1, 0.6]
            ]
        ])
    >>> context_, attention_weights = ParametricSelfAttention(3)(context)
    >>> print("Input: ", sentences.shape)
    Input:  torch.Size([1, 5, 3])
    >>> print("Output: ", context_.shape)
    Output:  torch.Size([1, 5, 3])
    """
    def __init__(self, dimensions):
        super(ParametricSelfAttention, self).__init__()
        self.dimensions = dimensions
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.linear_in = nn.Linear(dimensions, dimensions, bias=False)
        self.linear_out = nn.Linear(dimensions, dimensions, bias=False)

    def forward(self, context, return_weights=True):
        """
        context: [sequence_length, embedding_dimension]
        """
        context_ = self.linear_in(context)
        attention_scores  = torch.bmm(context_, context_.transpose(1, 2))
        attention_weights = self.softmax(attention_scores )
        context_ = torch.bmm(attention_weights, context_)
        context_ = self.tanh(context_)
        context_ = self.linear_out(context_)
        if return_weights:
            return context_ , attention_weights
        return context_ 

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
