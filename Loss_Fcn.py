import torch
import numpy as np
import pandas as pd

#define abs_softmax for tensor
def abs_softmax(x):
    """Returns weights with absolute values that sum to 1"""
    means = torch.mean(x, dim=1, keepdim=True)
    x_exp = torch.exp(x.abs()- means)
    x_exp_sum = torch.sum(x_exp,  dim=1, keepdims=True)
    
    return torch.sign(x)*x_exp/x_exp_sum

# define sharpe loss
def sharpe_loss(weights, returns):
    """Calculate Sharpe ratio given by model weights"""
    portfolio_returns = weights.multiply(returns).sum(dim=1)
    sharpe = portfolio_returns.mean() / portfolio_returns.std()
    return -sharpe
