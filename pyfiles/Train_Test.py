import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from datetime import datetime as dt
import decouple

from Data_Prep import get_targets, get_attn_targs, TS_Dataset, TS_Dataset_attn


def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    
    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss


def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Source:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Args:
        dim1: int, for both src and tgt masking, this must be target sequence
              length
        dim2: int, for src masking this must be encoder sequence length (i.e. 
              the length of the input sequence to the model), 
              and for tgt masking, this must be target sequence length 
    Return:
        A Tensor of shape [dim1, dim2]
    """
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)

def train_model_attn(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    
    for X, y, trg in data_loader:
        enc_seq_len = X.shape[1]

        # Output length
        output_sequence_length = 1
        
        tgt_mask = generate_square_subsequent_mask(
        dim1=output_sequence_length,
        dim2=output_sequence_length)
        
        src_mask = generate_square_subsequent_mask(
        dim1=output_sequence_length,
        dim2=enc_seq_len)
        
        output = model(
            src=X,
            tgt=trg,
            src_mask=src_mask,
            tgt_mask=tgt_mask)
        
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss

def test_model(data_loader, model, loss_function):
    
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()
    

    avg_loss = total_loss / num_batches
    return avg_loss

def test_model_attn(data_loader, model, loss_function):
    
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y, trg in data_loader:
            enc_seq_len = X.shape[1]

            # Output length
            output_sequence_length = 1
        
            tgt_mask = generate_square_subsequent_mask(
            dim1=output_sequence_length,
            dim2=output_sequence_length)
        
            src_mask = generate_square_subsequent_mask(
            dim1=output_sequence_length,
            dim2=enc_seq_len)
        
            output = model(
                src=X,
                tgt=trg,
                src_mask=src_mask,
                tgt_mask=tgt_mask)
            
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    return avg_loss

def early_stop(losses, tolerance=5):
    min_loss_idx = losses.index(min(losses))
    since_new_min = len(losses) - 1 - min_loss_idx
    if since_new_min >= tolerance:
        return True
    
def plot_losses(train_losses, val_losses):
    x = range(len(train_losses))
    plt.plot(x, train_losses, label='Train Loss')
    plt.plot(x, val_losses, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.show()

def plot_returns(p_return, bm_return, returns=True):
    x = range(len(p_return))
    if returns:
        plt.plot(x, p_return, label='Portfolio Return')
        plt.plot(x, bm_return, label='Benchmark Return')
    else:
        plt.plot(x, p_return, label='Portfolio Sharpe')
        plt.plot(x, bm_return, label='Benchmark Sharpe')
    plt.legend(loc='upper left')
    plt.show()
    
def predict(data_loader, model):

    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_star = model(X)
            output = torch.cat((output, y_star), 0)
    
    return output

def get_pred_df_sharpe(df_train, df_test, model, batch_size,
                       lookback, tickers, num_assets=315, iv=60, test_only=False):
    """Create dataframe with actual and un-normalized predicted return values """
    
    train_dataset = TS_Dataset(df_train, lookback=lookback, iv=iv)
    test_dataset = TS_Dataset(df_test, lookback=lookback, iv=iv)
    test_loader = DataLoader(test_dataset,
                        batch_size,
                        shuffle=False)
    
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    y_hat = '_port_weight'
    targets = get_targets(df_train, iv=iv)
    column_names = [col[21:] + y_hat for col in targets.columns]
    
    df_out_train = pd.DataFrame(predict(train_eval_loader, model).numpy(),
                         columns=column_names,
                         index=df_train.index)


    df_out_train = targets.join(df_out_train)
    
    df_out_test = pd.DataFrame(predict(test_loader, model).numpy(),
                              columns=column_names,
                              index=df_test.index)

    df_out_test = get_targets(df_test, iv=iv).join(df_out_test)
    
    df_out = pd.concat((df_out_train, df_out_test))

    df_out['return_next_portfolio'] = (df_out.iloc[:, num_assets:].values * df_out.iloc[:, :num_assets].values).sum(axis=1)
    df_out['return_portfolio'] = df_out['return_next_portfolio'].shift(1)
    
    # add benchmark column
    weight = 1 / num_assets
    weights = np.repeat(weight, num_assets)    
    df_out['return_next_benchmark'] = (df_out.iloc[:, :num_assets].values * weights).sum(axis=1)
    df_out['return_benchmark'] = df_out['return_next_benchmark'].shift(1)
    
    # add costs
#    df_out = turnover_cost(df_out, tickers, bps=1)
    
    if test_only:
        year = df_test.index[0].year + 1
        date = str(year) + '-01-01'
        return df_out[df_out.index >= date]
    else:
        return df_out
    
    
def predict_attn(data_loader, model):

    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _, trg in data_loader:
            enc_seq_len = X.shape[1]
            # Output length
            output_sequence_length = 1
            
            tgt_mask = generate_square_subsequent_mask(
            dim1=output_sequence_length,
            dim2=output_sequence_length)
        
            src_mask = generate_square_subsequent_mask(
            dim1=output_sequence_length,
            dim2=enc_seq_len)
        
            y_star = model(
                src=X,
                tgt=trg,
                src_mask=src_mask,
                tgt_mask=tgt_mask)
            
            output = torch.cat((output, y_star), 0)
    
    return output

def get_pred_df_sharpe_attn(df_train, df_test, model, batch_size, 
                            lookback, tickers, num_assets=315, iv=60, test_only=False):
    """Create dataframe with actual and un-normalized predicted return values """
    
    train_dataset = TS_Dataset_attn(df_train, lookback=lookback, iv=iv)
    test_dataset = TS_Dataset_attn(df_test, lookback=lookback, iv=iv)
    test_loader = DataLoader(test_dataset,
                        batch_size,
                        shuffle=False)
    
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    y_hat = '_port_weight'
    targets = get_targets(df_train, iv=iv)
    column_names = [col[21:] + y_hat for col in targets.columns]
    
    df_out_train = pd.DataFrame(predict_attn(train_eval_loader, model).numpy(),
                         columns=column_names,
                         index=df_train.index)


    df_out_train = targets.join(df_out_train)
    
    df_out_test = pd.DataFrame(predict_attn(test_loader, model).numpy(),
                              columns=column_names,
                              index=df_test.index)

    df_out_test = get_targets(df_test, iv=iv).join(df_out_test)
    
    df_out = pd.concat((df_out_train, df_out_test))

    df_out['return_next_portfolio'] = (df_out.iloc[:, num_assets:].values * df_out.iloc[:, :num_assets].values).sum(axis=1)
    df_out['return_portfolio'] = df_out['return_next_portfolio'].shift(1)
    
    # add benchmark column
    weight = 1 / num_assets
    weights = np.repeat(weight, num_assets)    
    df_out['return_next_benchmark'] = (df_out.iloc[:, :num_assets].values * weights).sum(axis=1)
    df_out['return_benchmark'] = df_out['return_next_benchmark'].shift(1)
    
    # add costs
#    df_out = turnover_cost(df_out, tickers, bps=1)
    
    if test_only:
        year = df_test.index[0].year + 1
        date = str(year) + '-01-01'
        return df_out[df_out.index >= date]
    else:
        return df_out

