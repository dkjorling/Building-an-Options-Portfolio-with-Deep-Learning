import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import Tensor
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from datetime import datetime as dt
from Loss_Fcn import abs_softmax, sharpe_loss
        
#  code from crosstab article
class LSTM_port(nn.Module):
    def __init__(self, num_sensors, hidden_units, out_feats=315):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.out_feats = out_feats
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=out_feats)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0])  # First dim of Hn is num_layers, which is set to 1 above.
        out = torch.flatten(out, 1)
        
        out = abs_softmax(out)
        return out

class GRU_port(nn.Module):
    def __init__(self, num_sensors, hidden_units, out_feats=315):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.out_feats = out_feats
        self.num_layers = 1

        self.gru = nn.GRU(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=out_feats)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        
        _, hn = self.gru(x, h0)
        out = self.linear(hn[0])  # First dim of Hn is num_layers, which is set to 1 above.
        out = torch.flatten(out, 1)
        
        out = abs_softmax(out)
        return out

class PositionalEncoder(nn.Module):
    def __init__(
        self, 
        dropout: float=0.1, 
        max_seq_len: int=50, 
        d_model: int=512,
        batch_first: bool=True
        ):

        """
        Parameters:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model 
                     (Vaswani et al, 2017)
        """
        super().__init__()
        self.d_model = d_model    
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        if  batch_first:
            self.x_dim = 1 
            position = torch.arange(max_seq_len).unsqueeze(1) # shape:  (20, 1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) #shape 256
            pe = torch.zeros(1, max_seq_len, d_model) # shape (20, 1, 512)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)
            
        else:
            self.x_dim = 0
            # copy pasted from PyTorch tutorial
            position = torch.arange(max_seq_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_seq_len, 1, d_model)  # shape (20, 1, 512)
            pe[:, 0, 0::2] = torch.sin(position * div_term) #shape (20, 256)
            pe[:, 0, 1::2] = torch.cos(position * div_term) #shape (20, 256)
            self.register_buffer('pe', pe)

        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or 
               [enc_seq_len, batch_size, dim_val]
        """

        x = x + self.pe[:x.size(self.x_dim)] # x shape is (32, 20, 512)

        return self.dropout(x)
    
    
class Transformer_port(nn.Module):
    def __init__(self,
        input_size: int,
        dec_seq_len: int,
        max_seq_len: int,
        batch_first: True,
        out_seq_len: int=58,
        dim_val: int=512,  
        n_encoder_layers: int=4,
        n_decoder_layers: int=4,
        n_heads: int=8,
        dropout_encoder: float=0.2, 
        dropout_decoder: float=0.2,
        dropout_pos_enc: float=0.1,
        dim_feedforward_encoder: int=2048,
        dim_feedforward_decoder: int=2048,
        num_predicted_features: int=13
        ):
        
        super().__init__()
        self.dec_seq_len = dec_seq_len
        self.max_seq_len = max_seq_len
        self.encoder_input_layer = nn.Linear(
            in_features=input_size,
            out_features=dim_val)

        self.decoder_input_layer = nn.Linear(
            in_features=num_predicted_features, # the number of features you want to predict
            out_features=dim_val) 

        self.linear = nn.Linear(
            in_features=dim_val,
            out_features=num_predicted_features)
        
        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_val,
            dropout=dropout_pos_enc,
            max_seq_len=max_seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val, 
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=batch_first)
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers,
            norm=None)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            batch_first=batch_first)
        
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers,
            norm=None)
        
    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor=None, 
                tgt_mask: Tensor=None) -> Tensor:
        
        src = self.encoder_input_layer(src) 
        src = self.positional_encoding_layer(src)
        src = self.encoder(src=src) # src shape: [batch_size, enc_seq_len, dim_val]
        
        decoder_output = self.decoder_input_layer(tgt) # outputs shpe (32, 512)
        decoder_output =  decoder_output.reshape(decoder_output.shape[0], 1, decoder_output.shape[1])
        
        decoder_output = self.decoder(
            tgt=decoder_output, #(32, 1, 512)
            memory=src, #(32, 20, 512)
            tgt_mask=tgt_mask, #(1, 1)
            memory_mask=src_mask) #(1, 20)
        
        decoder_output = self.linear(decoder_output)
        decoder_output = decoder_output.reshape(decoder_output.shape[0], decoder_output.shape[2])
        decoder_output = abs_softmax(decoder_output)
        
        return decoder_output