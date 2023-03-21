import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from datetime import datetime as dt



# define train/val/test split function
def train_val_test(data, lookback=30, train_years=5, val_years=1, test_years=1):
    
    # make train df
    year = data.index[0].year
    x = data[dt(year+1,1,1,0,0,0):].first_valid_index().strftime("%Y-%m-%d")
    index_start_train = data.index.get_loc(x) - lookback
    x = data[dt(year+1+train_years,1,1,0,0,0):].first_valid_index().strftime("%Y-%m-%d")
    index_end_train = data.index.get_loc(x) - lookback
    
    df_train = data.iloc[index_start_train:index_end_train, :]
    
    # make validation df
    index_start_val = data.index.get_loc(df_train.index[-1].strftime("%Y-%m-%d")) + 1
    x = data[dt(year+1+train_years+val_years,1,1,0,0,0):].first_valid_index().strftime("%Y-%m-%d")
    index_end_val = data.index.get_loc(x) - lookback
    
    df_val = data.iloc[index_start_val:index_end_val, :]
    
    # make test df
    index_start_test = data.index.get_loc(df_val.index[-1].strftime("%Y-%m-%d")) + 1
    if train_years == 13:
        x = data[dt(year+train_years+val_years+test_years,1,1,0,0,0):].last_valid_index().strftime("%Y-%m-%d")
        index_end_test = data.index.get_loc(x) + 1
    else:
        x = data[dt(year+1+train_years+val_years+test_years,1,1,0,0,0):].first_valid_index().strftime("%Y-%m-%d")
        index_end_test = data.index.get_loc(x) 
    
    df_test = data.iloc[index_start_test:index_end_test, :]
    
    return df_train.sort_index(), df_val.sort_index(),  df_test.sort_index()


# define targets/features/standardize functions

def get_targets(dataframe, iv=60, ten_day=False):
    if ten_day:
        x = 'IvMean' + str(iv) + '_return_ten'
        targets = [dataframe[col] for col in dataframe.columns if x in col]
        targets = pd.DataFrame(targets).transpose()
    else:
        x = 'IvMean' + str(iv) + '_return_next'
        targets = [dataframe[col] for col in dataframe.columns if x in col] 
        targets = pd.DataFrame(targets).transpose()
    return targets

def get_attn_targs(dataframe, iv=60, ten_day=False):
    if ten_day:
        x = 'IvMean' + str(iv) + '_return_ten'
        targets = [dataframe[col] for col in dataframe.columns if x in col]
        targets = pd.DataFrame(targets).transpose()
    else:
        x = 'IvMean' + str(iv) + '_return_prev'
        targets = [dataframe[col] for col in dataframe.columns if x in col] 
        targets = pd.DataFrame(targets).transpose()
    return targets

def get_features(dataframe):
    features = [dataframe[col] for col in dataframe.columns if 'return' not in col]
    features = pd.DataFrame(features).transpose()
    return features
    
def standardize(dataframe):
    df_standardized = pd.DataFrame()
    for col in dataframe.columns:
        col_mean = dataframe[col].mean()
        col_std = dataframe[col].std()
        df_standardized[col] = (dataframe[col] - col_mean) / col_std
    return df_standardized

def unstandardize(df_stand, df_orig):
    if len(df_stand.columns) == len(df_orig.columns):
        df_un = pd.DataFrame()
        for i, col in enumerate(df_stand.columns):
            col_og = df_orig.columns[i]
            df_un[col] = df_stand[col] * df_orig[col_og].std() + df_orig[col_og].mean()
        return df_un
    else:
        print("Value Error: Dataframes must have same column lengths")

def reduce_features(df, drop_columns):
    for col in df.columns:
        for dc in drop_columns:
            if dc in col:
                df = df.drop(col, axis=1)
    return df
        
class TS_Dataset(Dataset):
    def __init__(self, dataframe, lookback=30, iv=60):
        self.lookback = lookback
        self.X = torch.tensor(standardize(get_features(dataframe)).values).float()
        self.y = torch.tensor(standardize(get_targets(dataframe, iv=iv)).values).float()

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, i):
        if i >= self.lookback - 1:
            i_start = i - self.lookback + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.lookback - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]

class TS_Dataset_attn(Dataset):
    def __init__(self, dataframe, lookback=30, iv=60):
        self.lookback = lookback
        self.X = torch.tensor(standardize(get_features(dataframe)).values).float()
        self.y = torch.tensor(standardize(get_targets(dataframe, iv=iv)).values).float()
        self.tgt = torch.tensor(standardize(get_attn_targs(dataframe, iv=iv)).values).float()

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, i):
        if i >= self.lookback - 1:
            i_start = i - self.lookback + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.lookback - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i], self.tgt[i]
