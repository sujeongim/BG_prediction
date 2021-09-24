import torch
import pandas as pd
import numpy as np

def load_BG(config):
    bg = []

    for patient_id in range(16):
        bg+=[pd.read_csv(config.data_path.format(patient_id+1),header=None).values.ravel()]
    
    bg = np.array(bg)
    bg = torch.from_numpy(bg)

    return bg

def load_BG_predict(config):
    bg = []

    for patient_id in range(16,20):
        bg+=[pd.read_csv(config.data_path.format(patient_id+1),header=None).values.ravel()]
    
    bg = np.array(bg)
    bg = torch.from_numpy(bg)

    return bg

def split_data(data, device, train_ratio=.7):
    train_cnt = int(data.size(0)* train_ratio)
    valid_cnt = data.size(0) - train_cnt
    #test_cnt = data.size(0) - train_cnt - valid_cnt

    # Shuffle dataset to split into train/valid set.
    indices = torch.randperm(data.size(0)).to(device)
    data = torch.index_select(
        data, 
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)

    return data

# default : with 32 points, predict BG level after 30min.
def split_to_xy(data, device, input_size=32, output_term=6):
    x = []
    y = []

    for data_per_patient in data:
        for i in range(input_size, len(data_per_patient)-output_term):
            x.append(data_per_patient[i-input_size:i])
            y.append(data_per_patient[i+output_term-1])
    
    x, y = torch.stack(x).to(device), torch.stack(y).to(device)
    x, y = x.float(), y.float()
    return x, y


def get_hidden_sizes(input_size, output_size, n_layers):
    step_size = int((input_size - output_size) / n_layers) # (784 - 10) / 5 

    hidden_sizes = []
    current_size = input_size
    for i in range(n_layers-1):  # layer size를 등차수열로 만들어주기
        hidden_sizes += [current_size - step_size]
        current_size = hidden_sizes[-1]

    return hidden_sizes
