import argparse
import torch
import torch.nn as nn

import sys
import numpy as np
import matplotlib.pyplot as plt

from utils import load_BG_predict
from utils import get_hidden_sizes

# ml models
from model import MLP, LSTM
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

def load(fn, device):
    d = torch.load(fn, map_location=device) 
    
    return d['model'], d['config']

def plot( y, y_hat):
    #print(x.shape, y.shape)
    plt.title("Blood Glucose Level Prediction")
    plt.plot(y.detach().cpu().squeeze(), 'r')
    plt.plot(y_hat.detach().cpu().squeeze(), 'b')
    plt.show()


def test(n, model, x, y, to_be_shown=True):
    model.eval()

    with torch.no_grad():
        y_hat = model(x)
        mse_loss = nn.MSELoss()
        rmse = float(torch.sqrt(mse_loss(y_hat, y)))
        mape = float(torch.mean(torch.abs((y - y_hat) / y_hat)) * 100)
        print("patient %d   RMSE : %.4e , MAPE : %.4e" % (n, rmse, mape))
    if to_be_shown:
        plot(y, y_hat)
        
def test_ml(n, model, x, y, to_be_shown=True):
    y_hat = model.predict(x)
    mse_loss = nn.MSELoss()
    rmse = float(torch.sqrt(mse_loss(y_hat, y)))
    mape = float(torch.mean(torch.abs((y - y_hat) / y_hat)) * 100)
    print("patient %d   RMSE : %.4e , MAPE : %.4e" % (n, rmse, mape))
    if to_be_shown:
        plot(y, y_hat)

def define_argparser():
    p = argparse.ArgumentParser()

    # hyper-parameters
    p.add_argument("--model_fn", required=True)
    p.add_argument("--gpu_id", type=int, default=0 if torch.cuda.is_available() else -1)
    p.add_argument("--data_path", default="Virtual_patients_20/V_CGMS_{}.csv")

    config = p.parse_args()

    return config

def get_model(config, input_size, output_size, hidden_sizes, use_batch_norm, dropout_p):
    if config.model == "MLP":
        model = MLP(input_size, 
                    output_size, 
                    hidden_sizes, 
                    use_batch_norm, 
                    dropout_p)

    elif config.model == "LSTM":
        model = LSTM(input_size, 
                    output_size,
                    hidden_sizes,
                    use_batch_norm,
                    dropout_p,
                    lstm_hidden_size=500, 
                    n_layers=2)

    elif config.model == "SVR":
        model = SVR(kernel='rbf', gamma='auto', C=10**5)
    elif config.model == "KNN":
        model = KNeighborsRegressor(n_neighbors=7, weights='distance', metric='euclidian')
    elif config.model == "RFR":
        model = RandomForestRegressor(n_estimators=235, max_depth=8, min_sample_split=2, max_features='auto')
    return model

def main(config):
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device("cuda:%d" % config.gpu_id)

    data = load_BG_predict(config).to(device)
    
    print(data.shape)
    if config.model in ['SVR', 'KNN', 'RFR']:
        model, train_config = load(config.model_fn, device)
        
        for n, data_per_patient in enumerate(data):
            print(data_per_patient.shape)
            x_test = []
            y_test = []

            for i in range(train_config.points_to_see, len(data_per_patient)-(train_config.next_points_predict)):
                x_test.append(data_per_patient[i-train_config.points_to_see:i])
                y_test.append(data_per_patient[i+train_config.next_points_predict-1])

            x_test, y_test = torch.stack(x_test).float().to(device), torch.stack(y_test).float().to(device)
            test_ml(n, model, x_test, y_test, to_be_shown=True)
    
    elif config.model in ['LSTM', 'MLP']:
        model_dict, train_config = load(config.model_fn, device)

        input_size = train_config.points_to_see
        output_size = 1
        
        model = get_model(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=get_hidden_sizes(input_size, output_size, train_config.n_layers),
            use_batch_norm=not train_config.use_dropout,
            dropout_p=train_config.dropout_p,
            ).to(device)

        model.load_state_dict(model_dict)
    
        for n, data_per_patient in enumerate(data):
            print(data_per_patient.shape)
            x_test = []
            y_test = []

            for i in range(train_config.points_to_see, len(data_per_patient)-(train_config.next_points_predict)):
                x_test.append(data_per_patient[i-train_config.points_to_see:i])
                y_test.append(data_per_patient[i+train_config.next_points_predict-1])

            x_test, y_test = torch.stack(x_test).float().to(device), torch.stack(y_test).float().to(device)
            test(n, model, x_test, y_test, to_be_shown=True)



if __name__ == '__main__':
    config = define_argparser()
    main(config)