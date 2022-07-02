# -*- coding: utf-8 -*-
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from trainer import Trainer

from utils import load_BG
from utils import split_data
from utils import split_to_xy
from utils import get_hidden_sizes

# ml models
from model import MLP, LSTM
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

def define_argparser():
    p = argparse.ArgumentParser()

    # hyper-parameters
    #p.add_argument("--model_fn", required=True)
    p.add_argument("--gpu_id", type=int, default=0 if torch.cuda.is_available() else -1)
    p.add_argument("--model", type=str, default = 'MLP')

    p.add_argument("--data_path", default="Virtual_patients_20/V_CGMS_{}.csv")
    p.add_argument("--train_ratio", type=float, default=.7)
    p.add_argument("--points_to_see", type=int, default=32)
    p.add_argument("--next_points_predict", type=int, default=6)



    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_epochs", type=int, default=500)

    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--use_dropout", action='store_true')
    p.add_argument("--dropout_p", type=float, default=.3)

    p.add_argument("--verbose", type=int, default=1)

    config = p.parse_args()

    return config

def get_model(config, input_size=None, output_size=None, hidden_sizes=None, use_batch_norm=False, dropout_p=.5):
    if config.model == "MLP":
        model = MLP(input_size, 
                    output_size, 
                    hidden_sizes, 
                    use_batch_norm, 
                    dropout_p)

    elif config.model == "LSTM":
        model = LSTM(input_size=input_size, 
                    output_size=output_size,
                    hidden_sizes=hidden_sizes,
                    use_batch_norm=use_batch_norm,
                    dropout_p=dropout_p,
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
    config.model_fn = f"./model_results/{config.model}.pth"
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device("cuda:%d" % config.gpu_id)

    data = load_BG(config)
    data = split_data(data.to(device), device, config.train_ratio)
    
    x_train, y_train = split_to_xy(data[0], device, config.points_to_see, config.next_points_predict)
    x_valid, y_valid = split_to_xy(data[1], device, config.points_to_see, config.next_points_predict)
    #x_test, y_test = split_to_xy(data[2], device, config.points_to_see, config.next_points_predict)


    print("Train:", x_train.shape, y_train.shape)
    print("Valid:", x_valid.shape, y_valid.shape)

    input_size = config.points_to_see  # |x| = (9912, 32)
    output_size =1

    if config.model in ['SVR', 'KNN', 'RFR']:
        model = get_model(config)
        model.fit(x_train, y_train)
        
        # Save best model weights.
        torch.save({  # pickle로 저장
            'model' : model,
            'config': config,
        }, config.model_fn)


    elif config.model in ['LSTM', 'MLP']:
        model = get_model(
            config,
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=get_hidden_sizes(input_size, 
                                        output_size,
                                        config.n_layers),
            use_batch_norm=not config.use_dropout,
            dropout_p=config.dropout_p,
        ).to(device)
        optimizer = optim.Adam(model.parameters())
        crit = nn.MSELoss()

        if config.verbose >= 1:
            print(model)
            print(optimizer)
            print(crit)

        trainer = Trainer(model, optimizer, crit)

        trainer.train(
            train_data=(x_train, y_train),
            valid_data=(x_valid, y_valid),
            config=config
        )

        # Save best model weights.
        torch.save({  # pickle로 저장
            'model' : trainer.model.state_dict(),
            'opt' : optimizer.state_dict(),
            'config': config,
        }, config.model_fn)


if __name__ == '__main__':
    config = define_argparser()
    main(config)