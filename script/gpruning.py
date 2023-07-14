import pandas as pd
import os
import dataloader
import math
import numpy as np


def pruning(dataset, mean_rate, verdex):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset)
    data_col = pd.read_csv(os.path.join(dataset_path, 'vel.csv')).shape[0]
    train, _, _ = dataloader.load_data(dataset, data_col, 0)
    train = train.to_numpy()
    print(train.shape, type(train))
    mean_values = np.mean(train, axis=0)
    variance_values = np.var(train, axis=0)
    measure_values = mean_values * mean_rate + variance_values
    top_k_indices = np.argpartition(measure_values, -verdex)[-verdex:]

    print(type(top_k_indices))
    

# def 

if __name__ == "__main__":
    pruning("metr-la", 2, 20)