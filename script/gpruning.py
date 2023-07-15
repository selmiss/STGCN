import pandas as pd
import os
from script import dataloader
import math
import numpy as np


def pruning(dataset, adj_matrix, mean_rate, verdex, len_train, len_val, tf_rate=5):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset)
    data_col = pd.read_csv(os.path.join(dataset_path, 'vel.csv')).shape[0]
    train, _, _ = dataloader.load_data(dataset, data_col, 0)
    train = train.to_numpy()
    mean_values = np.mean(train, axis=0)
    variance_values = np.var(train, axis=0)
    measure_values = mean_values * mean_rate + variance_values
    measure_matrix_value = []
    for i in range(len(adj_matrix)):
        cnt = 0
        for j in range(i+1):
            cnt += adj_matrix[i][j]
        measure_matrix_value.append(cnt)
        
    # top_k_indices = np.argpartition(measure_values, -verdex)[-verdex:]
    top_k_indices = np.argpartition(measure_matrix_value, -verdex)[-verdex:]
    train = train[:, top_k_indices]
    adj_matrix = adj_matrix[top_k_indices][:, top_k_indices]
    trainr = train[: len_train]
    val = train[len_train: len_train + len_val]
    test = train[len_train + len_val:]
    return trainr, val, test, adj_matrix
    

def load_data(dataset_name, len_train, len_val):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    vel = pd.read_csv(os.path.join(dataset_path, 'vel.csv'))

    train = vel[: len_train]
    val = vel[len_train: len_train + len_val]
    test = vel[len_train + len_val:]
    return train, val, test


if __name__ == "__main__":
    arr = np.array([[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]])
    pruning("metr-la", arr, 2, 20)