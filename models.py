import numpy as np
from cmdstanpy import CmdStanModel
import random
import h5py


class AvgModel:

    def __init__(self, model_path, data_path, L=50, S=100, split=0.8):

        with h5py.File(data_path, 'r') as f:
            X = f['X'][:]
            y = f['y'][:]

        n = X.shape[0]
        idx = list(range(n))
        random.shuffle(idx)
        train = int(split * n)
        X_train, y_train = X[:train, :], y[:train]
        X_val, y_val = X[train:, :], y[train:]

        self.L = L
        self.S = S
        self.X_train, self.X_val = X_train, X_val
        self.y_train, self.y_val = y_train, y_val

        self.model = CmdStanModel(stan_file=model_path)

    def inference(self):

        d = {'N': self.X_train.shape[0],
             'M': self.X_val.shape[0],
             'H': self.y_train.shape[1],
             'K': self.X_train.shape[1],
             'L': self.L,
             'S': self.S,
             'X': self.X_train,
             'y': self.y_train,
             'X_val': self.X_val,
             'beta_mean': np.random.normal(size=self.X_train.shape[1]),
             'beta_sd': np.ones(self.X_train.shape[1])}

        return self.model.sample(data=d)


class MaxModel:

    def __init__(self, model_path, data_path, L=1000, S=100, split=0.8):

        with h5py.File(data_path, 'r') as f:
            X = f['X'][:]
            y = f['y'][:]

        n = X.shape[0]
        idx = list(range(n))
        random.shuffle(idx)
        train = int(split * n)
        X_train, y_train = X[:train, :], y[:train]
        X_val, y_val = X[train:, :], y[train:]

        self.L = L
        self.S = S
        self.X_train, self.X_val = X_train, X_val
        self.y_train, self.y_val = y_train, y_val

        self.cols = [
            'num_build500',
            'mean_fa_ratio',
            'min_distance_park',
            'num_trees_15m',
            'bias']

        self.model = CmdStanModel(stan_file=model_path)

    def inference(self):

        d = {'M': self.X_train.shape[0],
             'K': self.X_train.shape[1],
             'T': self.X_val.shape[0],
             'L': self.L,
             'S': self.S,
             'X': self.X_train,
             'y': self.y_train,
             'X_val': self.X_val}

        vb = self.model.variational(data=d)

        return vb
