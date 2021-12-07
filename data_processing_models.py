from tqdm import tqdm
from collections import defaultdict

import pandas as pd
import numpy as np
import h5py


DATA = 'data/data.csv'

if __name__ == '__main__':
    data = pd.read_csv(DATA)

    data['Day'] = pd.to_datetime(data.Day)
    data = data.sort_values(by=['Sensor.ID', 'Day'])
    data = data[data.Day.dt.month.isin([7, 8])].reset_index(
        0).drop(columns='index')
    data['is_august'] = (data.Day.dt.month == 8).astype(int)
    data['bias'] = 1

    print('~Creating 24h Dataset~')
    g = data.groupby(['Sensor.ID'])

    H = 24
    K = 4
    N = len(g.indices)
    W = 7

    y = []
    X = []
    i = 0
    for k, _ in tqdm(g.indices.items()):
        sub_group = g.get_group(k)
        unique_dates = sub_group.Day.unique()
        windows = [unique_dates[x:x + W]
                   for x in range(0, len(unique_dates), W)]
        for window in windows:
            hours = defaultdict(list)
            for day in window:
                day_group = sub_group[sub_group.Day == day]
                for _, row in day_group.iterrows():
                    hours[row.Hour].append(row.AirTemp)
            result = np.zeros(H)
            for k, v in hours.items():
                result[k] = np.mean(v)
            y.append(result)
            covariates = list(sub_group[['num_build500',
                                         'min_distance_park',
                                         'num_trees_15m',
                                         'is_august']].iloc[0].values) + [1]
            X.append(covariates)

    y = np.array(y)
    X = np.array(X)

    with h5py.File('data/avg_model_data.h5', 'w') as f:
        f.create_dataset('X', data=X)
        f.create_dataset('y', data=y)

    print('~Creating Max Model Dataset~')
    g = data.groupby(['Sensor.ID', 'Day'])
    X = []
    y = []
    for k, _ in tqdm(g.indices.items()):
        sub_group = g.get_group(k)
        y.append(np.max(sub_group['AirTemp']))
        X.append(sub_group[['num_build500',
                            'mean_fa_ratio',
                            'min_distance_park',
                            'num_trees_15m',
                            'is_august',
                            'bias']].iloc[0].values)

    X = np.array(X)
    y = np.array(y)

    with h5py.File('data/max_model_data.h5', 'w') as f:
        f.create_dataset('X', data=X)
        f.create_dataset('y', data=y)
