import cmdstanpy
import pandas as pd
import numpy as np
from cmdstanpy import cmdstan_path, CmdStanModel
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json
import scipy.stats as stats
import patsy

MODEL_PATH = 'max_model.stan'
sns.set_style("darkgrid", {"axes.facecolor": ".9"})


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


temps = pd.read_csv('data/Hyperlocal_Temperature_Monitoring.csv')
temps.head()


covariates = pd.read_csv('data/temp3.csv').round(8)
covariates2 = pd.read_csv('data/temp_1.csv').round(8)
covariates = covariates.merge(covariates2)
covariates['mean_fa_ratio'] = covariates['mean_fa_ratio'].fillna(0)
temps = temps.drop(index=np.where(temps['AirTemp'].isna())[0]).reset_index(0)
temps = temps.groupby(['Latitude','Longitude','Day','Year']).agg({'AirTemp':np.max}).reset_index(0).reset_index(0).reset_index(0).reset_index(0)
data = temps.merge(covariates, how='outer', on=['Latitude','Longitude'])


j = data.to_json(orient='columns')


N = data.shape[0]
X_new_size = 5000
X_size = N - X_new_size
shuffled_data = data.sample(frac=1)


d = {'N': N, 'M': X_size}

for col in shuffled_data.columns:
    if col not in ['Day','Year','Latitude','Longitude','AirTemp']:
        d[col] = shuffled_data[col].values

d['y'] = shuffled_data['AirTemp'].values


DATA = 'data/data.json'


from patsy import dmatrix

B = interpolate.BSpline
