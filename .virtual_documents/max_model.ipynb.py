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

MODEL_PATH = 'max_model.stan'
sns.set_style("darkgrid", {"axes.facecolor": ".9"})


temps = pd.read_csv('data/Hyperlocal_Temperature_Monitoring.csv')
temps.head()


covariates = pd.read_csv('data/temp3.csv').round(8)
covariates2 = pd.read_csv('data/temp_1.csv').round(8)
covariates = covariates.merge(covariates2)
covariates['mean_fa_ratio'] = covariates['mean_fa_ratio'].fillna(0)
temps = temps.drop(index=np.where(temps['AirTemp'].isna())[0]).reset_index(0)
temps = temps.groupby(['Latitude','Longitude','Day','Year']).agg({'AirTemp':np.max}).reset_index(0).reset_index(0).reset_index(0).reset_index(0)
data = temps.merge(covariates, how='outer', on=['Latitude','Longitude'])


N = data.shape[0]
X_new_size = 5000
X_size = N - X_new_size
shuffled_data = data.sample(frac=1)


d = {'N': N, 'M': X_size}

for col in shuffled_data.columns:
    if col not in ['Day','Year','Latitude','Longitude','AirTemp']:
        d[col] = shuffled_data[col].values

d['y'] = shuffled_data['AirTemp'].values
r = json.dumps(d,cls=NpEncoder)
with open('data/data.json', 'w') as f:
    json.dump(json.loads(r), f)


DATA = 'data/data.json'


model = CmdStanModel(stan_file=MODEL_PATH)


bern_vb = model.variational(data=d,require_converged=False)


mle = model.sample(data=d)


sns.distplot(result)
sns.distplot(test['AirTemp'])


list(bern_vb.variational_params_dict.items())[:10]


y_sims = mle.stan_variable(var='y_rep')


def check(simulated_data,y,agg_func,function_name):
    agg_data = agg_func(simulated_data,axis=1)
    ax = sns.displot(agg_data)
    ax.fig.suptitle(function_name)
    ax.axes[0][0].axvline(x = agg_func(y), color='red', linewidth=1,label='Original Data')
    plt.legend()
    plt.show()


n = 3000
y = shuffled_data['AirTemp'][:X_size] 
simulated_data = y_sims[:n]

check(simulated_data,y,np.mean,'Mean')
check(simulated_data,y,np.min,'Min')
check(simulated_data,y,np.max,'Max')
check(simulated_data,y,np.var,'Variance')
check(simulated_data,y,np.median,'Median')



