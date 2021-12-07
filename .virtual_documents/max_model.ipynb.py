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
from sklearn.metrics import mean_squared_error

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


data = data[pd.to_datetime(data.Day).dt.month.isin([8])].reset_index(0).drop(columns='index')
data['is_august'] = (pd.to_datetime(data.Day).dt.month == 8).astype(int)
data['bias'] = 1
X = data[['num_build500','mean_fa_ratio','min_distance_park','num_trees_15m','bias']].values
y = data['AirTemp'].values





A = [1,2,3]


def a_w_i(k,i):
    arr = list(range(k))
    arr.remove(i)
    return arr

def create_splits(n,k):
    arr = np.arange(n)
    random.shuffle(arr)
    splits = np.array(np.array_split(arr,k),dtype=object)
    return [(np.concatenate(splits[a_w_i(k,i)]),splits[i]) for i in range(k)]


MODEL_PATH_CV = 'models/max_model_cv.stan'
MODEL_PATH = 'models/max_model.stan'


N = data.shape[0]


mse = []

for train_idx, val_idx in tqdm(create_splits(N,5)):
    X_train, y_train = X[train_idx,:], y[train_idx]
    X_val, y_val = X[val_idx,:], y[val_idx]
    
    d = {'M': X_train.shape[0],
         'T': X_val.shape[0],
         'K': X_train.shape[1],
         'sigma_y': np.var(y_train),
         'X': X_train, 
         'y': y_train,
         'X_val': X_val}
    
    model = CmdStanModel(stan_file=MODEL_PATH_CV)
    vb = model.variational(data=d,iter=2500)
    
    out = vb.variational_params_dict
    y_pred = [out['y_out[' + str(i+1) + ']'] for i in range(len(val_idx))]
    
    mse.append(mean_squared_error(y_val,y_pred))


X.shape


n = X.shape[0]
idx = list(range(n))
random.shuffle(idx)
train = int(0.8 * n)
X_train,y_train = X[:train,:],y[:train]
X_val,y_val = X[train:,:],y[train:]


d = {'M': X_train.shape[0],
     'K': X.shape[1],
     'T': X_val.shape[0],
     'L': 1000,
     'S': 100,
     'X': X_train, 
     'y': y_train,
     'X_val':X_val}
    
model = CmdStanModel(stan_file=MODEL_PATH)


t1 = time()
mcmc = model.variational(data=d,output_dir='misc',save_diagnostics=True)
t2 = time()





y_sims = mcmc.stan_variable(var='y_rep')
y_out = mcmc.stan_variable(var='y_out')
b = mcmc.stan_variable(var='beta')


y_sims.shape


y_sims


mean_squared_error(y_val,y_out)


list(bern_vb.variational_params_dict.items())[:10]


y_sims = mle.stan_variable(var='y_rep')


def check(simulated_data,y,agg_func,function_name,ax):
    agg_data = agg_func(simulated_data,axis=1)
    ax = sns.histplot(agg_data,ax=ax).set_title(function_name)
    ax.axes[0][0].axvline(x = agg_func(y), color='red', linewidth=1,label='Original Data')
    plt.legend()
    plt.show()


y_p = y_train[:200] 

ax = plt.subplots(1,5,figsize=(10,10))
functions = [np.mean,np.min,np.max,np.var,np.median]
titles = ['Mean','Min','Max','Variance','Median']
for i,(func,title) in enumerate(zip(functions,titles)):
    check(y_sims,y_p,func,title,ax[i])


np.mean(y_sims,axis=1).shape



