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

MODEL_PATH = 'time_model_2.stan'

sns.set_style("darkgrid", {"axes.facecolor": ".9"})


temps = pd.read_csv('data/Hyperlocal_Temperature_Monitoring.csv')
temps.head()


covariates = pd.read_csv('data/temp3.csv').round(8)
covariates2 = pd.read_csv('data/temp_1.csv').round(8)
covariates = covariates.merge(covariates2)
covariates['mean_fa_ratio'] = covariates['mean_fa_ratio'].fillna(0)
temps = temps.drop(index=np.where(temps['AirTemp'].isna())[0])
data = temps.merge(covariates, how='outer', on=['Latitude','Longitude'])


data


data['month'] = pd.to_datetime(data['Day']).dt.month


data = pd.get_dummies(data,columns=['month'])


g = data.groupby(['Sensor.ID','Day'])


H = 24
K = 4
N = len(g.indices)

y = []#np.zeros([N,H])
X = []#np.zeros([N,K])
# INVESTIGATE DATA
i = 0

for k,_ in tqdm(g.indices.items()):
    sub_group = g.get_group(k)
    if sub_group['AirTemp'].shape[0] == H:
        y.append(sub_group['AirTemp'])
        covariates = sub_group[['num_build500','mean_fa_ratio','min_distance_park','num_trees_50m','month_6','month_7','month_8', 'month_9','month_10']].iloc[0].values
        X.append(covariates)

y = np.array(y)
X = np.array(X)


H = 24
K = 4
N = len(g.indices)

y = []#np.zeros([N,H])
X = []#np.zeros([N,K])
# INVESTIGATE DATA
i = 0
g = data.groupby(['Sensor.ID'])
for k,_ in tqdm(g.indices.items()):
    sub_group = g.get_group(k)
    for 
        
    y.append(np.mean())
    covariates = sub_group[['num_build500','mean_fa_ratio','min_distance_park','num_trees_50m']].iloc[0].values
    print(covariates)
    break
#         X.append(covariates)

# y = np.array(y)
# X = np.array(X)


data


X


l = random.sample(list(np.where(X[:,6])[0]),10)
for i in l:
    sns.lineplot(x=range(24),y=y[i,:])


N = y.shape[0]
X_new_size = 5000
K = X.shape[1]
# X_size = N - X_new_size
X_size = 1
shuff_idx = random.shuffle(list(range(N)))
shuff_y, shuff_X = y[shuff_idx,:][0], X[shuff_idx,:][0]
beta_mean = np.random.normal(size=X.shape[1])
beta_sd = np.random.uniform(size=X.shape[1])


d = {'N': N, 'M': X_size, 'H': H, 'K': K, 'X': shuff_X, 'y': shuff_y,'beta_mean': beta_mean,'beta_sd': beta_sd,'y_cov':shuff_y[:X_size,:].T}


model = CmdStanModel(stan_file=MODEL_PATH)


bern_vb = model.variational(data=d,require_converged=False)


b = bern_vb.stan_variable(var='beta')
for i in range(4):
    sns.lineplot(x=range(24),y=b[:,i])


y_sims = bern_vb.stan_variable(var='y_rep')
b = bern_vb.stan_variable(var='beta')


sns.lineplot(x=range(24),y=shuff_y[1,:],label='true')
sns.lineplot(x=range(24),y=y_sims[0,:])


mu = np.zeros(5)
cov = np.eye(5)
r = np.random.multivariate_normal(mu,cov) 
sns.lineplot(x=range(5),y=r)


n = 100
mu = np.zeros(n)
cov = np.eye(n)

cov = cov_off_axis(n,cov)
cov = cov_exp(n,mu,cov)

r = np.random.multivariate_normal(mu,cov) 
sns.lineplot(x=range(n),y=r[:])


def cov_off_axis(n,cov):
    for i in range(n):
        if i get_ipython().getoutput("= n - 1:")
            cov[i,i+1] = .5
        if i get_ipython().getoutput("= 0:")
            cov[i,i-1] = .5
    return cov
        
def cov_exp(n,mu,cov):
    l = list(range(n))
    for i in range(n):
        for j in range(n):
            cov[i,j] = np.exp(-(np.abs(l[i] - l[j]))*1/n)
    return cov


cov = bern_vb.stan_variable(var='Sigma')
sns.heatmap(cov)



