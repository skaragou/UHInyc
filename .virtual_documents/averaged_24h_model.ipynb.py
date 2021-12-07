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
from collections import defaultdict
import time

MODEL_PATH = 'models/time_model_2.stan'

sns.set_style("darkgrid", {"axes.facecolor": ".9"})


temps = pd.read_csv('data/Hyperlocal_Temperature_Monitoring.csv')
temps.head()


covariates = pd.read_csv('data/temp3.csv').round(8)
covariates2 = pd.read_csv('data/temp_1.csv').round(8)
covariates = covariates.merge(covariates2)
covariates['mean_fa_ratio'] = covariates['mean_fa_ratio'].fillna(0)
temps = temps.drop(index=np.where(temps['AirTemp'].isna())[0])
data = temps.merge(covariates, how='outer', on=['Latitude','Longitude'])


data['Day'] =pd.to_datetime(data.Day)
data = data.sort_values(by=['Sensor.ID','Day'])


data = data[data.Day.dt.month.isin([7,8])].reset_index(0).drop(columns='index')


data['is_august'] = (data.Day.dt.month == 8).astype(int)


data


g = data.groupby(['Sensor.ID'])

H = 24
K = 4
N = len(g.indices)
W = 7

y = []#np.zeros([N,H])
X = []#np.zeros([N,K])
i = 0
for k,_ in tqdm(g.indices.items()):
    sub_group = g.get_group(k)
    unique_dates = sub_group.Day.unique()
    windows = [unique_dates[x:x+W] for x in range(0, len(unique_dates), W)]
    for window in windows:
        hours = defaultdict(list)
        for day in window:
            day_group = sub_group[sub_group.Day == day]
            for _,row in day_group.iterrows():
                hours[row.Hour].append(row.AirTemp)
        result = np.zeros(H)
        for k,v in hours.items():
            result[k] = np.mean(v)
        y.append(result)
        # Add back mean floor area ratio
        covariates = list(sub_group[['num_build500','min_distance_park','num_trees_15m','is_august']].iloc[0].values)+[1] 
        X.append(covariates)

y = np.array(y)
X+ = np.array(X)


tmpx = X
tmpy = y


n = X.shape[0]
idx = list(range(n))
random.shuffle(idx)
train = int(0.8 * n)
X_train,y_train = tmpx[:train,:],tmpy[:train]
X_val,y_val = tmpx[train:,:],tmpy[train:]


d = {'N': X_train.shape[0],
     'M': X_val.shape[0], 
     'H': y_train.shape[1], 
     'K': X_train.shape[1], 
     'L': 50, 
     'S': 10,
     'X': X_train, 
     'y': y_train,
     'X_val': X_val,
     'beta_mean': np.random.normal(size=X.shape[1]),
     'beta_sd': np.ones(X.shape[1])}


MODEL_PATH = 'models/24h_model.stan'
model = CmdStanModel(stan_file=MODEL_PATH)


bern_vb = model.sample(data=d)


y_sims = bern_vb.stan_variable(var='y_rep')
# y_hat = bern_vb.stan_variable(var='y_hat')
b = bern_vb.stan_variable(var='beta')


y_sims.shape


for i in range(3):
    sns.lineplot(x=range(24),y=b[-1,:,i])


for i in range(30):
    sns.lineplot(x=range(24),y=y_sims[i,0,:],color='r',alpha=0.1)
    
sns.lineplot(x=range(24),y=shuff_y[0,:])


bern_vb = model.variational(data=d)


np.var(y_sims[-1,:,1,0])


cols = ['num_build500','min_distance_park','num_trees_15m']
i2v = {i:v for i,v in enumerate(cols)} 

df2 = pd.DataFrame()
for i in range(len(cols)):
    temp = pd.DataFrame(b[:,:,i])
    temp['var'] = i2v[i]
    df2 = df2.append(temp)
    
df2 = pd.melt(df2,id_vars=['var'], value_vars=list(range(24)))
sns.lineplot(data=df2,x='variable',y='value',hue='var')


model = CmdStanModel(stan_file=MODEL_PATH)


s = 100
for j in range(1):
    sns.lineplot(x=range(24),y=shuff_y[j,:],label='true')
    idx = random.sample(list(range(y_sims.shape[0])),s)
    df = pd.DataFrame()
    df['y'] = y_sims[-10:,j,:].reshape(-1)
    df['x'] = [i for _ in range(10) for i in range(24)]
    sns.lineplot(data=df,x='x',y='y')
    plt.show()


def check(simulated_data,y,agg_func,function_name):
    agg_data = agg_func(simulated_data,axis=1)
    df = pd.DataFrame()
    df['y'] = agg_data.reshape(-1)
    df['x'] = [i for _ in range(agg_data.shape[0]) for i in range(24)]
    sns.lineplot(data=df,x='x',y='y')
    ax = sns.lineplot(x=range(24),y=agg_func(y,axis=0),label='true')
    ax.get_figure().suptitle(function_name)
    plt.legend()
    plt.show()


n = 3000
y = shuff_y[:100] 
simulated_data = y_sims[-100:,:,:]

check(simulated_data,y,np.mean,'Mean')
check(simulated_data,y,np.min,'Min')
check(simulated_data,y,np.max,'Max')
check(simulated_data,y,np.var,'Variance')
check(simulated_data,y,np.median,'Median')


def calculate_mse(y_h,y):
    return np.mean([np.mean(np.mean(np.power(y_h[:,i,:] - shuff_y[i,:],2),axis=0)) for i in range(100)])


calculate_mse(y_hat[150:,:,:],shuff_y[-100:,:])


SMOOTHED_MODEL = 'smoothed_model.stan'
model1 = CmdStanModel(stan_file=SMOOTHED_MODEL)


d1 = {'N':50,'k': K,'n': H,'beta_mean': beta_mean,'beta_sd': beta_sd,'sigma_mean': np.zeros(6),'sigma_sd': np.ones(6),'y': shuff_y[:50,:],'X': shuff_X[:50,:]}


mcmc = model1.sample(data=d1)


y_sims = mcmc.stan_variable(var='y_rep')
# y_hat = bern_vb.stan_variable(var='y_hat')
b = mcmc.stan_variable(var='b')


y_sims.shape


for j in range(5):
    sns.lineplot(x=range(24),y=shuff_y[j,:],label='true')
    df = pd.DataFrame()
    df['y'] = y_sims[-1,j,:].reshape(-1)
    df['x'] = [i for _ in range(1) for i in range(24)]
    sns.lineplot(data=df,x='x',y='y')
    plt.show()


b = mcmc.stan_variable(var='sigma_b')


cols = ['num_build500','min_distance_park','num_trees_15m']
i2v = {i:v for i,v in enumerate(cols)} 

df2 = pd.DataFrame()
for i in range(len(cols)):
    temp = pd.DataFrame(b[-1,i,:].reshape(1,-1))
    temp['var'] = i2v[i]
    df2 = df2.append(temp)
    
df2 = pd.melt(df2,id_vars=['var'], value_vars=list(range(24)))
sns.lineplot(data=df2,x='variable',y='value',hue='var')


b[-1,i,:]


temp



