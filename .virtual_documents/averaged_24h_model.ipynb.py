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


data['Day'] =pd.to_datetime(data.Day)
data = data.sort_values(by=['Sensor.ID','Day'])


data


g = data.groupby(['Sensor.ID'])

H = 24
K = 4
N = len(g.indices)
W = 3

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
        covariates = list(sub_group[['num_build500','mean_fa_ratio','min_distance_park','num_trees_15m']].iloc[0].values)
        X.append(covariates)

y = np.array(y)
X = np.array(X)


from sklearn.preprocessing import StandardScaler


n = StandardScaler()
X_t = n.fit_transform(X)


X_t


N = y.shape[0]
X_new_size = 20
K = X.shape[1]
# X_size = N - X_new_size
X_size = 200
shuff_idx = random.shuffle(list(range(N)))
shuff_y, shuff_X = y[shuff_idx,:][0], X[shuff_idx,:][0]
beta_mean = np.random.normal(size=X.shape[1])
beta_sd = np.random.uniform(size=X.shape[1])


d = {'N': N, 'M': X_size, 'H': H, 'K': K, 'X': shuff_X, 'y': shuff_y,'beta_mean': beta_mean,'beta_sd': beta_sd,'y_cov':shuff_y[:X_size,:].T}


model = CmdStanModel(stan_file=MODEL_PATH)


bern_vb = model.sample(data=d)


y_sims = bern_vb.stan_variable(var='y_rep')
b = bern_vb.stan_variable(var='beta')
x_sims = [i for _ in range(4000) for i in range(24)]


cols = ['num_build500','mean_fa_ratio','min_distance_park','num_trees_15m']
i2v = {i:v for i,v in enumerate(cols)} 

df2 = pd.DataFrame()
for i in range(len(cols)):
    temp = pd.DataFrame(b[:,:,i])
    temp['var'] = i2v[i]
    df2 = df2.append(temp)
    
df2 = pd.melt(df2,id_vars=['var'], value_vars=list(range(24)))
sns.lineplot(data=df2,x='variable',y='value',hue='var')


s = 100
for j in range(3):
    sns.lineplot(x=range(24),y=shuff_y[j,:],label='true')
    idx = random.sample(list(range(y_sims.shape[0])),s)
    df = pd.DataFrame()
    df['y'] = y_sims[idx,j,:].reshape(-1)
    df['x'] = [i for _ in range(s) for i in range(24)]
    sns.lineplot(data=df,x='x',y='y')
    plt.show()








np.max(y_sims[:,0,:],axis=0).shape



