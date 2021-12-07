import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import mean_squared_error


class AvgModelPlots:

    def __init__(self, X_train, X_val, y_train, y_val, L, mcmc):
        self.cols = [
            'num_build500',
            'mean_fa_ratio',
            'min_distance_park',
            'num_trees_15m',
            'bias']
        self.y_sims = mcmc.stan_variable(var='y_rep')
        self.y_hat = mcmc.stan_variable(var='y_hat')
        self.b = mcmc.stan_variable(var='beta')
        self.L = L
        self.X_train, self.X_val = X_train, X_val
        self.y_train, self.y_val = y_train, y_val

    def coefficient_v_hour(self):
        titles = [
            'num_build500',
            'min_distance_park',
            'num_trees_15m',
            'is_august',
            'bias']
        fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
        for i, ax in enumerate(axes):
            chain_mean = [[]] * 4
            chain_x = [i for _ in range(4) for i in range(24)]
            for c, chain in enumerate(np.split(np.arange(4000), 4)):
                y_min, y_max = [], []
                for j in range(24):
                    chain_results = self.b[chain, j, i]
                    mu = np.mean(chain_results)
                    sigma = np.var(chain_results)
                    out = stats.norm.interval(0.95, loc=mu, scale=sigma)
                    y_min.append(out[0])
                    y_max.append(out[1])
                    chain_mean[c].append(mu)
                ax.fill_between(
                    range(24),
                    y_min,
                    y_max,
                    alpha=0.1,
                    color='red')
                ax.set_title(titles[i])
            df = pd.DataFrame(np.array(chain_mean).T, columns=[1, 2, 3, 4])
            df['Hour'] = chain_x
            df = pd.melt(
                df, id_vars=['Hour'], value_vars=[
                    1, 2, 3, 4]).rename(
                columns={
                    'variable': 'Chain', 'value': 'Value'})
            a = sns.lineplot(
                data=df,
                x='Hour',
                y='Value',
                hue='Chain',
                palette='crest',
                ax=ax,
                alpha=0.4)
            if i != 0:
                a.legend_.remove()
        fig.suptitle(
            'Coefficient vs Hour, CI and Mean from 4 Chains',
            fontsize=20,
            y=0.94)
        plt.show()

    def samples_from_posterior_predictive(self):
        titles = ['Sample ' + str(i + 1) for i in range(5)]
        fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
        fig.suptitle(
            '5 Samples and 95% CI from Posterior Predictive',
            fontsize=20,
            y=0.94)

        for i, ax in enumerate(axes):
            y_min, y_max = [], []
            for j in range(24):
                posterior_predictive = self.y_sims[:, i, :, j].reshape(-1)
                mu = np.mean(posterior_predictive)
                sigma = np.var(posterior_predictive)
                out = stats.norm.interval(0.95, loc=mu, scale=sigma)
                y_min.append(out[0])
                y_max.append(out[1])

            sns.lineplot(
                x=range(24), y=self.y_train[i, :], palette='crest', ax=ax)
            ax.fill_between(range(24), y_min, y_max, alpha=0.5, color='red')
            ax.set_title(titles[i])
        plt.show()

    def checking_mcmc_convergence(self):
        titles = [
            'num_build500',
            'min_distance_park',
            'num_trees_15m',
            'is_august',
            'bias']
        fig, axes = plt.subplots(24, 5, figsize=(15, 35), sharex=True)
        fig.suptitle(
            'Checking MCMC Convergence, 4 Chains',
            fontsize=20,
            y=0.90)
        for i in range(24):
            for j in range(5):
                for chain in np.split(np.arange(4000), 4):
                    p = sns.lineplot(
                        x=range(1000), y=self.b[chain, i, j], ax=axes[i, j])
                    p.set_title(titles[j] + ' hour=' + str(i))
        plt.show()

    def posterior_predictive_checks(self):
        functions = [np.mean, np.min, np.max, np.var, np.median]
        titles = ['Mean', 'Min', 'Max', 'Variance', 'Median']
        fig, axes = plt.subplots(5, 1, figsize=(15, 10), sharex=True)
        fig.suptitle('Posterior Predictive Checks', fontsize=20, y=0.94)
        for i, (func, title) in enumerate(zip(functions, titles)):
            hour, value = [], []
            for j in range(24):
                for p in range(self.L):
                    agg = func(self.y_sims[:, :, p, j])
                    hour.append(j)
                    value.append(agg)
            df = pd.DataFrame({'Hour': hour, 'Value': value})
            ax = sns.violinplot(data=df, x='Hour', y='Value', ax=axes[i])
            ax.set_title(title)
            if i < 4:
                ax.set(xlabel='')
            for p in range(24):
                x_axis = [p - 0.25, p + 0.25]
                y_axis = [func(self.y_train[:50, p])] * 2
                sns.lineplot(x=x_axis, y=y_axis, ax=axes[i], color='red')
        plt.show()

    def hour_v_mse(self):
        mse = []
        for i in range(24):
            mse.append(mean_squared_error(
                self.y_val[:, i], np.mean(self.y_hat[:, :, i], axis=0)))
        df = pd.DataFrame({'Hour': range(24), 'MSE': mse})
        g = sns.lineplot(data=df, y='MSE', x='Hour')
        g.get_figure().suptitle('Hour vs. MSE', fontsize=20, y=0.98)
        plt.show()
