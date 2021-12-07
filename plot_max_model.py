import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error


class MaxModelPlots:

    def __init__(self, X_train, X_val, y_train, y_val, L, vb):
        self.cols = [
            'num_build500',
            'mean_fa_ratio',
            'min_distance_park',
            'num_trees_15m',
            'bias']
        self.y_sims = vb.stan_variable(var='y_rep')
        self.y_out = vb.stan_variable(var='y_out')
        self.b = vb.stan_variable(var='beta')
        self.L = L
        self.X_train, self.X_val = X_train, X_val
        self.y_train, self.y_val = y_train, y_val
        self.vb = vb

    def get_params(self):
        out_cols = [
            'beta[1]',
            'beta[2]',
            'beta[3]',
            'beta[4]',
            'beta[5]',
            'sigma']
        dict_rename = dict(zip(out_cols, self.cols))
        result = self.vb.variational_params_pd[out_cols]
        return result.rename(columns=dict_rename)

    def mse(self):
        return mean_squared_error(self.y_val, self.y_out)

    def agg(self, simulated_data, y, agg_func):
        agg_data = agg_func(simulated_data, axis=1)
        return agg_data, agg_func(y)

    def posterior_predictive_checks(self):
        y_p = self.y_train[:self.L]

        functions = [np.mean, np.min, np.max, np.var, np.median]
        titles = ['Mean', 'Min', 'Max', 'Variance', 'Median']
        y_acc = []

        df = pd.DataFrame()

        for i, (func, title) in enumerate(zip(functions, titles)):
            agg_data, agg_y = self.agg(self.y_sims, y_p, func)
            df_temp = pd.DataFrame({'Value': agg_data, 'Function': titles[i]})
            y_acc.append(agg_y)
            df = df.append(df_temp)

        g = sns.FacetGrid(df, col="Function", sharex=False)
        g.map_dataframe(sns.histplot, x="Value")
        g.fig.suptitle('Posterior Predictive Checks', fontsize=20, y=1.1)

        for i, ax in enumerate(g.axes[0]):
            ax.axvline(
                x=y_acc[i],
                color='red',
                linewidth=1,
                label='Original Data')

        plt.show()

    def predicted_v_observed(self):
        plt.figure(figsize=(10, 10))
        ax = sns.scatterplot(
            y=self.y_out,
            x=self.y_val,
            alpha=0.5,
            color='red')
        ax.get_figure().suptitle('Predicted vs Observed', fontsize=20)
        ax.set(xlabel='Observed Air Temperature (F)',
               ylabel='Predicted Air Temperature (F)')
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.plot((x0, x1), (y0, y1), ':k')
        plt.show()
