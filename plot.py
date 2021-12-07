import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import graphviz

sns.set_style("darkgrid", {"axes.facecolor": ".9"})

DATA_PATH = 'data.csv'
DOT_GRAPH = 'misc/causal_graph.dot'

data = pd.read_csv('data.csv')


def causal_diagram():
    with open(DOT_GRAPH) as f:
        dot_graph = f.read()

    return graphviz.Source(dot_graph)


def hourly_temp_graph():
    avg = data.groupby(['Latitude', 'Longitude', 'Hour']).mean(
    ).reset_index(0).reset_index(0).reset_index(0)
    cols = [
        'num_trees15',
        'mean_fa_ratio',
        'min_distance_park',
        'num_build500']
    titles = [
        'Number of Trees within 15m',
        'Mean Floor-Area Ratio',
        'Minimum Distance to Park',
        'Number of Buildings within 50m']

    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(10, 10))
    fig.suptitle('Average Hourly Temperature Stratified by:', fontsize=16)

    for i in range(2):
        for j in range(2):
            sns.lineplot(data=avg,
                         ax=axes[i,
                                 j],
                         x='Hour',
                         y='AirTemp',
                         hue=cols[j + 2 * i],
                         palette='flare').set_title(titles[j + 2 * i])

    plt.show()


def pair_plot():
    sns.pairplot(data[['AirTemp',
                       'num_build500',
                       'mean_fa_ratio',
                       'min_distance_park',
                       'num_trees15']])
    plt.show()
