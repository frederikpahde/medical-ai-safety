import pandas as pd
import numpy as np
import seaborn as sns

def get_outlier_label(c, outlier_samples, new_color_every):
    if new_color_every is None:
        return 1 if c in outlier_samples else 0
    
    for i in range(4):
        if c in outlier_samples[i * new_color_every:(i + 1) * new_color_every]:
            return i + 1
            
    return 0

def plot_2d(data, label, ax, axis_labels={"x": "Dim 1", "y": "Dim 2"}):
    data = pd.DataFrame(data)
    data.columns = ['x', 'y']
    data['label'] = label
    
    palette =sns.color_palette()[:len(np.unique(data['label'].values))]

    # other samples
    sns.scatterplot(data=data[data['label'] == 0], x="x", y="y", 
                    color="grey", size=5, alpha=.4, ax=ax, legend=False)

    # highlighted samples
    sns.scatterplot(
        data=data[data['label'] != 0].sort_values("label"), x="x", y="y", 
        hue = 'label', 
        palette = palette[1:],
        s=150, alpha = 0.8, legend = False,
        ax = ax
    )

    ax.set(xlabel=axis_labels["x"], ylabel=axis_labels["y"])
    sns.despine()