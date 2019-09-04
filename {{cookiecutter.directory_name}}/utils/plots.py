'''
Plots used repeatedly in different projects.

Plots inspired by the following sources:
    https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0
    https://jovianlin.io/data-visualization-seaborn-part-1/
'''

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def density_plot(data: pd.DataFrame, inspected_var: str, target_var: str = None, bins: int = 100):
    '''
    Plot displaying density estimation of the selected variable, optionally highlighting separate classes.
    '''
    # TODO: Consider: kws support.

    # Univariate
    if target_var is None:
        sns.distplot(
            data[inspected_var],
            kde_kws={
                'linewidth': 3,
                'shade': True,
                'gridsize': bins
            },
            bins=bins
        )
        plt.title('Density Plot')
        plt.xlabel(inspected_var)
        plt.ylabel('Density')
        return

    # Univariate grouped by class variable
    target_values = data[target_var].unique()
    for val in target_values:
        # Subset to the airline
        subset = data[data[target_var] == val]

        # Draw the density plot
        sns.distplot(
            subset[inspected_var], hist=False, kde=True,
            kde_kws={
                'linewidth': 3,
                'shade': True,
                'gridsize': bins
            },
            label=val,
            #norm_hist=True
        )

    # Plot formatting
    plt.legend(prop={'size': 16}, title=target_var)
    plt.title('{} density grouped by {}'.format(inspected_var, target_var) )
    plt.xlabel(inspected_var)
    plt.ylabel('Density')

def histograms(data: pd.DataFrame):
    '''
    Plot small histograms of all numeric columns.
    '''
    # TODO: bool variables distribution as in:
    #     df['bool_var'].astype('int64').hist(bins=2)
    # TODO: Optional stacked/side by side graph showing proportions of a target variable

    # Select numeric columns (by their dtypes)
    plot_data = data.select_dtypes(['int64', 'float64'])

    fig = plot_data.hist(
        bins=15,
        color='steelblue',
        edgecolor='black', linewidth=1.0,
        xlabelsize=10, ylabelsize=10,
        xrot=45, yrot=0,
        figsize=(10,9),
        grid=False
    )

    plt.tight_layout(rect=(0, 0, 1.5, 1.5))

def scatter_colored(data: pd.DataFrame, var1: str, var2: str, target_var: str, plot_height: int=10, alpha: int=0.3):
    '''
    Scatter of two variables, colored by selected third class variable.
    Displays histogram for both variables.
    
    NOTE: Seems to have trouble with categorical variables :(
    '''
    target_values = data[target_var].unique()

    g = sns.jointplot(var1, var2, data=data, height=plot_height)

    # Clear the axes containing the scatter plot
    g.ax_joint.cla()

    # Generate colors and markers
    n = len(target_values)
    palette = sns.color_palette(n_colors=n)
    markers = ['x','o','v','^','<'] * (1 + int(n / 5))

    # Plot each individual point separately
    handles = []
    for i, n in enumerate(target_values):
        handle = g.ax_joint.scatter(
            data[data[target_var]==n][var1],
            data[data[target_var]==n][var2],
            color=palette[i],
            marker=markers[i],
            label=n,
            alpha=alpha
        )
        handles.append(handle)

    g.set_axis_labels(var1, var2)
    plt.legend(
        handles=handles,
        prop={'size': 16},
        title=target_var,
        loc='upper right',
        bbox_to_anchor=(1, 1.2)
    )