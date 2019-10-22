'''
Plots used repeatedly in different projects.

Plots inspired by the following sources:
    https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0
    https://jovianlin.io/data-visualization-seaborn-part-1/
'''

# TODO: Object facade instead of functions

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pprint


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

def list_as_string(l):
    s = pprint.pformat(l)
    s = s[1:-1].replace("'","")
    return s

def resample_period_as_string(resample_period):
    # See:
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    translations = {
        'D': 'days',
        'W': 'weeks',
        'M': 'months',
    }

    if resample_period in translations:
        return translations[resample_period]
    else:
        return resample_period

def timeseries_rolling_stats(
    data,
    target_variables,
    aggregate_functions = ['max', 'mean', 'min'],
    resample_period = 'D',
    period_lengths = [7, 30, 365],
    subplots=False,
    palette='muted',
    alphas=None,
    min_alpha=0.3,
    linewidths=None,
    min_linewidth=0.7,
    max_linewidth=2,
    auto_area=True,
    title=None
):
    """
    TODO
    data - DataFrame or resample
    """

    BY_UNKNOWN  = -1
    BY_PERIOD   = 0
    BY_FUNCTION = 1
    BY_VARIABLE = 2

    # 1. Prepare derived parameters
    # - linewidths
    # - alphas
    # - colors
    # - resample

    if type(target_variables) is not list:
        target_variables = [target_variables]

    if type(aggregate_functions) is not list:
        aggregate_functions = [aggregate_functions]

    if type(period_lengths) is not list:
        period_lengths = [period_lengths]

    # Plot and line count
    subplots_by = BY_UNKNOWN
    plotcount = 1

    if subplots:
        plotcount = len(target_variables)
        subplots_by = BY_VARIABLE
        if plotcount == 1:
            plotcount = len(aggregate_functions)
            subplots_by = BY_FUNCTION
            if plotcount == 1:
                plotcount = len(period_lengths)
                subplots_by = BY_PERIOD

    lines_per_plot = (len(aggregate_functions) * len(period_lengths) * len(target_variables)) / plotcount

    # linewidths
    linewidth_by = BY_UNKNOWN

    lengths = (len(period_lengths), len(aggregate_functions), len(target_variables))

    if len(period_lengths) > 1:
        linewidth_by = BY_PERIOD
    elif len(aggregate_functions) > 1:
        linewidth_by = BY_FUNCTION
    else:
        linewidth_by = BY_VARIABLE

    if linewidths is None:
        desired_list_len = lengths[linewidth_by]

        if desired_list_len == 1:
            linewidths = [max_linewidth]
        else:
            if lines_per_plot > 1:
                linewidths = np.linspace(min_linewidth, max_linewidth, desired_list_len)
            else:
                linewidths = [max_linewidth] * desired_list_len

    # Aplphas
    alpha_by = BY_UNKNOWN

    if len(period_lengths) > 1:
        alpha_by = BY_PERIOD
    elif len(aggregate_functions) > 1:
        alpha_by = BY_FUNCTION
    else:
        alpha_by = BY_VARIABLE

    if alphas is None:
        desired_list_len = lengths[alpha_by]

        if desired_list_len == 1:
            alphas = [1]
        else:
            if lines_per_plot > 1:
                alphas = np.linspace(min_alpha, 1, desired_list_len)
            else:
                alphas = [1] * desired_list_len

    # Colors
    color_by = BY_UNKNOWN

    desired_list_len = len(target_variables)
    color_by = BY_VARIABLE
    if desired_list_len == 1:
        desired_list_len = len(aggregate_functions)
        color_by = BY_FUNCTION
        if desired_list_len == 1:
            desired_list_len = len(period_lengths)
            color_by = BY_PERIOD

    pal = sns.color_palette(palette, desired_list_len)

    # Resampling
    data_not_resampled = isinstance(data, pd.DataFrame)

    if data_not_resampled:
        resample = {
            target_variable: data[target_variable].resample(resample_period)
            for target_variable in target_variables
        }
    else:
        resample = data

    # Debug messages
    '''
    print('lines_per_plot = (len(aggregate_functions) * len(period_lengths) * len(target_variables)) / plotcount')
    print('{} = ( {} * {} * {}) / {}'.format(lines_per_plot, len(aggregate_functions),len(period_lengths), len(target_variables), plotcount))

    print('BY_UNKNOWN: {}'.format(BY_UNKNOWN))
    print('BY_PERIOD: {}'.format(BY_PERIOD))
    print('BY_FUNCTION: {}'.format(BY_FUNCTION))
    print('BY_VARIABLE: {}'.format(BY_VARIABLE))

    print('target_variables: {}'.format(target_variables))
    print('plotcount: {}'.format(plotcount))
    print('subplots_by: {}'.format(subplots_by))
    print('lines_per_plot: {}'.format(lines_per_plot))
    print('linewidth_by: {}'.format(linewidth_by))
    print('linewidths: {}'.format(linewidths))
    print('alpha_by: {}'.format(alpha_by))
    print('alphas: {}'.format(alphas))
    print('color_by: {}'.format(color_by))
    print('all equal: {}'.format(linewidth_by == alpha_by and alpha_by == color_by))
    #print('pal: {}'.format(pal))
    '''

    ####################
    # 2. Draw the plot #
    ####################

    fig = plt.figure()

    if subplots:
        axes = []
        for i in range(1, plotcount+1):
            if i <= plotcount:
                axes.append(fig.add_subplot(plotcount*100 + 10 + i, xticklabels=[]))
            else:
                axes.append(fig.add_subplot(plotcount*100 + 10 + i))
    else:
        axes = [fig.add_subplot()]

    for j, target_variable in enumerate(target_variables):
        for i, aggregate_function in enumerate(aggregate_functions):
            for k, period_length in enumerate(period_lengths):
                df_rolling = resample[target_variable].agg(aggregate_function).rolling(period_length, center=True, min_periods=1)

                indexes = (k, i, j)

                alpha        = alphas[indexes[alpha_by]]
                linewidth    = linewidths[indexes[linewidth_by]]
                color        = pal[indexes[color_by]]
                plotindex    = indexes[subplots_by] if subplots else 0
                ax           = axes[plotindex]

                label = ''
                if len(target_variables) > 1:
                    label += ' {}'.format(target_variable)
                if len(period_lengths) > 1:
                    label += ' {} {}'.format(period_length, resample_period_as_string(resample_period))
                if len(aggregate_functions) > 1:
                    label += ' {}'.format(aggregate_function)
                label = label.strip()

                if lines_per_plot==1 and auto_area:
                    df_rolling.mean().plot.area(
                        ax=ax, label=label, alpha=alpha, color=color, linewidth=linewidth,
                        xticks=None if plotindex==plotcount-1 else []
                    )
                else:
                    ax.plot(df_rolling.mean(), label=label, alpha=alpha, color=color, linewidth=linewidth)
                ax.legend()
                ax.set_xlabel('')

    ax.set_xlabel('Date')
    if title is None:
        title = 'Rolling {} {} {} of {}.'.format(
            list_as_string(period_lengths), resample_period_as_string(resample_period), list_as_string(aggregate_functions), list_as_string(target_variables)
        )
    fig.suptitle(title, y=0.96)
    result = fig
    fig = axes[0].figure
    # Shared manual pseudo ylabel
    fig.text(0,0.5, list_as_string(target_variables), ha="center", va="center", rotation=90)
    plt.tight_layout()
    # return result
