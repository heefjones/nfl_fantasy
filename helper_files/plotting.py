# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# global random_state
random_state = 9





# function for plotting
def plot_mean_and_counts(df, x, y):

    '''
    Function to plot the number of observations and the mean of y over x.

    Args:
    df: DataFrame
    x: str
    y: str

    Returns:
    None
    
    '''

    # get pos
    pos = df.iloc[0]['Pos']

    # check variable
    data = df.groupby(x)[y].agg(['mean', 'count']).reset_index()

    fig, ax1 = plt.subplots(figsize=(12, 6))
    color = 'tab:red'
    ax1.set_xlabel(x)
    ax1.set_ylabel(f'Mean {y}', color=color)
    ax1.plot(data[x], data['mean'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Count', color=color)
    ax2.bar(data[x], data['count'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Mean {pos} {y} and Count by {x}')
    plt.grid()
    plt.show()





def plot_ranks_line(df, y, legend=True): 

    '''
    Function to plot VORP by a given y variable for each position

    Args:
    df: DataFrame
    y: str

    Returns:
    None
    
    '''

    # get pos
    pos = df.iloc[0]['Pos']

    # iterate through players and plot a line for each 'SeasonPosRank_ppr'
    plt.figure(figsize=(10, 6))
    for rank in sorted(df.SeasonPosRank_ppr.unique()):
        # # break on last iteration
        # if rank == df.SeasonPosRank_ppr.max():
        #     break

        # filter by rank
        temp = df[df.SeasonPosRank_ppr == rank]

        # plot
        plt.plot(temp.Year, temp[y], label=f'{pos}{rank}')

    # add labels
    plt.title(f'{y} by Positional Rank ({pos})')
    plt.xlabel('Year')
    plt.ylabel(f'{y}')
    if legend:
        plt.legend(title='Positional Rank')
    else:
        plt.legend().remove()
    plt.grid()
    plt.show()





def plot_ranks_boxplot(df, y):

    '''
    Function to plot VORP by a given y variable for each position

    Args:
    df: DataFrame
    y: str

    Returns:
    None

    '''

    # get pos
    pos = df.iloc[0]['Pos']

    # get number of replacement-level players
    num_players = df.SeasonPosRank_ppr.max()

    # plot boxplot of y over 'Year
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Year', y=y, data=df, palette='viridis')
    plt.title(f'{y} (Top {num_players} {pos})')
    plt.ylabel(f'{y}')
    plt.xlabel('Year')
    plt.xticks(rotation=90)
    plt.grid()
    plt.show()