# data science
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# system
import os
import gc

# machine learning
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict, KFold
from sklearn.metrics import r2_score, root_mean_squared_error
from xgboost import XGBRegressor

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# display
pd.set_option('display.max_columns', None)
# pl.Config.set_tbl_rows(n=50)
# pl.Config.set_tbl_cols(-1)
sns.set(style='whitegrid', font='Average')

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# global vars


# numpy seed
SEED = 9
np.random.seed(SEED)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# cleaning.ipynb

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def load_data(file_paths, multi_header=False, pff=False):
    # list to hold dfs
    dfs = []

    # read each file into a dataframe
    for file_path in file_paths:      
        # load the season into a df
        data = pd.read_csv(file_path)

        # if multi_header is True, make the second row the column names and drop the first row
        if multi_header:
            data.columns = data.loc[0]
            data = data.drop(0)
        
        # get year from filename and add as column
        year = file_path[-8:-4]
        data['Year'] = int(year)
        
        # add df to the list
        dfs.append(data)

    # stack dataframes together
    df = pd.concat(dfs, axis=0, ignore_index=True)

    # fantasy seasons
    if not pff:
        # drop rank/point columns (we will be recalculating these)
        df = df.drop(columns=['Rk', 'FantPt', 'PPR', 'DKPt', 'FDPt', 'VBD', 'PosRank', 'OvRank'])

    return df

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def show_shape_and_nulls(df):
    """
    Display the shape of a DataFrame and the number of null values in each column.

    Args:
    - df (pd.DataFrame): The DataFrame to analyze.

    Returns:
    - None
    """

    # print shape
    print(f'Shape: {df.shape}')

    # check for missing values
    print('Null values:')

    # display null values
    display(df.isnull().sum().to_frame().T)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def fill_experience(group):
    # get first experience value for a player
    first_exp = group['Exp'].iloc[0]
    
    # if value is null, set to 0 (rookie season)
    if pd.isna(first_exp):
        first_exp = 0
    
    # define range of years to fill each player's experience column
    experience = range(int(first_exp), int(first_exp) + len(group))
    group['Exp'] = list(experience)
    return group

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def add_rank_cols(df, points_type):
    """
    Extract the points column from df and add total Points, PPG, and PPT rank columns to df.

    Args:
    - df (pd.DataFrame): DataFrame containing player data.
    - points_type (str): The type of point system to use (standard, half-ppr, ppr, 6pt passing TD).

    Returns:
    - pd.DataFrame: DataFrame with added rank columns.
    """

    # some groups we will be using
    year_groups = df.groupby('Year')
    pos_groups = df.groupby(['Year', 'Pos'])
    
    # create mapping for each metric and corresponding grouping
    metrics = ['Points', 'PPG', 'PPT']
    metric_to_group = {}
    for metric in metrics:
        metric_to_group[f'{metric}OvrRank_{points_type}'] = year_groups
        metric_to_group[f'{metric}PosRank_{points_type}'] = pos_groups
    
    # iterate over each rank column, compute the ranking, and assign to df
    for rank_col, group in metric_to_group.items():
        # determine the base metric by removing the rank part from the rank column name
        if 'OvrRank' in rank_col:
            base_metric = rank_col.replace(f'OvrRank_{points_type}', '')
        elif 'PosRank' in rank_col:
            base_metric = rank_col.replace(f'PosRank_{points_type}', '')
        
        # construct the source column name
        source_col = f"{base_metric}_{points_type}"

        # compute rank
        df[rank_col] = group[source_col].transform(lambda x: x.rank(ascending=False, method='min')).astype(int)
    
    return df

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# preds.ipynb

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#