# data science
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# system
import os
import gc
from tqdm import tqdm

# machine learning
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
fantasy_seasons = './data/fantasy_seasons'
passing_data = './data/pff/passing_data'
rushing_data = './data/pff/rushing_data'
receiving_data = './data/pff/receiving_data'
blocking_data = './data/pff/blocking_data'
team_data = './data/pff/team_data'

# numpy seed
SEED = 9
np.random.seed(SEED)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# eda.ipynb

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

def add_point_cols(df, points):
    """
    Add point columns to the dataframe.

    Args:
    - df (pd.DataFrame): The dataframe to add point columns to.
    - points (str): The type of point system to use (standard, half-ppr, ppr, 6pt passing TD).

    Returns:
    - df (pd.DataFrame): The dataframe with point column added.
    """

    # error handling
    if points not in ['standard', 'half-ppr', 'ppr', '6']:
        raise ValueError("Invalid points type. Choose from 'standard', 'half-ppr', 'ppr', or '6'.")

    # calculate standard points without passing TDs
    standard_points = (df['Pass_Yds'] * 0.04) + (df['Pass_Int'] * -1) + (df['Rush_Yds'] * 0.1) + \
        (df['Rush_TD'] * 6) + (df['Rec_Yds'] * 0.1) + (df['Rec_TD'] * 6) + (df['FmbLost'] * -2)

    # standard points
    if points == 'standard':
        df['Points_standard'] = standard_points + (df['Pass_TD'] * 4)
        
    # half-ppr    
    elif points == 'half-ppr':
        df['Points_half-ppr'] = standard_points + (df['Pass_TD'] * 4) + (df['Rec_Rec'] * 0.5)

    # ppr    
    elif points == 'ppr':
        df['Points_ppr'] = standard_points + (df['Pass_TD'] * 4) + (df['Rec_Rec'] * 1)

    # PPR scoring with 6pt passing TDs
    elif points == '6':
        df['Points_6'] = standard_points + (df['Pass_TD'] * 6) + (df['Rec_Rec'] * 1)

    # point-per-game column
    df['PPG_' + points] = (df['Points_' + points] / df['G']).fillna(0)

    # point-per-touch column
    df['PPT_' + points] = (df['Points_' + points] / df['Touches']).fillna(0)

    return df

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def add_rank_cols(df):
    """
    Extract the points column from df and add total Points, PPG, and PPT rank columns to df.

    Args:
    - df (pd.DataFrame): DataFrame containing player data.

    Returns:
    - pd.DataFrame: DataFrame with added rank columns.
    """

    # some groups we will be using
    year_groups = df.groupby('Year')
    pos_groups = df.groupby(['Year', 'Pos'])

    # extract the suffix from the points column (e.g., "_half-ppr")
    points_col = next(col for col in df.columns if col.startswith('Points_'))
    points_type = points_col.replace('Points', '')
    
    # create mapping for each metric and corresponding grouping
    metrics = ['Points', 'PPG', 'PPT']
    metric_to_group = {}
    for metric in metrics:
        metric_to_group[f'{metric}OvrRank{points_type}'] = year_groups
        metric_to_group[f'{metric}PosRank{points_type}'] = pos_groups
    
    # iterate over each rank column, compute the ranking, and assign to df
    for rank_col, group in metric_to_group.items():
        # determine the base metric by removing the rank part from the rank column name
        if 'OvrRank' in rank_col:
            base_metric = rank_col.replace(f'OvrRank{points_type}', '')
        elif 'PosRank' in rank_col:
            base_metric = rank_col.replace(f'PosRank{points_type}', '')
        
        # construct the source column name
        source_col = f"{base_metric}{points_type}"

        # compute rank
        df[rank_col] = group[source_col].transform(lambda x: x.rank(ascending=False, method='min')).astype(int)
    
    return df

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def add_vorp_cols(df, num_teams=12, wr3=True):
    """
    Add VORP columns to the dataframe using the default Underdog fantasy lineup (12 teams, 3 WRs).

    Args:
    - df (pd.DataFrame): The dataframe containing player data.
    - num_teams (int): The number of teams in the league.
    - wr3 (bool): Whether to include a 3rd WR in the lineup.

    Returns:
    - df (pd.DataFrame): The dataframe with VORP columns added.
    """

    # define the replacement rank based on num teams and 3WR format
    replacement_ranks = {'QB': num_teams, 'RB': int(num_teams * 2.5), 'WR': int(num_teams * 2.5 + True), 'TE': num_teams}

    # extract the suffix from the points column (e.g., "_half-ppr")
    points_col = next(col for col in df.columns if col.startswith('Points_'))
    points_type = points_col.replace('Points', '')

    # iterate through the position groups
    for (year, pos), group in df.groupby(['Year', 'Pos']):

        # iterate for both season total and PPG VORP
        for rank_type in ['Points', 'PPG']:

            # get the replacement rank for the current position, subtract 1 to get the index
            rank = int(replacement_ranks[pos] - 1)

            # sort group
            group = group.sort_values(rank_type + points_type, ascending=False)

            # get replacement player points for the current position and scoring type
            replacement = group.iloc[rank][rank_type + points_type]

            # add VORP column
            df.loc[(df['Year'] == year) & (df['Pos'] == pos), rank_type + '_' + 'VORP' + points_type] = \
            df.loc[(df['Year'] == year) & (df['Pos'] == pos), rank_type + points_type] - replacement

    return df

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def add_target_cols(df):
    """
    Add Total and PPG target columns to the dataframe.

    Args:
    - df (pd.DataFrame): The dataframe containing player data.

    Returns:
    - df (pd.DataFrame): The dataframe with target columns added.
    """

    # extract the suffix from the points column (e.g., "_half-ppr")
    points_col = next(col for col in df.columns if col.startswith('Points_'))
    points_type = points_col.replace('Points', '')

    # group by each player and shift the points column by 1
    df['PointsTarget' + points_type] = df.groupby('Key')['Points' + points_type].shift(-1)
    df['PPGTarget' + points_type] = df.groupby('Key')['PPG' + points_type].shift(-1)

    return df

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def add_percent_columns(df, formulas):
    # iterate through formulas
    for new_col, (numerator, denominator) in formulas.items():
        # normalize
        df[new_col] = df[numerator] / df[denominator]
        
    return df

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def prefix_df(df, prefix):
    # rename all columns with prefix
    df.columns = [f'{prefix}_{col}' for col in df.columns]

    # restore Player and Year columns
    df['Player'] = df[f'{prefix}_player']
    df['Year'] = df[f'{prefix}_Year']

    # drop prefixed Player and Year
    return df.drop([f'{prefix}_player', f'{prefix}_Year'], axis=1)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def clean_pff_data(pff_data, prefix):
    # passing data
    if prefix == 'Pass':
        formulas = {'Dropback%': ('dropbacks', 'passing_snaps'), 'Aimed_passes%': ('aimed_passes', 'attempts'), 
                        'Dropped_passes%': ('drops', 'aimed_passes'), 'Batted_passes%': ('bats', 'aimed_passes'), 
                        'Thrown_away%': ('thrown_aways', 'passing_snaps'), 'Pressure%': ('def_gen_pressures', 'passing_snaps'), 
                        'Scramble%': ('scrambles', 'passing_snaps'), 'Sack%': ('sacks', 'passing_snaps'), 
                        'Pressure_to_sack%': ('sacks', 'def_gen_pressures'), 'BTT%': ('big_time_throws', 'aimed_passes'), 
                        'TWP%': ('turnover_worthy_plays', 'aimed_passes'), 'First_down%': ('first_downs', 'attempts')}
        cols_to_drop = ['player_id', 'position', 'team_name', 'player_game_count', 'aimed_passes', 'attempts', 'bats', 'big_time_throws', 'btt_rate', 'completion_percent', 
             'completions', 'declined_penalties', 'def_gen_pressures', 'drop_rate', 'drops', 'grades_offense', 'grades_run', 'grades_hands_fumble', 'franchise_id', 
             'hit_as_threw', 'interceptions', 'penalties', 'pressure_to_sack_rate', 'qb_rating', 'sack_percent', 'sacks', 'scrambles', 'spikes', 'thrown_aways', 
             'touchdowns', 'turnover_worthy_plays', 'twp_rate', 'yards', 'ypa', 'first_downs']
    
    # rushing data
    elif prefix == 'Rush':
        formulas = {'Team_Rush%': ('attempts', 'run_plays'), 'Avoided_tackles_per_attempt': ('avoided_tackles', 'attempts'), 
                 '10+_yard_run%': ('explosive', 'attempts'), '15+_yard_run%': ('breakaway_attempts', 'attempts'), 
                 '15+_yard_run_yards%': ('breakaway_yards', 'yards'), 'First_down%': ('first_downs', 'attempts'), 
                 'Gap%': ('gap_attempts', 'attempts'), 'Zone%': ('zone_attempts', 'attempts'), 
                 'YCO_per_attempt': ('yards_after_contact', 'attempts')}
        cols_to_drop = ['player_id', 'position', 'team_name', 'player_game_count', 'attempts', 'avoided_tackles', 'breakaway_attempts', 'breakaway_percent', 'breakaway_yards', 
             'declined_penalties', 'designed_yards', 'drops', 'elu_recv_mtf', 'elu_rush_mtf', 'elu_yco', 'explosive', 'first_downs', 'franchise_id', 'fumbles', 
             'gap_attempts', 'grades_offense_penalty', 'grades_pass', 'grades_pass_block', 'grades_pass_route', 'grades_run_block', 'penalties', 'rec_yards', 'receptions', 
             'routes', 'scramble_yards', 'scrambles', 'targets', 'total_touches', 'touchdowns', 'yards', 'yards_after_contact', 'yco_attempt', 'ypa', 'yprr', 'run_plays', 'zone_attempts']
    
    # receiving data
    elif prefix == 'Rec':
        formulas = {'Avoided_tackles_per_reception': ('avoided_tackles', 'receptions'), 'First_down%': ('first_downs', 'receptions'), 
                              'Int_per_target': ('interceptions', 'targets'), 'YAC%': ('yards_after_catch', 'yards')}
        cols_to_drop = ['player_id', 'position', 'team_name', 'player_game_count', 'avoided_tackles', 'contested_receptions', 'contested_targets', 'declined_penalties', 'drops', 
                        'first_downs', 'franchise_id', 'fumbles', 'grades_pass_block', 'inline_snaps', 'pass_blocks', 'pass_plays', 'penalties', 'receptions', 'routes', 'slot_snaps', 
                        'targets', 'touchdowns', 'wide_snaps', 'yards', 'yards_after_catch']

    # normalize
    pff_data = add_percent_columns(pff_data, formulas)

    # drop columns
    pff_data = pff_data.drop(columns=cols_to_drop)

    # add prefix
    pff_data = prefix_df(pff_data, prefix)

    return pff_data

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# preds.ipynb

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#