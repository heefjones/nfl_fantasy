# data science
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# system
import os
import gc

# machine learning
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
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

# set numpy seed
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

def add_point_cols(df):
    # calculate standard points
    df['Points_standard'] = (df['Pass_Yds'] * 0.04) + (df['Pass_TD'] * 4) + (df['Pass_Int'] * -1) + \
        (df['Rush_Yds'] * 0.1) + (df['Rush_TD'] * 6) + \
        (df['Rec_Yds'] * 0.1) + (df['Rec_TD'] * 6) + \
        (df['FmbLost'] * -2)

    # calculate half-ppr points
    df['Points_half-ppr'] = df['Points_standard'] + (df['Rec_Rec'] * 0.5)

    # calculate ppr points
    df['Points_ppr'] = df['Points_standard'] + (df['Rec_Rec'] * 1)

    # PPR scoring with 6pt passing TDs
    df['Points_6'] = (df['Pass_Yds'] * 0.04) + (df['Pass_TD'] * 6) + (df['Pass_Int'] * -1) + \
        (df['Rush_Yds'] * 0.1) + (df['Rush_TD'] * 6) + \
        (df['Rec_Rec'] * 1) + (df['Rec_Yds'] * 0.1) + (df['Rec_TD'] * 6) + \
        (df['FmbLost'] * -2)

    # list for scoring
    scoring = ['standard', 'half-ppr', 'ppr', '6']

    # add point-per-game columns
    for scoring_type in scoring:
        df['PPG_' + scoring_type] = (df['Points_' + scoring_type] / df['G']).fillna(0)

    # add point-per-touch columns
    for scoring_type in scoring:
        df['PPT_' + scoring_type] = (df['Points_' + scoring_type] / df['Touches']).fillna(0)

    return df

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def add_rank_cols(df):
    # some groups we will be using
    year_groups = df.groupby('Year')
    pos_groups = df.groupby(['Year', 'Pos'])

    # define metric types and corresponding groups
    metric_to_group = {'SeasonOvrRank': (year_groups, 'Points_{}'), 'SeasonPosRank': (pos_groups, 'Points_{}'), 
                    'PPGOvrRank':    (year_groups, 'PPG_{}'), 'PPGPosRank':    (pos_groups, 'PPG_{}'), 
                    'PPTOvrRank':    (year_groups, 'PPT_{}'), 'PPTPosRank':    (pos_groups, 'PPT_{}')}
    
    # list for scoring
    scoring = ['standard', 'half-ppr', 'ppr', '6']

    # calculate all ranks
    for metric, (group, col_template) in metric_to_group.items():
        for k in scoring:
            col_name = col_template.format(k)
            rank_col = f'{metric}_{k}'
            df[rank_col] = group[col_name].transform(lambda x: x.rank(ascending=False, method='min')).astype(int)

    return df

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def add_vorp_cols(df):
    # define the replacement rank based on league
    replacement_ranks_10 = {'QB': 10, 'RB': 25, 'WR': 25, 'TE': 10}
    replacement_ranks_12 = {'QB': 12, 'RB': 30, 'WR': 30, 'TE': 12}
    replacement_ranks_10_3WR = {'QB': 10, 'RB': 25, 'WR': 35, 'TE': 10}
    replacement_ranks_12_3WR = {'QB': 12, 'RB': 30, 'WR': 42,'TE': 12}

    # list for scoring
    scoring = ['standard', 'half-ppr', 'ppr', '6']

    # iterate through the position groups
    for (year, pos), group in df.groupby(['Year', 'Pos']):

        # iterate through the replacement ranks
        for replacement_ranks, col_name in [(replacement_ranks_10, '10tm'), (replacement_ranks_12, '12tm'), (replacement_ranks_10_3WR, '10tm_3WR'), (replacement_ranks_12_3WR, '12tm_3WR')]:

            # iterate for both seasonal and PPG VORP
            for rank_type in ['Points', 'PPG']:

                # get the replacement rank for the current position, subtract 1 to get the index
                rank = int(replacement_ranks[pos] - 1)

                # iterate through the scoring types
                for scoring_type in scoring:

                    # sort group
                    group = group.sort_values(rank_type + '_' + scoring_type, ascending=False)

                    # get replacement player points for the current position and scoring type
                    replacement = group.iloc[rank][rank_type + '_' + scoring_type]

                    # add VORP column
                    df.loc[(df['Year'] == year) & (df['Pos'] == pos), rank_type + '_' + 'VORP_' + scoring_type + '_' + col_name] = \
                    df.loc[(df['Year'] == year) & (df['Pos'] == pos), rank_type + '_' + scoring_type] - replacement

    return df

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def add_target_cols(df):

    # list for scoring
    scoring = ['standard', 'half-ppr', 'ppr', '6']
    
    # add seasonal target cols
    for scoring_type in scoring:
        # group by each player and shift the points column by 1
        df['SeasonTarget_' + scoring_type] = df.groupby('Key')['Points_' + scoring_type].shift(-1)

    # add ppg target cols
    for scoring_type in scoring:
        # group by each player and shift the points column by 1
        df['PPGTarget_' + scoring_type] = df.groupby('Key')['PPG_' + scoring_type].shift(-1)

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

def create_features(df):
    """
    Create features for each player.

    Args:
    - df (pd.dataframe): Contains the 

    Returns:
    - (pl.dataframe): Dataframe with new features added.
    """

    # convert to polars dataframe
    df = pl.from_pandas(df)

    # sort by player key and year
    df = df.sort(["Key", "Year"])

    # convert to lazy frame for better performance
    lazy_df = df.lazy()

    # define the rolling aggregation functions
    agg_funcs = {"mean": lambda col, w: pl.col(col).rolling_mean(w, min_samples=1, center=True), 
                 "std":  lambda col, w: pl.col(col).rolling_std(w, min_samples=1, center=True).fill_nan(0)}

    # list of windows
    windows = []

    # loop over each window duration, creating rolling features
    for m in windows:
        # create a list to hold the expressions for this window size
        exprs = []

        # create rolling features for 
        for col in ['anglez', 'enmo']:
            for stat, func in agg_funcs.items():
                alias_name = f'{col}_{m}m_{stat}'
                exprs.append(func(col, m).alias(alias_name).cast(pl.Int16))
            
            # difference features
            diff_col = f'{col}_diff'
            for stat, func in agg_funcs.items():
                alias_name = f'{diff_col}_{m}m_{stat}'
                exprs.append(func(diff_col, m).alias(alias_name).cast(pl.UInt16))
        
        # add the rolling features
        lazy_df = lazy_df.with_columns(exprs)

    # collect the results back into a standard df
    return lazy_df.collect()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#