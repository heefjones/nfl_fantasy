# data science
import pandas as pd
import polars as pl
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

# system
import os
import gc

# machine learning
from tqdm import tqdm
from sklearn.model_selection import cross_validate, KFold, cross_val_score
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
sns.set(style='whitegrid', font='Average')

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# global vars
TEAM_COLORS = {'GNB': '#203731', 'CHI': '#0B162A', 'DET': '#0076B6', 'MIN': '#4F2683', 
    'NOR': '#D3BC8D', 'ATL': '#A71930', 'CAR': '#0085CA', 'TAM': '#D50A0A', 
    'LAR': '#003594', 'SEA': '#002244', 'ARI': '#97233F', 'SFO': '#AA0000', 
    'DAL': '#041E42', 'NYG': '#0B2265', 'PHI': '#004C54', 'WAS': '#773141', 
    'LVR': '#000000', 'LAC': '#0080C6', 'KAN': '#E31837', 'DEN': '#FB4F14', 
    'BUF': '#00338D', 'NWE': '#002244', 'MIA': '#008E97', 'NYJ': '#125740', 
    'CIN': '#FB4F14', 'PIT': '#FFB612', 'BAL': '#241773', 'CLE': '#311D00',
    'TEN': '#0C2340', 'JAX': '#006778', 'HOU': '#002244', 'IND': '#003594'}

# numpy seed
SEED = 9
np.random.seed(SEED)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# eda.ipynb

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def load_data(path, pff=False):
    """
    Combine CSV files from a data directory into a single DataFrame.

    Args:
    - path (str): The path to the directory containing the CSV files.
    - pff (bool): If True, load PFF data. If False, load non-PFF data.

    Returns:
    - df (pd.DataFrame): A DataFrame containing the combined data from all CSV files.
    """

    # format path
    path = f'./data/{path}'

    # get all csv files in the 'pass_data' directory
    file_paths = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.csv')]

    # list to hold dfs
    dfs = []

    # read each file into a dataframe
    for file_path in file_paths:      
        # load the season into a df
        data = pd.read_csv(file_path)

        # for non-pff data, handle multi-header
        if not pff:
            data.columns = data.loc[0]
            data = data.drop(0)
        
        # get year from filename and add as column
        year = file_path[-8:-4]
        data['Year'] = int(year)
        
        # add df to the list
        dfs.append(data)

    # stack dataframes together
    df = pd.concat(dfs, axis=0, ignore_index=True)

    # non-pff data
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
    # add 'Key' back manually
    group['Key'] = group.name

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

def add_point_cols(df, points_type):
    """
    Add point columns to the dataframe.

    Args:
    - df (pd.DataFrame): The dataframe to add point columns to.
    - points_type (str): The type of point system to use (standard, half-ppr, ppr, 6pt passing TD).

    Returns:
    - df (pd.DataFrame): The dataframe with point column added.
    """

    # error handling
    if points_type not in ['standard', 'half-ppr', 'ppr', '6']:
        raise ValueError("Invalid points_type type. Choose from 'standard', 'half-ppr', 'ppr', or '6'.")

    # calculate standard points_type (excluding passing TDs)
    standard_points = (df['Pass_Yds'] * 0.04) + (df['Pass_Int'] * -1) + (df['Rush_Yds'] * 0.1) + \
        (df['Rush_TD'] * 6) + (df['Rec_Yds'] * 0.1) + (df['Rec_TD'] * 6) + (df['FmbLost'] * -2)

    # standard points
    if points_type == 'standard':
        df['Points_standard'] = standard_points + (df['Pass_TD'] * 4)
        
    # half-ppr    
    elif points_type == 'half-ppr':
        df['Points_half-ppr'] = standard_points + (df['Pass_TD'] * 4) + (df['Rec_Rec'] * 0.5)

    # ppr    
    elif points_type == 'ppr':
        df['Points_ppr'] = standard_points + (df['Pass_TD'] * 4) + (df['Rec_Rec'] * 1)

    # PPR scoring with 6pt passing TDs
    elif points_type == '6':
        df['Points_6'] = standard_points + (df['Pass_TD'] * 6) + (df['Rec_Rec'] * 1)

    # point-per-game column
    df['PPG_' + points_type] = (df['Points_' + points_type] / df['G']).fillna(0)

    # point-per-touch column
    df['PPT_' + points_type] = (df['Points_' + points_type] / df['Touches']).fillna(0)

    return df

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

def add_vorp_cols(df, points_type, num_teams=12, wr3=True):
    """
    Add VORP columns to the dataframe using the default Underdog fantasy lineup (12 teams, 3 WRs).

    Args:
    - df (pd.DataFrame): The dataframe containing player data.
    - points_type (str): The type of point system to use (standard, half-ppr, ppr, 6pt passing TD).
    - num_teams (int): The number of teams in the league.
    - wr3 (bool): Whether to include a 3rd WR in the lineup. If false, a 2WR format is implied.

    Returns:
    - df (pd.DataFrame): The dataframe with VORP columns added.
    """

    # define the replacement rank based on num teams and 3WR format
    replacement_ranks = {'QB': num_teams, 'RB': int(num_teams * 2.5), 'WR': int(num_teams * 2.5 + True), 'TE': num_teams}

    # iterate through the position groups
    for (year, pos), group in df.groupby(['Year', 'Pos']):

        # iterate for both season total and PPG VORP
        for rank_type in ['Points', 'PPG']:

            # get the replacement rank for the current position, subtract 1 to get the index
            rank = int(replacement_ranks[pos] - 1)

            # format col name
            col_name = rank_type + '_' + points_type

            # sort group
            group = group.sort_values(col_name, ascending=False)

            # get replacement player points for the current position and scoring type
            replacement = group.iloc[rank][col_name]

            # add VORP column
            df.loc[(df['Year'] == year) & (df['Pos'] == pos), rank_type + '_VORP_' + points_type] = \
            df.loc[(df['Year'] == year) & (df['Pos'] == pos), col_name] - replacement

    return df

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def add_target_cols(df, points_type):
    """
    Add Total and PPG target columns to the dataframe.

    Args:
    - df (pd.DataFrame): The dataframe containing player data.
    - points_type (str): The type of point system to use (standard, half-ppr, ppr, 6pt passing TD).

    Returns:
    - df (pd.DataFrame): The dataframe with target columns added.
    """

    # group by each player and shift the points column by 1
    df['PointsTarget_' + points_type] = df.groupby('Key')['Points_' + points_type].shift(-1)
    df['PPGTarget_' + points_type] = df.groupby('Key')['PPG_' + points_type].shift(-1)

    return df

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def add_percent_columns(df, formulas):
    """
    Add percentage columns to the DataFrame based on given formulas.

    Args:
    - df (pd.DataFrame): The DataFrame to add percentage columns to.
    - formulas (dict): A dictionary where keys are new column names and values are tuples of (numerator, denominator).
    
    Returns:
    - df (pd.DataFrame): The DataFrame with added percentage columns.
    """

    # iterate through formulas
    for new_col, (numerator, denominator) in formulas.items():
        # normalize
        df[new_col] = df[numerator] / df[denominator]
        
    return df

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def prefix_df(df, prefix):
    """
    Prefix all columns in the DataFrame with a given prefix.

    Args:
    - df (pd.DataFrame): The DataFrame to prefix.
    - prefix (str): The prefix to add to the columns.

    Returns:
    - df (pd.DataFrame): The DataFrame with prefixed columns.
    """

    # rename all columns with prefix
    df.columns = [f'{prefix}_{col}' for col in df.columns]

    # restore Player and Year columns
    df['Player'] = df[f'{prefix}_player']
    df['Year'] = df[f'{prefix}_Year']

    # drop prefixed Player and Year
    return df.drop([f'{prefix}_player', f'{prefix}_Year'], axis=1)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def clean_pff_player_data(pff_data, prefix):
    """
    Clean PFF player data by normalizing certain columns and dropping unnecessary ones.

    Args:
    - pff_data (pd.DataFrame): The PFF data to clean.
    - prefix (str): The prefix to add to the columns.

    Returns:
    - pff_data (pd.DataFrame): The cleaned PFF data.
    """

    # passing data
    if prefix == 'Pass':
        formulas = {'Dropback%': ('dropbacks', 'passing_snaps'), 
                    'Aimed_passes%': ('aimed_passes', 'attempts'), 
                    'Dropped_passes%': ('drops', 'aimed_passes'), 
                    'Batted_passes%': ('bats', 'aimed_passes'), 
                    'Thrown_away%': ('thrown_aways', 'passing_snaps'), 
                    'Pressure%': ('def_gen_pressures', 'passing_snaps'), 
                    'Scramble%': ('scrambles', 'passing_snaps'), 
                    'Sack%': ('sacks', 'passing_snaps'), 
                    'Pressure_to_sack%': ('sacks', 'def_gen_pressures'), 
                    'BTT%': ('big_time_throws', 'aimed_passes'), 
                    'TWP%': ('turnover_worthy_plays', 'aimed_passes'), 
                    'First_down%': ('first_downs', 'attempts')}
        cols_to_drop = ['player_id', 'position', 'team_name', 'player_game_count', 'bats', 'big_time_throws', 'btt_rate', 'completion_percent', 
             'completions', 'declined_penalties', 'drop_rate', 'drops', 'grades_run', 'franchise_id', 
             'interceptions', 'penalties', 'pressure_to_sack_rate', 'qb_rating', 'sack_percent', 'scrambles', 'spikes', 'thrown_aways', 
             'touchdowns', 'turnover_worthy_plays', 'twp_rate', 'yards', 'ypa', 'first_downs', 'attempts']
    
    # rushing data
    elif prefix == 'Rush':
        formulas = {'Team_Rush%': ('attempts', 'run_plays'), 
                    'Avoided_tackles_per_attempt': ('avoided_tackles', 'attempts'), 
                    '10+_yard_run%': ('explosive', 'attempts'), 
                    '15+_yard_run%': ('breakaway_attempts', 'attempts'), 
                    '15+_yard_run_yards%': ('breakaway_yards', 'yards'), 
                    'First_down%': ('first_downs', 'attempts'), 
                    'Gap%': ('gap_attempts', 'attempts'), 
                    'Zone%': ('zone_attempts', 'attempts'), 
                    'YCO_per_attempt': ('yards_after_contact', 'attempts')}
        cols_to_drop = ['player_id', 'position', 'team_name', 'player_game_count', 'breakaway_attempts', 'breakaway_percent', 'breakaway_yards', 
             'declined_penalties', 'designed_yards', 'drops', 'elu_recv_mtf', 'elu_rush_mtf', 'elu_yco', 'first_downs', 'franchise_id', 'fumbles', 
             'grades_offense_penalty', 'grades_pass', 'grades_pass_block', 'grades_pass_route', 'grades_run_block', 'penalties', 'rec_yards', 'receptions', 
             'routes', 'scramble_yards', 'scrambles', 'targets', 'total_touches', 'touchdowns', 'yards', 'yards_after_contact', 'yco_attempt', 'ypa', 'yprr', 'attempts', 'run_plays']
    
    # receiving data
    elif prefix == 'Rec':
        formulas = {'Avoided_tackles_per_reception': ('avoided_tackles', 'receptions'), 
                    'First_down%': ('first_downs', 'receptions'), 
                    'Int_per_target': ('interceptions', 'targets'), 
                    'YAC%': ('yards_after_catch', 'yards'), 
                    'Contested_catch_rate': ('contested_receptions', 'contested_targets')}
        cols_to_drop = ['player_id', 'position', 'team_name', 'player_game_count', 'declined_penalties', 'drops', 'first_downs', 'franchise_id', 'fumbles', 
                        'grades_pass_block', 'pass_blocks', 'pass_plays', 'penalties', 'receptions', 'targets', 'touchdowns', 'yards', 'yards_after_catch', 'interceptions']

    # normalize
    pff_data = add_percent_columns(pff_data, formulas)

    # drop columns
    pff_data = pff_data.drop(columns=cols_to_drop)

    # add prefix
    pff_data = prefix_df(pff_data, prefix)

    return pff_data.fillna(0)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def clean_pff_team_data(pff_data):
    """
    Clean PFF team data by normalizing certain columns and dropping unnecessary ones.

    Args:
    - pff_data (pd.DataFrame): The PFF data to clean.

    Returns:
    - pff_data (pd.DataFrame): The cleaned PFF data.
    """

    # add 'Win%' col
    pff_data['Win%'] = pff_data['Wins'] / (pff_data['Wins'] + pff_data['Losses'])

    # calculate PPG and PPG allowed
    pff_data['PPG'] = pff_data['Points For'] / (pff_data['Wins'] + pff_data['Losses'])
    pff_data['PPG_allowed'] = pff_data['Points Against'] / (pff_data['Wins'] + pff_data['Losses'])

    # create 'Pass Defense Grade' col as average of 'Pass Rush' and 'Pass Coverage' grades
    pff_data['Pass Defense Grade'] = ((pff_data['Pass Rush Grade'] + pff_data['Coverage Grade'])) / 2

    # add 'Team' to the beginning of each column name
    pff_data.columns = ['Team_' + col for col in pff_data.columns]

    # remove 'Team' from 'Tm' and 'Year'
    pff_data['Tm'] = pff_data['Team_Tm']
    pff_data['Year'] = pff_data['Team_Year']

    # drop columns
    return pff_data.drop(columns=['Team_Tm', 'Team_Year', 'Team_Points For', 'Team_Points Against', 'Team_Wins', 'Team_Losses']).fillna(0)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def consolidate_pos_columns(df, col_name):
    """
    Consolidate duplicate columns like 'Pass_X', 'Rush_X', 'Rec_X' into a single column
    based on player position.

    Args:
    - df (pd.DataFrame): your dataframe
    - col_name (str): e.g. 'grades_offense' or 'grades_hands_fumble'

    Returns:
    - df (pd.DataFrame): dataframe with the new consolidated column and dropped duplicates
    """

    # get column names
    pass_col, rush_col, rec_col = f'Pass_{col_name}', f'Rush_{col_name}', f'Rec_{col_name}'

    # map positional conditions to the correct column 
    conditions = [df['Pos'] == 'QB', df['Pos'] == 'RB', df['Pos'].isin(['WR', 'TE'])]
    choices = [df[pass_col], df[rush_col], df[rec_col]]
    df[col_name] = np.select(conditions, choices, default=np.nan)

    # drop the original columns
    return df.drop([pass_col, rush_col, rec_col], axis=1)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ppg.ipynb

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def drop_total_volume_cols(df):
    """
    Drop total-season volume columns from the DataFrame.

    Args:
    - df (pd.DataFrame): The DataFrame to drop columns from.

    Returns:
    - df (pd.DataFrame): The DataFrame with dropped columns.
    """

    # drop non-normalized columns and a few redundant normalized columns
    dropped_cols = ['G', 'GS', 'ProBowl', 'AllPro', 'Pass_Cmp', 'Pass_Att', 'Pass_Yds', 'Pass_TD', 'Pass_Int', 'Rush_Att', 'Rush_Yds', 'Rush_TD', 'Pass_Cmp%', 'Rec_Catch%', 'num_games', 'Touches', 
                'Rec_Tgt', 'Rec_Rec', 'Rec_Yds', 'Rec_TD', 'Fmb', 'FmbLost', 'Scrim_TD', 'Scrim_Yds', 'Rush_Y/A', 'Rec_Y/R', 'Pass_Y/A', 
                'Points_half-ppr', 'PointsOvrRank_half-ppr', 'PointsPosRank_half-ppr', 'Points_VORP_half-ppr', 'PointsTarget_half-ppr', 'PPG_VORP_half-ppr']
    return df.drop(columns=dropped_cols)


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def create_features(df):
    """
    Create features for each player.

    Args:
    - df (pd.dataframe): Player data.

    Returns:
    - (pl.dataframe): Dataframe with new features added.
    """

    # convert to polars dataframe and sort
    df = pl.from_pandas(df).sort(["Key", "Year"])

    # convert to a lazy frame for efficiency
    lazy_df = df.lazy()

    # define cols to aggregate
    non_agg_cols = ['Player', 'Tm', 'Pos', 'Key', 'Year', 'Age', 'Exp', 'target']
    agg_cols = [col for col in df.columns if col not in non_agg_cols]

    # list of expressions for original columns
    base_exprs = [pl.col('*')]

    # expressions that rely on prior aliases
    post_exprs = []

    # iterate through each column to be aggregated
    for col in agg_cols:
        # rolling stats (n years)
        for n in [3]:
            base_exprs.extend([
                pl.col(col)
                .rolling_mean(window_size=n, min_samples=1)
                .over('Key')
                .alias(f'{col}_{n}y_mean'),
                pl.col(col)
                .rolling_std(window_size=n, min_samples=1)
                .over('Key')
                .alias(f'{col}_{n}y_std')])

        # cumulative career mean & std
        cum_sum = pl.col(col).cum_sum().over('Key')
        cum_count = (pl.col('Exp') + 1)
        cum_mean = (cum_sum / cum_count).alias(f'{col}_career_mean')
        sum_sq = pl.col(col).pow(2).cum_sum().over('Key')
        cum_var = ((sum_sq - cum_sum.pow(2) / cum_count) / cum_count)
        cum_std = cum_var.sqrt().alias(f'{col}_career_std')
        base_exprs.extend([cum_mean, cum_std])

        # trend slope relative to career (expanding linear regression)
        sum_year = pl.col('Year').cum_sum().over('Key')
        sum_year_sq = pl.col('Year').pow(2).cum_sum().over('Key')
        sum_y = cum_sum
        sum_xy = (pl.col(col) * pl.col('Year')).cum_sum().over('Key')
        x_mean = (sum_year / cum_count)
        y_mean = (sum_y / cum_count)
        cov = (sum_xy / cum_count) - (x_mean * y_mean)
        var_x = (sum_year_sq / cum_count) - x_mean.pow(2)
        slope = (cov / var_x).alias(f'{col}_career_trend_slope')
        base_exprs.append(slope)

        # momentum: difference between recent (3y) mean and career mean
        post_exprs.append((pl.col(f'{col}_3y_mean') - pl.col(f'{col}_career_mean')).alias(f'{col}_momentum'))

    # add the new columns to df
    lazy_df = lazy_df.with_columns(base_exprs)
    lazy_df = lazy_df.with_columns(post_exprs)

    # collect results back into a pandas df
    df_pandas = lazy_df.collect().to_pandas()

    # fill nulls and infs with 0
    non_target_cols = [col for col in df_pandas.columns if col != 'target']
    df_pandas[non_target_cols] = df_pandas[non_target_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # sort columns 
    return df_pandas[sorted(df_pandas.columns)]

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def get_pos_subsets(features):
    """
    Create subsets of the features DataFrame for each position group (QB, RB, WR/TE).

    Args:
    - features (pd.DataFrame): The DataFrame containing player features.

    Returns:
    - qb (pd.DataFrame): Subset of features for quarterbacks.
    - rb (pd.DataFrame): Subset of features for running backs.
    - wr_te (pd.DataFrame): Subset of features for wide receivers and tight ends.
    """

    # create the 4 positional subsets
    qb = features.query('Pos == "QB"')
    rb = features.query('Pos == "RB"')
    wr_te = features.query('Pos == "WR" or Pos == "TE"')

    # drop 'Rec' cols for QBs
    rec_cols = [col for col in features.columns if col.startswith('Rec_')]
    qb = qb.drop(columns=rec_cols)

    # drop 'Pass' cols for RBs and WRs/TEs
    pass_cols = [col for col in features.columns if col.startswith('Pass_')]
    rb = rb.drop(columns=pass_cols)
    wr_te = wr_te.drop(columns=pass_cols)

    return qb, rb, wr_te

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def cross_val(df, estimator, folds=5):
    """
    Perform KFold cross validation on a given estimator and store evaluation metrics.
    
    Args:
    - df (pd.DataFrame): Data to model.
    - estimator (sklearn estimator): Estimator to use for modeling.
    - folds (int): Number of cross-validation folds to use. Default is 5.
    
    Returns:
    - pd.DataFrame: DataFrame containing the mean and standard deviation of RMSE and R2 scores for training and validation sets.
    """

    # non-feature cols
    non_feat_cols = ['Player', 'Pos', 'Tm', 'Key', 'Year', 'target']

    # define X and y
    X = df.drop(non_feat_cols, axis=1)
    y = df['target']

    # cross_validate
    cv = KFold(n_splits=folds, shuffle=True, random_state=SEED)
    scoring = {'rmse': 'neg_root_mean_squared_error', 'r2': 'r2'}
    results = cross_validate(estimator, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1)
    
    # assemble into a df
    scores = pd.DataFrame({'train_rmse': -results['train_rmse'], 'val_rmse': -results['test_rmse'], 'train_r2': results['train_r2'], 'val_r2': results['test_r2']})
    
    # aggregate
    summary = scores.agg(['mean', 'std'])
    return summary[['train_rmse', 'train_r2', 'val_rmse', 'val_r2']]

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def get_X_y(pos_subset):
    """
    Create X and y datasets for modeling.

    Args:
    - pos_subset (pd.DataFrame): The DataFrame containing player features for a specific position.

    Returns:
    - X (pd.DataFrame): Feature set.
    - y (pd.Series): Target variable.
    """

    # non-feature cols
    non_feat_cols = ['Player', 'Pos', 'Tm', 'Key', 'Year', 'target']

    # define X and y
    X = pos_subset.drop(non_feat_cols, axis=1)
    y = pos_subset['target']
    return X, y

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def xgb_cross_val(max_depth, learning_rate, gamma, subsample, colsample_bytree, min_child_weight, X, y):
    """
    Objective function for XGBoost hyperparameter tuning using Bayesian Optimization.

    Args:
    - XGBRegressor parameters: max_depth, learning_rate, gamma, subsample, colsample_bytree, min_child_weight
    - X (pd.DataFrame): Feature set.
    - y (pd.Series): Target variable.

    Returns:
    - scores.mean() (float): Mean RMSE from 5-fold cross-validation.
    """

    # define XGBoost parameters
    params = {'n_estimators': 1000,
        'max_depth': int(max_depth),
        'learning_rate': learning_rate,
        'gamma': gamma,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'min_child_weight': int(min_child_weight),
        'random_state': SEED, 
        'n_jobs': -1}

    # create pipeline
    xgb = XGBRegressor(**params)

    # 10-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(xgb, X, y, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)

    # return mean cv score
    return scores.mean()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def run_bayes_opt(X, y, param_bounds, seed, init_points=10, n_iter=100, verbose=2):
    """
    Run BayesianOptimization on xgb_cross_val for one (X, y) dataset.

    Args:
      - X (pd.DataFrame or np.ndarray): Features
      - y (pd.Series or np.ndarray): Target
      - param_bounds (dict): Boundaries for BO
      - seed (int): random_state for reproducibility
      - init_points (int): number of random initial points
      - n_iter (int): number of Bayesian iters
      - verbose (int): verbosity level

    Returns:
      - optimizer (BayesianOptimization): fitted optimizer object
    """
    
    # black‐box function wrapping xgb_cross_val
    def _f(max_depth, learning_rate, gamma, subsample, colsample_bytree, min_child_weight):
        return xgb_cross_val(
            max_depth=int(max_depth),
            learning_rate=learning_rate,
            gamma=gamma,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=int(min_child_weight),
            X=X, y=y)

    # set up the optimizer
    optimizer = BayesianOptimization(f=_f, pbounds=param_bounds, random_state=seed, verbose=verbose)
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    return optimizer

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def get_2024_preds(df, model):
    """
    Train model on all data prior to 2023 and use 2023 data to predict 2024 results.

    Args:
    - df (pd.DataFrame): DataFrame containing player data.
    - model (sklearn estimator): The model to use for predictions.

    Returns:
    - preds_df (pd.DataFrame): DataFrame containing predictions and true values for the specified position group.
    """

    # get training data (before 2023) and test data (2023)
    X_train, y_train = get_X_y(df.query('Year < 2023'))
    X_test, y_test = get_X_y(df.query('Year == 2023'))

    # train model
    model.fit(X_train, y_train)

    # make predictions
    y_pred = model.predict(X_test)

    # evaluate model
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'RMSE: {rmse:.4f}')
    print(f'R2: {r2:.4f}')
    print()

    # create a df for our predictions
    preds_df = pd.DataFrame(data={'player': df.query('Year == 2023')['Player'].values, 'team': df.query('Year == 2023')['Tm'].values, 
                              'y_true': y_test, 'y_pred': y_pred, 'error': (y_pred - y_test), 'pos': df.query('Year == 2023')['Pos'].values})
    
    # map colors to our preds_df, fill nans with gray
    preds_df['team_color'] = preds_df['team'].map(TEAM_COLORS).fillna('gray')

    # sort by true ppg values
    return preds_df.sort_values('y_true', ascending=False).reset_index(drop=True)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def plot_2024_preds(preds_df):
    """
    Visualize the model's predictions against the true values.

    Args:
    - preds_df (pd.DataFrame): DataFrame containing the model's predictions and true values.

    Returns:
    - None
    """

    # create fig
    plt.figure(figsize=(14, 8))

    # title, labels
    plt.title('2024 Fantasy PPG Predictions', fontsize=22)
    plt.xlabel('True PPG', fontsize=22)
    plt.ylabel('Predicted PPG', fontsize=22)

    # define full palette
    full_palette = {'QB': '#1f39a6', 'RB': '#B22222', 'WR': '#9B30FF', 'TE': '#708090'}

    # filter palette to only positions present
    unique_positions = preds_df['pos'].unique()
    active_palette = {pos: full_palette[pos] for pos in unique_positions if pos in full_palette}

    # scatter the points
    sns.scatterplot(data=preds_df, x='y_true', y='y_pred', hue='pos', palette=active_palette, legend=True)

    # perfect-prediction line
    plt.plot([5, 30], [5, 30], color='black', linewidth=1)

    # annotate player names
    texts = []
    for i, row in preds_df.iterrows():
        texts.append(plt.text(row['y_true'], row['y_pred'], row['player'], fontsize=10, va='center', ha='center'))

    # adjust the text positions to avoid overlaps
    adjust_text(texts, x=preds_df['y_true'].values, y=preds_df['y_pred'].values, arrowprops=dict(arrowstyle='-', color='black', lw=0.2), min_arrow_len=0)

    # set limits for x and y axes
    xmin, xmax = preds_df['y_true'].min() - 2, preds_df['y_true'].max() + 2
    ymin, ymax = preds_df['y_pred'].min() - 2, preds_df['y_pred'].max() + 2
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    # annotations for over and under-predictions
    plt.text(xmin + 3, ymax - 3, 'Over-predictions', fontsize=20, weight='semibold', color='red')
    plt.text(xmax - 5, ymin + 3, 'Under-predictions', fontsize=20, weight='semibold', color='red')
    plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def get_2025_preds(df, model):
    """
    Train model on all data prior to 2024 and use 2024 data to predict 2025 results.

    Args:
    - df (pd.DataFrame): DataFrame containing player data.
    - model (sklearn estimator): The model to use for predictions.

    Returns:
    - preds_df (pd.DataFrame): DataFrame containing 2025 predictions.
    """

    # get X and y
    X, y = get_X_y(df)

    # predict
    y_pred = model.predict(X)

    # create a df for our predictions
    preds_df = pd.DataFrame(data={'player': df['Player'].values, 'team': df['Tm'].values, 'y_pred': y_pred, 'pos': df['Pos'].values})

    # sort by prediction
    preds_df = preds_df.sort_values('y_pred', ascending=False).reset_index(drop=True)

    # add rank
    preds_df['rank'] = [(i+1) for i in range(preds_df.shape[0])]
    
    # map colors to our preds_df, fill nans with gray
    preds_df['team_color'] = preds_df['team'].map(TEAM_COLORS).fillna('gray')

    return preds_df

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def plot_2025_preds(preds_df, pos, xlabel, xmin=0, xmax=1):
    """
    Visualize the model's 2025 predictions.

    Args:
    - preds_df (pd.DataFrame): DataFrame containing the model's predictions.
    - xmin (int): Minimum x-axis value for the plot.
    - xmax (int): Maximum x-axis value for the plot.

    Returns:
    - None
    """

    # reverse the df for proper display
    preds_df = preds_df[::-1]

    # get length of preds
    num_players = preds_df.shape[0]

    # dynamic line and marker sizes
    linesize = max(2, min(12, 144 / num_players))
    markersize = linesize * 2

    # plot
    plt.figure(figsize=(14, 10))
    plt.hlines(y=preds_df['player'], xmin=xmin, xmax=preds_df['y_pred'], color=preds_df['team_color'], lw=linesize)
    plt.plot(preds_df['y_pred'], preds_df['player'], 'o', color='black', markersize=markersize)
    plt.title(f'2025 Predictions: {pos}', fontsize=24)
    plt.xlabel(xlabel, fontsize=24)
    plt.xticks(fontsize=18)
    plt.xlim(xmin, xmax)
    plt.margins(x=0, y=0.04)

    # add rank to y-tick labels
    ytick_labels = [f'{name} ({int(rank)})' for name, rank in zip(preds_df['player'], preds_df['rank'])]
    plt.yticks(ticks=range(num_players), labels=ytick_labels, fontsize=markersize)

    # annotate
    for i, row in preds_df.iterrows():
        plt.text(row['y_pred'] + 0.3, row['player'], f'{row["y_pred"]:.2f}', va='center', fontsize=markersize)
    plt.tight_layout()
    plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# games_played.ipynb

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def create_injury_features(df):
    """
    Create features for each player.

    Args:
    - df (pd.dataframe): Player data.

    Returns:
    - (pl.dataframe): Dataframe with new features added.
    """

    # convert to polars dataframe and sort
    df = pl.from_pandas(df).sort(["Key", "Year"])

    # convert to a lazy frame for efficiency
    lazy_df = df.lazy()

    # define cols to aggregate
    non_agg_cols = ['Player', 'Tm', 'Pos', 'Key', 'Year', 'Age', 'Exp', 'target']
    agg_cols = [col for col in df.columns if col not in non_agg_cols]

    # list of expressions for original columns
    base_exprs = [pl.col('*')]

    # expressions that rely on prior aliases
    post_exprs = []

    # iterate through each column to be aggregated
    for col in agg_cols:
        # rolling stats (n years)
        for n in [2, 3]:
            base_exprs.extend([
                pl.col(col)
                .rolling_mean(window_size=n, min_samples=1)
                .over('Key')
                .alias(f'{col}_{n}y_mean'),
                pl.col(col)
                .rolling_std(window_size=n, min_samples=1)
                .over('Key')
                .alias(f'{col}_{n}y_std')])

        # cumulative career mean
        cum_sum = pl.col(col).cum_sum().over('Key')
        cum_count = (pl.col('Exp') + 1)
        cum_mean = (cum_sum / cum_count).alias(f'{col}_career_mean')
        base_exprs.extend([cum_mean])

    # add the new columns to df
    lazy_df = lazy_df.with_columns(base_exprs)

    # collect results back into a pandas df
    df_pandas = lazy_df.collect().to_pandas()

    # fill nulls and infs with 0
    non_target_cols = [col for col in df_pandas.columns if col != 'target']
    df_pandas[non_target_cols] = df_pandas[non_target_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # sort columns 
    return df_pandas[sorted(df_pandas.columns)]

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def plot_adj_preds(preds_df, pos):
    """
    Visualize the model's predictions against the true values.

    Args:
    - preds_df (pd.DataFrame): DataFrame containing ppg and games_played_pct predictions.

    Returns:
    - None
    """

    # map colors to our preds_df, fill nans with gray
    preds_df['team_color'] = preds_df['team'].map(TEAM_COLORS).fillna('gray')

    # create fig
    plt.figure(figsize=(14, 8))

    # title, labels
    plt.title(f'2025 Predictions: Top {preds_df.shape[0]} {pos}s', fontsize=22)
    plt.xlabel('PPG', fontsize=22)
    plt.ylabel('Games Played %', fontsize=22)

    # scatter the points
    
    plt.scatter(
        preds_df['ppg_pred'],
        preds_df['games_played_pct_pred'],
        color=preds_df['team_color'],
        s=30
    )

    # annotate player names
    texts = []
    for i, row in preds_df.iterrows():
        texts.append(plt.text(row['ppg_pred'], row['games_played_pct_pred'], row['player'], fontsize=10, va='center', ha='center'))

    # adjust the text positions to avoid overlaps
    adjust_text(texts, force_points=0.5, arrowprops=dict(arrowstyle='-', color='black', lw=0.2))

    plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# rankings.ipynb

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def show_top_players(df, pos, num_players):
    """
    Show the top players for a given position based on the predictions.
    """

    # filter the predictions for the specified position
    pos_preds = df.query('pos == @pos').copy()[['player', 'adp_rank_2025', 'pred_rank_2025', 'rank_diff', 'ppg_pred']]
    
    # sort by predicted points and select the top players
    top_players = pos_preds.sort_values(by='adp_rank_2025', ascending=True).reset_index(drop=True)

    return top_players.head(num_players).T

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def clean_rankings(rankings):
    # combine firstName and lastName
    rankings['player'] = rankings['firstName'] + ' ' + rankings['lastName']

    # drop cols
    rankings = rankings.drop(columns=['id', 'firstName', 'lastName', 'teamName', 'slotName', 'lineupStatus'])

    # remove first 2 chars from positionRank col
    rankings['positionRank'] = rankings['positionRank'].str[2:]

    # rename positionRank to adp_rank_2025
    rankings = rankings.rename(columns={'positionRank': 'adp_rank_2025'})

    # drop players with null adp_rank_2025
    rankings = rankings[rankings['adp_rank_2025'].notnull()]

    # cast adp_rank_2025 to int
    rankings['adp_rank_2025'] = rankings['adp_rank_2025'].astype(int)

    # remove "." and " Jr" from player col
    rankings['player'] = rankings['player'].str.replace('.', '', regex=False).str.replace(' Jr', '', regex=False)

    return rankings