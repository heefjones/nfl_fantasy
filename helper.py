# data science
import pandas as pd
import polars as pl
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
# pl.Config.set_tbl_rows(n=50)
# pl.Config.set_tbl_cols(-1)
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

def clean_pff_data(pff_data, prefix):
    """
    Clean PFF data by normalizing certain columns and dropping unnecessary ones.

    Args:
    - pff_data (pd.DataFrame): The PFF data to clean.
    - prefix (str): The prefix to add to the columns.

    Returns:
    - pff_data (pd.DataFrame): The cleaned PFF data.
    """

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

# ppg.ipynb

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
    non_agg_cols = ['Player', 'Tm', 'Pos', 'Key', 'Year', 'PPGTarget_half-ppr', 'Age', 'Exp']
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
                .rolling_mean(window_size=n, min_periods=1)
                .over('Key')
                .alias(f'{col}_{n}y_mean'),
                pl.col(col)
                .rolling_std(window_size=n, min_periods=1)
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
    df_pandas = df_pandas.replace([np.inf, -np.inf], np.nan).fillna(0)

    # sort columns 
    return df_pandas[sorted(df_pandas.columns)]

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def cross_val(df, target_col, estimator, folds=5):
    """
    Perform KFold cross validation on a given estimator and store evaluation metrics.
    
    Args:
    - df (pd.DataFrame): Data to model.
    - target_col (str): Name of the target column in the DataFrame.
    - estimator (sklearn estimator): Estimator to use for modeling.
    - folds (int): Number of cross-validation folds to use. Default is 5.
    
    Returns:
    - pd.DataFrame: DataFrame containing the mean and standard deviation of RMSE and R2 scores for training and validation sets.
    """

    # non-feature cols
    non_feat_cols = ['Player', 'Pos', 'Tm', 'Key', 'Year'] + [target_col]

    # define X and y
    X = df.drop(non_feat_cols, axis=1)
    y = df[target_col]

    # cross_validate
    cv = KFold(n_splits=folds, shuffle=True, random_state=SEED)
    scoring = {'rmse': 'neg_root_mean_squared_error', 'r2': 'r2'}
    results = cross_validate(estimator, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1)
    
    # assemble into a df
    scores = pd.DataFrame({'train_rmse': -results['train_rmse'], 'val_rmse': -results['test_rmse'], 'train_r2': results['train_r2'], 'val_r2': results['test_r2']})
    
    # aggregate
    summary = scores.agg(['mean', 'std'])
    return summary

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



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# injury.ipynb

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#