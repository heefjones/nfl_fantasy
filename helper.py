# data science
import pandas as pd
import polars as pl
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

# machine learning
from tqdm import tqdm
from sklearn.model_selection import cross_validate, KFold, cross_val_score
from sklearn.metrics import r2_score, root_mean_squared_error
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization

# system
import os
import gc
from functools import reduce

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
TEAM_NAMES = {
    'Arizona Cardinals': 'ARI',
    'Atlanta Falcons': 'ATL',
    'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF',
    'Carolina Panthers': 'CAR',
    'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN',
    'Cleveland Browns': 'CLE',
    'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN',
    'Detroit Lions': 'DET',
    'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU',
    'Indianapolis Colts': 'IND',
    'Jacksonville Jaguars': 'JAX',
    'Kansas City Chiefs': 'KC',
    'Las Vegas Raiders': 'LV',
    'Los Angeles Chargers': 'LAC',
    'Los Angeles Rams': 'LAR',
    'Miami Dolphins': 'MIA',
    'Minnesota Vikings': 'MIN',
    'New England Patriots': 'NE',
    'New Orleans Saints': 'NO',
    'New York Giants': 'NYG',
    'New York Jets': 'NYJ',
    'Philadelphia Eagles': 'PHI',
    'Pittsburgh Steelers': 'PIT',
    'San Francisco 49ers': 'SF',
    'Seattle Seahawks': 'SEA',
    'Tampa Bay Buccaneers': 'TB',
    'Tennessee Titans': 'TEN',
    'Washington Commanders': 'WAS'}


# numpy seed
SEED = 9
np.random.seed(SEED)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# preprocessing.ipynb

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
        data['year'] = int(year)
        
        # add df to the list
        dfs.append(data)

    # stack dataframes together
    df = pd.concat(dfs, axis=0, ignore_index=True)

    # non-pff data
    if not pff:
        # drop rank/point columns (we will be recalculating these)
        df = df.drop(columns=['Rk', 'FantPt', 'PPR', 'DKPt', 'FDPt', 'VBD', 'PosRank', 'OvRank'])

        # replace 3-letter team codes with 2-letter codes
        team_map = {'GNB': 'GB', 'KAN': 'KC', 'LVR': 'LV', 'NWE': 'NE', 'NOR': 'NO', 'SFO': 'SF', 'TAM': 'TB'}
        df['team'] = df['team'].replace(team_map)

    return df

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def clean_names(df):
    # lowercase the player col
    df['player'] = df['player'].str.lower()

    # remove punctuation
    df['player'] = df['player'].str.replace(r"[.\-']", '', regex=True)
    
    # remove sirnames
    df['player'] = df['player'].str.replace(r'\b(jr|sr|ii|iii|iv|v)\b', '', regex=True)

    # replace all 'joshua' -> 'josh',  'mitch' -> 'mitchell', 'ben' -> 'benjamin', 'gabe' -> 'gabriel', dave' -> 'david'
    for name_pair in [('joshua', 'josh'), ('mitch', 'mitchell'), ('ben', 'benjamin'), ('gabe', 'gabriel'), ('dave', 'david')]:
        df['player'] = df['player'].str.replace(name_pair[0] + ' ', name_pair[1] + ' ', regex=False)

    # remove trailing spaces
    df['player'] = df['player'].str.strip()

    # rename players
    df.loc[df['player'] == 'marquise brown', 'player'] = 'hollywood brown'
    df.loc[df['player'] == 'chigoziem okonkwo', 'player'] = 'chig okonkwo'
    df.loc[df['player'] == 'stevie johnson', 'player'] = 'steve johnson'
    return df

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def show_duplicates(df, subset):
    """
    Show duplicates in the DataFrame.
    """

    # identify duplicates
    dup = df[df.duplicated(subset=subset, keep=False)]

    # show
    print(f'Number of duplicate rows: {dup.shape[0]}')
    display(dup[subset + ['team']].sort_values('year').T)

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
    # add 'key' back manually
    group['key'] = group.name

    # get first experience value for a player
    first_exp = group['exp'].iloc[0]
    
    # if value is null, set to 0 (rookie season)
    if pd.isna(first_exp):
        first_exp = 0
    
    # define range of years to fill each player's experience column
    experience = range(int(first_exp), int(first_exp) + len(group))
    group['exp'] = list(experience)
    return group

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def add_point_cols(df, points_type):
    """
    Add point columns to the dataframe.

    Args:
    - df (pd.DataFrame): The dataframe to add point columns to.
    - points_type (str): The type of point system to use (standard, half-ppr, ppr, 6pt passing td).

    Returns:
    - df (pd.DataFrame): The dataframe with point column added.
    """

    # error handling
    if points_type not in ['standard', 'half-ppr', 'ppr', '6']:
        raise ValueError("Invalid points_type type. Choose from 'standard', 'half-ppr', 'ppr', or '6'.")

    # calculate standard points_type (excluding passing tds)
    standard_points = (df['pass_yds'] * 0.04) + (df['pass_int'] * -1) + (df['rush_yds'] * 0.1) + \
        (df['rush_td'] * 6) + (df['rec_yds'] * 0.1) + (df['rec_td'] * 6) + (df['fmb_lost'] * -2)

    # standard points
    if points_type == 'standard':
        df['points_standard'] = standard_points + (df['pass_td'] * 4)
        
    # half-ppr    
    elif points_type == 'half-ppr':
        df['points_half-ppr'] = standard_points + (df['pass_td'] * 4) + (df['rec_rec'] * 0.5)

    # ppr    
    elif points_type == 'ppr':
        df['points_ppr'] = standard_points + (df['pass_td'] * 4) + (df['rec_rec'] * 1)

    # PPR scoring with 6pt passing tds
    elif points_type == '6':
        df['points_6'] = standard_points + (df['pass_td'] * 6) + (df['rec_rec'] * 1)

    # point-per-game column
    df['ppg_' + points_type] = (df['points_' + points_type] / df['g']).fillna(0)

    # point-per-touch column
    df['ppt_' + points_type] = (df['points_' + points_type] / df['touches']).fillna(0)

    return df

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def add_rank_cols(df, points_type):
    """
    Extract the points column from df and add total points, ppg, and ppt rank columns to df.

    Args:
    - df (pd.DataFrame): DataFrame containing player data.
    - points_type (str): The type of point system to use (standard, half-ppr, ppr, 6pt passing td).

    Returns:
    - pd.DataFrame: DataFrame with added rank columns.
    """

    # some groups we will be using
    year_groups = df.groupby('year')
    pos_groups = df.groupby(['year', 'pos'])
    
    # create mapping for each metric and corresponding grouping
    metrics = ['points', 'ppg', 'ppt']
    metric_to_group = {}
    for metric in metrics:
        metric_to_group[f'{metric}_ovr_rank_{points_type}'] = year_groups
        metric_to_group[f'{metric}_pos_rank_{points_type}'] = pos_groups
    
    # iterate over each rank column, compute the ranking, and assign to df
    for rank_col, group in metric_to_group.items():
        # determine the base metric by removing the rank part from the rank column name
        if 'ovr_rank' in rank_col:
            base_metric = rank_col.replace(f'ovr_rank_{points_type}', '')
        elif 'pos_rank' in rank_col:
            base_metric = rank_col.replace(f'pos_rank_{points_type}', '')
        
        # construct the source column name
        source_col = f"{base_metric}{points_type}"

        # compute rank
        df[rank_col] = group[source_col].transform(lambda x: x.rank(ascending=False, method='min')).astype(int)
    
    return df

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def add_vorp_cols(df, points_type, num_teams=12, wr3=True):
    """
    Add vorp columns to the dataframe using the default Underdog fantasy lineup (12 teams, 3 WRs).

    Args:
    - df (pd.DataFrame): The dataframe containing player data.
    - points_type (str): The type of point system to use (standard, half-ppr, ppr, 6pt passing td).
    - num_teams (int): The number of teams in the league.
    - wr3 (bool): Whether to include a 3rd WR in the lineup. If false, a 2WR format is implied.

    Returns:
    - df (pd.DataFrame): The dataframe with vorp columns added.
    """

    # define the replacement rank based on num teams and 3WR format
    replacement_ranks = {'QB': num_teams, 'RB': int(num_teams * 2.5), 'WR': int(num_teams * 2.5 + True), 'TE': num_teams}

    # iterate through the position groups
    for (year, pos), group in df.groupby(['year', 'pos']):

        # iterate for both season total and ppg vorp
        for rank_type in ['points', 'ppg']:

            # get the replacement rank for the current position, subtract 1 to get the index
            rank = int(replacement_ranks[pos] - 1)

            # format col name
            col_name = rank_type + '_' + points_type

            # sort group
            group = group.sort_values(col_name, ascending=False)

            # get replacement player points for the current position and scoring type
            replacement = group.iloc[rank][col_name]

            # add vorp column
            df.loc[(df['year'] == year) & (df['pos'] == pos), rank_type + '_vorp_' + points_type] = \
            df.loc[(df['year'] == year) & (df['pos'] == pos), col_name] - replacement

    return df

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def add_target_cols(df, points_type):
    """
    Add Total and ppg target columns to the dataframe.

    Args:
    - df (pd.DataFrame): The dataframe containing player data.
    - points_type (str): The type of point system to use (standard, half-ppr, ppr, 6pt passing td).

    Returns:
    - df (pd.DataFrame): The dataframe with target columns added.
    """

    # group by each player and shift the points column by 1
    df['points_target_' + points_type] = df.groupby('key')['points_' + points_type].shift(-1)
    df['ppg_target_' + points_type] = df.groupby('key')['ppg_' + points_type].shift(-1)

    return df

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def clean_rankings(rankings):
    # combine firstName and lastName
    rankings['player'] = rankings['firstName'] + ' ' + rankings['lastName']

    # drop cols
    rankings = rankings.drop(columns=['id', 'firstName', 'lastName', 'slotName', 'lineupStatus'])

    # clean player names
    rankings = clean_names(rankings)

    # remove pos from positionRank
    rankings['positionRank'] = rankings['positionRank'].str[2:]

    # rename positionRank
    rankings = rankings.rename(columns={'positionRank': 'adp_rank_2025'})

    # drop unranked players
    rankings = rankings[rankings['adp_rank_2025'].notnull()]

    # cast adp_rank_2025 to int
    rankings['adp_rank_2025'] = rankings['adp_rank_2025'].astype(int)

    # remove "." and " Jr" from player col
    rankings['player'] = rankings['player'].str.replace('.', '', regex=False).str.replace(' Jr', '', regex=False)

    # map teamName col to abbreviations
    rankings['teamName'] = rankings['teamName'].map(TEAM_NAMES)

    # add year col
    rankings['year'] = 2024

    # sort by rank
    return rankings.sort_values(by='adp_rank_2025', ascending=True).reset_index(drop=True)

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

    # restore player and year columns
    df['player'] = df[f'{prefix}_player']
    df['year'] = df[f'{prefix}_year']

    # drop prefixed player and year
    return df.drop([f'{prefix}_player', f'{prefix}_year'], axis=1)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def clean_pff_player_data(df, prefix):
    """
    Clean PFF player data by normalizing certain columns and dropping unnecessary ones.

    Args:
    - df (pd.DataFrame): The PFF data to clean.
    - prefix (str): The prefix to add to the columns.

    Returns:
    - df (pd.DataFrame): The cleaned PFF data.
    """

    # passing
    if prefix == 'pass':
        formulas = {'dropback_pct': ('dropbacks', 'passing_snaps'), 
                    'aimed_passes_pct': ('aimed_passes', 'attempts'), 
                    'dropped_passes_pct': ('drops', 'aimed_passes'), 
                    'batted_passes_pct': ('bats', 'aimed_passes'), 
                    'thrown_away_pct': ('thrown_aways', 'passing_snaps'), 
                    'pressure_pct': ('def_gen_pressures', 'passing_snaps'), 
                    'scramble_pct': ('scrambles', 'passing_snaps'), 
                    'sack%': ('sacks', 'passing_snaps'), 
                    'pressure_to_sack%': ('sacks', 'def_gen_pressures'), 
                    'btt_pct': ('big_time_throws', 'aimed_passes'), 
                    'twp_pct': ('turnover_worthy_plays', 'aimed_passes'), 
                    'first_down_pct': ('first_downs', 'attempts')}
    
    # rushing
    elif prefix == 'rush':
        formulas = {'team_rush_pct': ('attempts', 'run_plays'), 
                    'avoided_tackles_per_attempt': ('avoided_tackles', 'attempts'), 
                    '10_yard_run_pct': ('explosive', 'attempts'), 
                    '15_yard_run_pct': ('breakaway_attempts', 'attempts'), 
                    '15_yard_run_yards_pct': ('breakaway_yards', 'yards'), 
                    'first_down_pct': ('first_downs', 'attempts'), 
                    'gap_pct': ('gap_attempts', 'attempts'), 
                    'zone_pct': ('zone_attempts', 'attempts'), 
                    'yds_contact/a': ('yards_after_contact', 'attempts')}
    
    # receiving
    elif prefix == 'rec':
        formulas = {'avoided_tackles/r': ('avoided_tackles', 'receptions'), 
                    'first_down_pct': ('first_downs', 'receptions'), 
                    'int_per_target': ('interceptions', 'targets'), 
                    'yac_pct': ('yards_after_catch', 'yards'), 
                    'contested_pct': ('contested_targets', 'targets'),
                    'contested_catch_pct': ('contested_receptions', 'contested_targets')}

    # normalize
    df = add_percent_columns(df, formulas)

    # drop columns
    df = df.drop(columns=['player_id', 'position', 'player_game_count', 'declined_penalties', 'penalties', 'drops', 'franchise_id', 'touchdowns', 'yards'])

    # add prefix
    df = prefix_df(df, prefix)

    # fix team col
    df['team'] = df[f'{prefix}_team_name']
    df = df.drop(columns=[f'{prefix}_team_name'])

    # clean player names
    df = clean_names(df)

    return df.fillna(0)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def clean_pff_team_data(df):
    """
    Clean PFF team data by normalizing certain columns and dropping unnecessary ones.

    Args:
    - df (pd.DataFrame): The PFF data to clean.

    Returns:
    - df (pd.DataFrame): The cleaned PFF data.
    """

    # calculate games played
    games_played = df['Wins'] + df['Losses']

    # add win %
    df['win_pct'] = df['Wins'] / games_played

    # calculate ppg and ppg against
    df['ppg'] = df['Points For'] / games_played
    df['ppga'] = df['Points Against'] / games_played

    # create 'pass defense grade' col as average of 'pass rush' and 'pass coverage' grades
    df['pass_def_grade'] = ((df['Pass Rush Grade'] + df['Coverage Grade'])) / 2

    # fix team col
    df['team'] = df['team']

    # drop columns
    df = df.drop(columns=['team', 'Points For', 'Points Against', 'Wins', 'Losses'])

    # add 'team' to the beginning of each column name
    df.columns = ['team_' + col if col not in ['team', 'year'] else col for col in df.columns]

    # lowercase and replace spaces with underscores
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # fill nulls
    return df.fillna(0)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def consolidate_pos_columns(df, col_name):
    """
    Consolidate duplicate columns like 'pass_X', 'rush_X', 'rec_X' into a single column
    based on player position.

    Args:
    - df (pd.DataFrame): your dataframe
    - col_name (str): e.g. 'grades_offense' or 'grades_hands_fumble'

    Returns:
    - df (pd.DataFrame): dataframe with the new consolidated column and dropped duplicates
    """

    # get column names
    pass_col, rush_col, rec_col = f'pass_{col_name}', f'rush_{col_name}', f'rec_{col_name}'

    # map positional conditions to the correct column 
    conditions = [df['pos'] == 'QB', df['pos'] == 'RB', df['pos'].isin(['WR', 'TE'])]
    choices = [df[pass_col], df[rush_col], df[rec_col]]
    df[col_name] = np.select(conditions, choices, default=np.nan)

    # drop the original columns
    return df.drop([pass_col, rush_col, rec_col], axis=1)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# eda.ipynb

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def plot_heatmap(data, title):
    # Compute the correlation between positions for each team and year
    correlations = data.corr()

    # only keep the upper triangle of the correlation matrix
    mask = np.tril(np.ones_like(correlations, dtype=bool))
    correlations = correlations.mask(mask)

    # Plot a heatmap of the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, fmt=".2f", cmap='coolwarm', center=0)
    plt.title(title)
    plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ppg.ipynb

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def create_features(df):
    """
    Create features for each player.

    Args:
    - df (pd.dataframe): player data.

    Returns:
    - (pl.dataframe): Dataframe with new features added.
    """

    # convert to polars dataframe and sort
    df = pl.from_pandas(df).sort(["key", "year"])

    # convert to a lazy frame for efficiency
    lazy_df = df.lazy()

    # define cols to aggregate
    non_agg_cols = ['player', 'team', 'pos', 'key', 'year', 'age', 'exp', 'pro_bowl', 'all_pro', 'team_next', 'new_team', 'target']
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
                .over('key')
                .alias(f'{col}_{n}y_mean'),
                pl.col(col)
                .rolling_std(window_size=n, min_samples=1)
                .over('key')
                .alias(f'{col}_{n}y_std')])

        # cumulative career mean & std
        cum_sum = pl.col(col).cum_sum().over('key')
        cum_count = (pl.col('exp') + 1)
        cum_mean = (cum_sum / cum_count).alias(f'{col}_career_mean')
        sum_sq = pl.col(col).pow(2).cum_sum().over('key')
        cum_var = ((sum_sq - cum_sum.pow(2) / cum_count) / cum_count)
        cum_std = cum_var.sqrt().alias(f'{col}_career_std')
        base_exprs.extend([cum_mean, cum_std])

        # trend slope relative to career (expanding linear regression)
        sum_year = pl.col('year').cum_sum().over('key')
        sum_year_sq = pl.col('year').pow(2).cum_sum().over('key')
        sum_y = cum_sum
        sum_xy = (pl.col(col) * pl.col('year')).cum_sum().over('key')
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
    - wr (pd.DataFrame): Subset of features for wide receivers.
    - te (pd.DataFrame): Subset of features for tight ends.
    """

    # create the 4 positional subsets
    qb = features.query('pos == "QB"')
    rb = features.query('pos == "RB"')
    wr = features.query('pos == "WR"')
    te = features.query('pos == "TE"')

    # define positional cols
    pass_cols = [col for col in features.columns if col.startswith('pass_')]
    rush_cols = [col for col in features.columns if col.startswith('rush_')]
    rec_cols = [col for col in features.columns if col.startswith('rec_')]

    # drop non-positional cols
    qb = qb.drop(columns=rec_cols)
    rb = rb.drop(columns=pass_cols)
    wr = wr.drop(columns=pass_cols + rush_cols)
    te = te.drop(columns=pass_cols + rush_cols)
    
    return qb, rb, wr, te

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
    non_feat_cols = ['player', 'pos', 'team', 'team_next', 'key', 'year', 'target']

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
    non_feat_cols = ['player', 'pos', 'team', 'key', 'year', 'target']

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

def run_bayes_opt(X, y, param_bounds, seed, init_points=5, n_iter=20, verbose=2):
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

def get_xgb_model(params):
    """
    Create an XGBoost model with the given parameters.

    Args:
    - params (dict): Dictionary of parameters for the XGBoost model.

    Returns:
    - model (XGBRegressor): Configured XGBoost model.
    """

    return XGBRegressor(**params, n_estimators=1000, random_state=SEED, n_jobs=-1)

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
    X_train, y_train = get_X_y(df.query('year < 2023'))
    X_test, y_test = get_X_y(df.query('year == 2023'))

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
    preds_df = pd.DataFrame(data={'player': df.query('year == 2023')['player'].values, 'team': df.query('year == 2023')['team'].values, 
                              'y_true': y_test, 'y_pred': y_pred, 'error': (y_pred - y_test), 'pos': df.query('year == 2023')['pos'].values})
    
    # map colors to our preds_df, fill nans with gray
    preds_df['team_color'] = preds_df['team'].map(TEAM_COLORS).fillna('gray')

    # sort by true ppg values
    return preds_df.sort_values('y_true', ascending=False).reset_index(drop=True)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def plot_2024_preds(preds_df, pos):
    """
    Visualize the model's predictions against the true values.

    Args:
    - preds_df (pd.DataFrame): DataFrame containing the model's predictions and true values.
    - pos (str): position group (e.g., 'QB', 'RB', 'WR', 'TE').

    Returns:
    - None
    """

    # create fig
    plt.figure(figsize=(14, 8))

    # title, labels
    plt.title(f'2024 Predictions: {pos}', fontsize=24)
    plt.xlabel('True ppg', fontsize=22)
    plt.ylabel('Predicted ppg', fontsize=22)

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
    preds_df = pd.DataFrame(data={'player': df['player'].values, 'team': df['team_next'].values, 'y_pred': y_pred, 'pos': df['pos'].values})

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
    - pos (str): position group (e.g., 'QB', 'RB', 'WR', 'TE').
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

# games_played.ipynb

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def create_injury_features(df):
    """
    Create features for each player.

    Args:
    - df (pd.dataframe): player data.

    Returns:
    - (pl.dataframe): Dataframe with new features added.
    """

    # convert to polars dataframe and sort
    df = pl.from_pandas(df).sort(["key", "year"])

    # convert to a lazy frame for efficiency
    lazy_df = df.lazy()

    # define cols to aggregate
    non_agg_cols = ['player', 'team', 'pos', 'key', 'year', 'Age', 'Exp', 'target']
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
                .over('key')
                .alias(f'{col}_{n}y_mean'),
                pl.col(col)
                .rolling_std(window_size=n, min_samples=1)
                .over('key')
                .alias(f'{col}_{n}y_std')])

        # cumulative career mean
        cum_sum = pl.col(col).cum_sum().over('key')
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
    plt.xlabel('ppg', fontsize=22)
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

# rankings.ipynb

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def compute_rank_diff(df):
    # make a copy and reset index
    df = df.copy().reset_index(drop=True)

    # ensure adp ranks are unique by adding a small offset per duplicate
    df['adp_rank_2025_no_ties'] = df['adp_rank_2025'] + df.groupby('adp_rank_2025').cumcount()

    # compute expected rank based on order
    df['expected_rank'] = df.index + 1

    # calculate how many ranks were skipped at each adp position
    df['skipped_ranks_cumulative'] = df['adp_rank_2025_no_ties'] - df['expected_rank']

    # add cumulative ranks skipped to pred_rank
    df['pred_rank_2025'] = df['pred_rank_2025']

    # compute adjusted rank difference
    df['rank_diff'] = df['adp_rank_2025_no_ties'] - (df['pred_rank_2025'] + df['skipped_ranks_cumulative'])

    # drop old cols
    df = df.drop(columns=['adp_rank_2025', 'expected_rank', 'skipped_ranks_cumulative'])

    # rename adp_rank col
    return df.rename(columns={'adp_rank_2025_no_ties': 'adp_rank_2025'})

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def show_top_players(df, pos, num_players):
    """
    Show the top players for a given position based on the predictions.
    """

    # filter the predictions for the specified position
    pos_preds = df.query('pos == @pos').copy()[['player', 'teamName', 'adp_rank_2025', 'pred_rank_2025', 'rank_diff', 'rank_diff/adp_rank', 'ppg_pred']]#.sort_values('rank_diff', ascending=False).reset_index(drop=True)

    return pos_preds.head(num_players).T

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

