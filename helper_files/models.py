# data science stuff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# models
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# data preprocessing, performance metrics
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error

# global random_state
random_state = 9





def create_features(df, target_col):
    '''

    Create features for the model. This includes cumulative stats, rolling stats, and year-over-year stats.
    The final dataset is returned, sorted by player and year.

    Parameters:
    df (pandas.Dataframe) - dataframe to create features from

    Returns:
    features (pandas.Dataframe) - final dataset with features

    '''

    # ensure we are sorted properly
    df = df.sort_values(['Key', 'Year'], ascending=True)

    # group by player
    player_group = df.groupby('Key')

    # metadata cols
    bool_cols = df.select_dtypes(include=['bool']).columns
    meta_cols = ['Key', 'Year', 'Age', 'Exp']

    # team_cols are the cols that start with 'Team_'
    team_cols = [col for col in df.columns if col.startswith('Team_')]

    # agg_cols are all columns that are not boolean, metadata, or target
    agg_cols = [col for col in df.columns if col not in bool_cols and col not in meta_cols and col not in team_cols and col != target_col]

    # function to calculate aggregate stats
    def calculate_group_stats(group):
        new_cols = {}
        
        for col in agg_cols:
            # Career mean and standard deviation including current and prior rows
            new_cols[f'{col}_career_mean'] = group[col].expanding().mean()
            new_cols[f'{col}_career_std'] = group[col].expanding().std()

            # Rolling mean and std over the last 3 seasons including current row
            new_cols[f'{col}_rolling_mean_2'] = group[col].rolling(window=2, min_periods=1).mean()
            new_cols[f'{col}_rolling_std_2'] = group[col].rolling(window=2, min_periods=1).std()

            # Difference from the previous season
            new_cols[f'{col}_diff'] = group[col].diff()
        
        # Convert the dictionary to a DataFrame and concatenate it with the original group
        new_df = pd.concat([group, pd.DataFrame(new_cols, index=group.index)], axis=1)
        
        return new_df

    # Apply the function to each group
    features = player_group.apply(calculate_group_stats)

    # fill nulls
    features = features.fillna(0)

    # fill infinities
    features = features.replace([np.inf, -np.inf], 0)

    return features





def cross_val(df, pos, target, estimator, models_df, random_state=random_state):
    '''
    
    Scale data and perform 5-fold cross validation.
    Append the estimator, RMSE, and R-squared to models_df.
    
    Parameters:
    df (pandas.Dataframe) - dataframe to create X and y from
    pos (str) - position to filter by
    target (str) - target column
    estimator (sklearn.estimator) - estimator to cross validate
    models_df (pandas.Dataframe) - dataframe that holds model results
    
    Returns:
    rmse (float) - average Root-Mean_Squared-Error from cross validation
    r2 (float) - average R^2 from cross validation
    
    '''
    
    # features and target
    X = df.drop(columns=target)
    y = df[target]

    # define the numeric and binary features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    binary_features = X.select_dtypes(include=['bool']).columns
    preprocessor = ColumnTransformer(transformers=[('num', RobustScaler(), numeric_features), ('binary', 'passthrough', binary_features)])

    # create pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), 
                               ('estimator', estimator)])
    
    # 5-fold cross validation
    results = cross_validate(pipeline, X, y, cv=5, scoring=['neg_root_mean_squared_error', 'r2'], n_jobs=-1, verbose=1)
    
    # get rmse and r-squared
    rmse = results['test_neg_root_mean_squared_error'].mean() * -1
    r2 = results['test_r2'].mean()
    
    # append results to models_df
    models_df.loc[len(models_df.index)] = [pos, X.columns, target, str(estimator), rmse, r2]
    
    return



def train(df, pos, target, estimator, models_df, random_state=random_state):
    '''
    Train multiple models by adding a new ferature each iteration. Assumes that features are most important -> least important (left to right).

    Parameters:
    df (pandas.Dataframe) - dataframe to create features from
    pos (str) - position to filter by
    target (str) - target column
    estimator (sklearn.estimator) - estimator to cross validate
    models_df (pandas.Dataframe) - dataframe that holds model results

    Returns:
    None
    
    '''

    # get target
    y = df[target]

    # iterate through columns
    for i in range(len(df.columns)):
        X = df.iloc[:, :i+1].drop(columns=target)

        # define the numeric and binary features
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        binary_features = X.select_dtypes(include=['bool']).columns
        percent_features = [col for col in X.columns if 'rate' in col or '%' in col]
        preprocessor = ColumnTransformer(transformers=[('num', RobustScaler(), numeric_features), ('binary', 'passthrough', binary_features), ('percent', 'passthrough', percent_features)])

        # create pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), 
                                ('estimator', estimator)])
        
        # 5-fold cross validation
        results = cross_validate(pipeline, X, y, cv=5, scoring=['neg_root_mean_squared_error', 'r2'], n_jobs=-1, verbose=1)
    
        # get rmse and r-squared
        rmse = results['test_neg_root_mean_squared_error'].mean() * -1
        r2 = results['test_r2'].mean()
        
        # append results to models_df
        models_df.loc[len(models_df.index)] = [pos, X.columns, target, str(estimator), rmse, r2]