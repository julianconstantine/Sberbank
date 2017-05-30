import pandas as pd
import numpy as np


def rmsle(y_true, y_pred):
    # Root-mean squared log error
    return np.sqrt(np.mean((np.log(y_pred + 1) - np.log(y_true + 1))**2))


def prepare_submission(y_pred, id_list, version):
    # Prepare a CSV to be submitted for evaluation on Kaggle
    f = open('src/submissions/sub' + str(version) + '/submission' + str(version) + '.csv', mode='w')
    f.writelines('id,price_doc' + '\n')

    for i in range(len(id_list)):
        id = id_list[i]
        y_i = y_pred[i]

        linestr = str(id) + ',' + str(y_i)

        f.writelines(linestr + '\n')

    f.close()


def floor_date(x, unit='month'):
    # Round a Timestamp down to the nearest month (first of the month)
    if unit == 'month':
        y = str(x.year)
        m = str(x.month)

        s = y + '-' + m + '-01'

        return pd.to_datetime(s)
    else:
        raise ValueError("Not yet implemented")


def prepare_ts(data, level):
    # Create time-dependent features (counts of property sales) at a specified level
    if level == 'city':
        grouped = data.groupby('year_month')
    elif level == 'raion':
        grouped = data.groupby(['year_month', 'sub_area'])
    elif level == 'building':
        grouped = data.groupby(['year_month', 'apartment_id'])
    else:
        raise ValueError("level must be either city, raion, or building")

    ts = grouped.agg({'price_doc': np.size})

    ts = ts['price_doc'].reset_index()
    ts.rename(columns={'price_doc': level + '_units_sold'}, inplace=True)

    # Need to implement expansion to include periods with zero observations
    all_time_periods = pd.date_range(start='2011-08-01', end='2016-05-31', freq='1 M').tolist()
    all_time_periods = [floor_date(x, 'month') for x in all_time_periods]

    # Take only unique values
    all_time_periods = list(set(all_time_periods))

    if level == 'city':
        temp_df = pd.DataFrame()
        temp_df['year_month'] = all_time_periods

        # Expand the time series to include months with no sales
        ts = pd.merge(temp_df, ts, on='year_month', how='left').fillna(0)
    elif level == 'raion':
        temp_df = pd.DataFrame()

        all_raions = data['sub_area'].unique().tolist()

        temp_df['sub_area'] = all_raions*len(all_time_periods)

        l_time = all_time_periods * len(all_raions)
        l_raion = []

        for r in all_raions:
            l_raion += [r]*len(all_time_periods)

        temp_df['year_month'] = l_time
        temp_df['sub_area'] = l_raion

        # Expand the time series to include months with no sales
        ts = pd.merge(temp_df, ts, on=['year_month', 'sub_area'], how='left').fillna(0)
    elif level == 'building':
        temp_df = pd.DataFrame()

        all_buildings = data['apartment_id'].unique().tolist()

        l_time = all_time_periods * len(all_buildings)
        l_building = []

        for r in all_buildings:
            l_building += [r] * len(all_time_periods)

        temp_df['year_month'] = l_time
        temp_df['apartment_id'] = l_building

        # Expand the time series to include months with no sales
        ts = pd.merge(temp_df, ts, on=['year_month', 'apartment_id'], how='left').fillna(0)

    # Add an underscore for naming purposes
    level += '_'

    # Create lagged variables
    ts[level + 'units_sold_lag1'] = ts[level + 'units_sold'].shift(periods=1)
    ts[level + 'units_sold_lag2'] = ts[level + 'units_sold'].shift(periods=2)

    # Median replace lagged values (this is crude, want to fix for raion/building-level)
    ts.loc[pd.isnull(ts[level + 'units_sold_lag1']), level + 'units_sold_lag1'] = ts[level + 'units_sold_lag1'].dropna().median()
    ts.loc[pd.isnull(ts[level + 'units_sold_lag2']), level + 'units_sold_lag2'] = ts[level + 'units_sold_lag2'].dropna().median()

    # Create moving averages and sums
    ts[level + 'units_sold_MA2'] = ts[level + 'units_sold'].rolling(window=2, min_periods=0).mean()
    ts[level + 'units_sold_SUM2'] = ts[level + 'units_sold'].rolling(window=2, min_periods=0).sum()

    ts[level + 'units_sold_MA3'] = ts[level + 'units_sold'].rolling(window=3, min_periods=0).mean()
    ts[level + 'units_sold_SUM3'] = ts[level + 'units_sold'].rolling(window=3, min_periods=0).sum()

    ts[level + 'units_sold_MA6'] = ts[level + 'units_sold'].rolling(window=6, min_periods=0).mean()
    ts[level + 'units_sold_SUM6'] = ts[level + 'units_sold'].rolling(window=6, min_periods=0).sum()

    return ts


def split_data(data, ignore_cols, log_y=False):
    # Split into training, testing, validation and take logs of prices
    y_train = data.loc[data['subset'] == 'train', 'price_doc']
    y_val = data.loc[data['subset'] == 'val', 'price_doc']
    y_full = data.loc[data['subset'].isin(['train', 'val']), 'price_doc']

    if log_y:
        # Take the base-10 logarithm of the y values
        y_train = np.log10(y_train)
        y_val = np.log10(y_val)
        y_full = np.log10(y_full)

    X_train = data.loc[data['subset'] == 'train'].drop(labels=ignore_cols, axis=1)
    X_val = data.loc[data['subset'] == 'val'].drop(labels=ignore_cols, axis=1)
    X_test = data.loc[data['subset'] == 'test'].drop(labels=ignore_cols, axis=1)

    X_full = X_train.append(X_val, ignore_index=True)

    X_dict = {'train': X_train, 'val': X_val, 'full': X_full, 'test': X_test}
    y_dict = {'train': y_train, 'val': y_val, 'full': y_full}

    return X_dict, y_dict



