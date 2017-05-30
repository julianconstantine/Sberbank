from src.functions import floor_date

import pandas as pd
import numpy as np

import json
import re

# Load raw datasets
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')
macro = pd.read_csv('data/raw/macro.csv')

# Parameters
VAL_FRAC = 0.3

RECOMPUTE_TRAIN_VAL_SPLIT = False

# Split training and validation data
if RECOMPUTE_TRAIN_VAL_SPLIT:
    train_val_split = ['val' if np.random.rand(1) <= VAL_FRAC else 'train' for i in range(train.shape[0])]

    pd.to_pickle(obj=train_val_split, path='data/interim/train_val_split.pkl')

    train['subset'] = train_val_split
    test['subset'] = 'test'
else:
    train_val_split = pd.read_pickle(path='data/interim/train_val_split.pkl')
    train['subset'] = train_val_split
    test['subset'] = 'test'

# Combine training and testing data
combined = train.append(test, ignore_index=True)

# Macro columns to keep
macro_cols = ["timestamp", "balance_trade", "balance_trade_growth", "eurrub", "average_provision_of_build_contract",
              "micex_rgbi_tr", "micex_cbi_tr", "deposits_rate", "mortgage_value", "mortgage_rate", "income_per_cap",
              "museum_visitis_per_100_cap", "apartment_build"]

# Merge on macroeconomic data
combined = pd.merge(combined, macro[macro_cols], on='timestamp')

print(combined.shape)  # 38,133 x 392
print(train.shape)  # 30,471 x 293
print(test.shape)  # 7,662 x 292


# RENAME VARIABLES: ekder_all, ekder_male, ekder_female -> elder_all, elder_male, elder_female
combined = combined.rename(columns={'ekder_all': 'elder_all', 'ekder_male': 'elder_male', 'ekder_female': 'elder_female'})

# SET DATA TYPES
# Load the SCS
with open('references/scs.json', mode='r') as f:
    scs = json.load(fp=f)

for var in scs:
    if var not in combined.columns:
        continue

    if 'Mapping' in scs[var]:
        m = scs[var]['Mapping']

        for old_value in m:
            new_value = m[old_value]
            combined.loc[combined[var] == old_value, var] = new_value

    var_type = scs[var]['Type']

    if var_type == 'logical':
        if combined[var].isnull().any():
            # If there are missing values cast as a factor, with 1 -> "yes", 0 -> "no", nan -> "missing
            combined.loc[combined[var] == 1, var] = 'yes'
            combined.loc[combined[var] == 0, var] = 'no'
            combined.loc[pd.isnull(combined[var]), var] = 'missing'

            combined[var] = combined[var].astype(object)
        else:
            # Otherwise just as as a zero/one float
            combined[var] = combined[var].astype(bool).astype(float)
    elif var_type == 'numeric':
        # Strip out any symbols that are not periods, minus signs, numbers, or scientific notation
        combined[var] = combined[var].apply(lambda x: re.sub(pattern='[^0-9.e-]', repl='', string=str(x)) if not pd.isnull(x) else x)

        # Replace any empty strings with NaNs
        combined[var] = combined[var].apply(lambda x: np.nan if x == '' else x)

        # Cast as float
        combined[var] = combined[var].astype(float)
    elif var_type == 'factor':
        combined[var] = combined[var].astype(object)
    elif var_type == 'datetime':
        combined[var] = pd.to_datetime(combined[var])


###################
# HELPER FEATURES #
###################

# CREATE VARIABLE: year_month
# timestamp rounded down to the first of each month
combined['year_month'] = combined['timestamp'].apply(lambda x: floor_date(x, unit='month'))

# CREATE VARIABLE: year_month_lag1
combined['year_month_lag1'] = combined['year_month'].apply(lambda x: floor_date(x - pd.Timedelta('30 days')))

# CREATE VARIABLE: year_month_lag2
combined['year_month_lag2'] = combined['year_month_lag1'].apply(lambda x: floor_date(x - pd.Timedelta('30 days')))


#############################
# FLAG OBSERVATIONS TO DROP #
#############################

# CREATE VARIABLE: drop
combined['drop'] = False

# DROP OBSERVATIONS: Observations in TRAINING SET with life_sq > 5000
combined.loc[((combined['subset'] == 'train') & (combined['life_sq'] > 500)), 'drop'] = True


# DROP OBSERVATIONS: Observations in TRAINING/VALIDATION SET with product_type == 'Investment' and price_doc <= 1000000
combined.loc[(combined['subset'] != 'test') & (combined['product_type'] == 'Investment') & (combined['price_doc'] <= 1000000), 'drop'] = True

# DROP OBSERVATIONS: Observations in TRAINING/VALIDATION SET with product_type == 'Investment' and price_doc = 2000000
combined.loc[(combined['subset'] != 'test') & (combined['product_type'] == 'Investment') & (combined['price_doc'] == 2000000), 'drop'] = True

# DROP OBSERVATIONS: Observations in TRAINING/VALIDATION SET with product_type == 'Investment' and price_doc = 3000000
combined.loc[(combined['subset'] != 'test') & (combined['product_type'] == 'Investment') & (combined['price_doc'] == 3000000), 'drop'] = True

print(combined['drop'].sum())  # 2,067

# Save the combined dataset
pd.to_pickle(obj=combined, path='data/interim/combined_clean.pkl')