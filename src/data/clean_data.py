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

# Merge on macroeconomic data
combined = pd.merge(combined, macro, on='timestamp')

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


# Save the combined dataset
pd.to_pickle(obj=combined, path='data/interim/combined_clean.pkl')