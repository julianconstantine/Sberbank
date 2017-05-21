import pandas as pd
import numpy as np

# Load raw datasets
train = pd.read_csv('data/raw/train.csv', parse_dates=['timestamp'])
test = pd.read_csv('data/raw/test.csv', parse_dates=['timestamp'])
macro = pd.read_csv('data/raw/macro.csv', parse_dates=['timestamp'])

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


# Save the combined dataset
pd.to_pickle(obj=combined, path='data/interim/combined.pkl')