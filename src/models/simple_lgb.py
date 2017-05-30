from src.functions import rmsle, prepare_submission, split_data

import lightgbm as lgb
import seaborn as sns
import pandas as pd
import numpy as np

import os

ATTEMPT = 17

if not os.path.isdir('src/submissions/sub' + str(ATTEMPT)):
    os.mkdir('src/submissions/sub' + str(ATTEMPT))
else:
    raise ValueError("Submission #%i has already been created" % ATTEMPT)


#########################
# READ/PREPARE DATASETS #
#########################

# Read dataset
combined = pd.read_pickle(path='data/processed/combined_no_med_replace_with_ts_data.pkl')
# combined = pd.read_pickle(path='data/interim/combined_clean.pkl')

print(combined.shape)   # 38,133 x 361

# Drop bad observations
combined.drop(combined[combined['drop'] == True].index, axis=0, inplace=True)

print(combined.shape)  # 36,066 x 361


# Variables to drop from training data
vars_to_drop = ['id', 'timestamp', 'year_month', 'year_month_lag1', 'year_month_lag2', 'price_doc', 'apartment_id',
                'subset']

X_dict, y_dict = split_data(data=combined, ignore_cols=vars_to_drop, log_y=True)

print(X_dict['train'].shape)  # 21,582 rows x 352 columns
print(X_dict['val'].shape)  # 9,049 rows x 352 columns
print(X_dict['test'].shape)  # 7,662 rows x 352 columns


# LIGHTGBM MODEL
SEED = 19920429

ROUNDS = 450

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting': 'gbdt',
    'learning_rate': 0.04,
    'verbose': 0,
    'num_leaves': 2 ** 5,
    'bagging_fraction': 0.95,
    'bagging_freq': 1,
    'bagging_seed': SEED,
    'feature_fraction': 0.7,
    'feature_fraction_seed': SEED,
    'max_bin': 100,
    'max_depth': 5,
    'num_rounds': ROUNDS
}

train_lgb = lgb.Dataset(X_dict['train'], y_dict['train'])
model = lgb.train(params, train_lgb, num_boost_round=ROUNDS)

# Save feature importance scores
imp_df = pd.DataFrame()
imp_df['variable'] = X_dict['train'].columns
imp_df['importance'] = model.feature_importance()

imp_df = imp_df.sort_values('importance', ascending=False)

imp_df.to_csv('src/submissions/sub' + str(ATTEMPT) + '/importance' + str(ATTEMPT) + '.csv', index=False)


# Get predictions
y_val_pred = model.predict(X_dict['val'])
y_train_pred = model.predict(X_dict['train'])


# SUB 11: 0.47242
# SUB 13: 0.47367
# SUB 15: 0.27376
# SUB 16: 0.27339
# SUB 17: 0.25714
rmsle(y_true=10**y_dict['val'], y_pred=10**y_val_pred)

# SUB 11: 0.37516
# SUB 13: 0.35083
# SUB 15: 0.21902
# SUB 16: 0.21882
# SUB 17: 0.19959
rmsle(y_true=10**y_dict['train'], y_pred=10**y_train_pred)


# The training/validation y values are basically the same
sns.distplot(y_dict['train'])
ax = sns.distplot(y_dict['val'])
ax.set(xlabel='Log Price', ylabel='Density', title='Training/Validation Target Distribution')

# Visualize training predictions
sns.distplot(y_dict['train'])
ax = sns.distplot(y_train_pred)
ax.set(xlabel='Log Price', ylabel='Density', title='Training Distribution/Predictions')

# Visualize validation predictions
sns.distplot(y_dict['val'])
ax = sns.distplot(y_val_pred)
ax.set(xlabel='Log Price', ylabel='Density', title='Validation Distribution/Predictions')


# Train model on full dataset
full_lgb = lgb.Dataset(X_dict['full'], y_dict['full'])
model = lgb.train(params, full_lgb, num_boost_round=ROUNDS)

# Save model parameters
f = open('src/submissions/sub' + str(ATTEMPT) + '/params.txt', mode='w')
f.write(str(model))
f.close()

# Get final predictions
y_test_pred = model.predict(X_dict['test'])

# List of IDs
ids = combined.loc[combined['subset'] == 'test', 'id'].tolist()

# Prepare submission
prepare_submission(y_pred=10**y_test_pred, id_list=ids, version=ATTEMPT)

