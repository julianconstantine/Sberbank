from src.functions import rmsle, prepare_submission, split_data

from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.ensemble import GradientBoostingRegressor

import seaborn as sns
import pandas as pd
import numpy as np

import os

#####################
# SCRIPT PARAMETERS #
#####################

ATTEMPT = 14

if not os.path.isdir('src/submissions/sub' + str(ATTEMPT)):
    os.mkdir('src/submissions/sub' + str(ATTEMPT))
else:
    raise ValueError("Submission #%i has already been created" % ATTEMPT)

#########################
# READ/PREPARE DATASETS #
#########################

# Read dataset
# combined = pd.read_pickle(path='data/processed/combined.pkl')
combined = pd.read_pickle(path='data/processed/combined_no_med_replace_with_ts_data.pkl')

# Variables to drop from training data
vars_to_drop = ['id', 'timestamp', 'year_month', 'year_month_lag1', 'year_month_lag2', 'price_doc', 'apartment_id', 'subset']

X_dict, y_dict = split_data(data=combined[combined['drop'] == False], ignore_cols=vars_to_drop, log_y=True)

print(X_dict['train'].shape)  # 21,419 rows x 416 columns
print(X_dict['val'].shape)  # 9,049 rows x 416 columns
print(X_dict['test'].shape)  # 7,662 rows x 416 columns

# Examine distribution
# Note: This has weird tails, so let's try Huber M-regression as Friedman (1999) suggested
sns.distplot(y_dict['train'])


# COARSE-GRID CROSS-VALIDATION
# Perform cross-validation first over a coarse grid
coarse_grid = {
    'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500],
    'max_depth': [2, 4, 6, 8],
    'learning_rate': [0.1, 0.08, 0.06, 0.04, 0.02]
}

# Training/validation split
indices = np.concatenate((np.ones(shape=X_dict['train'].shape[0]), np.zeros(shape=X_dict['val'].shape[0])))

# Use PredefinedSplit to implement grid search on a dedicated validation set
ps = PredefinedSplit(test_fold=indices)

gscv = GridSearchCV(estimator=GradientBoostingRegressor(loss='huber', verbose=True), param_grid=coarse_grid, cv=ps,
                    scoring='neg_mean_squared_error', verbose=True)
gscv.fit(X=X_dict['full'], y=y_dict['full'])

# Best Parameters:
#   learning_rate: 0.02
#   max_depth: 2
#   n_estimators: 100
gscv.best_params_

# Best Score: -0.045
gscv.best_score_

# Save the cross-validation scores
pd.to_pickle(obj=gscv, path='src/submissions/sub' + str(ATTEMPT) + '/gscv_coarse.pkl')


# FINE-GRID CROSS-VALIDATION
# Fine grid for hyperparameter tuning
fine_grid = {
    'n_estimators': [60, 80, 100, 120, 140],
    'max_depth': [2, 3],
    'learning_rate': [0.01, 0.02, 0.03]
}

gscv_fine = GridSearchCV(estimator=GradientBoostingRegressor(loss='huber', verbose=True), param_grid=fine_grid, cv=ps,
                         scoring='neg_mean_squared_error', verbose=True)
gscv_fine.fit(X=X_dict['full'], y=y_dict['full'])

# Best Parameters:
#   learning_rate: 0.01
#   max_depth: 2
#   n_estimators: 60
gscv_fine.best_params_


model = GradientBoostingRegressor(loss='huber', verbose=True, n_estimators=60, learning_rate=0.01, max_depth=2)
model.fit(X=X_dict['train'], y=y_dict['train'])

# DataFrame of most important features
imp_df = pd.DataFrame()
imp_df['variable'] = X_dict['train'].columns
imp_df['importance'] = model.feature_importances_

imp_df = imp_df.sort_values(by='importance', ascending=False)

# Save feature importance scores
imp_df.to_csv('src/submissions/sub' + str(ATTEMPT) + '/importance' + str(ATTEMPT) + '.csv', index=False)

# Get predictions
y_val_pred = model.predict(X_dict['val'])
y_train_pred = model.predict(X_dict['train'])

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

# SUB 11: 0.211
# SUB 14: 0.3926
rmsle(y_true=10**y_dict['val'], y_pred=10**y_val_pred)

# SUB 11: 0.179
# SUB 14: 0.3936
rmsle(y_true=10**y_dict['train'], y_pred=10**y_train_pred)

# Train on full data
model.fit(X=X_dict['full'], y=y_dict['full'])

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



