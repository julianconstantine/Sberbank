from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

import seaborn as sns
import xgboost as xgb
import pandas as pd
import numpy as np

import json

def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log(y_pred + 1) - np.log(y_true + 1))**2))

# Read dataset
combined = pd.read_pickle(path='data/processed/combined.pkl')

# Split into training, testing, validation
y_train = combined.loc[combined['subset'] == 'train', 'price_doc']
y_val = combined.loc[combined['subset'] == 'val', 'price_doc']
y_full = combined.loc[combined['subset'].isin(['train', 'val']), 'price_doc']

subsets = combined['subset']

# Variables to drop from training data
vars_to_drop = ['id', 'timestamp', 'year_month', 'year_month_lag1', 'year_month_lag2', 'price_doc', 'price_doc',
                'apartment_id', 'subset']

# Read the SCS and binarize all factor variables or columns with type object
with open('references/scs.json', mode='r') as f:
    scs = json.load(fp=f)

vars_to_binarize = []

for var in scs:
    if var not in vars_to_drop:
        if scs[var]['Type'] == 'factor' or combined[var].dtype == object:
            print(var, combined[var].nunique())
            vars_to_binarize.append(var)


# Binarize the categorical features
combined_binarized = pd.get_dummies(combined.drop(labels=vars_to_drop, axis=1), columns=vars_to_binarize, drop_first=True)

# Make sure everything is a float
for var in combined_binarized.columns:
    combined_binarized[var] = combined_binarized[var].astype(float)

# Add back in the subsets
combined_binarized['subset'] = subsets

X_train = combined_binarized.loc[combined_binarized['subset'] == 'train'].drop(labels='subset', axis=1)
X_val = combined_binarized.loc[combined_binarized['subset'] == 'val'].drop(labels='subset', axis=1)
X_test = combined_binarized.loc[combined_binarized['subset'] == 'test'].drop(labels='subset', axis=1)

X_full = X_train.append(X_val, ignore_index=True)

print(X_train.shape)  # 21,476 rows x 1,329 columns
print(X_val.shape)  # 8,995 rows x 1,329 columns
print(X_test.shape)  # 7,662 rows x 1,329 columns

# Train the GBM model
model = GradientBoostingRegressor(loss='ls', learning_rate=0.01, n_estimators=300, max_depth=8, verbose=True)
model.fit(X=X_train, y=y_train)

# DataFrame of most important features
imp_df = pd.DataFrame()
imp_df['variable'] = X_train.columns
imp_df['importance'] = model.feature_importances_

imp_df = imp_df.sort_values(by='importance', ascending=False)

y_val_pred = model.predict(X_val)
y_train_pred = model.predict(X_train)

# The training/validation y values are basically the same
sns.distplot(y_train)
sns.distplot(y_val)

# Visualize training predictions
sns.distplot(y_train)
sns.distplot(y_train_pred)

# Visualize validation predictions
sns.distplot(y_val)
sns.distplot(y_val_pred)

sns.distplot(np.log(y_val))
sns.distplot(np.log(y_val_pred[y_val_pred >= 1]))

# 0.56123 (learning_rate=0.01, n_estimators=400, max_depth=5)
# 0.56290 (learning_rate=0.01, n_estimators=300, max_depth=8)
rmsle(y_true=y_val, y_pred=y_val_pred)


# Train on full data
model.fit(X=X_full, y=y_full)

y_test_pred = model.predict(X_test)

id_test_list = combined.loc[combined['subset'] == 'test', 'id'].tolist()


def prepare_submission(y_pred, id_list, version):
    f = open('src/submissions/submission' + str(version) + '.csv', mode='w')
    f.writelines('id,price_doc' + '\n')

    for i in range(len(id_list)):
        id = id_list[i]
        y_i = y_pred[i]

        linestr = str(id) + ',' + str(y_i)

        f.writelines(linestr + '\n')

    f.close()


prepare_submission(y_pred=y_test_pred, id_list=id_test_list, version=6)



