from sklearn.model_selection import GridSearchCV

import xgboost as xgb
import pandas as pd
import numpy as np


def rmsle(y_true, y_pred):
    return np.mean((np.log(y_pred + 1) - np.log(y_true + 1))**2)

# Read dataset
combined = pd.read_pickle(path='data/processed/combined.pkl')

# Split into training, testing, validation
y_train = combined.loc[combined['subset'] == 'train', 'price_doc']
y_val = combined.loc[combined['subset'] == 'val', 'price_doc']
y_full = combined.loc[combined['subset'].isin(['train', 'val']), 'price_doc']

subsets = combined['subset']

combined_binarized = pd.get_dummies(combined.drop(labels=['id', 'apartment_id', 'timestamp', 'year_month', 'price_doc',
                                                          'subset'], axis=1))
combined_binarized['subset'] = subsets

X_train = combined_binarized.loc[combined_binarized['subset'] == 'train'].drop(labels='subset', axis=1)
X_val = combined_binarized.loc[combined_binarized['subset'] == 'val'].drop(labels='subset', axis=1)
X_test = combined_binarized.loc[combined_binarized['subset'] == 'test'].drop(labels='subset', axis=1)

X_full = X_train.append(X_val, ignore_index=True)

print(X_train.shape)  # 21,476 rows x 579 columns
print(X_val.shape)  # 8,995 rows x 579 columns
print(X_test.shape)  # 7,662 rows x 579 columns

# Train basic GBM model
# params = {
#     'objective': 'reg:linear',
#     'eval_metric': 'rmse'
# }

model = xgb.XGBRegressor(objective='reg:linear', learning_rate=0.05, subsample=0.7,
                         colsample_bytree=0.7, max_depth=5, n_estimators=350, silent=False)


# param_grid = {
#     'reg_lambda': [0.05, 1, 2],
#     'max_depth': [4, 6, 8],
#     'n_estimators': [150, 200, 250]
# }
#
# kfcv = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
#
# kfcv.fit(X=X_full, y=y_full)

model.fit(X=X_train, y=y_train)

y_val_pred = model.predict(X_val)

# RMSLE #1: 0.39504
# RMSLE #2: 0.21298 (max_depth=4, n_estimators=200)
# RMSLE #3: 0.21337 (max_depth=4, n_estimators=150)
# RMSLE #4: 0.21315 (max_depth=4, n_estimators=150, id dropped)
# RMSLE #5: 0.21104 (max_depth=5, n_estimators=350, learning_rate=0.05, subsample=0.7, colsample_bytree=0.7, id dropped,
#                    copying parameters from: https://www.kaggle.com/bguberfain/sberbank-russian-housing-market/naive-xgb-lb-0-317/notebook)
rmsle(y_true=y_val, y_pred=y_val_pred)


# Train on full data
model.fit(X=X_full, y=y_full)

y_test_pred = model.predict(X_test)

id_test_list = combined.loc[combined['subset'] == 'test', 'id'].tolist()


def prepare_submission(y_pred, id_list, version):
    # y_test_pred = model.predict(X_test)

    # id_list = X_test['id'].tolist()

    f = open('src/submissions/submission' + str(version) + '.csv', mode='w')
    f.writelines('id,price_doc' + '\n')

    for i in range(len(id_list)):
        id = id_list[i]
        y_i = y_pred[i]

        linestr = str(id) + ',' + str(y_i)

        f.writelines(linestr + '\n')

    f.close()


prepare_submission(y_pred=y_test_pred, id_list=id_test_list, version=5)



