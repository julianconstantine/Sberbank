# USING PARTIAL PCA FOR COLLINEARITY
# Link: https://www.kaggle.com/optidatascience/use-partial-pca-for-collinearity-lb-0-328-w-xgb/comments/notebook

# Load packages
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from scipy.stats import norm
from scipy import stats
from ggplot import *

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd
import numpy as np

import warnings
import bisect

warnings.filterwarnings('ignore')

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', 500)


################
# INTRODUCTION #
################

# Load data
train = pd.read_csv('data/raw/train.csv', parse_dates=['timestamp'])
test = pd.read_csv('data/raw/test.csv', parse_dates=['timestamp'])
macro = pd.read_csv('data/raw/macro.csv', parse_dates=['timestamp'])

print(train.columns)
print(train.shape)
print(test.shape)

# Describe the output field
print(train['price_doc'].describe())

sns.distplot(train['price_doc'])

# Take the logarithm of the target because the distribution is right skewed
train['log_price_doc'] = np.log(train['price_doc'])

print(train['log_price_doc'].describe())

sns.distplot(train['log_price_doc'])


# Merge data into a single dataset
train_copy = train.copy()
train_copy['source'] = 'train'

test_copy = test.copy()
test_copy['source'] = 'test'

all_data = pd.concat((train_copy, test_copy), ignore_index=True)

macro_columns = ['mac_' + c if c != 'timestamp' else 'timestamp' for c in macro.columns]
macro.columns = macro_columns
all_data = all_data.merge(macro, on='timestamp', how='left')

# 38,133 rows x 393 columns
print(all_data.shape)

# Numeric and categorical data types
all_data_dtype = all_data.dtypes
display_nvar = len(all_data.columns)

# NOTE: The to_dict() function converts a pandas directly to a Python dictionary
all_data_dtype_dict = all_data_dtype.to_dict()

# Most variables are numeric, some are object
all_data.dtypes.value_counts()


########################################
# TRANSFORM VARIABLES AND MISSING DATA #
########################################

# This function compares variables in the training/testing set to check for ill-behaved variables

def var_desc(dt, all_data):
    print('--------------------------------------------')

    for c in all_data.columns:
        if all_data[c].dtype == dt:
            t1 = all_data[all_data['source'] == 'train'][c]
            t2 = all_data[all_data['source'] == 'test'][c]

            if dt == 'object':
                f1 = t1[~pd.isnull(t1)].value_counts()
                f2 = t2[~pd.isnull(t2)].value_counts()
            else:
                f1 = t1[~pd.isnull(t1)].describe()
                f2 = t2[~pd.isnull(t2)].describe()

            m1 = t1.isnull().value_counts()
            m2 = t2.isnull().value_counts()

            f = pd.concat((f1, f2), axis=1)
            m = pd.concat((m1, m2), axis=1)

            f.columns = ['train', 'test']
            m.columns = ['train', 'test']

            print(dt + ' - ' + c)
            print('Unique values: ', len(t1.value_counts()), len(t2.value_counts()))
            print(f.sort_values(by='train', ascending=False))
            print()

            m_print = m[m.index == True]

            if len(m_print) > 0:
                print('missing - ' + c)
                print(m_print)
            else:
                print('NO Missing values - ' + c)
            if dt != "object":
                if len(t1.value_counts()) <= 10:
                    c1 = t1.value_counts()
                    c2 = t2.value_counts()
                    c = pd.concat([c1, c2], axis=1)
                    f.columns = ['train', 'test']
                    print(c)
            print('--------------------------------------------')

# Uncomment to run variable description
var_desc(dt='object', all_data=all_data)

# Convert the yes/no object variables to 1/0
for c in all_data.columns:
    if all_data[c].dtype == 'object' and c not in ['sub_area', 'timestamp', 'source']:
        if len(all_data[c].value_counts()) == 2:
            all_data['num_' + c] = [0 if x in ['no','OwnerOccupier'] else 1 for x in all_data[c]]
        if len(all_data[c].value_counts()) == 5:
            all_data['num_' + c] = 0
            all_data['num_' + c].loc[all_data[c] == 'poor'] = 0
            all_data['num_' + c].loc[all_data[c] == 'satisfactory'] = 1
            all_data['num_' + c].loc[all_data[c] == 'good'] = 2
            all_data['num_' + c].loc[all_data[c] == 'excellent'] = 3
            all_data['num_' + c].loc[all_data[c] == 'no data'] = 1

# Missing values
missing_col = [[c, sum(all_data[all_data['source'] == 'train'][c].isnull()), sum(all_data[all_data['source'] == 'test'][c].isnull())] for c in all_data.columns]
missing_col = pd.DataFrame(missing_col, columns=['var', 'missing_train', 'missing_test'])

# Plot the number of missing values per column
missing_df = missing_col[(missing_col['missing_train'] + missing_col['missing_test']) > 0]
missing_df = missing_df.sort_values('missing_train')

# Create the plot
f, ax = plt.subplots(figsize=(6, 15))
sns.barplot(y=missing_df['var'], x=missing_df['missing_train'])


#################################
# PRINCIPAL COMPONENTS ANALYSIS #
#################################

# Group data into small categories than apply PCA to each category

# Columns to exclude
exclude_columns = ['id', 'timestamp', 'sub_area'] + [c for c in all_data.columns if all_data[c].dtype == 'object']

# Columns to reserve
reserve_columns = ['price_doc', 'log_price_doc', 'source', 'cafe_sum_500_max_price_avg', 'cafe_sum_500_min_price_avg',
                   'cafe_avg_price_500', 'hospital_beds_raion']


def select_group(keys):
    list_all = list()

    for k in keys:
        l = [c for c in all_data.columns if c.find(k) != -1 and c not in exclude_columns and c not in reserve_columns]
        l = list(set(l))
        list_all += l

    return list_all

column_groups = dict()

column_groups['people'] = select_group(keys=['_all', 'male'])
column_groups['id'] = select_group(keys=['ID_'])
column_groups['church'] = select_group(keys=['church'])
column_groups['build'] = select_group(keys=['build_count_'])
column_groups['cafe'] = select_group(keys=['cafe_count'])
column_groups['cafeprice'] = select_group(keys=['cafe_sum', 'cafe_avg'])
column_groups['km'] = select_group(keys=['_km', 'metro_min', '_avto_min', '_walk_min', '_min_walk'])
column_groups['mosque'] = select_group(keys=['mosque_count'])
column_groups['market'] = select_group(keys=['market_count'])
column_groups['office'] = select_group(keys=['office_count'])
column_groups['leisure'] = select_group(keys=['leisure_count'])
column_groups['sport'] = select_group(keys=['sport_count'])
column_groups['green'] = select_group(keys=['green_part'])
column_groups['prom'] = select_group(keys=['prom_part'])
column_groups['trc'] = select_group(keys=['trc_count'])
column_groups['sqm'] = select_group(keys=['_sqm_'])
column_groups['raion'] = select_group(keys=['_raion'])
column_groups['macro'] = select_group(keys=['mac_'])
column_groups.keys()

col_temp = list()

for d in column_groups:
    col_temp += column_groups[d]

column_groups['other'] = [c for c in all_data.columns if c not in col_temp and c not in exclude_columns and c not in reserve_columns]

# These columns are to be excluded from PCA
print(column_groups['other'])

# Remove variables in macro data with too many missing columns
macro_missing_2 = pd.DataFrame([[c, sum(all_data[c].isnull())] for c in column_groups['macro']],
                               columns=['var', 'missing'])

macro_missing_3 = macro_missing_2[macro_missing_2['missing'] > 5000]
print(macro_missing_3)
exclude_columns += list(macro_missing_3['var'].tolist())
print(exclude_columns)

column_groups['macro'] = select_group(['mac_'])

loopkeys = list(column_groups.keys())
print(loopkeys)


# Perform partial PCA on each of the column groups
def partial_pca(var, data, column_groups):
    pca = PCA()

    df = data[column_groups[var]].dropna()

    print([len(data[column_groups[var]]), len(df)])

    # Normalize: subtract mean and divide by variance
    df = (df - df.mean()) / df.std(ddof=0)
    pca.fit(df)

    varexp = pca.explained_variance_ratio_.cumsum()
    cutoff = bisect.bisect(varexp, 0.95)

    newcol = pd.DataFrame(pca.fit_transform(X=df)[:, 0:(cutoff + 1)],
                          columns=['PCA_' + var + '_' + str(i) for i in range(cutoff + 1)], index=df.index)
    # print(newcol)
    column_groups['PCA_' + var] = list(newcol.columns)

    return newcol, column_groups, pca

for c in loopkeys:
    if c != 'other':
        print(c)
        newcol, column_groups, pca = partial_pca(var=c, data=all_data, column_groups=column_groups)
        all_data = all_data.join(newcol)
        print(all_data.shape)

# 38,133 row x 509 columns
print(all_data.shape)


###############
# CORRELATION #
###############


