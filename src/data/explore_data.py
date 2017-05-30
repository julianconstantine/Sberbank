import seaborn as sns

import pandas as pd
import numpy as np

# Read datasets
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

macro = pd.read_csv('data/raw/macro.csv')

# 30,471 rows x 292 columns
print(train.shape)

# 7,662 rows x 291 columns
print(test.shape)

# 2,484 rows x 100 columns
print(macro.shape)


########################################
# MANUAL INSPECTION OF BASE PREDICTORS #
########################################

# Create list of training variables
with open('references/train_variables.txt', mode='w') as f:
    f.writelines('Target\n')
    f.writelines('\tprice_doc')
    f.writelines('\n')
    f.writelines('\n')
    f.writelines('Predictors')
    f.writelines('\n')

    for col in train.columns.tolist()[:-1]:
        f.writelines('\t' + col + '\n')


# EXAMINE VARIABLE: price_doc
# This is our target
train['price_doc'].describe()

sns.distplot(train['price_doc'])


# EXAMINE VARIABLE: id
#   Num Missing: 0
train['id'].isnull().sum()

#   Num Unique: 30,471 (all uniquq)
train['id'].nunique()


# EXAMINE VARIABLE: timestamp
#   Num Missing: 0
train['timestamp'].isnull().sum()

# count                   30471
# unique                   1161
# top       2014-12-16 00:00:00
# freq                      160
# first     2011-08-20 00:00:00
# last      2015-06-30 00:00:00
pd.to_datetime(train['timestamp']).describe()


# EXAMINE VARIABLE: full_sq
#   Num Missing: 0
train['full_sq'].isnull().sum()

#   Num Unique: 211
train['full_sq'].nunique()

# count    30471.000000
# mean        54.214269
# std         38.031487
# min          0.000000
# 25%         38.000000
# 50%         49.000000
# 75%         63.000000
# max       5326.000000
train['full_sq'].describe()

sns.distplot(train['full_sq'])


# EXAMINE VARIABLE: life_sq
#   Num Missing: 6,383
train['life_sq'].isnull().sum()

#   Num Unique: 175
train['life_sq'].nunique()

# count    24088.000000
# mean        34.403271
# std         52.285733
# min          0.000000
# 25%         20.000000
# 50%         30.000000
# 75%         43.000000
# max       7478.000000
train['life_sq'].describe()

sns.distplot(train['life_sq'].dropna())


# EXAMINE VARIABLE: floor
#   Num Missing: 167
train['floor'].isnull().sum()

#   Num Unique: 41
train['floor'].nunique()

# count    30304.000000
# mean         7.670803
# std          5.319989
# min          0.000000
# 25%          3.000000
# 50%          6.500000
# 75%         11.000000
# max         77.000000
train['floor'].describe()

sns.distplot(train['floor'].dropna())


# EXAMINE VARIABLE: max_floor
#   Num Missing: 9,572
train['max_floor'].isnull().sum()

#   Num Unique: 49
train['max_floor'].nunique()

# count    20899.000000
# mean        12.558974
# std          6.756550
# min          0.000000
# 25%          9.000000
# 50%         12.000000
# 75%         17.000000
# max        117.000000
train['max_floor'].describe()


# EXAMINE VARIABLE: material
#   Num Missing: 9,572
train['material'].isnull().sum()

#   Num Unique: 6
train['material'].nunique()

#  1.0    14197
# NaN      9572
#  2.0     2993
#  5.0     1561
#  4.0     1344
#  6.0      803
#  3.0        1
train['material'].value_counts(dropna=False)


# EXAMINE VARIABLE: build_year
#   Num Missing: 13,605
train['build_year'].isnull().sum()

#   Num Unique: 119
train['build_year'].nunique()

# A lot of them are coded as impossible numbers
# NaN            13605
#  2014.0          919
#  2015.0          824
#  0.0             530
#  2013.0          464
#  1970.0          418
train['build_year'].value_counts(dropna=False)

sns.distplot(train['build_year'].dropna())


# EXAMINE VARIABLE: num_room
#   Num Missing: 9,572
train['num_room'].isnull().sum()

#   Num Unique: 13
train['num_room'].nunique()

# NaN      9572
#  2.0     8132
#  1.0     7602
#  3.0     4675
#  4.0      418
#  5.0       40
#  0.0       14
#  6.0        9
#  8.0        3
#  10.0       2
#  7.0        1
#  19.0       1
#  9.0        1
#  17.0       1
train['num_room'].value_counts(dropna=False)


# EXAMINE VARIABLE: kitch_sq
#   Num Missing: 9,572
train['kitch_sq'].isnull().sum()

#   Num Unique: 74
train['kitch_sq'].nunique()

# count    20899.000000
# mean         6.399301
# std         28.265979
# min          0.000000
# 25%          1.000000
# 50%          6.000000
# 75%          9.000000
# max       2014.000000
train['kitch_sq'].describe()


# EXAMINE VARIABLE: state
#   Num Missing: 13,559
train['state'].isnull().sum()

#   Num Unique: 5
train['state'].nunique()

# NaN      13559
#  2.0      5844
#  3.0      5790
#  1.0      4855
#  4.0       422
#  33.0        1
train['state'].value_counts(dropna=False)


# EXAMINE VARIABLE: product_type
#   Num Missing: 0
train['product_type'].isnull().sum()

#   Num Unique: 2
train['product_type'].nunique()

# Investment       19448
# OwnerOccupier    11023
train['product_type'].value_counts(dropna=False)


# EXAMINE VARIABLE: sub_area
#   Num Missing: 0
train['sub_area'].isnull().sum()

#   Nim Unique: 146
train['sub_area'].nunique()

# Poselenie Sosenskoe               1776
# Nekrasovka                        1611
# Poselenie Vnukovskoe              1372
# Poselenie Moskovskij               925
# Poselenie Voskresenskoe            713
# ...
train['sub_area'].value_counts()


#######################
# PRICE DISTRIBUTIONS #
#######################

import matplotlib.pyplot as plt

import seaborn as sns

sns.distplot(np.log10(train.loc[train['product_type'] == 'Investment', 'price_doc']))
sns.distplot(np.log10(train.loc[train['product_type'] != 'Investment', 'price_doc']))
