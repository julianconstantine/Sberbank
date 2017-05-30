from src.functions import floor_date, prepare_ts

import pandas as pd
import numpy as np

import json


#####################
# SCRIPT PARAMETERS #
#####################

# Flag to replace infinite values
REPLACE_INFINITE = True

# Method to replace infinite values
REPLACE_INFINITE_METHOD = 'Null'

# Flag to replace null values
REPLACE_NULL_NUMERIC = False

# Method to replace null values
REPLACE_NULL_METHOD = 'Median'

# Flag to create categorical dummies
CREATE_CATEGORICAL_DUMMIES = False

# Flag to encode categorical variables with numeric values
ENCODE_CATEGORICAL = True

# Flag to include time-series data
INCLUDE_TS_DATA = True


#################
# LOAD DATASETS #
#################

# Read combined dataset
combined = pd.read_pickle(path='data/interim/combined_clean.pkl')

# Read in SCS
with open('references/scs.json', mode='r') as f:
    scs = json.load(fp=f)


#####################
# PROPERTY FEATURES #
#####################

# CREATE VARIABLE: year
combined['year'] = combined['timestamp'].apply(lambda x: x.year)


# CREATE VARIABLE: month
combined['month'] = combined['timestamp'].apply(lambda x: x.month)


# CREATE VARIABLE: day_of_week
combined['day_of_week'] = combined['timestamp'].apply(lambda x: x.dayofweek)


# CREATE VARIABLE: day_of_month
combined['day_of_month'] = combined['timestamp'].apply(lambda x: x.day)


# CREATE VARIABLE: floors_from_top
combined['floors_from_top'] = combined['max_floor'] - combined['floor']


# CREATE VARIABLE: relative_floor
combined['relative_floor'] = combined['floor']/combined['max_floor']


# CREATE VARIABLE: avg_room_size
combined['avg_room_size'] = (combined['life_sq'] - combined['kitch_sq'])/combined['num_room']


# CREATE VARIABLE: life_proportion
combined['life_proportion'] = combined['life_sq']/combined['full_sq']


# CREATE VARIABLE: kitch_proportion
combined['kitch_proportion'] = combined['kitch_sq']/combined['full_sq']


# CREATE VARIABLE: extra_sq
combined['extra_sq'] = combined['full_sq'] - combined['life_sq'] - combined['kitch_sq']


# CREATE VARIABLE: age_at_sale
combined['age_at_sale'] = combined['timestamp'].apply(lambda x: x.year) - combined['build_year']


# CREATE VARIABLE: avg_max_floor_per_raion
avg_raion_max_floors = combined.loc[combined['subset'] == 'train'].groupby('sub_area', as_index=False)['max_floor'].mean()
avg_raion_max_floors = avg_raion_max_floors.rename(columns={'max_floor': 'avg_max_floor_per_raion'})

combined = pd.merge(combined, avg_raion_max_floors, on='sub_area', how='left')


# CREATE VARIABLE: apartment_id
# Use raion (sub_area) + distance from metro station to infer apartment building identity
# Link: https://www.kaggle.com/c/sberbank-russian-housing-market/discussion/33269
combined['apartment_id'] = (combined['sub_area'] + combined['metro_km_avto'].astype(str))


###########################
# TIME-DEPENDENT FEATURES #
###########################

# Time-dependent features are aggregated at three levels:
#   1. Building level
#   2. Raion level
#   3. Market level (city level)

if INCLUDE_TS_DATA:
    city_ts = prepare_ts(data=combined, level='city')
    raion_ts = prepare_ts(data=combined, level='raion')
    building_ts = prepare_ts(data=combined, level='building')

    # Merge on time-dependent features
    combined = pd.merge(combined, city_ts, on='year_month')
    combined = pd.merge(combined, raion_ts, on=['year_month', 'sub_area'])
    combined = pd.merge(combined, building_ts, on=['year_month', 'apartment_id'])


#############################
# RAION POPULATION FEATURES #
#############################

# CREATE VARIABLE: population_density
# Population density of the raion
combined['population_density'] = combined['raion_popul']/combined['area_m']


# CREATE VARIABLE: young_proportion
combined['young_proportion'] = combined['young_all']/combined['full_all']


# CREATE VARIABLE: work_proportion
combined['work_proportion'] = combined['work_all']/combined['full_all']


# CREATE VARIABLE: elder_proportion
combined['elder_proportion'] = combined['elder_all']/combined['full_all']


# CREATE VARIABLE: sex_ratio_all
combined['sex_ratio_all'] = combined['male_f']/combined['female_f']


# CREATE VARIABLE: sex_ratio_young
combined['sex_ratio_young'] = combined['young_male']/combined['young_female']


# CREATE VARIABLE: sex_ratio_work
combined['sex_ratio_work'] = combined['work_male']/combined['work_female']


# CREATE VARIABLE: sex_ratio_elder
combined['sex_ratio_elder'] = combined['elder_male']/combined['elder_female']


# CREATE VARIABLE: preschool_ratio
combined['preschool_ratio'] = combined['children_preschool']/combined['preschool_quota']


# CREATE VARIABLE: school_ratio
combined['school_ratio'] = combined['children_school']/combined['school_quota']


# CREATE VARIABLE: school_centers_per_capita
combined['school_centers_per_capita'] = combined['children_school']/combined['school_education_centers_raion']


# CREATE VARIABLE: preschool_centers_per_capita
combined['preschool_centers_per_capita'] = combined['children_preschool']/combined['preschool_education_centers_raion']


# Some plots
# out = combined.groupby(['year_month', 'product_type'], as_index=False).agg({'price_doc': np.mean})
# sns.pointplot(x='year_month', y='price_doc', hue='product_type', data=out)

# out = combined.groupby(['year_month', 'product_type']).size().reset_index()
# out.columns = ['year_month', 'product_type', 'size']
# sns.pointplot(x='year_month', y='size', hue='product_type', data=out)

# out = combined.loc[(combined['product_type'] != 'Investment') | (combined['price_doc'] >= 1000000)].groupby(['year_month', 'product_type'], as_index=False).agg({'price_doc': np.mean})
# sns.pointplot(x='year_month', y='price_doc', hue='product_type', data=out)


########################
# CATEGORICAL FEATURES #
########################

# 38,130 x 360
print(combined.shape)

# Categorical features to ignore
ignore_vars = ['id', 'subset', 'apartment_id']

categorical_features = []

# Get list of categorical features
for var in scs:
    if var in combined.columns and var not in ignore_vars:
        if scs[var]['Type'] == 'factor' or combined[var].dtype == object:
            print(var, combined[var].nunique())
            categorical_features.append(var)

# Add a "missing" category to categorical features that are null
for var in categorical_features:
    combined.loc[combined[var].isnull(), var] = 'missing'

if CREATE_CATEGORICAL_DUMMIES:
    # Create dummies for every categorical variable
    combined = pd.get_dummies(data=combined, columns=categorical_features, drop_first=True)

    # Make sure everything is a float
    for var in combined:
        combined[var] = combined[var].astype(float)

if ENCODE_CATEGORICAL:
    # Go through and encode each variable categorical variable as a number
    for var in categorical_features:
        combined[var + '_numeric'] = pd.factorize(combined[var])[0]

# Drop the original categorical variables
combined.drop(labels=categorical_features, axis=1, inplace=True)

# 38,130 x 360
print(combined.shape)


########################
# INFINITE REPLACEMENT #
########################

if REPLACE_INFINITE:
    if REPLACE_INFINITE_METHOD == 'Null':
        # Replace infinite values with NaNs
        for var in combined.columns:
            # Replace any infinite values with NaNs
            try:
                combined.loc[combined[var] == np.inf, var] = np.nan
                combined.loc[combined[var] == -np.inf, var] = np.nan
            except TypeError:
                pass


####################
# NULL REPLACEMENT #
####################

if REPLACE_NULL_NUMERIC:
    if REPLACE_NULL_METHOD == 'Median':
        # Median replace all null values and create dummies indicating where the variable was missing; if variable is
        # categorical, just add a 'missing' category
        for var in combined.columns:
            if combined[var].isnull().any():
                if var in scs:
                    if scs[var]['Type'] == 'factor':
                        print("Adding 'missing' category to %s" % var)
                        combined.loc[pd.isnull(combined[var]), var] = 'missing'
                    else:
                        print("Median-replacing null values of %s" % var)
                        combined[var + '_null'] = pd.isnull(combined[var]).astype(float)

                        med = np.median(combined.loc[(combined['subset'] == 'train') & ~pd.isnull(combined[var]), var])
                        combined.loc[pd.isnull(combined[var]), var] = med
                else:
                    # If not in SCS, assume it's numeric
                    print("Median-replacing null values of %s" % var)
                    combined[var + '_null'] = pd.isnull(combined[var]).astype(float)

                    med = np.median(combined.loc[(combined['subset'] == 'train') & ~pd.isnull(combined[var]), var])
                    combined.loc[pd.isnull(combined[var]), var] = med


DATASET_NAME = 'combined'

# Save the final dataset
if not REPLACE_NULL_NUMERIC:
    DATASET_NAME += '_no_med_replace'

if INCLUDE_TS_DATA:
    DATASET_NAME += '_with_ts_data'

DATASET_NAME += '.pkl'

# Save the data
pd.to_pickle(obj=combined, path='data/processed/' + DATASET_NAME)