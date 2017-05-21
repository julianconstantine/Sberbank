import pandas as pd
import numpy as np

import copy

# Read combined dataset
combined = pd.read_pickle(path='data/interim/combined.pkl')


def floor_date(x, unit='month'):
    if unit == 'month':
        y = str(x.year)
        m = str(x.month)

        s = y + '-' + m + '-01'

        return pd.to_datetime(s)
    else:
        raise ValueError("Not yet implemented")


def med_replace(x):
    z = copy.copy(x)
    z[pd.isnull(z)] = np.median(x[~pd.isnull(x)])

    return z


###################
# HELPER FEATURES #
###################

# CREATE VARIABLE: year_month
# timestamp rounded down to the first of each month
combined['year_month'] = combined['timestamp'].apply(lambda x: floor_date(x, unit='month'))


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


###################
# BUILDING FEATURES
###################

# CREATE VARIABLE: apartment_id
# Use raion (sub_area) + distance from metro station to infer apartment building identity
# Link: https://www.kaggle.com/c/sberbank-russian-housing-market/discussion/33269
combined['apartment_id'] = (combined['sub_area'] + combined['metro_km_avto'].astype(str))

# Alternatives for inferring apartment_id (I think the original way is fine)
temp = combined.groupby(['metro_km_avto', 'build_year']).size()
temp.shape  # 13,626

temp = combined.groupby(['metro_km_avto', 'school_km']).size()
temp.shape  # 13,640

temp = combined.groupby(['metro_km_avto', 'school_km', 'catering_km']).size()
temp.shape  # 13,640

temp = combined.groupby(['metro_km_avto', 'school_km', 'exhibition_km']).size()
temp.shape  # 13,640

temp = combined.groupby(['sub_area', 'metro_km_avto', 'school_km', 'catering_km', 'exhibition_km']).size()
temp.shape  # 13,640

temp = combined.groupby(['sub_area', 'metro_km_avto', 'school_km', 'catering_km', 'exhibition_km']).size()
temp.shape  # 13,640


# CREATE VARIABLE: sales_per_building_per_month
df = combined.loc[combined['subset'] == 'train'].groupby(['apartment_id', 'year_month'], as_index=False).size()
df = df.reset_index()
df.columns = ['apartment_id', 'year_month', 'sales_per_building_per_month']

# Merge together
combined = pd.merge(combined, df, on=['apartment_id', 'year_month'], how='left')

# Replace all NaNs with zeros
combined.loc[pd.isnull(combined['sales_per_building_per_month']), 'sales_per_building_per_month'] = 0


# CREATE VARIABLE: avg_sale_price_per_building_per_month
df = combined.loc[combined['subset'] == 'train'].groupby(['apartment_id', 'year_month'], as_index=False)['price_doc'].mean()
df.columns = ['apartment_id', 'year_month', 'avg_price_per_building_per_month']

# Merge together (leave NaNs alone for now)
combined = pd.merge(combined, df, on=['apartment_id', 'year_month'], how='left')


###################
# MARKET FEATURES #
###################

# CREATE VARIABLE: properties_sold_per_month
# Compute only on the training data!
sales_per_month_train = combined.loc[combined['subset'] == 'train'].groupby('year_month', as_index=False).size()
sales_per_month_train = sales_per_month_train.reset_index()
sales_per_month_train.columns = ['year_month', 'properties_sold_per_month']

# Merge on variable
combined = pd.merge(combined, sales_per_month_train, on='year_month', how='left')

# Fill NaNs with zeros
combined.loc[pd.isnull(combined['properties_sold_per_month']), 'properties_sold_per_month'] = 0


# CREATE VARIABLE: avg_sale_price_per_month
avg_price_per_month_train = combined.loc[combined['subset'] == 'train'].groupby('year_month', as_index=False)['price_doc'].mean()
avg_price_per_month_train = avg_price_per_month_train.rename(columns={'price_doc': 'avg_sale_price_per_month'})

# Merge on variable
combined = pd.merge(combined, avg_price_per_month_train, on='year_month', how='left')

# Median replace all missing values
combined['avg_sale_price_per_month'] = med_replace(x=combined['avg_sale_price_per_month'])


# CREATE VARIABLE: med_sale_price_per_month
avg_price_per_month_train = combined.loc[combined['subset'] == 'train'].groupby('year_month', as_index=False)['price_doc'].median()
avg_price_per_month_train = avg_price_per_month_train.rename(columns={'price_doc': 'med_sale_price_per_month'})

# Merge on variable
combined = pd.merge(combined, avg_price_per_month_train, on='year_month', how='left')

# Median replace all missing values
combined['med_sale_price_per_month'] = med_replace(x=combined['med_sale_price_per_month'])


##################
# RAION FEATURES #
##################

# CREATE VARIABLE: distance_from_kremlin
# Latitude/longitude of the Kremlin
# kremlin = (55.7520, 37.6175)


# CREATE VARIABLE: avg_price_per_raion
avg_raion_prices = combined.groupby('sub_area', as_index=False)['price_doc'].mean()
avg_raion_prices = avg_raion_prices.rename(columns={'price_doc': 'avg_price_per_raion'})

combined = pd.merge(combined, avg_raion_prices, on='sub_area')


# CREATE VARIABLE: avg_price_per_sqkm_per_raion
# avg_raion_prices = combined.groupby('sub_area', as_index=False)['price_doc'].mean()
# avg_raion_prices = avg_raion_prices.rename(columns={'price_doc': 'avg_price_per_raion'})

# combined = pd.merge(combined, avg_raion_prices, on='sub_area')


# CREATE VARIABLE: avg_max_floor_per_raion
avg_raion_max_floors = combined.groupby('sub_area', as_index=False)['max_floor'].mean()
avg_raion_max_floors = avg_raion_max_floors.rename(columns={'max_floor': 'avg_max_floor_per_raion'})

combined = pd.merge(combined, avg_raion_max_floors, on='sub_area')


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
combined['elder_proportion'] = combined['ekder_all']/combined['full_all']


# Save the final dataset
pd.to_pickle(obj=combined, path='data/processed/combined.pkl')