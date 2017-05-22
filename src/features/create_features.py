import pandas as pd
import numpy as np

import copy
import json

# Read combined dataset
combined = pd.read_pickle(path='data/interim/combined_clean.pkl')

# Read in SCS
with open('references/scs.json', mode='r') as f:
    scs = json.load(fp=f)


def floor_date(x, unit='month'):
    if unit == 'month':
        y = str(x.year)
        m = str(x.month)

        s = y + '-' + m + '-01'

        return pd.to_datetime(s)
    else:
        raise ValueError("Not yet implemented")


###################
# HELPER FEATURES #
###################

# CREATE VARIABLE: year_month
# timestamp rounded down to the first of each month
combined['year_month'] = combined['timestamp'].apply(lambda x: floor_date(x, unit='month'))

# CREATE VARIABLE: year_month_lag1
combined['year_month_lag1'] = combined['year_month'].apply(lambda x: floor_date(x-pd.Timedelta('30 days')))

# CREATE VARIABLE: year_month_lag2
combined['year_month_lag2'] = combined['year_month_lag1'].apply(lambda x: floor_date(x-pd.Timedelta('30 days')))


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
# MARKET FEATURES #
###################

# CREATE VARIABLE: properties_sold_current_month
sales_per_month_train = combined.loc[combined['subset'] == 'train'].groupby('year_month', as_index=False).size()
sales_per_month_train = sales_per_month_train.reset_index()
sales_per_month_train.columns = ['year_month', 'properties_sold_current_month']

# Merge on variable
combined = pd.merge(combined, sales_per_month_train, on='year_month', how='left')

# Fill NaNs with zeros
combined.loc[pd.isnull(combined['properties_sold_current_month']), 'properties_sold_current_month'] = 0


# CREATE VARIABLE: properties_sold_last_month
sales_per_month_train = combined.loc[combined['subset'] == 'train'].groupby('year_month_lag1', as_index=False).size()
sales_per_month_train = sales_per_month_train.reset_index()
sales_per_month_train.columns = ['year_month_lag1', 'properties_sold_last_month']

# Merge on variable
combined = pd.merge(combined, sales_per_month_train, on='year_month_lag1', how='left')

# Fill NaNs with zeros
combined.loc[pd.isnull(combined['properties_sold_last_month']), 'properties_sold_last_month'] = 0


# CREATE VARIABLE: properties_sold_last2_month
sales_per_month_train = combined.loc[combined['subset'] == 'train'].groupby('year_month_lag2', as_index=False).size()
sales_per_month_train = sales_per_month_train.reset_index()
sales_per_month_train.columns = ['year_month_lag2', 'properties_sold_last2_month']

# Merge on variable
combined = pd.merge(combined, sales_per_month_train, on='year_month_lag2', how='left')

# Fill NaNs with zeros
combined.loc[pd.isnull(combined['properties_sold_last2_month']), 'properties_sold_last2_month'] = 0


# CREATE VARIABLE: avg_sale_price_current_month
avg_price_per_month_train = combined.loc[combined['subset'] == 'train'].groupby('year_month', as_index=False)['price_doc'].mean()
avg_price_per_month_train = avg_price_per_month_train.rename(columns={'price_doc': 'avg_sale_price_current_month'})

# Merge on variable
combined = pd.merge(combined, avg_price_per_month_train, on='year_month', how='left')


# CREATE VARIABLE: avg_sale_price_last_month
avg_price_per_month_train = combined.loc[combined['subset'] == 'train'].groupby('year_month_lag1', as_index=False)['price_doc'].mean()
avg_price_per_month_train = avg_price_per_month_train.rename(columns={'price_doc': 'avg_sale_price_last_month'})

# Merge on variable
combined = pd.merge(combined, avg_price_per_month_train, on='year_month_lag1', how='left')


# CREATE VARIABLE: avg_sale_price_last2_month
avg_price_per_month_train = combined.loc[combined['subset'] == 'train'].groupby('year_month_lag2', as_index=False)['price_doc'].mean()
avg_price_per_month_train = avg_price_per_month_train.rename(columns={'price_doc': 'avg_sale_price_last2_month'})

# Merge on variable
combined = pd.merge(combined, avg_price_per_month_train, on='year_month_lag2', how='left')


##################
# RAION FEATURES #
##################

# CREATE VARIABLE: avg_price_per_raion
avg_raion_prices = combined.loc[combined['subset'] == 'train'].groupby('sub_area', as_index=False)['price_doc'].mean()
avg_raion_prices = avg_raion_prices.rename(columns={'price_doc': 'avg_price_per_raion'})

combined = pd.merge(combined, avg_raion_prices, on='sub_area')


# CREATE VARIABLE: avg_max_floor_per_raion
avg_raion_max_floors = combined.loc[combined['subset'] == 'train'].groupby('sub_area', as_index=False)['max_floor'].mean()
avg_raion_max_floors = avg_raion_max_floors.rename(columns={'max_floor': 'avg_max_floor_per_raion'})

combined = pd.merge(combined, avg_raion_max_floors, on='sub_area', how='left')


# CREATE VARIABLE: avg_price_per_raion_current_month
avg_raion_prices = combined.loc[combined['subset'] == 'train'].groupby(['sub_area', 'year_month'], as_index=False)['price_doc'].mean()
avg_raion_prices = avg_raion_prices.rename(columns={'price_doc': 'avg_price_per_raion_current_month'})

combined = pd.merge(combined, avg_raion_prices, on=['sub_area', 'year_month'], how='left')


# CREATE VARIABLE: avg_price_per_raion_last_month
avg_raion_prices = combined.loc[combined['subset'] == 'train'].groupby(['sub_area', 'year_month_lag1'], as_index=False)['price_doc'].mean()
avg_raion_prices = avg_raion_prices.rename(columns={'price_doc': 'avg_price_per_raion_last_month'})

combined = pd.merge(combined, avg_raion_prices, on=['sub_area', 'year_month_lag1'], how='left')


# CREATE VARIABLE: avg_price_per_raion_last2_month
avg_raion_prices = combined.loc[combined['subset'] == 'train'].groupby(['sub_area', 'year_month_lag2'], as_index=False)['price_doc'].mean()
avg_raion_prices = avg_raion_prices.rename(columns={'price_doc': 'avg_price_per_raion_last2_month'})

combined = pd.merge(combined, avg_raion_prices, on=['sub_area', 'year_month_lag2'], how='left')


#####################
# BUILDING FEATURES #
#####################

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


# CREATE VARIABLE: sales_per_building_current_month
df = combined.loc[combined['subset'] == 'train'].groupby(['apartment_id', 'year_month'], as_index=False).size()
df = df.reset_index()
df.columns = ['apartment_id', 'year_month', 'sales_per_building_current_month']

# Merge together
combined = pd.merge(combined, df, on=['apartment_id', 'year_month'], how='left')

# Replace all NaNs with zeros
combined.loc[pd.isnull(combined['sales_per_building_current_month']), 'sales_per_building_current_month'] = 0


# CREATE VARIABLE: sales_per_building_last_month
df = combined.loc[combined['subset'] == 'train'].groupby(['apartment_id', 'year_month_lag1'], as_index=False).size()
df = df.reset_index()
df.columns = ['apartment_id', 'year_month_lag1', 'sales_per_building_last_month']

# Merge together
combined = pd.merge(combined, df, on=['apartment_id', 'year_month_lag1'], how='left')

# Replace all NaNs with zeros
combined.loc[pd.isnull(combined['sales_per_building_last_month']), 'sales_per_building_last_month'] = 0


# CREATE VARIABLE: sales_per_building_last2_month
df = combined.loc[combined['subset'] == 'train'].groupby(['apartment_id', 'year_month_lag2'], as_index=False).size()
df = df.reset_index()
df.columns = ['apartment_id', 'year_month_lag2', 'sales_per_building_last2_month']

# Merge together
combined = pd.merge(combined, df, on=['apartment_id', 'year_month_lag2'], how='left')

# Replace all NaNs with zeros
combined.loc[pd.isnull(combined['sales_per_building_last2_month']), 'sales_per_building_last2_month'] = 0



# CREATE VARIABLE: avg_sale_price_per_building_current_month
df = combined.loc[combined['subset'] == 'train'].groupby(['apartment_id', 'year_month'], as_index=False)['price_doc'].mean()
df.columns = ['apartment_id', 'year_month', 'avg_price_per_building_current_month']

# Merge together (leave NaNs alone for now)
combined = pd.merge(combined, df, on=['apartment_id', 'year_month'], how='left')


# CREATE VARIABLE: avg_sale_price_per_building_last_month
df = combined.loc[combined['subset'] == 'train'].groupby(['apartment_id', 'year_month_lag1'], as_index=False)['price_doc'].mean()
df.columns = ['apartment_id', 'year_month_lag1', 'avg_price_per_building_last_month']

# Merge together
combined = pd.merge(combined, df, on=['apartment_id', 'year_month_lag1'], how='left')


# CREATE VARIABLE: avg_sale_price_per_building_last2_month
df = combined.loc[combined['subset'] == 'train'].groupby(['apartment_id', 'year_month_lag2'], as_index=False)['price_doc'].mean()
df.columns = ['apartment_id', 'year_month_lag2', 'avg_price_per_building_last2_month']

# Merge together
combined = pd.merge(combined, df, on=['apartment_id', 'year_month_lag2'], how='left')


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


########################
# INFINITE REPLACEMENT #
########################

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

# Save the final dataset
pd.to_pickle(obj=combined, path='data/processed/combined.pkl')