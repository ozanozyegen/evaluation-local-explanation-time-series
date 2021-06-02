# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import scipy as sp
import sys, os
from scipy import stats
from datetime import datetime
from data.helpers import week_of_month


# %%
DATA_DIR = 'data/raw/walmart/'
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
covariates = pd.read_csv(os.path.join(DATA_DIR, 'features.csv'))
stores = pd.read_csv(os.path.join(DATA_DIR, 'stores.csv')) # 45 Store types and sizes


# %%
covariates['Temperature'] = covariates['Temperature'].astype('float32')
covariates['Fuel_Price'] = covariates['Fuel_Price'].astype('float32')
covariates['Unemployment'] = covariates['Unemployment'].astype('float32')
covariates['CPI'] = covariates['CPI'].astype('float32')
covariates['Date'] = pd.to_datetime(covariates['Date'])
covariates['IsHoliday'] = covariates['IsHoliday'].astype('int')


# %%
train['Date'] = pd.to_datetime(train['Date'])
train['year'] = train['Date'].dt.year
train['month'] = train['Date'].dt.month
train['weekofmonth'] = train['Date'].apply(week_of_month)
train['day'] = train['Date'].dt.day
train.drop('IsHoliday', axis=1, inplace=True)


# %%
mapping = {'A':1, 'B':2, 'C':3}
stores['Type'].replace(mapping, inplace=True)


# %%
data = pd.merge(train, stores, how='left', left_on='Store', right_on='Store')
data = pd.merge(data, covariates, how='left', left_on=['Date', 'Store'], right_on=['Date', 'Store'])


# %%
print(data.isna().any())


# %%
def LowCorr_red(val):
    color = 'salmon' if np.abs(val) <= 0.2 else ''
    return 'background-color:' + color  

corr_table = data.corr().style.applymap(LowCorr_red)
corr_table

# %% [markdown]
# It at least seems like that there is no noticeable linear relationship between Markdown and weeklysales. Although this does not give us any clue regarding possible non-linear relationship between those features, for the sake of simplicity fo analysis and to deal with Missing data, we decided to drop those columns.
# 
# Due to future blindness assumption, beside Fuel_Price and Temperature feature which are known to be somewhat predictable, we drop all not foreseeable features.

# %%
data = data.drop(['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'], axis=1)

# %% [markdown]
# ### Check missing data for departments of stores

# %%
def Nan_red(val):
    color = 'salmon' if np.isnan(val) else ''
    return 'background-color:' + color  


# %%
counting_type = data.groupby(['Store', 'Dept']).count()[["Date"]]
counting_type["Counter"] = counting_type["Date"]
counting_type = counting_type[["Counter"]]
pivotted = pd.pivot_table(counting_type, values='Counter', index='Store', columns='Dept')


# %%
pivotted = pivotted.style.applymap(Nan_red)


# %%
pivot_reset = pd.DataFrame(data = pd.pivot_table(counting_type, values='Counter', index='Store', columns='Dept')).reset_index(drop=True)
pivot_reset = pd.DataFrame(data = pivot_reset.reset_index(drop = True).values)

# %% [markdown]
# - Complete series have 143 steps, 73% of the data is complete. We will only use the complete data

# %%
Full_TS_Indices = np.where(pivot_reset >= 142.9) or np.where(np.isnan(pivot_reset))
Full_TS_Indices = np.asarray([np.asarray(pivotted.index)[Full_TS_Indices[0]], np.asarray(pivotted.columns)[Full_TS_Indices[1]]])
Full_TS_Indices = pd.DataFrame(data = Full_TS_Indices.T, columns = ["Store", "Dept"])


# %%
# Drop store_dept with missing values
data = pd.merge(data, Full_TS_Indices, left_on=['Store', 'Dept'], right_on=['Store', 'Dept'], how='right')


# %%
# Put Weekly Sales to the beginning
data = data.reindex(columns=['Weekly_Sales'] + list([a for a in data.columns if a != 'Weekly_Sales']))


# %%
data.columns

# %%
data.sort_values(by='Date', inplace=True)
data.to_csv('data/processed/walmart.csv')



# %%
