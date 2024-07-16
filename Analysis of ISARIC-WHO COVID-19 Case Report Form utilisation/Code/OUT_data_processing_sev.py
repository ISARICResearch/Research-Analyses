# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:20:20 2023

@author: SDuque
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the path to the data directory
path = 'D:/ISARIC/JAN 23/'

# Load the descriptions and data files
descrip = pd.read_excel(os.path.join(path, 'CRF_outcomes.xlsx'))
data_HO = pd.read_csv(os.path.join(path, 'Internal_HO_2023-01-10.csv'))
data_DM = pd.read_csv(os.path.join(path, 'usub_db_sev.csv'))
data_DM_ns = pd.read_csv(os.path.join(path, 'usub_db.csv'))
data_DS = pd.read_csv(os.path.join(path, 'Internal_DS_2023-01-10.csv'))

# Modify 'HOEVINTX2' based on conditions
data_HO['HOEVINTX2'] = data_HO['HOEVINTX']
data_HO['HOEVINTX2'].loc[(data_HO['HOEVINTX'].isin(['AT ANY TIME DURING HOSPITALIZATION', 'AT ANY TIME']) & data_HO['HOTPT'].isna())] = 'any'
data_HO['HOEVINTX2'].loc[((data_HO['HOEVINTX'].isin(['00:00-24:00 ON DAY OF ASSESSMENT'])) | data_HO['HOEVINTX'].isna()) & data_HO['HOTPT'].isna()] = 'daily'

# Process 'data_DS' to make necessary modifications
data_DS['DSTERM'] = data_DS['DSTERM'].str.upper()

# Merge datasets
data_DS_HO = pd.merge(data_DS[['USUBJID', 'DSTERM', 'DSDECOD', 'DSSTDTC']],
                      data_HO[['USUBJID', 'HODECOD', 'HOOCCUR', 'HOSTDTC', 'HOENDTC', 'HODISOUT', 'SELFCARE', 'HOEVINTX2', 'HOENDY']],
                      on='USUBJID', how='outer')

data_DS_HO = pd.merge(data_DS_HO, data_DM, on='USUBJID', how='left')
data_HO = pd.merge(data_HO, data_DM, on='USUBJID', how='left')

# Identify ICU patients
ICU_PATIENTS = data_HO['USUBJID'].loc[(data_HO['HODECOD'] == 'INTENSIVE CARE UNIT') & (data_HO['HOOCCUR'] == 'Y')].unique()

# Initialize DataFrame to store results
df2 = pd.DataFrame()

# Iterate over unique sites
for j in data_DM_ns['DB'].unique():
    table_counts = []
    table_SATERM = []
    
    data_HO_site = data_DS_HO.loc[data_DS_HO['DB'] == j]
    
    # Iterate over the description rows
    for i in range(len(descrip)):
        den = len(data_DM['USUBJID'].loc[data_DM['DB'] == j].unique())
        
        if descrip['term'].iloc[i] == 'INTENSIVE CARE UNIT':
            total = len(data_HO_site['USUBJID'].loc[(data_HO_site['HODECOD'] == 'INTENSIVE CARE UNIT') & (data_HO_site['HOOCCUR'].isin(['Y', 'N'])) & (data_HO_site['HOEVINTX2'] == descrip['when'].iloc[i])].unique())
        elif pd.isna(descrip['term'].iloc[i]):
            total = len(data_HO_site['USUBJID'].loc[data_HO_site[descrip['var'].iloc[i]].notna()].unique())
        elif descrip['Dependant'].iloc[i] == 'icu':
            total = len(data_HO_site['USUBJID'].loc[(data_HO_site[descrip['var'].iloc[i]].notna()) & (data_HO_site['HODECOD'] == 'INTENSIVE CARE UNIT') & (data_HO_site['HOOCCUR'] == 'Y') & (data_HO_site['HOEVINTX2'] == descrip['when'].iloc[i])].unique())
            den = len(data_HO_site['USUBJID'].loc[(data_HO_site['HODECOD'] == 'INTENSIVE CARE UNIT') & (data_HO_site['HOOCCUR'] == 'Y') & (data_HO_site['HOEVINTX2'] == descrip['when'].iloc[i])].unique())
        elif descrip['Dependant'].iloc[i] == 'death':
            total = len(data_HO_site['USUBJID'].loc[(data_HO_site[descrip['var'].iloc[i]] == descrip['term'].iloc[i]) & (data_HO_site['DSDECOD'] != 'DEATH') & (data_HO_site['DSDECOD'] != 'LOST TO FOLLOW-UP')].unique())
            den = len(data_HO_site['USUBJID'].loc[(data_HO_site['DSDECOD'] != 'DEATH') & (data_HO_site['DSDECOD'] != 'LOST TO FOLLOW-UP')].unique())  # Alive patients
        elif descrip['Dependant'].iloc[i] == 'selfcare':
            total = len(data_HO_site['USUBJID'].loc[(data_HO_site[descrip['var'].iloc[i]] == descrip['term'].iloc[i]) & (data_HO_site['HODISOUT'] == 'HOME, SELF CARE') & (data_HO_site['DSDECOD'] != 'DEATH') & (data_HO_site['DSDECOD'] != 'LOST TO FOLLOW-UP')].unique())
            den = len(data_HO_site['USUBJID'].loc[(data_HO_site['HODISOUT'] == 'HOME, SELF CARE') & (data_HO_site['DSDECOD'] != 'DEATH') & (data_HO_site['DSDECOD'] != 'LOST TO FOLLOW-UP')].unique())  # Alive patients
        
        if den == 0:
            table_counts.append(0)
        else:
            table_counts.append(100 * total / den)
        table_SATERM.append(str(descrip['name'].iloc[i]))
    
    df2['Extracted TERM'] = table_SATERM
    df2[j] = table_counts

# Set index and calculate statistics
df2.set_index('Extracted TERM', inplace=True)
numeric_columns = data_DM['DB'].unique()
df2 = df2[numeric_columns]
df2['mean'] = df2[numeric_columns].mean(axis=1)

df3 = df2.copy()
df3 = df3.reset_index()
df3['mean(excluding 0)'] = np.zeros(len(df3))

df2[(df2 == 0)] = np.nan
df3['mean(excluding 0)'] = df2[numeric_columns].mean(axis=1)
df3['% sites included'] = (100 * (df2[numeric_columns] > 0).sum(axis=1)) / len(numeric_columns)
df3['Coefficient_of_Variation'] = df3[numeric_columns].std(axis=1) / df3[numeric_columns].mean(axis=1)

# Save the final results to an Excel file
df3.to_excel(os.path.join(path, 'example_db_DSHO_means_sev.xlsx'), index=False)
