# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 08:52:47 2023

@author: SDuque
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the path to the data directory
path = 'D:/ISARIC/JAN 23/'

# Load the primary and demographic data
table = 'Internal_SA_20230110.csv'
data_SA = pd.read_csv(os.path.join(path, table))
data_DM = pd.read_csv(os.path.join(path, 'usub_db_sev.csv'))
data_DM_ns = pd.read_csv(os.path.join(path, 'usub_db.csv'))

# Filter primary data to include only severe cases
data_SA = data_SA.loc[data_SA['USUBJID'].isin(data_DM['USUBJID'].unique())]

# Load the description file and filter for included items
descrip = pd.read_excel(os.path.join(path, 'CRF_SA.xlsx'))
descrip = descrip[descrip['Include'] == 'X']

# Merge primary data with demographic data
data_SA = pd.merge(data_SA, data_DM, on='USUBJID', how='left')

# Create 'SAMODIFY2' column and perform necessary replacements
data_SA['SAMODIFY2'] = data_SA['SAMODIFY']
data_SA['SAMODIFY'].loc[data_SA['SAMODIFY2'].isin(['DIABETES MELLITUS - TYPE NOT SPECIFIED'])] = 'DIABETES MELLITUS - TYPE NOT SPECIFIED'
data_SA['SAMODIFY'].loc[data_SA['SAMODIFY2'].isin(['DIABETES MELLITUS - TYPE 1', 'DIABETES MELLITUS - TYPE 2', 'DIABETES MELLITUS - GESTATIONAL'])] = 'DIABETES MELLITUS - TYPE SPECIFIED'

# Identify and adjust for patients with multiple diabetes types specified
a = data_SA.loc[data_SA['SAMODIFY'].isin(['DIABETES MELLITUS - TYPE NOT SPECIFIED', 'DIABETES MELLITUS - TYPE SPECIFIED'])]
usub_no = set(a['USUBJID'].loc[a['SAMODIFY'] == 'DIABETES MELLITUS - TYPE NOT SPECIFIED'].unique())
usub_sp = set(a['USUBJID'].loc[a['SAMODIFY'] == 'DIABETES MELLITUS - TYPE SPECIFIED'].unique())
usub_nosp_sp = usub_no.intersection(usub_sp)
data_SA['SAMODIFY'].loc[(data_SA['USUBJID'].isin(usub_nosp_sp)) & (data_SA['SAMODIFY2'].isin(['DIABETES MELLITUS - TYPE NOT SPECIFIED']))] = 'DIABETES MELLITUS - TYPE SPECIFIED'

# Drop rows with missing 'Extracted SATERM'
descrip.dropna(subset=['Extracted SATERM'], inplace=True)

# Initialize DataFrame for results
df2 = pd.DataFrame()

# Process data for each database site
group_sites = 'DB'
for j in data_DM_ns[group_sites].unique():
    table_counts = []
    table_SATERM = []
    
    data_SA_site = data_SA.loc[data_SA[group_sites] == j]
          
    for i in range(len(descrip)):
        den = len(data_DM['USUBJID'].loc[data_DM[group_sites] == j].unique())
        total = len(data_SA_site['USUBJID'].loc[
            (data_SA_site['SACAT'] == descrip["CAT"].iloc[i]) & 
            (data_SA_site[descrip["TERM"].iloc[i]] == descrip["Extracted SATERM"].iloc[i]) & 
            (data_SA_site['SAOCCUR'].isin(['Y', 'N']))
        ].unique())
        name = descrip['Extracted SATERM'].iloc[i] + '_' + descrip['SUFIX'].iloc[i]
        
        if descrip['SALOC'].iloc[i] == 'X':
            total = len(data_SA_site['USUBJID'].loc[
                (data_SA_site['SALOC'].notna()) & 
                (data_SA_site['SACAT'] == descrip["CAT"].iloc[i]) & 
                (data_SA_site[descrip["TERM"].iloc[i]] == descrip["Extracted SATERM"].iloc[i]) & 
                (data_SA_site['SAOCCUR'].isin(['Y']))
            ].unique())
            den = len(data_SA_site['USUBJID'].loc[
                (data_SA_site['SACAT'] == descrip["CAT"].iloc[i]) & 
                (data_SA_site[descrip["TERM"].iloc[i]] == descrip["Extracted SATERM"].iloc[i]) & 
                (data_SA_site['SAOCCUR'].isin(['Y']))
            ].unique())
            name = descrip['Extracted SATERM'].iloc[i] + '_' + descrip['SUFIX'].iloc[i] + '_LOC'
        
        if descrip['SAOCCUR'].iloc[i] == 'X':
            total = len(data_SA_site['USUBJID'].loc[
                (data_SA_site['SACAT'] == descrip["CAT"].iloc[i]) & 
                (data_SA_site[descrip["TERM"].iloc[i]] == descrip["Extracted SATERM"].iloc[i]) & 
                (data_SA_site['SAOCCUR'].isin(['Y']))
            ].unique())
            name = descrip['Extracted SATERM'].iloc[i] + '_' + descrip['SUFIX'].iloc[i]
        
        if descrip['Cough'].iloc[i] == 'X':
            usubj_num = set(data_SA_site['USUBJID'].loc[
                (data_SA_site['SACAT'] == descrip["CAT"].iloc[i]) & 
                (data_SA_site[descrip["TERM"].iloc[i]] == descrip["Extracted SATERM"].iloc[i]) & 
                (data_SA_site['SAOCCUR'].isin(['Y', 'N']))
            ].unique())
            usubj_den = set(data_SA_site['USUBJID'].loc[
                (data_SA_site['SACAT'] == descrip["CAT"].iloc[i]) & 
                (data_SA_site['SAMODIFY'] == 'COUGH') & 
                (data_SA_site['SAOCCUR'].isin(['Y']))
            ].unique())
            den = len(usubj_den)
            total = len(usubj_num.intersection(usubj_den))  # Convert the result to a set  

        if den == 0:
            table_counts.append(0)
        else:
            table_counts.append(100 * total / den)
        table_SATERM.append(name)
       
    df2['Extracted SATERM'] = table_SATERM
    df2[j] = table_counts

# Set index and calculate statistics
df2.set_index('Extracted SATERM', inplace=True)
numeric_columns = data_DM_ns['DB'].unique()
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
df3.to_excel(os.path.join(path, 'example_db_SA_means_sev.xlsx'), index=False)
