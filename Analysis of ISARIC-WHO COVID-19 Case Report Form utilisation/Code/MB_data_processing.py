# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 10:49:15 2023

@author: SDuque
"""

import os
import pandas as pd
import numpy as np

# Define the path and file names
path = 'D:/ISARIC/JAN 23/'
table = 'Internal_MB_2023-01-10.csv'
data_dm_file = 'usub_db.csv'
descrip_file = 'CRF_MB.xlsx'

# Read data
data_MB = pd.read_csv(os.path.join(path, table))
data_DM = pd.read_csv(os.path.join(path, data_dm_file))
descrip = pd.read_excel(os.path.join(path, descrip_file))

# Filter and clean description data
descrip = pd.read_excel(os.path.join(path, descrip_file))
descrip = descrip.loc[descrip['Include'] == 'X']

# Data cleaning and transformation
group_MB = 'MBCAT'
data_MB[group_MB] = data_MB[group_MB].str.strip().str.upper()

# Merge with data_DM
data_MB = pd.merge(data_MB, data_DM, on='USUBJID', how='left')

# Initialize DataFrame for results
df2 = pd.DataFrame()

# Function to calculate percentages
def calculate_percentages(data_site, descrip, group, group_sites):
    table_counts = []
    table_SATERM = []
    for i in range(len(descrip)):
        den = len(data_DM['USUBJID'].loc[data_DM[group_sites] == group].unique())
        total = len(data_site['USUBJID'].loc[
            (data_site[group_MB] == descrip["Extracted MBTERM"].iloc[i]) &
            (data_site['MBSTRESC'].notna()) &
            (data_site['MBTSTDTL'] == descrip['MBTSTDTL'].iloc[i])
        ].unique())
        name = descrip['Extracted MBTERM'].iloc[i]
        if descrip['Positive'].iloc[i] == 'X':
            total = len(data_site['USUBJID'].loc[
                (data_site[group_MB] == descrip["Extracted MBTERM"].iloc[i]) &
                (data_site['MBTSTDTL'] == descrip['MBTSTDTL'].iloc[i]) &
                (data_site['MBORRES'].notna())
            ].unique())
            den = len(data_site['USUBJID'].loc[
                (data_site[group_MB] == descrip["Extracted MBTERM"].iloc[i]) &
                (data_site['MBORRES'].notna())
            ].unique())
            name = descrip['Extracted MBTERM'].iloc[i] + '_specified'
        if den == 0:
            table_counts.append(0)
        else:
            table_counts.append(100 * total / den)
        table_SATERM.append(name)
    return table_SATERM, table_counts

# Calculate percentages for each group site
group_sites = 'DB'
for j in data_DM[group_sites].unique():
    data_MB_site = data_MB.loc[data_MB[group_sites] == j]
    terms, counts = calculate_percentages(data_MB_site, descrip, j, group_sites)
    df_temp = pd.DataFrame({'Extracted TERM': terms, j: counts})
    df2 = pd.concat([df2, df_temp], ignore_index=True)

# Additional description fields
descrip_additional = ['MBSPEC', 'MBMETHOD', 'MBDTC', 'MBSTRESC', 'MBTEST']
descrip_labels = ['Biospecimen', 'Lab Method', 'Collection Date', 'Result', 'Pathogen tested']

# Function to calculate additional percentages
def calculate_additional_percentages(data_site, descrip, descrip_labels, group):
    table_counts = []
    table_SATERM = []
    for i in range(len(descrip)):
        den = len(data_DM['USUBJID'].loc[data_DM[group_sites] == group].unique())
        strings_to_check = ['UNKNOWN', 'UNSPECIFIED', 'INDETERMINATE', 'NOT SPECIFIED']
        MB_str = data_site[descrip[i]].astype(str)
        mask = ~MB_str.str.contains('|'.join(strings_to_check), case=False)
        total = len(data_site['USUBJID'].loc[
            (data_site[descrip[i]].notna()) & mask
        ].unique())
        name = descrip_labels[i].upper()
        if den == 0:
            table_counts.append(0)
        else:
            table_counts.append(100 * total / den)
        table_SATERM.append(name)
    return table_SATERM, table_counts

# Calculate additional percentages for each group site
for j in data_DM[group_sites].unique():
    data_MB_site = data_MB.loc[data_MB[group_sites] == j]
    terms, counts = calculate_additional_percentages(data_MB_site, descrip_additional, descrip_labels, j)
    df_temp = pd.DataFrame({'Extracted TERM': terms, j: counts})
    df2 = pd.concat([df2, df_temp], ignore_index=True)

# Save the result to Excel
df2.set_index('Extracted TERM', inplace=True)

# Calculate mean and additional statistics
numeric_columns = data_DM['DB'].unique()
df2 = df2[numeric_columns]
df2['mean'] = df2.mean(axis=1)
df3 = df2.reset_index()
df3['mean(excluding 0)'] = df2.replace(0, np.nan).mean(axis=1)
df3['% sites included'] = (100 * (df2 > 0).sum(axis=1) / len(numeric_columns))
df3['Coefficient_of_Variation'] = df2.std(axis=1) / df2.mean(axis=1)

# Save the final result to Excel
df3.to_excel(os.path.join(path, 'example_db_MB_means.xlsx'), index=False)
