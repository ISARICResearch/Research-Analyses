# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 20:20:03 2023

@author: SDuque
"""

import os
import pandas as pd
import numpy as np

# Define the path and file names
path = 'D:/ISARIC/JAN 23/'
data_inmodify2_file = 'data_INMODIFY2.csv'
data_dm_file = 'usub_db.csv'
descrip_file = 'CRF_IN.xlsx'

# Read data
data_IN = pd.read_csv(os.path.join(path, data_inmodify2_file))
data_DM = pd.read_csv(os.path.join(path, data_dm_file))

# Filter data to include only rows with USUBJID present in data_DM
data_IN = data_IN.loc[data_IN['USUBJID'].isin(data_DM['USUBJID'].unique())]

# Read and filter description data
descrip = pd.read_excel(os.path.join(path, descrip_file))
descrip = descrip.loc[descrip['Include'] == 'X']
descrip = descrip[['INEVINTX', 'INDUR', 'INSTDTC', 'INENDTC', 'TYPE', 'INDOSTOT', 'INDOSU', 'INDOSE', 'DIFF_TYPE', 'DIFF_TYPE_TRT', 'INROUTE', 'INMODIFY2']]
descrip.loc[descrip['TYPE'] == 'YES', ['DIFF_TYPE', 'DIFF_TYPE_TRT']] = np.nan
descrip.dropna(subset=['INEVINTX', 'INMODIFY2'], inplace=True)
descrip.drop_duplicates(inplace=True)

# Initialize an empty DataFrame to store results
df2 = pd.DataFrame()
group_sites = 'DB'

# Calculate statistics for each group site
for j in data_DM[group_sites].unique():
    table_counts = []
    table_SATERM = []
    
    data_IN_site = data_IN.loc[data_IN[group_sites] == j]
          
    for i in range(len(descrip)):
        den = len(data_DM['USUBJID'].loc[data_DM[group_sites] == j].unique())
        total = len(data_IN_site['USUBJID'].loc[
            (data_IN_site['INEVINTX'] == descrip["INEVINTX"].iloc[i]) &
            (data_IN_site['INMODIFY2'] == descrip["INMODIFY2"].iloc[i]) &
            (data_IN_site['INOCCUR'].isin(['Y', 'N']))
        ].unique())
        name = descrip['INMODIFY2'].iloc[i]
        
        if descrip['INDUR'].iloc[i] == 'YES':           
            total = len(data_IN_site['USUBJID'].loc[
                (data_IN_site['INDUR'].notna()) &
                (data_IN_site['INEVINTX'] == descrip["INEVINTX"].iloc[i]) &
                ((data_IN_site['INMODIFY2'] == descrip["INMODIFY2"].iloc[i]) | 
                (data_IN_site[descrip["DIFF_TYPE"].iloc[i]] == descrip["DIFF_TYPE_TRT"].iloc[i]))
            ].unique())
            den = len(data_IN_site['USUBJID'].loc[
                (data_IN_site['INEVINTX'] == descrip["INEVINTX"].iloc[i]) &
                ((data_IN_site['INMODIFY2'] == descrip["INMODIFY2"].iloc[i]) | 
                (data_IN_site[descrip["DIFF_TYPE"].iloc[i]] == descrip["DIFF_TYPE_TRT"].iloc[i]))
            ].unique())
            name = descrip['INMODIFY2'].iloc[i] + '_DUR'
        
        if descrip['INSTDTC'].iloc[i] == 'YES':
            total = len(data_IN_site['USUBJID'].loc[
                (data_IN_site['INSTDTC'].notna()) &
                (data_IN_site['INEVINTX'] == descrip["INEVINTX"].iloc[i]) &
                ((data_IN_site['INMODIFY2'] == descrip["INMODIFY2"].iloc[i]) | 
                (data_IN_site[descrip["DIFF_TYPE"].iloc[i]] == descrip["DIFF_TYPE_TRT"].iloc[i]))
            ].unique())
            den = len(data_IN_site['USUBJID'].loc[
                (data_IN_site['INEVINTX'] == descrip["INEVINTX"].iloc[i]) &
                ((data_IN_site['INMODIFY2'] == descrip["INMODIFY2"].iloc[i]) | 
                (data_IN_site[descrip["DIFF_TYPE"].iloc[i]] == descrip["DIFF_TYPE_TRT"].iloc[i]))
            ].unique())
            name = descrip['INMODIFY2'].iloc[i] + '_STDTC'
        
        if descrip['INENDTC'].iloc[i] == 'YES':
            total = len(data_IN_site['USUBJID'].loc[
                (data_IN_site['INENDTC'].notna()) &
                (data_IN_site['INEVINTX'] == descrip["INEVINTX"].iloc[i]) &
                ((data_IN_site['INMODIFY2'] == descrip["INMODIFY2"].iloc[i]) | 
                (data_IN_site[descrip["DIFF_TYPE"].iloc[i]] == descrip["DIFF_TYPE_TRT"].iloc[i]))
            ].unique())
            den = len(data_IN_site['USUBJID'].loc[
                (data_IN_site['INEVINTX'] == descrip["INEVINTX"].iloc[i]) &
                ((data_IN_site['INMODIFY2'] == descrip["INMODIFY2"].iloc[i]) | 
                (data_IN_site[descrip["DIFF_TYPE"].iloc[i]] == descrip["DIFF_TYPE_TRT"].iloc[i]))
            ].unique())
            name = descrip['INMODIFY2'].iloc[i] + '_ENDTC'
        
        if descrip['INDOSTOT'].iloc[i] == 'YES':
            total = len(data_IN_site['USUBJID'].loc[
                (data_IN_site['INDOSTOT'].notna()) &
                (data_IN_site['INEVINTX'] == descrip["INEVINTX"].iloc[i]) &
                ((data_IN_site['INMODIFY2'] == descrip["INMODIFY2"].iloc[i]) | 
                (data_IN_site[descrip["DIFF_TYPE"].iloc[i]] == descrip["DIFF_TYPE_TRT"].iloc[i]))
            ].unique())
            den = len(data_IN_site['USUBJID'].loc[
                (data_IN_site['INEVINTX'] == descrip["INEVINTX"].iloc[i]) &
                ((data_IN_site['INMODIFY2'] == descrip["INMODIFY2"].iloc[i]) | 
                (data_IN_site[descrip["DIFF_TYPE"].iloc[i]] == descrip["DIFF_TYPE_TRT"].iloc[i]))
            ].unique())
            name = descrip['INMODIFY2'].iloc[i] + '_DOSTOT'
        
        if descrip['INDOSU'].iloc[i] == 'YES':
            total = len(data_IN_site['USUBJID'].loc[
                (data_IN_site['INDOSU'].notna()) &
                (data_IN_site['INEVINTX'] == descrip["INEVINTX"].iloc[i]) &
                ((data_IN_site['INMODIFY2'] == descrip["INMODIFY2"].iloc[i]) | 
                (data_IN_site[descrip["DIFF_TYPE"].iloc[i]] == descrip["DIFF_TYPE_TRT"].iloc[i])) &
                (data_IN_site['INPRESP'].isin(['Y', 'N']))
            ].unique())
            den = len(data_IN_site['USUBJID'].loc[
                (data_IN_site['INEVINTX'] == descrip["INEVINTX"].iloc[i]) &
                ((data_IN_site['INMODIFY2'] == descrip["INMODIFY2"].iloc[i]) | 
                (data_IN_site[descrip["DIFF_TYPE"].iloc[i]] == descrip["DIFF_TYPE_TRT"].iloc[i])) &
                (data_IN_site['INPRESP'].isin(['Y', 'N']))
            ].unique())
            name = descrip['INMODIFY2'].iloc[i] + '_DOSU'
        
        if descrip['TYPE'].iloc[i] == 'YES':
            usubjid_typ = set(data_IN_site['USUBJID'].loc[
                (data_IN_site['INEVINTX'] == descrip["INEVINTX"].iloc[i]) &
                (data_IN_site["INMODIFY2"] == descrip["INMODIFY2"].iloc[i])
            ].unique())
            usubjid_bef = set(data_IN_site['USUBJID'].loc[
                (data_IN_site['INEVINTX'] == descrip["INEVINTX"].iloc[i - 1]) &
                (data_IN_site['INMODIFY2'] == descrip["INMODIFY2"].iloc[i - 1])
            ].unique())
            den = len(usubjid_bef)
            total = len(set(usubjid_bef).intersection(usubjid_typ))
        
        if descrip['INROUTE'].iloc[i] == 'YES':
            total = len(data_IN_site['USUBJID'].loc[
                (data_IN_site['INROUTE'].notna()) &
                (data_IN_site['INEVINTX'] == descrip["INEVINTX"].iloc[i]) &
                ((data_IN_site['INMODIFY2'] == descrip["INMODIFY2"].iloc[i]) | 
                (data_IN_site[descrip["DIFF_TYPE"].iloc[i]] == descrip["DIFF_TYPE_TRT"].iloc[i]))
            ].unique())
            den = len(data_IN_site['USUBJID'].loc[
                (data_IN_site['INEVINTX'] == descrip["INEVINTX"].iloc[i]) &
                ((data_IN_site['INMODIFY2'] == descrip["INMODIFY2"].iloc[i]) | 
                (data_IN_site[descrip["DIFF_TYPE"].iloc[i]] == descrip["DIFF_TYPE_TRT"].iloc[i]))
            ].unique())
            name = descrip['INMODIFY2'].iloc[i] + '_ROUTE'
        
        if descrip['INDOSE'].iloc[i] == 'YES':
            total = len(data_IN_site['USUBJID'].loc[
                (data_IN_site['INDOSE'].notna()) &
                (data_IN_site['INEVINTX'] == descrip["INEVINTX"].iloc[i]) &
                ((data_IN_site['INMODIFY2'] == descrip["INMODIFY2"].iloc[i]) | 
                (data_IN_site[descrip["DIFF_TYPE"].iloc[i]] == descrip["DIFF_TYPE_TRT"].iloc[i]))
            ].unique())
            den = len(data_IN_site['USUBJID'].loc[
                (data_IN_site['INEVINTX'] == descrip["INEVINTX"].iloc[i]) &
                ((data_IN_site['INMODIFY2'] == descrip["INMODIFY2"].iloc[i]) | 
                (data_IN_site[descrip["DIFF_TYPE"].iloc[i]] == descrip["DIFF_TYPE_TRT"].iloc[i]))
            ].unique())
            name = descrip['INMODIFY2'].iloc[i] + '_DOSE'
        
        if descrip['INMODIFY2'].iloc[i] == 'VASOPRESSOR/INOTROPIC_DAILY':
            usubjid_dop = set(data_IN_site['USUBJID'].loc[
                (data_IN_site['INMODIFY2'] == 'DOPAMINE') &
                (data_IN_site['INEVINTX'] == descrip["INEVINTX"].iloc[i]) &
                (data_IN_site['INOCCUR'] == 'Y')
            ])
            usubjid_dep = set(data_IN_site['USUBJID'].loc[
                (data_IN_site['INMODIFY2'] == 'VASOPRESSOR/INOTROPIC_DAILY') &
                (data_IN_site['INEVINTX'] == descrip["INEVINTX"].iloc[i]) &
                (data_IN_site['INPRESP'].isin(['Y', 'N']))
            ])
            den = len(usubjid_dop)
            total = len(set(usubjid_dop).intersection(usubjid_dep))
        
        if den == 0:
            table_counts.append(0)
        else:
            table_counts.append(100 * total / den)
        table_SATERM.append(name)
       
    df2['Extracted INTRT'] = table_SATERM
    df2[j] = table_counts

df2.set_index('Extracted INTRT', inplace=True)

# Calculate mean and additional statistics
numeric_columns = data_DM['DB'].unique()
df2 = df2[numeric_columns]
df2['mean'] = df2.mean(axis=1)
df3 = df2.reset_index()
df3['mean(excluding 0)'] = df2.replace(0, np.nan).mean(axis=1)
df3['% sites included'] = (100 * (df2 > 0).sum(axis=1) / len(numeric_columns))
df3['Coefficient_of_Variation'] = df2.std(axis=1) / df2.mean(axis=1)

# Save the final result to Excel
df3.to_excel(os.path.join(path, 'example_db_IN_means.xlsx'), index=False)
