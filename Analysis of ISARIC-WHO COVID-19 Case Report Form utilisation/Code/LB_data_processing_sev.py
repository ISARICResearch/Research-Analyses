# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 09:51:51 2023

@author: SDuque
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the path to the data directory
path = 'D:/ISARIC/JAN 23/'

# Load Laboratory data
table = 'Internal_LB_20230110_v2.csv'
data_LB = pd.read_csv(os.path.join(path, table)).loc[:, ['USUBJID', 'LBTESTCD', 'LBORRES', 'LBORRESU', 'LBSPEC', 'LBEVINTX', 'LBDTC', 'LBCAT', 'LBDY']]

# Display value counts for 'LBCAT' and 'LBEVINTX' columns for debugging purposes
print(data_LB['LBCAT'].value_counts())
print(data_LB['LBEVINTX'].value_counts())

# Standardize and clean 'LBTESTCD' column
data_LB['LBTESTCD'] = data_LB['LBTESTCD'].str.strip().str.upper().astype(str)

# Load the description file and filter for included items
descrip = pd.read_excel(os.path.join(path, 'CRF_LB.xlsx'))
descrip = descrip.loc[descrip['Include'] == 'X']

# Load demographic data
data_DM = pd.read_csv(os.path.join(path, 'usub_db_sev.csv'))
data_DM_ns = pd.read_csv(os.path.join(path, 'usub_db.csv'))

# Filter laboratory data to include only severe cases
data_LB = data_LB.loc[data_LB['USUBJID'].isin(data_DM['USUBJID'].unique())]

# Merge laboratory data with demographic data
data_LB = pd.merge(data_LB, data_DM, on='USUBJID', how='left')

# Categorize time for laboratory data
data_LB['time'] = np.zeros(len(data_LB))
data_LB['time'].loc[((data_LB['LBEVINTX'] == '00:00-24:00 ON DAY OF ASSESSMENT') & (data_LB['LBDY'] <= 1)) | 
                    (data_LB['LBCAT'] == 'LABORATORY RESULTS ON ADMISSION') | 
                    (data_LB['LBEVINTX'].isin(['WITHIN 24 HOURS OF ADMISSION', 'BEFORE HOSPITAL ADMISSION']))] = 'Admi'
data_LB['time'].loc[((data_LB['LBDY'] > 1) & (data_LB['LBEVINTX'] == '00:00-24:00 ON DAY OF ASSESSMENT') & 
                     (data_LB['LBCAT'] != 'LABORATORY RESULTS ON ADMISSION')) | 
                    ((data_LB['LBCAT'] == 'LABORATORY RESULTS ON ICU ADMISSION') & (data_LB['LBDY'] > 1)) | 
                    (data_LB['LBEVINTX'].isin(['LATEST', 'AFTER ADMISSION']))] = 'Daily'

# Initialize DataFrame for results
df2 = pd.DataFrame()

# Process data for each database site
group_sites = 'DB'

for j in data_DM_ns[group_sites].unique():
    table_counts = []
    table_SATERM = []
    
    data_LB_site = data_LB.loc[data_LB[group_sites] == j]
          
    for i in range(len(descrip)):
        den = len(data_DM['USUBJID'].loc[data_DM[group_sites] == j].unique())
        total = len(data_LB_site['USUBJID'].loc[(data_LB_site['LBTESTCD'] == descrip["LBTESTCD_value"].iloc[i]) & 
                                                (data_LB_site['LBORRES'].notna()) & 
                                                (data_LB_site['time'] == descrip['time'].iloc[i])].unique())
        name = descrip['LBTESTCD_value'].iloc[i] + '_' + descrip['time'].iloc[i]
        
        if descrip['LBORRESU'].iloc[i] == 'X':
            total = len(data_LB_site['USUBJID'].loc[(data_LB_site['LBTESTCD'] == descrip["LBTESTCD_value"].iloc[i]) & 
                                                    (data_LB_site['LBORRES'].notna()) & 
                                                    (data_LB_site['LBORRESU'].notna()) & 
                                                    (data_LB_site['time'] == descrip['time'].iloc[i])].unique())
            name = descrip['LBTESTCD_value'].iloc[i] + '_UNIT_' + descrip['time'].iloc[i]
            
            if isinstance(descrip['LBORRESU_value'].iloc[i], str):
                total = len(data_LB_site['USUBJID'].loc[(data_LB_site['LBTESTCD'] == descrip["LBTESTCD_value"].iloc[i]) & 
                                                        (data_LB_site['LBORRES'].notna()) & 
                                                        (data_LB_site['LBORRESU'] == descrip["LBORRESU_value"].iloc[i]) & 
                                                        (data_LB_site['time'] == descrip['time'].iloc[i])].unique())
                name = descrip['LBTESTCD_value'].iloc[i] + '_UNIT_' + descrip["LBORRESU_value"].iloc[i] + '_' + descrip['time'].iloc[i]
        
        if descrip['LBSPEC'].iloc[i] == 'X':
            total = len(data_LB_site['USUBJID'].loc[(data_LB_site['LBTESTCD'] == descrip["LBTESTCD_value"].iloc[i]) & 
                                                    (data_LB_site['LBORRES'].notna()) & 
                                                    (data_LB_site['LBSPEC'].notna()) & 
                                                    (data_LB_site['time'] == descrip['time'].iloc[i])].unique())
            name = descrip['LBTESTCD_value'].iloc[i] + '_SPEC_' + descrip['time'].iloc[i]
        
        if den == 0:
            table_counts.append(0)
        else:
            table_counts.append(100 * total / den)
        table_SATERM.append(name)
       
    df2['Extracted LBTESTCD'] = table_SATERM
    df2[j] = table_counts

# Set index and calculate statistics
df2.set_index('Extracted LBTESTCD', inplace=True)
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
df3.to_excel(os.path.join(path, 'example_db_LB_means_sev.xlsx'), index=False)
