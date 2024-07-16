# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:20:20 2023

@author: SDuque
"""

import os
import pandas as pd
import numpy as np

# Define the path to the data directory
path = 'D:/ISARIC/JAN 23/'

# List of files in the directory
arr = os.listdir(path)

# Define filenames and tables
filename = 'Internal_DM_site_2023-01-10_v2.csv'
tables = ['Internal_CQ_2023-01-10.csv', 'Internal_DS_2023-01-10.csv', 'Internal_ER_2023-01-10.csv', 'Internal_HO_2023-01-10.csv', 'Internal_IE_2023-01-10.csv', 'Internal_IN_20230110.csv', 'Internal_LB_20230110_v2.csv', 'Internal_MB_2023-01-10.csv', 'Internal_RELREC_2023-01-10.csv', 'Internal_RP_2023-01-10.csv', 'Internal_RS_2023-01-10.csv', 'Internal_SA_20230110.csv', 'Internal_VS_20230110.csv']

# Load the dictionary and description files
dictio = pd.read_excel(os.path.join(path, 'CORE RedCAP to SDTM mapping_FU_v2.1_2021-07-08 (2).xlsx'))
descrip = pd.read_excel(os.path.join(path, 'CRF_DM.xlsx'))

# Load the main dataset and merge with description
data_DEM = pd.read_csv(os.path.join(path, 'Internal_DM_2023-01-10_v2.csv'))
data_DM_ns = pd.read_csv(os.path.join(path, 'usub_db.csv'))
data_DM = pd.read_csv(os.path.join(path, 'usub_db_sev.csv'))

# Unique database names
data_bases = data_DM_ns['DB'].unique()

# Merge data_DEM with data_DM
data_DEM = pd.merge(data_DEM, data_DM, on='USUBJID', how='left')

# Initialize DataFrame to store results
df2 = pd.DataFrame()

# Iterate over unique database sites
for j in data_bases:
    table_counts = []
    table_SATERM = []
    den = len(data_DM['USUBJID'].loc[data_DM['DB'] == j].unique())
    data_DEM_site = data_DEM.loc[data_DEM['DB'] == j]
          
    # Iterate over the description rows
    for i in range(len(descrip)):
        total = len(data_DEM_site['USUBJID'].loc[(data_DEM_site[descrip['Extracted Term'].iloc[i]].notna())].unique())
        
        if den == 0:
            table_counts.append('no site')
        else:
            table_counts.append(100 * total / den)
        table_SATERM.append(str(descrip['Extracted Term'].iloc[i]).upper())
       
    df2['Extracted TERM'] = table_SATERM
    df2[j] = table_counts

# Load additional datasets for further processing
table2 = 'Internal_DM_2023-01-10_v2.csv'
data_sex = pd.read_csv(os.path.join(path, table2)).loc[:, ['USUBJID', 'SEX']]
data_age = pd.read_csv(os.path.join(path, table2)).loc[:, ['USUBJID', 'AGE', 'AGEU']]

# Process 'Internal_RP_2023-01-10.csv'
table = 'Internal_RP_2023-01-10.csv'
data_DEM = pd.read_csv(os.path.join(path, table))
data_DEM = pd.merge(data_DEM, data_sex, on='USUBJID', how='left')
data_DEM = pd.merge(data_DEM, data_age, on='USUBJID', how='left')
data_DEM = data_DEM.loc[(data_DEM['SEX'] == 'F') & ((data_DEM['AGEU'] == 'YEARS') & (data_DEM['AGE'] >= 12) & (data_DEM['AGE'] <= 50))]
data_DEM = data_DEM.drop(columns=['SEX', 'AGE', 'AGEU'])

# Clean and prepare data for analysis
data_DEM['RPTEST'] = data_DEM['RPTEST'].str.strip()
data_DEM['RPORRES'] = data_DEM['RPORRES'].str.strip()

# Merge with severity data

data_DM = pd.merge(data_DM, data_sex, on='USUBJID', how='left')
data_DM = pd.merge(data_DM, data_age, on='USUBJID', how='left')
data_DM = data_DM.loc[(data_DM['SEX'] == 'F') & ((data_DM['AGEU'] == 'YEARS') & (data_DM['AGE'] >= 12) & (data_DM['AGE'] <= 50))]
data_DM = data_DM.drop(columns=['SEX', 'AGE', 'AGEU'])

data_DEM = pd.merge(data_DEM, data_DM, on='USUBJID', how='left')

# Define descriptions
descrip_list = ['Pregnant Indicator', 'Estimated Gestational Age']

df2_temp = pd.DataFrame()
for j in data_bases:
    table_counts = []
    table_SATERM = []
    den = len(data_DM['USUBJID'].loc[data_DM['DB'] == j].unique())
    data_DEM_site = data_DEM.loc[data_DEM['DB'] == j]
    for i in range(len(descrip_list)):
        total = len(data_DEM_site['USUBJID'].loc[(data_DEM_site['RPTEST'] == descrip_list[i]) & (data_DEM_site['RPORRES'].notna())].unique())
        if descrip_list[i] == 'Estimated Gestational Age':
            den = len(data_DEM_site['USUBJID'].loc[(data_DEM_site['RPTEST'] == 'Pregnant Indicator') & (data_DEM_site['RPORRES'] == 'Yes')].unique())
            series1 = pd.Series(data_DEM_site['USUBJID'].loc[(data_DEM_site['RPTEST'] == descrip_list[i]) & (data_DEM_site['RPORRES'].notna())].unique())
            series2 = pd.Series(data_DEM_site['USUBJID'].loc[(data_DEM_site['RPTEST'] == 'Pregnant Indicator') & (data_DEM_site['RPORRES'] == 'Yes')].unique())
            total = len(series1[series1.isin(series2)])
        if den == 0:
            table_counts.append('no site')
        else:
            table_counts.append(100 * total / den)
        table_SATERM.append(str(descrip_list[i]).upper())
    
    df2_temp['Extracted TERM'] = table_SATERM
    df2_temp[j] = table_counts

df2 = pd.concat([df2, df2_temp], ignore_index=True)

# Process 'Internal_IE_2023-01-10.csv'
table = 'Internal_IE_2023-01-10.csv'
data_DEM = pd.read_csv(os.path.join(path, table))
data_DEM['IETEST'] = data_DEM['IETEST'].str.strip()
data_DEM = pd.merge(data_DEM, data_DM, on='USUBJID', how='left')

df2_temp = pd.DataFrame()
for j in data_bases:
    table_counts = []
    den = len(data_DM['USUBJID'].loc[data_DM['DB'] == j].unique())
    data_DEM_site = data_DEM.loc[data_DEM['DB'] == j]
    total = len(data_DEM_site['USUBJID'].loc[data_DEM_site['IETESTCD'].notna()].unique())
    if den == 0:
        table_counts.append('no site')
    else:
        table_counts.append(100 * total / den)
    
    df2_temp['Extracted TERM'] = ('Inclusion Criteria').upper()
    df2_temp[j] = table_counts

df2 = pd.concat([df2, df2_temp], ignore_index=True)

# Process 'Internal_ER_2023-01-10.csv'
table = 'Internal_ER_2023-01-10.csv'
data_DEM = pd.read_csv(os.path.join(path, table))
data_DEM = pd.merge(data_DEM, data_age, on='USUBJID', how='left')
data_DEM = data_DEM.loc[(data_DEM['AGEU'] == 'YEARS') & (data_DEM['AGE'] >= 17)]
data_DEM = data_DEM.drop(columns=['AGE', 'AGEU'])
data_DEM['ERTERM'] = data_DEM['ERTERM'].str.lower().str.strip()

data_DM = pd.merge(data_DM, data_age, on='USUBJID', how='left')
data_DM = data_DM.loc[(data_DM['AGEU'] == 'YEARS') & (data_DM['AGE'] >= 17)]
data_DM = data_DM.drop(columns=['AGE', 'AGEU'])

data_DEM = pd.merge(data_DEM, data_DM, on='USUBJID', how='left')

descrip_list = ['healthcare', 'laboratory']
df2_temp = pd.DataFrame()
for j in data_bases:
    table_counts = []
    table_SATERM = []
    den = len(data_DM['USUBJID'].loc[data_DM['DB'] == j].unique())
    data_DEM_site = data_DEM.loc[data_DEM['DB'] == j]
    for i in range(len(descrip_list)):
        total = len(data_DEM_site['USUBJID'].loc[(data_DEM_site['ERTERM'].str.contains(descrip_list[i])) & (data_DEM_site['EROCCUR'].isin(['Y', 'N']))].unique())
        if den == 0:
            table_counts.append('no site')
        else:
            table_counts.append(100 * total / den)
        table_SATERM.append(str(descrip_list[i] + ' worker').upper())
    
    df2_temp['Extracted TERM'] = table_SATERM
    df2_temp[j] = table_counts

df2 = pd.concat([df2, df2_temp], ignore_index=True)

# Process first symptom date
table = 'Internal_SA_20230110.csv'
data_DEM = pd.read_csv(os.path.join(path, table)).loc[:, ['USUBJID', 'SASTDTC', 'SATERM']]
data_DEM = pd.merge(data_DEM, data_DM, on='USUBJID', how='left')

df2_temp = pd.DataFrame()
for j in data_bases:
    table_counts = []
    den = len(data_DM['USUBJID'].loc[data_DM['DB'] == j].unique())
    data_DEM_site = data_DEM.loc[data_DEM['DB'] == j]
    total = len(data_DEM_site['USUBJID'].loc[(data_DEM_site['SASTDTC'].notna()) & (data_DEM_site['SATERM'] == 'COVID-19 SYMPTOMS')].unique())
    if den == 0:
        table_counts.append('no site')
    else:
        table_counts.append(100 * total / den)
    
    df2_temp['Extracted TERM'] = ('First Symptom date').upper()
    df2_temp[j] = table_counts

df2 = pd.concat([df2, df2_temp], ignore_index=True)

# Replace 'no site' with 0 and save results
df2 = df2.replace({'no site': 0})
df2.to_excel(os.path.join(path, 'example_sites_DM.xlsx'), index=False)

df = df2.copy()
df2.set_index('Extracted TERM', inplace=True)

numeric_columns = data_bases
df2 = df2[numeric_columns]

df2['mean'] = df2[numeric_columns].mean(axis=1)

df3 = df2.copy()
df3 = df3.reset_index()

df3['mean(excluding 0)'] = np.zeros(len(df3))

df2[(df2 == 0)] = np.nan

df3['mean(excluding 0)'] = df2[numeric_columns].mean(axis=1)
df3['% sites included'] = (100 * (df2[numeric_columns] > 0).sum(axis=1)) / len(numeric_columns)
df3['Coefficient_of_Variation'] = df3[numeric_columns].std(axis=1) / df3[numeric_columns].mean(axis=1)

df3.to_excel(os.path.join(path, 'example_db_DM_means_sev.xlsx'), index=False)
