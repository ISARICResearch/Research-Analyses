# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 09:51:51 2023

@author: SDuque
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce
import math
import re
import random

# Define the path to the data directory
path = 'D:/ISARIC/JAN 23/'

# Load the necessary datasets
data_VS = pd.read_csv(os.path.join(path, 'Internal_VS_20230110.csv'))
data_DM = pd.read_csv(os.path.join(path, 'usub_db.csv'))

# Standardize 'VSEVINTX' and 'VSCAT' columns
data_VS['VSEVINTX'].loc[data_VS['VSEVINTX'] == 'Within 24 hours of admission'] = 'WITHIN 24 HOURS OF ADMISSION'
data_VS['VSCAT'] = data_VS['VSCAT'].str.upper()

# Standardize 'VSTESTCD' column
data_VS['VSTESTCD'] = data_VS['VSTESTCD'].str.strip().str.upper().astype(str)

# Load the description file for vital signs
descrip = pd.read_excel(os.path.join(path, 'CRF_VS.xlsx'))

# Merge 'data_VS' with 'data_DM' on 'USUBJID'
data_VS = pd.merge(data_VS, data_DM, on='USUBJID', how='left')

# Initialize an empty DataFrame to store the results
df2 = pd.DataFrame()

# Load and prepare laboratory data
data_LB = pd.read_csv(os.path.join(path, 'Internal_LB_20230110_v2.csv')).loc[:, ['USUBJID', 'LBTESTCD', 'LBORRES', 'LBORRESU', 'LBSPEC', 'LBEVINTX']]
data_LB = data_LB.loc[data_LB['LBEVINTX'] == '00:00-24:00 ON DAY OF ASSESSMENT']
data_LB['LBTESTCD'] = data_LB['LBTESTCD'].str.strip().str.upper().astype(str)

# Merge 'data_LB' with 'data_DM' on 'USUBJID'
data_LB = pd.merge(data_LB, data_DM, on='USUBJID', how='left')

# Load the description file for laboratory data
descrip2 = pd.read_excel(os.path.join(path, 'CRF_LB.xlsx'))
descrip2 = descrip2[descrip2['Vs'] == 'X']

# Combine description data
descrip = pd.concat([descrip, descrip2], axis=0, ignore_index=True)

# Load and prepare response data
data_RS = pd.read_csv(os.path.join(path, 'Internal_RS_2023-01-10.csv')).loc[:, ['USUBJID', 'RSTEST', 'RSSCAT', 'RSEVINTX', 'RSSTRESC']]
data_RS = pd.merge(data_RS, data_DM, on='USUBJID', how='left')

# Process data for each group site
for j in data_DM['DB'].unique():
    table_counts = []
    table_SATERM = []

    data_VS_site = data_VS.loc[data_VS['DB'] == j]
    data_LB_site = data_LB.loc[data_LB['DB'] == j]
    data_RS_site = data_RS.loc[data_RS['DB'] == j]

    for i in range(len(descrip)):
        den = len(data_DM['USUBJID'].loc[data_DM['DB'] == j].unique())
        total = len(data_VS_site['USUBJID'].loc[
            ((data_VS_site['VSEVINTX'] == descrip["VSEVINTX"].iloc[i]) | (data_VS_site['VSCAT'] == descrip["VSCAT"].iloc[i])) &
            (data_VS_site['VSTESTCD'] == descrip["VSTESTCD_value"].iloc[i]) &
            (data_VS_site['VSORRES'].notna())
        ].unique())
        name = descrip['VSTESTCD_value'].iloc[i]

        if descrip['VSORRESU'].iloc[i] == 'X':
            total = len(data_VS_site['USUBJID'].loc[
                ((data_VS_site['VSEVINTX'] == descrip["VSEVINTX"].iloc[i]) | (data_VS_site['VSCAT'] == descrip["VSCAT"].iloc[i])) &
                (data_VS_site['VSTESTCD'] == descrip["VSTESTCD_value"].iloc[i]) &
                (data_VS_site['VSORRES'].notna()) &
                (data_VS_site['VSSTRESU'].notna())
            ].unique())
            name += '_UNIT'

        if descrip['VSO2SRC'].iloc[i] == 'X':
            total = len(data_VS_site['USUBJID'].loc[
                ((data_VS_site['VSEVINTX'] == descrip["VSEVINTX"].iloc[i]) | (data_VS_site['VSCAT'] == descrip["VSCAT"].iloc[i])) &
                (data_VS_site['VSTESTCD'] == descrip["VSTESTCD_value"].iloc[i]) &
                (data_VS_site['VSO2SRC'] != 'UNKNOWN')
            ].unique())
            name += '_VSO2SRC'

        if descrip['Vs'].iloc[i] == 'X':
            total = len(data_LB_site['USUBJID'].loc[
                (data_LB_site['LBTESTCD'] == descrip["LBTESTCD_value"].iloc[i]) &
                (data_LB_site['LBORRES'].notna())
            ].unique())
            name = descrip['LBTESTCD_value'].iloc[i] + '_DAILY'

            if descrip['LBORRESU'].iloc[i] == 'X':
                total = len(data_LB_site['USUBJID'].loc[
                    (data_LB_site['LBTESTCD'] == descrip["LBTESTCD_value"].iloc[i]) &
                    (data_LB_site['LBORRES'].notna()) &
                    (data_LB_site['LBORRESU'].notna())
                ].unique())
                name += '_UNIT_DAILY'

            if descrip['LBSPEC'].iloc[i] == 'X':
                total = len(data_LB_site['USUBJID'].loc[
                    (data_LB_site['LBTESTCD'] == descrip["LBTESTCD_value"].iloc[i]) &
                    (data_LB_site['LBORRES'].notna()) &
                    (data_LB_site['LBSPEC'].notna())
                ].unique())
                name += '_SPEC_DAILY'

            if descrip['Depends'].iloc[i] == 'PO2':
                usubjid_po2 = set(data_LB_site['USUBJID'].loc[
                    (data_LB_site['LBTESTCD'] == 'PO2') &
                    (data_LB_site['LBORRES'].notna())
                ])
                usubjid_dep = set(data_LB_site['USUBJID'].loc[
                    (data_LB_site['LBTESTCD'] == descrip["LBTESTCD_value"].iloc[i]) &
                    (data_LB_site['LBORRES'].notna())
                ])
                den = len(usubjid_po2)
                total = len(usubjid_po2.intersection(usubjid_dep))

        if descrip['SDTM Domain(s)'].iloc[i] == 'RS':
            total = len(data_RS_site['USUBJID'].loc[
                (data_RS_site['RSEVINTX'] == descrip["VSEVINTX"].iloc[i]) &
                (data_RS_site['RSTEST'] == descrip["VSTESTCD_value"].iloc[i]) &
                (data_RS_site['RSSTRESC'].notna())
            ].unique())
            name = descrip['VSTEST_value'].iloc[i]

        suf = ''
        if descrip["VSEVINTX"].iloc[i] == 'WITHIN 24 HOURS OF ADMISSION':
            suf = '_INHOSP'
        elif descrip["VSEVINTX"].iloc[i] == '00:00-24:00 ON DAY OF ASSESSMENT':
            suf = '_DAILY'
        elif descrip["VSEVINTX"].iloc[i] == 'BEFORE ACUTE COVID-19 ILLNESS':
            suf = '_ADMIT'

        if den == 0:
            table_counts.append(0)
        else:
            table_counts.append(100 * total / den)
        table_SATERM.append(name + suf)

    df2['Extracted VSTESTCD'] = table_SATERM
    df2[j] = table_counts


df2.set_index('Extracted VSTESTCD', inplace=True)

numeric_columns = data_DM['DB'].unique()
df2 = df2[numeric_columns]
df2['mean'] = df2[numeric_columns].mean(axis=1)

df3 = df2.copy()
df3.reset_index(inplace=True)

df3['mean(excluding 0)'] = np.zeros(len(df3))
df3.loc[(df3 == 0).any(axis=1), 'mean(excluding 0)'] = df2[numeric_columns].replace(0, np.nan).mean(axis=1)

df3['% sites included'] = (100 * (df2[numeric_columns] > 0).sum(axis=1)) / len(numeric_columns)
df3['Coefficient_of_Variation'] = df3[numeric_columns].std(axis=1) / df3[numeric_columns].mean(axis=1)

# Save the final statistics to an Excel file
df3.to_excel(os.path.join(path, 'example_db_VS_means.xlsx'), index=False)