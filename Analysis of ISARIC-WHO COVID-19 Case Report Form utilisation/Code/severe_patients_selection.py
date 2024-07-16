# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:08:30 2023

@author: SDuque
"""

import os
import pandas as pd
import numpy as np

# Define the path
path = 'D:/ISARIC/JAN 23/'

# Load datasets
data_IN = pd.read_csv(os.path.join(path, 'data_INMODIFY2.csv')) # this file is obtained in 'IN_data_pre-processing.py'
data_DM = pd.read_csv(os.path.join(path, 'usub_db.csv'))
data_HO = pd.read_csv(os.path.join(path, 'Internal_HO_2023-01-10.csv'))

# Extract ICU patients
ICU_PATIENTS = data_HO['USUBJID'].loc[
    (data_HO['HODECOD'] == 'INTENSIVE CARE UNIT') & (data_HO['HOOCCUR'] == 'Y')
].unique()

# Define severity conditions
severity = [
    'HIGH FLOW OXYGEN NASAL CANNULA_ANY', 'ARTIFICIAL RESPIRATION_ANY', 'INOTROPES_ANY',
    'NON-INVASIVE VENTILATION_ANY', 'VASOPRESSOR/INOTROPIC_DAILY', 'HIGH FLOW NASAL CANNULA_DAILY',
    'NON-INVASIVE VENTILATION_DAILY', 'ARTIFICIAL RESPIRATION_DAILY', 'DOPAMINE', 
    'NONINVASIVE VENTILATION_TYPE_DAILY'
]

# Extract severe patients
SEVERE_PATIENTS = data_IN['USUBJID'].loc[
    (data_IN['INMODIFY2'].isin(severity)) & (data_IN['INOCCUR'] == 'Y')
].unique()

# Combine ICU and severe patients
severe_set = set(SEVERE_PATIENTS)
icu_set = set(ICU_PATIENTS)
all_SEVERE = list(severe_set.union(icu_set))

# Calculate severity percentages for each database
table_counts = []
for j in data_DM['DB'].unique():
    den = len(data_DM['USUBJID'].loc[data_DM['DB'] == j].unique())
    data_SA_site = data_DM.loc[data_DM['DB'] == j]
    total = len(data_SA_site['USUBJID'].loc[data_SA_site['USUBJID'].isin(all_SEVERE)].unique())
    if den == 0:
        table_counts.append([j, 0, total, den])
    else:
        table_counts.append([j, 100 * total / den, total, den])

# Create DataFrame with results
df2 = pd.DataFrame(table_counts, columns=['DataBase', 'Percentage', 'Total Severe', 'Total in DB'])
df2['Subs'] = df2['Total in DB'] - df2['Total Severe']

# Save severe patients' data to a new CSV file
data_DM_sev = data_DM.loc[data_DM['USUBJID'].isin(all_SEVERE)]
data_DM_sev.to_csv(os.path.join(path, 'usub_db_sev.csv'), index=False)
