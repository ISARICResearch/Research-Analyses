# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 10:13:33 2023

@author: SDuque
"""

import os
import pandas as pd
import numpy as np

# Define the path and file names
path = 'D:/ISARIC/JAN 23/'
table = 'Internal_IN_20230110.csv'
data_dm_file = 'usub_db.csv'
descrip_file = 'crf_IN.xlsx'

# Read the main table and description file
data_IN = pd.read_csv(os.path.join(path, table))
descrip = pd.read_excel(os.path.join(path, descrip_file))
descrip = descrip.loc[descrip['Include'] == 'X']

# Define the group column
group_IN = 'INMODIFY'

# Clean and standardize the data
data_IN[group_IN] = data_IN[group_IN].str.strip().str.upper()
data_IN['INCLAS'] = data_IN['INCLAS'].str.strip().str.upper()
data_IN['INTRT'] = data_IN['INTRT'].str.strip().str.upper()

# Update 'INMODIFY' column based on specific conditions
data_IN.loc[data_IN['INMODIFY'].isin(['RITONAVIR', 'LOPINAVIR AND RITONAVIR', 'LOPINAVIR']), 'INMODIFY'] = 'LOPINAVIR/RITONAVIR'
data_IN.loc[data_IN['INMODIFY'].isin(['CHLOROQUINE']), 'INMODIFY'] = 'CHLOROQUINE / HYDROXYCHLOROQUINE'
data_IN.loc[(~data_IN['INMODIFY'].isin(['LOPINAVIR/RITONAVIR', 'INTERFERON BETA', 'REMDESIVIR', 'CHLOROQUINE / HYDROXYCHLOROQUINE'])) & 
            (data_IN['INCLAS'] == 'ANTIVIRALS FOR SYSTEMIC USE'), 'INMODIFY'] = 'OTHER ANTIVIRALS'

data_IN['INOCCUR2'] = data_IN['INOCCUR'].replace({'Y': 'YES/NO', 'N': 'YES/NO'})

data_IN.loc[data_IN['INMODIFY'].isin(['BIPAP', 'CPAP']), 'INMODIFY'] = 'BIPAP/CPAP'
data_IN.loc[data_IN['INMODIFY'].isin(['NON-INVASIVE RESPIRATORY SUPPORT']), 'INMODIFY'] = 'NON-INVASIVE VENTILATION'
data_IN.loc[(data_IN['INCLAS'] == 'PSYCHOLEPTICS') & (data_IN['INMODIFY'] != 'SEDATION'), 'INMODIFY'] = 'OTHER SEDATION'

covid_vaccine_conditions = [
    'COVID-19 VACCINATION', 'COVID-19 VACCINE CANSINBIO', 'COVID-19 VACCINE TYPE UNKNOWN',
    'COVID-19 VACCINE PFIZER-BIONTECH', 'COVID-19 VACCINE JANSSENS (JOHNSON AND JOHNSON)',
    'COVID-19 VACCINE ASTRAZENECA/UNIVERSITY OF OXFORD', 'COVID-19 VACCINE SPUTNIK V', 
    'COVID-19 VACCINE MODERNA', 'COVID-19 VACCINE NOVAVAX', 'COVID-19 VACCINE SINOVAC',
    'COVID-19 VACCINE SINOPHARM', 'COVID-19 VACCINE COVAXIN'
]
data_IN.loc[data_IN['INMODIFY'].isin(covid_vaccine_conditions), 'INMODIFY'] = 'COVID VACCINATION'

dopamine_conditions = [
    'DOPAMINE < 5 UG/KG/MIN OR DOBUTAMINE OR MILRINONE OR LEVOSIMENDAN',
    'DOPAMINE 5-15 UG/KG/MIN OR EPINEPHRINE/NOREPINEPHRINE <0.1 UG/KG/MIN OR VASOPRESSIN OR PHENYLEPHRINE',
    'DOPAMINE > 15 UG/KG/MIN OR EPINEPHRINE/NOREPINEPRINE > 0.1. UG/KG/MIN',
    'DOPAMINE > 15 UG/KG/MIN OR EPINEPHRINE/NOREPINEPHRINE > 0.1 UG/KG/MIN',
    'DOBUTAMINE OR MILRINONE OR LEVOSIMENDAN (ANY DOSE) OR DOPAMINE < 5 MCG/MIN',
    'EPINEPHRINE/NOREPINEPHRINE < 0.1 MCG/KG.MIN OR VASOPRESSIN (ANY DOSE) OR PHENYLEPHRINE OR DOPAMINE 5-15 MCG/MIN',
    'EPINEPHRINE/NOREPINEPHRINE > 0.1 MCG/KG/MIN OR DOPAMINE > 15 MCG/MIN',
    'DOPAMINE 5-15 UG/MIN OR EPINEPHRINE/NOREPINEPHRINE < 0.1 UG/KG/MIN OR VASOPRESSIN OR PHENYLEPHRINE'
]
data_IN.loc[data_IN['INMODIFY'].isin(dopamine_conditions), 'INMODIFY'] = 'DOPAMINE'

# Filter rows based on 'INEVINTX'
data_IN = data_IN.loc[data_IN['INEVINTX'].isin(['00:00-24:00 ON DAY OF ASSESSMENT', 'WITHIN 14 DAYS OF ADMISSION', 'AT ANY TIME DURING HOSPITALIZATION'])]

# Merge with additional data
data_DM = pd.read_csv(os.path.join(path, data_dm_file))
data_IN = pd.merge(data_IN, data_DM, on='USUBJID', how='left')


# Further processing based on descriptions
descrip_short = descrip[['Extracted INTRT', 'TRT', 'INEVINTX', 'DIFF_TYPE', 'DIFF_TYPE_TRT', 'INMODIFY2']].dropna(subset=['Extracted INTRT', 'INEVINTX', 'INMODIFY2']).drop_duplicates()

data_IN = data_IN.loc[data_IN['INEVINTX'].isin(descrip_short['INEVINTX'].unique())]
data_IN['INMODIFY2'] = np.zeros(len(data_IN))

# Update 'INMODIFY2' based on conditions from descrip_short
for i in range(len(descrip_short)):
    if pd.isna(descrip_short['DIFF_TYPE'].iloc[i]) or descrip_short['DIFF_TYPE'].iloc[i] == 'INMODIFY2':
        data_IN['INMODIFY2'].loc[(data_IN['INEVINTX'] == descrip_short['INEVINTX'].iloc[i]) & (data_IN[descrip_short['Extracted INTRT'].iloc[i]] == descrip_short['TRT'].iloc[i])] = descrip_short['INMODIFY2'].iloc[i]
    else:
        data_IN['INMODIFY2'].loc[(data_IN['INEVINTX'] == descrip_short['INEVINTX'].iloc[i]) & (data_IN[descrip_short['Extracted INTRT'].iloc[i]] == descrip_short['TRT'].iloc[i]) & (data_IN[descrip_short['DIFF_TYPE'].iloc[i]] != descrip_short['DIFF_TYPE_TRT'].iloc[i])] = descrip_short['INMODIFY2'].iloc[i]

# Save the final processed data to CSV
data_IN.to_csv(os.path.join(path, 'data_INMODIFY2.csv'), index=False)
