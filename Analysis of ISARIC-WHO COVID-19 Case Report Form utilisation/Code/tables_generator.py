import os
import pandas as pd
import numpy as np

# Define the path and list the files in the directory
path = 'D:/ISARIC/JAN 23/'
arr = os.listdir(path)
# df and dfsev are generated in Scatter_plots_domains.py
# Load data
df = pd.read_excel(path + 'example_db_all_means.xlsx')
df['distance'] = 100 * df['distance']

df_sev = pd.read_excel(path + 'example_db_all_means_sev.xlsx')
df_sev = df_sev.rename(columns={'distance': 'distance severe'})
df_sev['distance severe'] = 100 * df_sev['distance severe']

# Correlation data preparation
correlation_all_data = df[['mean(excluding 0)', '% sites included', 'Coefficient_of_Variation', 'distance']]
correlation_all_data = correlation_all_data.rename(columns={'distance': 'Distance', 'mean(excluding 0)': 'Completion rate',
                                                            '% sites included': 'Inclusion rate', 'Coefficient_of_Variation': 'Coefficient of variation'})

correlation_severe_data = df_sev[['mean(excluding 0)', '% sites included', 'Coefficient_of_Variation', 'distance severe']]
correlation_severe_data = correlation_severe_data.rename(columns={'distance severe': 'Distance', 'mean(excluding 0)': 'Completion rate',
                                                                  '% sites included': 'Inclusion rate', 'Coefficient_of_Variation': 'Coefficient of variation'})

# Calculate correlations and save to Excel
correlation_all = correlation_all_data.corr().round(2)
correlation_severe = correlation_severe_data.corr().round(2)
correlation_all.to_excel(path + 'correlation_all.xlsx')
correlation_severe.to_excel(path + 'correlation_severe.xlsx')

# Load group information and merge with main data
groups = pd.read_excel(path + 'variable_names.xlsx', sheet_name='groups')
df = pd.merge(df, groups, left_on='Extracted TERM', right_on='New Name', how='left')

# Load and process data
table = 'Internal_DM_site_2023-01-10_v2.csv'
data = pd.read_csv(path + table)
data_DM = pd.read_csv(path + 'usub_db.csv')
data = pd.merge(data, data_DM, on='USUBJID', how='left')

# Load and process severe data
data_DM_sev = pd.read_csv(path + 'usub_db_sev.csv')
data_sev = data_DM_sev[['DB', 'USUBJID']].rename(columns={'USUBJID': 'N Severe Patients (%)'})
data_sev = data_sev.groupby(['DB']).nunique().reset_index()

# Aggregate data by DB
data1 = data[['DB', 'siteid_final', 'COUNTRY', 'USUBJID', 'RFSTDTC']]
data1['RFSTDTC'] = pd.to_datetime(data1['RFSTDTC'])

# Filter data based on specific criteria
data1.loc[(data1['DB'] == 'CVVCORE') & (data1['RFSTDTC'] <= pd.to_datetime('2020-01-24')), 'RFSTDTC'] = pd.to_datetime('2020-01-24')
data1.loc[(data1['DB'] == 'CVCCPUK') & (data1['RFSTDTC'] <= pd.to_datetime('2020-01-30')), 'RFSTDTC'] = pd.to_datetime('2020-01-30')
data1 = data1.loc[((~data1['DB'].isin(['CVCCPUK', 'CVVCORE'])) & (data1['RFSTDTC'] > pd.to_datetime('2020-01-30'))) | (data1['DB'].isin(['CVCCPUK', 'CVVCORE']))]
data1 = data1.loc[(data1['RFSTDTC'] <= '2023-03-20') & (data1['RFSTDTC'].notna())]

# Calculate collection dates
fechas_max = data1[['DB', 'RFSTDTC']].groupby(['DB']).max().reset_index()
fechas_min = data1[['DB', 'RFSTDTC']].groupby(['DB']).min().reset_index()
fechas = pd.merge(fechas_max, fechas_min, how='inner', on='DB')
fechas['Collection'] = fechas['RFSTDTC_y'].dt.strftime('%Y-%m-%d') + '_' + fechas['RFSTDTC_x'].dt.strftime('%Y-%m-%d')

# Merge and calculate severe patient data
data2 = data1.groupby(['DB']).nunique().reset_index().drop(columns='RFSTDTC')
data3 = pd.merge(data2, fechas[['DB', 'Collection']], how='left', on='DB')
data3 = pd.merge(data3, data_sev, how='left', on='DB')
data3['N Severe Patients (%)'] = data3['N Severe Patients (%)'].fillna(0)
data3['N Severe Patients (%)'] = data3['N Severe Patients (%)'].astype('Int64').astype(str) + ' (' + (round(100 * data3['N Severe Patients (%)'] / data3['USUBJID'], 1)).astype(str) + ')'

# Append median and IQR values
median_values = data3.median(axis=0).rename('Median')
quantile_025 = data3.quantile(q=0.25, axis=0)
quantile_075 = data3.quantile(q=0.75, axis=0)
IQR = (quantile_025.astype(str) + '-' + quantile_075.astype(str)).rename('IQR')

data3 = data3.append(median_values).append(IQR)
data3.to_excel(path + 'table1.xlsx')

# Group data by 'Group crf'
df2 = df[[col for col in data3.columns if col in df.columns] + ['Group crf']]
df2['total'] = 1
df2[df2.columns.difference(['Group crf', 'total'])] = df2[df2.columns.difference(['Group crf', 'total'])].applymap(lambda x: 1 if x > 0 else x)
df2 = df2.groupby('Group crf').sum().reset_index()

# Add total row
suma_filas_anteriores = df2.sum()
df2 = df2.append(suma_filas_anteriores, ignore_index=True)
df2['Group crf'].iloc[-1] = f'Total Variables (%) (N= {df2["total"].iloc[-1]})'

# Calculate and format statistics
df2.set_index('Group crf', inplace=True)
median_values = df2.drop(columns=['total']).median(axis=1)
quantile_025 = df2.drop(columns=['total']).quantile(q=0.25, axis=1)
quantile_075 = df2.drop(columns=['total']).quantile(q=0.75, axis=1)
df2['Median'] = median_values.astype(str) + ' (' + (round(100 * median_values / df2['total'], 1)).astype(str) + '%)'
df2['IQR'] = quantile_025.astype(str) + '-' + quantile_075.astype(str)
mean_values = df2.drop(columns=['total']).mean(axis=1)
std_deviation = df2.drop(columns=['total']).std(axis=1)
df2['Mean'] = round(mean_values, 1).astype(str) + ' (' + (round(100 * mean_values / df2['total'], 1)).astype(str) + '%)'
df2['SD'] = round(std_deviation, 1).astype(str)

# Format columns
for column in df2.columns.difference(['total', 'Median', 'IQR', 'Mean', 'SD']):
    df2[column] = df2[column].astype('Int64').astype(str) + ' (' + (round(100 * df2[column] / df2['total'], 1)).astype(str) + '%)'

df2 = df2.drop(columns=['total']).transpose()

# Reorder columns
df2 = df2[['Total Variables (%) (N= 243)', 'Demograhics N=13', 'Vital signs (A) N=11', 'Signs & symptoms N=29', 
           'Comorbidities N=21', 'Pre-admission medication N=2', 'Lab tests (A) N=27', 'Vitals & assessments N=21', 
           'Interventions (D) N=15', 'Lab tests (D) N=27', 'Interventions(S) N=25', 'Diagnostics N=14', 
           'Complications N=32', 'Outcome status N=2', 'Post-outcome N=4']]

df2.to_excel(path + 'table2.xlsx')

# Function to assign color based on distance values
def assign_color(row):
    if row['distance severe'] > 70:
        return '#8A1A0C'
    elif row['distance severe'] < 70 and row['distance'] > 70 and row['distance'] != row['distance severe']:
        return '#fc7703'
    elif row['distance severe'] < 30 and row['distance'] > 30 and row['distance'] != row['distance severe']:
        return '#7CE81C'
    elif row['distance severe'] < 30:
        return '#016C11'
    elif 30 <= row['distance severe'] <= 70:
        return '#fff419'

# Merge data and calculate delta
df3 = pd.merge(groups, df[['Extracted TERM', 'distance', 'mean(excluding 0)', '% sites included']],
               right_on='Extracted TERM', left_on='New Name', how='left').drop(columns=['Extracted TERM'])
df3 = pd.merge(df3, df_sev[['Extracted TERM', 'distance severe', 'mean(excluding 0)', '% sites included']],
               right_on='Extracted TERM', left_on='New Name', how='left')
df3['delta'] = df3['distance'] - df3['distance severe']
df3 = df3.drop(columns=['Extracted TERM'])
df3['Color'] = df3.apply(assign_color, axis=1)
df3 = df3.sort_values(by=['distance'])
df3['distance'] = round(df3['distance'], 1)
df3['distance severe'] = round(df3['distance severe'], 1)

# Create and format table3
table3 = df3.rename(columns={'New Name': 'Variable', 'distance': 'Distance(all)', 'mean(excluding 0)_x': 'Completeness %(all)',
                             '% sites included_x': 'sites included % (all)', 'distance severe': 'Distance(severe)', 
                             'mean(excluding 0)_y': 'Completeness %(severe)', '% sites included_y': 'sites included % (severe)'})
table3['Distance (All)'] = table3['Distance(all)'].astype(str) + ' (' + round(table3['Completeness %(all)'], 1).astype(str) + ',' + round(table3['sites included % (all)'], 1).astype(str) + ')'
table3['Distance (Severe)'] = table3['Distance(severe)'].astype(str) + ' (' + round(table3['Completeness %(severe)'], 1).astype(str) + ',' + round(table3['sites included % (severe)'], 1).astype(str) + ')'
table3['delta'] = round(table3['delta'], 1)
table3 = table3[['General group', 'Group crf', 'Variable', 'Distance (All)', 'Distance (Severe)', 'delta', 'Color']]
table3.to_excel(path + 'table3_supplementary.xlsx')

# Group data and calculate statistics for summary table
df4 = df3.drop(columns=['General group', 'New Name', 'delta', 'Color']).fillna(0)
grouped = df4.groupby('Group crf')[['distance', 'mean(excluding 0)_x', '% sites included_x', 'distance severe', 'mean(excluding 0)_y', '% sites included_y']]

mean_all = df4.mean()
std_all = df4.std()
median = grouped.mean()
median.loc['All variables'] = mean_all
iqr_range = grouped.std()
iqr_range.loc['All variables'] = std_all

# Apply the function to create the new column
median['Color'] = median.apply(assign_color, axis=1)

# Combine median and IQR into a DataFrame
table4 = pd.DataFrame({
    'Group crf': median.index,
    'Distance mean all(SD)': round(median['distance'], 1).astype(str) + ' (' + round(iqr_range['distance'], 1).astype(str) + ')',
    'Completion mean all (SD)': round(median['mean(excluding 0)_x'], 1).astype(str) + ' (' + round(iqr_range['mean(excluding 0)_x'], 1).astype(str) + ')',
    'Inclusion mean all(SD)': round(median['% sites included_x'], 1).astype(str) + ' (' + round(iqr_range['% sites included_x'], 1).astype(str) + ')',
    'Distance mean severe(SD)': round(median['distance severe'], 1).astype(str) + ' (' + round(iqr_range['distance severe'], 1).astype(str) + ')',
    'Completion mean severe (SD)': round(median['mean(excluding 0)_y'], 1).astype(str) + ' (' + round(iqr_range['mean(excluding 0)_y'], 1).astype(str) + ')',
    'Inclusion mean severe (SD)': round(median['% sites included_y'], 1).astype(str) + ' (' + round(iqr_range['% sites included_y'], 1).astype(str) + ')',
    'Delta (SD)': (round(median['distance'] - median['distance severe'], 1)).astype(str) + ' (' + (round(iqr_range['distance'] - iqr_range['distance severe'], 1)).astype(str) + ')',
    'Color': median['Color']
})

table4 = table4.sort_values(by=['distance'])
table4 = table4.drop(columns=['distance'])

# Merge with group information and save to Excel
groups = groups[['General group', 'Group crf']].drop_duplicates()
table4 = pd.merge(groups[['General group', 'Group crf']], table4, on='Group crf', how='right')
table4.to_excel(path + 'table3_summarized.xlsx')
