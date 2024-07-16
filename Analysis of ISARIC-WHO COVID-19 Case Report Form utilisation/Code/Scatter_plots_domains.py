# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 09:53:07 2023

@author: SDuque
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# Define the path and list the files in the directory
path = 'D:/ISARIC/JAN 23/'
arr = os.listdir(path)

# Function to calculate scores
def calculate_scores_updated(df):
    df['normalized_mean'] = df['mean(excluding 0)'] / 100
    df['normalized_sites_included'] = df['% sites included'] / 100
    df['normalized_coeff_variation'] = np.exp(-df['Coefficient_of_Variation'])
    df['combined_score'] = df[['normalized_mean', 'normalized_sites_included', 'normalized_coeff_variation']].mean(axis=1)
    return df

# Function to calculate Euclidean distance
def calculate_distance(row):
    target_point = (100, 100)
    a = euclidean([row['mean(excluding 0)'], row['% sites included']], target_point)
    b = a / ((2 * (100 ** 2)) ** 0.5)
    return b

# Function to plot scatter distances
def scatter_distances(df, domain):
    df = df.rename(columns={'mean(excluding 0)': 'Mean', '% sites included': 'Sites_Included', 'distance': 'Distance'})

    def get_color(distance):
        if distance < 0.3: 
            return '#016C11'
        elif distance > 0.7:
            return '#8A1A0C'
        else:
            return '#fff419'

    df['Color'] = df['Distance'].apply(get_color)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=500)
    scatter = ax.scatter(df['Mean'], df['Sites_Included'], c=df['Color'], s=100)

    sorted_terms = df[['Extracted TERM', 'Distance']].drop_duplicates().sort_values(by='Distance')['Extracted TERM']
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=term,
                                  markerfacecolor=get_color(df[df['Extracted TERM'] == term]['Distance'].iloc[0]), markersize=10)
                       for term in sorted_terms]

    if len(sorted_terms) > 30:
        ax.legend(handles=legend_elements, title='Variable', loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2)
    else:
        ax.legend(handles=legend_elements, title='Variable', loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=len(sorted_terms) // 5 + 1)
        
    ax.set_xlabel('Mean completed proportion (% patients)', fontsize=20)
    ax.set_ylabel('Mean included proportion (% sites)', fontsize=20)
    ax.set_title(f'Utility distance in all patients, n=950064', fontsize=25)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()

# Function to plot comparative scatter distances
def scatter_distances_comp(df, domain, df_comp):
    df = df.rename(columns={'mean(excluding 0)': 'Mean', '% sites included': 'Sites_Included', 'distance': 'Distance'})
    df_comp = df_comp.rename(columns={'mean(excluding 0)': 'Mean', '% sites included': 'Sites_Included', 'distance': 'Distance'})

    def get_color(distance, distance_comp):
        if distance > 0.7:
            return '#8A1A0C'
        elif distance < 0.7 and distance_comp > 0.7 and distance_comp != distance:
            return '#fc7703'
        elif distance < 0.3 and distance_comp > 0.3 and distance_comp != distance:
            return '#7CE81C'
        elif distance < 0.3:
            return '#016C11'
        elif 0.3 <= distance <= 0.7:
            return '#fff419'

    df['Color'] = df.apply(lambda row: get_color(row['Distance'], df_comp.at[row.name, 'Distance']), axis=1)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    scatter = ax.scatter(df['Mean'], df['Sites_Included'], c=df['Color'], s=100)

    sorted_terms = df_comp[['Extracted TERM', 'Distance']].drop_duplicates().sort_values(by='Distance')['Extracted TERM']
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=term,
                                  markerfacecolor=get_color(df[df['Extracted TERM'] == term]['Distance'].iloc[0], df_comp[df_comp['Extracted TERM'] == term]['Distance'].iloc[0]), markersize=10)
                       for term in sorted_terms]

    if len(sorted_terms) > 20:
        legend = ax.legend(handles=legend_elements, title='Variable', loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2)
    else:
        legend = ax.legend(handles=legend_elements, title='Variable', loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=len(sorted_terms) // 3 + 1)
    for text in legend.get_texts():
        text.set_color('white')

    ax.set_xlabel('Mean completed proportion (% patients)', fontsize=20)
    ax.set_ylabel('Mean included proportion (% sites)', fontsize=20)
    ax.set_title(f'Utility distance in severe patients, n=256529', fontsize=25)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()

# Load and process data
domains = ['DM', 'VS', 'SA', 'IN', 'LB', 'MB', 'DSHO']
thr = [170, 60, 150, 80, 35, 60, 100]
df = pd.DataFrame()
cont = 0

for i in domains:
    df_temp = pd.read_excel(path + 'example_db_' + i + '_means.xlsx')
    df_temp = df_temp.rename(columns={df_temp.columns[1]: 'Extracted TERM'})
    df_temp = df_temp.drop_duplicates()
    df = pd.concat([df, df_temp], ignore_index=True)
    df_temp = df_temp.fillna(0)
    df_temp['distance'] = df_temp.apply(calculate_distance, axis=1)
    cont += 1

df2 = df.fillna(0)
df['distance'] = df2.apply(calculate_distance, axis=1)
df = df.drop_duplicates(subset=['Extracted TERM'])
names = pd.read_excel(path + 'variable_names.xlsx', sheet_name='rename')
df = pd.merge(df, names, on='Extracted TERM', how='left')
df['Extracted TERM'] = df['New Name']
df = df.drop(columns='New Name')
df.to_excel(path + 'example_db_all_means.xlsx')

# Calculate scores and plot distances
scored_data_updated = calculate_scores_updated(df[['Extracted TERM', 'mean(excluding 0)', '% sites included', 'Coefficient_of_Variation']])
data = df[['Extracted TERM', 'mean(excluding 0)', '% sites included', 'Coefficient_of_Variation']]
data2 = data.copy().fillna(0)
data2['distance'] = data2.apply(calculate_distance, axis=1)
data['distance'] = data2['distance']

scatter_distances(data2, 'all')
scatter_distances_comp(data, 'all severe', data2)

# Merging with groups data and plotting
groups = pd.read_excel(path + 'variable_names.xlsx', sheet_name='groups')
groups['General group'].loc[groups['General group'] == 'Outcome'] = 'Summary'
data2 = pd.merge(data2, groups, right_on='New Name', left_on='Extracted TERM', how='left')
data3 = pd.merge(data, groups, right_on='New Name', left_on='Extracted TERM', how='left')

# Function to plot scatter distances for various groups
def scatter_distances_various(ax, df, domain):
    df = df.rename(columns={'mean(excluding 0)': 'Mean', '% sites included': 'Sites_Included', 'distance': 'Distance'})

    def get_color(distance):
        if distance < 0.3:
            return '#016C11'
        elif distance > 0.7:
            return '#8A1A0C'
        else:
            return '#fff419'

    df['Color'] = df['Distance'].apply(get_color)
    scatter = ax.scatter(df['Mean'], df['Sites_Included'], c=df['Color'], s=100)
    ax.set_title(f'Utility distance in {domain}', fontsize=20)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True)

unique_groups = data2['General group'].unique()

for group in unique_groups:
    df = data2.loc[data2['General group'] == group]
    scatter_distances(df, group)
    df2 = data3.loc[data3['General group'] == group]
    scatter_distances_comp(df2, group, df)
