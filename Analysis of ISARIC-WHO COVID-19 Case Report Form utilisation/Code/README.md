# Data Processing and Analysis Scripts

## Overview
This repository contains scripts for processing and analyzing clinical data from various internal sources. The primary focus is on merging, cleaning, and calculating statistical metrics for different datasets related to clinical outcomes.

## Directory Structure
- `PATH/ISARIC/JAN 23/`: The directory containing all the data files.

## Files
### Main Data Files
- `Internal_CQ_2023-01-10.csv`
- `Internal_DS_2023-01-10.csv`
- `Internal_ER_2023-01-10.csv`
- `Internal_HO_2023-01-10.csv`
- `Internal_IE_2023-01-10.csv`
- `Internal_IN_20230110.csv`
- `Internal_LB_20230110_v2.csv`
- `Internal_MB_2023-01-10.csv`
- `Internal_RP_2023-01-10.csv`
- `Internal_RS_2023-01-10.csv`
- `Internal_SA_20230110.csv`
- `Internal_VS_20230110.csv`
- `usub_db.csv`: Database file containing unique subject IDs and the sites IDs.
- `usub_db_sev.csv`: Database file for severe cases subject IDs and the sites IDs.
- `CRF_DM.xlsx`: Description file for demographic data.
- `CRF_outcomes.xlsx`: Description file for clinical outcomes.
- `CRF_IN.xlsx`: Description file for Interventions and treatments.
- `CRF_LB.xlsx`: Description file for laboratories events.
- `CRF_MB.xlsx`: Description file for microbiological events.
- `CRF_SA.xlsx`: Description file for Signs and Symptoms, comorbidities and complications.
- `CRF_VS.xlsx`: Description file for Vital Signs.
- `CRF_outcomes.xlsx`: Description file for clinical outcomes.

## Scripts
### Script 1: Data Processing and Merging
#### Description
- This the processing scripts processes and merges clinical data from multiple sources.

#### Steps
1. **Import Libraries**: Import necessary libraries like pandas, numpy, etc.
2. **Define Paths**: Set the path to the data directory.
3. **Load Files**: Load the main dataset and merge it with other datasets.
4. **Data Cleaning**: Clean and preprocess the data.
5. **Merge Data**: Merge data based on unique subject IDs and other relevant columns.
6. **Calculate Metrics**: Calculate statistical metrics and store them in a DataFrame.
7. **Save Results**: Save the processed data and calculated metrics to an Excel file.


## Usage
### Prerequisites
- Python 3.x
- Required libraries: pandas, numpy, seaborn, matplotlib

### How to Run
1. **Set the Data Path**: Ensure the `path` variable points to the directory containing your data files.
2. **Run the Script**: Execute the script using a Python interpreter.

```sh
python script_name.py