# Mpox South Kivu 2024 study

These are the figures and tables for the Mpox study in South Kivu, DRC. The attached code is to process the raw data (saving a clean CSV and a data dictionary) and to perform the data analyses (saving data and metadata files used by ISARIC VERTEX tool, commit [0a41749](https://github.com/ISARICResearch/VERTEX/tree/0a417492c342475836c83a8009677edb7f81d85f)). 

This does not include the raw data. Multiple versions of the raw data exist (e.g. after data quality queries were raised and resolved). The version of the raw data that was used to create the outputs was called 'Check-Kamituga_mpox_dataset_MRC_French_(6)_LMM_Final_075818(2).xlsx". The code may or may not work for previous or later versions of the data. 

To reproduce the outputs, you will need access to the raw data. If you have access to the raw data, you should be able to run both scripts in order (`data_processing.py` and `analysis.py`) after changing the filepaths at the top of each script. 

After this, you should be able to run an ISARIC VERTEX dashboard (at the time of writing, the commit is [0a41749](https://github.com/ISARICResearch/VERTEX/tree/0a417492c342475836c83a8009677edb7f81d85f)), which will take the files saved and create the figures.

Package requirements are listed in `requirements.txt`, please ensure these are installed first. Alternatively you can use the requirements from VERTEX.

Add a link to the paper once here once published.
