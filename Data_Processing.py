# Standard library imports
import os
import re
import random
from datetime import datetime, timedelta
import warnings
from math import floor
import shutil
import statistics as st
from os import listdir
from os.path import isfile, join

# Third-party imports for data handling and visualization
import pandas as pd
import numpy as np
import tsfel
import openpyxl
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Machine learning and optimization libraries
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import RFECV, SelectFromModel
import xgboost as xgb
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real

# Deep learning libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D, Input
from tensorflow.keras.optimizers import Adam, Adam as TfAdam
from tensorflow.keras.metrics import AUC, BinaryAccuracy, Precision, Recall, Accuracy
from tensorflow.keras.callbacks import Callback
from tensorflow_addons.metrics import F1Score

# PyTorch libraries
import torch
from torch import nn
import torch.nn.functional as F
import torchvision

# Optimization and experiment tracking tools
import optuna
from optuna.exceptions import TrialPruned
import joblib
import wandb

# To ignore all warnings
warnings.filterwarnings("ignore")

import random
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

# Check if any GPUs are available
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # Set TensorFlow to use the first GPU if available
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Using GPU:", gpus[0])
    except RuntimeError as e:
        # Memory growth must be set before initializing GPUs
        print(e)
else:
    print("No GPU found, using CPU instead.")



# %%
def update_csv_date(csv_filename, filepath):
    """
    Updates the date and time in a CSV file based on the filename or 'Time' column.

    Args:
        csv_filename (str): The name of the CSV file.
        filepath (str): The path where the CSV file is located.

    Returns:
        None
    """
    # Check if the CSV file exists
    if os.path.exists(os.path.join(filepath, csv_filename)):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(os.path.join(filepath, csv_filename))
        length = len(csv_filename)

        # Extract year, month, and day from the CSV filename
        year_frm_csv = int(csv_filename[length - 12:length - 8])
        month_frm_csv = int(csv_filename[length - 8:length - 6])
        day_frm_csv = int(csv_filename[length - 6:length - 4])

        try:
            # Try to parse 'Time' column as datetime using ISO8601 format
            df['Time'] = pd.to_datetime(df['Time'], format='ISO8601')
        except ValueError:
            # If parsing with the time_format fails, fall back to "%H:%M:%S"
            df['Time'] = pd.to_datetime(df['Time'], format="%H:%M:%S")

        # Iterate through DataFrame rows
        for index, row in df.iterrows():
            time_from_df = row['Time']
            
            # Adjust the date for 'sleep' CSV files where time is exactly midnight
            if 'sleep' in csv_filename.lower():
                if time_from_df.hour == 0 and time_from_df.minute == 0 and time_from_df.second == 0:
                    day_frm_csv += 1
                    if month_frm_csv == 12:
                        days_in_month = pd.Timestamp(year_frm_csv + 1, 1, 1) - pd.Timestamp(year_frm_csv, month_frm_csv, 1)
                    else:
                        days_in_month = pd.Timestamp(year_frm_csv, month_frm_csv + 1, 1) - pd.Timestamp(year_frm_csv, month_frm_csv, 1)
                    if day_frm_csv > days_in_month.days:
                        day_frm_csv = 1
                        month_frm_csv += 1
                        if month_frm_csv > 12:
                            month_frm_csv = 1
                            year_frm_csv += 1
            
            # Update the 'Time' column with the adjusted date and time
            df.at[index, 'Time'] = datetime(year_frm_csv, month_frm_csv, day_frm_csv, time_from_df.hour, time_from_df.minute, time_from_df.second)
            
        # Write the modified DataFrame back to the CSV file
        df.to_csv(os.path.join(filepath, csv_filename), encoding="utf-8", index=False)
        # Uncomment the line below if you want to print a message indicating the update
        # print("Updated: " + csv_filename + " located at: " + filepath)
    else:
        # Print an error message if the CSV file does not exist
        print("Error in updating the time & date of the following CSV file: " + csv_filename + " located at: " + filepath)

def survey_score_calc(input_file):
    """
    Calculate survey scores from an input Excel file and return a modified DataFrame.

    Parameters:
    - input_file (str): Path to the input Excel file containing survey data.

    Returns:
    - pd.DataFrame: Modified DataFrame with calculated survey scores.
    """

    # Read the Excel file into a DataFrame
    df = pd.read_excel(input_file)
    
    # Check if the required columns exist in the DataFrame
    required_columns = ['PSS_1', 'PSS_2', 'PSS_3', 'PSS_4']
    if not set(required_columns).issubset(df.columns):
        print(f"Required columns not found in {input_file}")
        return pd.DataFrame()
    else:
        # Check if 'PSS (Sum)' column already exists
        if 'PSS (Sum)' not in df.columns:
            # Specify the columns to sum for PSS
            PSS_individual = ['PSS_1', 'PSS_2', 'PSS_3', 'PSS_4']
            
            # Add a new column 'PSS (Sum)' with the sum across specified columns
            df['PSS (Sum)'] = df[PSS_individual].sum(axis=1)
            
        # Check if 'STAI (Sum)' column already exists
        if 'STAI (Sum)' not in df.columns:
            # Specify the columns to sum for STAI
            STAI_individual =  [f'STAI_St_{i}' for i in range(1, 21)]
            
            # Add a new column 'STAI (Sum)' with the sum across specified columns
            df['STAI (Sum)'] = df[STAI_individual].sum(axis=1)

            # Replace 0 sum in 'STAI (Sum)' with NaN
            df.loc[df['STAI (Sum)'] == 0, 'STAI (Sum)'] = np.nan
    
        # Return a subset of the DataFrame with specific columns
        return df

def get_Modality(file):
    """
    Extracts a specific Modality pattern from a string.

    This function searches for a pattern in the input string where a sequence of alphanumeric characters is immediately followed by '202'. 
    If such a pattern is found, it returns the preceding alphanumeric characters. 
    If no such pattern is found, it returns 'Unknown'.

    Parameters:
    - file (str): A string, typically representing a filename or file content, where the pattern is to be searched.

    Returns:
    - str: The extracted pattern (alphanumeric characters before '202') if found; otherwise, 'Unknown'.
    """

    # Search for a pattern '\w+202' in the 'file' string
    mod_match = re.search(r'(\w+)202', file)

    return mod_match.group(1)

def get_student_id(student):
    """
    Extracts a student ID from a string based on predefined patterns.

    This function searches the input string for a sequence of digits preceded by "20." or "21." and returns these digits as the student ID.
    If no such pattern is found, it returns None.

    Parameters:
    - student (str): A string containing the student information, expected to include a pattern like '20.XXXX' or '21.XXXX'.

    Returns:
    - str or None: The extracted student ID as a string if a matching pattern is found; otherwise, None.
    """

    # Search for a pattern '20.\d+' in the 'student' string
    match = re.search(r'20\.(\d+)', student)

    # If a match is found, return the first group (digits after '20.')
    if match:
        student_id= match.group(1)

    # If no match is found for the first pattern, search for a pattern '21.\d+'
    else:
        match = re.search(r'21\.(\d+)', student)

        # If a match is found, return the first group (digits after '21.')
        if match:
            student_id= match.group(1)

    # Return None if no patterns are matched
    return student_id

def processing_excel_to_dict(data_folder, aggregation_level, time_period):
    """
    Processes and merges student data from various modalities and surveys from specified directories.

    This function iterates over directories containing student modality data (e.g., heart rate, calories) and survey data,
    extracting and processing each into a structured format. The data from each modality is resampled according to a given 
    aggregation level and summarized over specified time periods. The processed modality and survey data are returned as 
    separate DataFrames.

    Parameters:
        data_folder (str): The path to the folder containing student data directories.
        aggregation_level (str): The granularity of data aggregation, e.g., '1H' for hourly.
        time_period (str): The length of the interval for summarizing data, e.g., '2W' for two weeks.

    Returns:
        tuple: A tuple containing two pandas DataFrames:
            - The first DataFrame contains resampled and pivoted modality data for each student.
            - The second DataFrame contains processed survey data for each student.

    Processing Steps:
        1. The function first initializes empty DataFrames for modality and survey data.
        2. It iterates through each student's directory, processing modality data files by reading, resampling, and pivoting the data
           based on specified time periods.
        3. Survey data is extracted from Excel files, processed, and concatenated into a DataFrame.
        4. Both DataFrames are returned for further use or analysis.

    Notes:
        - The function assumes that the directory structure and file naming conventions are consistent and that the required data
          exists in the expected format.
        - The modality data is processed through resampling and interpolating to handle missing or irregular time series data.
        - Survey data files are expected to be in Excel format and contain specific columns like 'Student ID' and 'CESD'.
    """
    # Initialize dictionaries and dataframes to store processed data
    All_Student_Dict = {}
    all_student_mod_data = pd.DataFrame()
    all_student_survey_data = pd.DataFrame()
    all_student_weekly_data = pd.DataFrame()

   # Define columns for the weekly data DataFrame, including ID, modality, dates, and hourly slots for each day of the week
    columns = ['Student ID', 'Modality', 'Start Date', 'End Date'] + [f'{d}_{h}' for d in range(7) for h in range(24)]
    all_student_weekly_data = pd.DataFrame(columns=columns)


    for terms in os.listdir(data_folder):
        terms_dir = os.path.join(data_folder, terms)
        if os.path.isdir(terms_dir):
            for student in os.listdir(terms_dir):
                student_dir = os.path.join(terms_dir, student)
                if os.path.isdir(student_dir):
                    student_df = pd.DataFrame()
                    student_weekly_df = pd.DataFrame()
                    student_id = get_student_id(student)
                    print(student)

                    fitbit_dir = os.path.join(student_dir, 'Fitbit')
                    if not os.path.exists(fitbit_dir):
                        print(f"Directory not found: {fitbit_dir}")
                        continue  # Skip to the next iteration if directory is not found

                    for file in os.listdir(fitbit_dir):
                        file_dir = os.path.join(fitbit_dir, file)
                        if file.endswith(".csv"):
                            # Extract mod from the file name
                            mod = get_Modality(file)
                            
                            if mod in ['heart', 'calories', 'step', 'distance', 'sleep']:
                                update_csv_date(file, os.path.join(student_dir, 'Fitbit/'))
                                daily_df = pd.read_csv(file_dir)
                                if mod.lower() == "heart":
                                    mod_col = "Heart Rate"
                                elif mod.lower() == "sleep":
                                    mod_col = "State"
                                else:
                                    mod_col = mod.title()

                                # Create a subset DataFrame with only 'modcolumn', 'Time', and 'Student ID'
                                daily_df = daily_df[['Time', mod_col]]
                                daily_df['Mod'] = mod
                                daily_df = daily_df.rename(columns={mod_col: 'Mod Values'})

                                # Concatenate daily data to the mod DataFrame
                                student_df = pd.concat([student_df, daily_df], ignore_index=True)



                    # Check if 'Mod' column is present in the final DataFrame
                    if 'Mod' in student_df.columns:
                        
                        for mod in ['heart', 'calories', 'step', 'distance', 'sleep']:
                            mod_df=student_df[student_df['Mod'] == mod]
                            if not mod_df.empty:

                                mod_df.drop('Mod', axis=1, inplace=True)

                                # Convert 'mod_col' to numeric data type
                                mod_df['Mod Values'] = pd.to_numeric(mod_df['Mod Values'], errors='coerce')
                                
                                if mod.lower()=='heart':
                                    # Organize by time index
                                    mod_df['Mod Values'] = mod_df['Mod Values'].replace(0, np.nan)
                                    # Convert 'mod_col' to numeric data type
                                    mod_df=mod_df.dropna()
                                    # Organize by time index
                                    mod_df['Time'] = pd.to_datetime(mod_df['Time'])
                                    mod_df.set_index('Time', inplace=True)
                                    mod_df_resampled = mod_df.resample(f'{aggregation_level}').mean()

                                    # Replace 0s with NaNs before applying the window mean
                                    mod_df_resampled['Mod Values'] = mod_df_resampled['Mod Values'].replace(0, np.nan)

                                    mod_df_resampled['Mod Values'].interpolate(method='linear', inplace=True)
                                    mod_df_resampled['Mod'] = mod
                                    mod_df_resampled.reset_index(inplace=True)
                                    mod_df_resampled.loc[:, 'Time'] = pd.to_datetime(mod_df_resampled['Time'])

        
                                    # print(mod_df_resampled)
                                elif mod.lower() == 'sleep':
                                    # Organize by time index
                                    mod_df['Time'] = pd.to_datetime(mod_df['Time'])
                                    mod_df.set_index('Time', inplace=True)

                                    # Convert Mod Values to 1 for the presence of 1 and 0 otherwise
                                    mod_df['Mod Values'] = (mod_df['Mod Values'] == 1).astype(int)

                                    # Resample and sum the 1s in the Mod Values
                                    mod_df_resampled = mod_df.resample(f'{aggregation_level}').sum()
                                    mod_df_resampled['Mod'] = mod

                                    # Replace NaNs after resampling, as the resampling may introduce new NaNs
                                    mod_df_resampled['Mod Values'] = mod_df_resampled['Mod Values'].fillna(0)
                                    mod_df_resampled.reset_index(inplace=True)
                                    mod_df_resampled.loc[:, 'Time'] = pd.to_datetime(mod_df_resampled['Time'])

                                # Resample by sum for 'calories', 'step', or 'distance'
                                elif mod.lower() in ['calories','step', 'distance']:

                                    # Replace 0s with NaNs before applying the window mean
                                    mod_df['Mod Values'] = mod_df['Mod Values'].replace(0, np.nan)
                                    # Interpolate missing values
                                    mod_df['Mod Values'].interpolate(method='linear', inplace=True)
                                #     # Organize by time index
                                    mod_df['Time'] = pd.to_datetime(mod_df['Time'])
                                    mod_df.set_index('Time', inplace=True)

                                    # Resample by mean
                                    mod_df_resampled = mod_df.resample(f'{aggregation_level}').sum()
                                    mod_df_resampled['Mod'] = mod
                                    mod_df_resampled.reset_index(inplace=True)
                                    mod_df_resampled.loc[:, 'Time'] = pd.to_datetime(mod_df_resampled['Time'])
                                else:
                                    continue

                                student_df_resampled=mod_df_resampled
                                print(mod_df_resampled)
                                # Get the start and end dates from the DataFrame
                                start_date = student_df_resampled['Time'].min()
                                end_date = student_df_resampled['Time'].max()

                                # Create a two-week interval date range
                                date_range = pd.date_range(start=start_date, end=end_date, freq=f'{time_period}')
                                import re

                                
                                aggregation = int(re.findall(r'\d+', aggregation_level)[0])  # Finds all sequences of digits and converts the first match to an integer

                                for i in range(len(date_range) - 1):
                                    interval_start = date_range[i]
                                    interval_end = date_range[i + 1]

                                    # Filter the data within the two-week interval
                                    interval_data = student_df_resampled[
                                        (student_df_resampled['Time'] >= interval_start) & (student_df_resampled['Time'] < interval_end)].copy()
                                    # print('1')
                                    # print(f"Interval data for student {student_id}, mod {mod} between {interval_start} and {interval_end}:\n{interval_data}")
                                    # Calculate the day of the week and hour
                                    interval_data.loc[:, 'Day'] = interval_data['Time'].dt.dayofweek
                                    interval_data.loc[:, 'Hour'] = interval_data['Time'].dt.hour
                                    # print('2')

                                    # Pivot the data to create the desired format
                                    pivot_data = interval_data.pivot_table(index='Hour', columns='Day', values='Mod Values')

                                    if pivot_data.empty:
                                        continue
                                
                                    # # Create a row for the result DataFrame
                                    row_data = {
                                        'Student ID': student_id,
                                        'Modality': mod,
                                        'Start Date': interval_start,
                                        'End Date': interval_end,
                                        **{f'{d}_{h}': pivot_data.loc[h, d] for d in range(7) for h in range(0,24,aggregation)}
                                        }

                                    # print(4)
                                    # Convert the row into a DataFrame
                                    row_df = pd.DataFrame([row_data])
                                    # print(row_df)
                                    # Append the row to the result list
                                    all_student_weekly_data=pd.concat([all_student_weekly_data,row_df], axis=0, ignore_index=True)
                                    # print(all_student_weekly_data)
                                
                        

                    for file in os.listdir(os.path.join(student_dir, 'Survey/')):
                        file_dir = os.path.join(os.path.join(student_dir, 'Survey/'), file)
                        if file.endswith(".xlsx") and os.path.getsize(file_dir) > 0:
                            individ_stud_scores = survey_score_calc(file_dir)
                            individ_stud_scores['Student ID'] = student_id
                            all_student_survey_data = pd.concat([all_student_survey_data, individ_stud_scores],
                                                                ignore_index=True)
                            

    all_student_survey_data = all_student_survey_data[['Student ID', 'StartDate', 'CESD', 'PSS (Sum)', 'STAI (Sum)']]
    all_student_survey_data = all_student_survey_data.rename(columns={'StartDate': 'SurveyDate', 'CESD': 'CESD (Sum)'})


    return all_student_weekly_data, all_student_survey_data

def processing_excel_to_dict_std(data_folder, aggregation_level, time_period):
    """
    Processes and merges student data from various modalities and surveys from specified directories.

    This function iterates over directories containing student modality data (e.g., heart rate, calories) and survey data,
    extracting and processing each into a structured format. The data from each modality is resampled according to a given 
    aggregation level and summarized over specified time periods. The processed modality and survey data are returned as 
    separate DataFrames.

    Parameters:
        data_folder (str): The path to the folder containing student data directories.
        aggregation_level (str): The granularity of data aggregation, e.g., '1H' for hourly.
        time_period (str): The length of the interval for summarizing data, e.g., '2W' for two weeks.

    Returns:
        tuple: A tuple containing two pandas DataFrames:
            - The first DataFrame contains resampled and pivoted modality data for each student.
            - The second DataFrame contains processed survey data for each student.

    Processing Steps:
        1. The function first initializes empty DataFrames for modality and survey data.
        2. It iterates through each student's directory, processing modality data files by reading, resampling, and pivoting the data
           based on specified time periods.
        3. Survey data is extracted from Excel files, processed, and concatenated into a DataFrame.
        4. Both DataFrames are returned for further use or analysis.

    Notes:
        - The function assumes that the directory structure and file naming conventions are consistent and that the required data
          exists in the expected format.
        - The modality data is processed through resampling and interpolating to handle missing or irregular time series data.
        - Survey data files are expected to be in Excel format and contain specific columns like 'Student ID' and 'CESD'.
    """
    # Initialize dictionaries and dataframes to store processed data
    All_Student_Dict = {}
    all_student_mod_data = pd.DataFrame()
    all_student_survey_data = pd.DataFrame()
    all_student_weekly_data = pd.DataFrame()

   # Define columns for the weekly data DataFrame, including ID, modality, dates, and hourly slots for each day of the week
    columns = ['Student ID', 'Modality', 'Start Date', 'End Date'] + [f'{d}_{h}' for d in range(7) for h in range(24)]
    all_student_weekly_data = pd.DataFrame(columns=columns)

    for terms in os.listdir(data_folder):
        terms_dir = os.path.join(data_folder, terms)
        if os.path.isdir(terms_dir):
            for student in os.listdir(terms_dir):
                student_dir = os.path.join(terms_dir, student)
                if os.path.isdir(student_dir):
                    student_df = pd.DataFrame()
                    student_weekly_df = pd.DataFrame()
                    student_id = get_student_id(student)
                    print(student)

                    fitbit_dir = os.path.join(student_dir, 'Fitbit')
                    if not os.path.exists(fitbit_dir):
                        print(f"Directory not found: {fitbit_dir}")
                        continue  # Skip to the next iteration if directory is not found

                    for file in os.listdir(fitbit_dir):
                        file_dir = os.path.join(fitbit_dir, file)
                        if file.endswith(".csv"):
                            # Extract mod from the file name
                            mod = get_Modality(file)
                            if mod in ['heart', 'calories', 'step', 'distance', 'sleep']:
                                update_csv_date(file, os.path.join(student_dir, 'Fitbit/'))  # Update CSV dates if necessary
                                try:
                                    daily_df = pd.read_csv(file_dir)
                                    if mod.lower() == "heart":
                                        mod_col = "Heart Rate"
                                    elif mod.lower() == "sleep":
                                        mod_col = "State"
                                    else:
                                        mod_col = mod.title()

                                    # Create a subset DataFrame with only 'mod_col', 'Time', and 'Student ID'
                                    daily_df = daily_df[['Time', mod_col]]
                                    daily_df['Mod'] = mod
                                    daily_df = daily_df.rename(columns={mod_col: 'Mod Values'})

                                    # Concatenate daily data to the student DataFrame
                                    student_df = pd.concat([student_df, daily_df], ignore_index=True)
                                except pd.errors.EmptyDataError:
                                    print(f"Warning: The file '{file_dir}' is empty. Skipping.")

                    # Check if 'Mod' column is present in the final DataFrame
                    if 'Mod' in student_df.columns:
                        for mod in ['heart', 'calories', 'step', 'distance', 'sleep']:
                            mod_df=student_df[student_df['Mod'] == mod]
                            if not mod_df.empty:
                                mod_df.drop('Mod', axis=1, inplace=True)

                                # Convert 'mod_col' to numeric data type and filter out zero values
                                mod_df['Mod Values'] = pd.to_numeric(mod_df['Mod Values'], errors='coerce')
                                mod_df = mod_df[mod_df['Mod Values'] != 0]  # Exclude rows where 'Mod Values' are zero

                                # Ensure there are still data points left after removing zeros
                                if not mod_df.empty:
                                    if mod in ['heart', 'calories']:
                                        mod_df['Mod Values'].replace(0, np.nan, inplace=True)
                                    # Organize by time index
                                    mod_df['Time'] = pd.to_datetime(mod_df['Time'])
                                    mod_df.set_index('Time', inplace=True)

                                    # Resample and calculate standard deviation, excluding NaNs
                                    mod_df_resampled = mod_df.resample(aggregation_level).std()  # min_count=1 ensures at least one valid value is needed
                                    # Impute NaN values in the resampled data with the mean std of non-NaN periods
                                    mean_std = mod_df_resampled['Mod Values'].mean()
                                    mod_df_resampled['Mod Values'].fillna(mean_std, inplace=True)

                                    # Check if there's any data left after resampling and cleaning
                                    if not mod_df_resampled.empty:
                                        mod_df_resampled['Mod'] = mod
                                        mod_df_resampled.reset_index(inplace=True)
                                        mod_df_resampled.loc[:, 'Time'] = pd.to_datetime(mod_df_resampled['Time'])

                                        # Optionally, print or log the results
                                        print("Resampled Data with Imputed STD:\n", mod_df_resampled)


                                student_df_resampled=mod_df_resampled
                                print(mod_df_resampled)
                                # Get the start and end dates from the DataFrame
                                start_date = student_df_resampled['Time'].min()
                                end_date = student_df_resampled['Time'].max()

                                # Create a two-week interval date range
                                date_range = pd.date_range(start=start_date, end=end_date, freq=f'{time_period}')

                                for i in range(len(date_range) - 1):
                                    interval_start = date_range[i]
                                    interval_end = date_range[i + 1]

                                    # Filter the data within the two-week interval
                                    interval_data = student_df_resampled[
                                        (student_df_resampled['Time'] >= interval_start) & (student_df_resampled['Time'] < interval_end)].copy()
                                    # print('1')
                                    # print(f"Interval data for student {student_id}, mod {mod} between {interval_start} and {interval_end}:\n{interval_data}")
                                    # Calculate the day of the week and hour
                                    interval_data.loc[:, 'Day'] = interval_data['Time'].dt.dayofweek
                                    interval_data.loc[:, 'Hour'] = interval_data['Time'].dt.hour
                                    # print('2')

                                    # Pivot the data to create the desired format
                                    pivot_data = interval_data.pivot_table(index='Hour', columns='Day', values='Mod Values')

                                    if pivot_data.empty:
                                        continue
                                
                                    # # Create a row for the result DataFrame
                                    row_data = {
                                        'Student ID': student_id,
                                        'Modality': mod,
                                        'Start Date': interval_start,
                                        'End Date': interval_end
                                    }

                                    for d in range(7):
                                        for h in range(0, 24, 1):
                                            # Check if the hour (h) and day (d) exist in the DataFrame's index and columns
                                            if h in pivot_data.index and d in pivot_data.columns:
                                                value = pivot_data.loc[h, d]
                                            else:
                                                value = np.nan
                                            # Assign the value to row_data dictionary
                                            row_data[f'{d}_{h}'] = value

                                    # print(4)
                                    # Convert the row into a DataFrame
                                    row_df = pd.DataFrame([row_data])
                                    # print(row_df)
                                    # Append the row to the result list
                                    all_student_weekly_data=pd.concat([all_student_weekly_data,row_df], axis=0, ignore_index=True)
                                    # print(all_student_weekly_data)
                                
                        

                    for file in os.listdir(os.path.join(student_dir, 'Survey/')):
                        file_dir = os.path.join(os.path.join(student_dir, 'Survey/'), file)
                        if file.endswith(".xlsx") and os.path.getsize(file_dir) > 0:
                            individ_stud_scores = survey_score_calc(file_dir)
                            individ_stud_scores['Student ID'] = student_id
                            all_student_survey_data = pd.concat([all_student_survey_data, individ_stud_scores],
                                                                ignore_index=True)
                            

    all_student_survey_data = all_student_survey_data[['Student ID', 'StartDate', 'CESD', 'PSS (Sum)', 'STAI (Sum)']]
    all_student_survey_data = all_student_survey_data.rename(columns={'StartDate': 'SurveyDate', 'CESD': 'CESD (Sum)'})


    return all_student_weekly_data, all_student_survey_data

def merge_survey_mod_data(all_student_weekly_data, all_student_survey_data):
    """
    Merges weekly student data with survey data using an 'asof' merge technique. This function ensures that each student's weekly
    data is merged with their most recent survey data up to the end of each week.

    The function first converts 'Start Date', 'End Date', and 'SurveyDate' to datetime objects and ensures that 'Student ID' is
    consistent across both datasets. It then performs a backward 'asof' merge on 'End Date' and 'SurveyDate', where each entry
    from the weekly data is merged with the last survey data on or before the end date.

    Parameters:
        all_student_weekly_data (pd.DataFrame): A DataFrame containing weekly data for students, including columns for 'Student ID',
                                                'Start Date', and 'End Date'.
        all_student_survey_data (pd.DataFrame): A DataFrame containing survey data for students, with columns including 'Student ID',
                                                'SurveyDate', and survey scores such as 'CESD (Sum)', 'PSS (Sum)', and 'STAI (Sum)'.

    Returns:
        pd.DataFrame: A DataFrame resulting from merging the weekly and survey data, with survey data aligned to the corresponding
                      weekly record based on the latest survey data before or on the end date of the weekly record.

    Notes:
        - This function can drop columns that have all NaN values after the merge, which can simplify the resulting DataFrame.
        - The merge is sensitive to the order of dates and assumes that 'End Date' in the weekly data and 'SurveyDate' in the
          survey data are properly sorted and formatted.
    """
    # Convert 'End Date' and 'SurveyDate' to datetime if not already
    all_student_weekly_data['Start Date'] = pd.to_datetime(all_student_weekly_data['Start Date']).dt.date
    all_student_weekly_data['End Date'] = pd.to_datetime(all_student_weekly_data['End Date']).dt.date
    all_student_weekly_data['End Date'] = pd.to_datetime(all_student_weekly_data['End Date'])
    all_student_survey_data['SurveyDate'] = pd.to_datetime(all_student_survey_data['SurveyDate'])

    # Ensure 'Student ID' is of the same type in both DataFrames, for example, converting to string if necessary
    all_student_weekly_data['Student ID'] = all_student_weekly_data['Student ID'].astype(str)
    all_student_survey_data['Student ID'] = all_student_survey_data['Student ID'].astype(str)

    # Perform the asof merge on 'End Date'
    merged_df = pd.merge_asof(all_student_weekly_data.sort_values('End Date'), 
                              all_student_survey_data[['Student ID', 'SurveyDate', 'CESD (Sum)', 'PSS (Sum)', 'STAI (Sum)']].sort_values('SurveyDate'),
                              by='Student ID',
                              left_on='End Date',
                              right_on='SurveyDate',
                              direction='backward')

    # Optionally, drop columns with all NaN values
    merged_df = merged_df.dropna(axis=1, how='all')
    return merged_df

def apply_mask(merged_df):
    """
    Applies a masking technique to a DataFrame by creating mask columns for missing values in specific score columns.
    
    This function takes a DataFrame and adds three new columns that serve as masks for missing values in the PSS, CESD, and STAI score columns.
    Each of these new columns ('PSS Score Mask', 'CESD Score Mask', 'STAI Score Mask') will have a value of 1 where the corresponding score column has a missing value (NaN), and a value of 0 otherwise.
    
    Parameters:
        merged_df (pd.DataFrame): The DataFrame to which the masks will be applied. It must contain the columns 'PSS (Sum)', 'CESD (Sum)', and 'STAI (Sum)'.
    
    Returns:
        pd.DataFrame: The modified DataFrame with the new mask columns added.
    """
    # STAI Score Mask
    merged_df['PSS Score Mask'] = 0
    merged_df.loc[merged_df['PSS (Sum)'].isnull(), 'PSS Score Mask'] = 1

    # PSS Score Mask
    merged_df['CESD Score Mask'] = 0
    merged_df.loc[merged_df['CESD (Sum)'].isnull(), 'CESD Score Mask'] = 1

    # STAI Score Mask
    merged_df['STAI Score Mask'] = 0
    merged_df.loc[merged_df['STAI (Sum)'].isnull(), 'STAI Score Mask'] = 1
    return merged_df

def all_modalities_filter(merged_df):
    """
    Filters a DataFrame to select rows where each student has all five modalities recorded between the 'Start Date' and 'End Date'.

    This function processes a DataFrame by converting date-related columns to datetime objects and extracting the date part. It sorts the DataFrame by 'Student ID' and calculates the number of unique 'Modality' entries for each student on each start date. Students who have records with all five modalities on a given date are identified.

    The resulting subset of the DataFrame, containing only records of students with all five modalities, is then sorted by 'Student ID' and 'Start Date' before being returned.

    Parameters:
        merged_df (pd.DataFrame): The DataFrame to process, expected to contain columns 'Start Date', 'End Date', 'SurveyDate', 'Student ID', and 'Modality'.

    Returns:
        pd.DataFrame: A subset of the original DataFrame, including only the rows where each student has all five modalities recorded within a week.
    
    Notes:
        This function prints the subset DataFrame before returning it, which can be helpful for debugging or verification purposes.
    """
    merged_df['Start Date'] = pd.to_datetime(merged_df['Start Date'])
    merged_df['End Date'] = pd.to_datetime(merged_df['End Date'])
    merged_df['SurveyDate'] = pd.to_datetime(merged_df['SurveyDate'])

    merged_df = merged_df.sort_values(by='Student ID')

    # Extract only the date part from 'Start Date'
    merged_df['Start Date'] = merged_df['Start Date'].dt.date
    merged_df['End Date'] = merged_df['End Date'].dt.date

    # Count the number of unique modalities for each student on a given date
    modalities_count = merged_df.groupby(['Student ID', 'Start Date'])['Modality'].nunique()

    # Identify Student IDs with all 5 modalities in a week
    student_ids_with_all_modalities = modalities_count[modalities_count == 5].index.get_level_values('Student ID')

    # Create a subset DataFrame with only the rows where a Student ID has all 5 modalities
    selected_col = merged_df[merged_df['Student ID'].isin(student_ids_with_all_modalities)]
    print(selected_col)

    # Sort the subset DataFrame by Student ID and Start Date
    selected_col = selected_col.sort_values(by=['Student ID', 'Start Date'])

    return selected_col

def time_series_to_extracted_features(time_series_df, aggregation_level, time_period):
    """
    Function to extract statistical features from a time series DataFrame.

    Parameters:
    - time_series_df: DataFrame containing time series data.
    - aggregation_level: The aggregation level for the time series data (e.g., '1H' for 1 hour).
    - time_period: The time period for which the features are extracted (e.g., 'W-MON' for weekly on Mondays).

    Returns:
    - feature_df: DataFrame containing extracted features and non-time series data.
    """

    # Initialize an empty DataFrame to store processed data
    data_processed = pd.DataFrame()

    # List of columns that are not time series
    nonnumeric_col = ['Student ID', 'Modality', 'Start Date', 'End Date', 'SurveyDate', 
                      'CESD (Sum)', 'PSS (Sum)', 'STAI (Sum)', 'PSS Score Mask', 
                      'CESD Score Mask', 'STAI Score Mask', 'Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0']
    
    # Iterate over each row in the time series DataFrame
    for i in range(time_series_df.shape[0]):
        row = time_series_df.iloc[[i]]
        
        # Extract features from columns that are not in the non-time series list
        features = row.loc[:, [item for item in row.columns if item not in nonnumeric_col]].values.flatten()
        features = features.reshape(1, -1)

        # Ensure the feature length is suitable for feature extraction
        # if len(features) < 2:
        #     print("Skipping feature extraction due to insufficient data length.")
        #     continue

        # Creating a dictionary to hold calculated features
        extracted_features = {
            'mean': np.mean(features),
            'mode': stats.mode(features)[0][0],  # Mode might return multiple values; we take the first
            'median': np.median(features),
            'std': np.std(features, ddof=1),  # Sample standard deviation
            'variance': np.var(features, ddof=1),  # Sample variance
            '25%': np.percentile(features, 25),
            '50%': np.percentile(features, 50),  # Same as median
            '75%': np.percentile(features, 75),
            'range': np.max(features) - np.min(features),
            'IQR': np.percentile(features, 75) - np.percentile(features, 25),
            'sum': np.sum(features),
            'unique_values': len(np.unique(features)),
            'min': np.min(features),
            'max': np.max(features),
            'RMS': np.sqrt(np.mean(np.square(features))),
            'entropy': stats.entropy(np.histogram(features, bins=10)[0])
        }
        
        # Convert the dictionary to a DataFrame
        feature_df = pd.DataFrame([extracted_features])

        # Concatenate the new features with the processed data
        data_processed = pd.concat([data_processed, feature_df], ignore_index=True)
        print(data_processed)

    # Reset index of the processed data
    data_processed = data_processed.reset_index(drop=True)

    # Concatenate non-time series data with extracted features
    part1_columns = ['Student ID', 'Modality', 'Start Date', 'End Date']
    part2_columns = ['CESD (Sum)', 'PSS (Sum)', 'STAI (Sum)', 
                     'PSS Score Mask', 'CESD Score Mask', 'STAI Score Mask', 'SurveyDate']
    feature_df = pd.concat([
        time_series_df[part1_columns].reset_index(drop=True), 
        data_processed, 
        time_series_df[part2_columns].reset_index(drop=True)
    ], axis=1)

    # Define column names explicitly
    column_names = part1_columns + list(data_processed.columns) + part2_columns
    feature_df.columns = column_names

    # Save the extracted features to a CSV file
    output_file_name = f'extracted_features_{aggregation_level}_{time_period}.csv'
    feature_df.to_csv(output_file_name, index=False)
    print(f'Saved extracted features to {output_file_name}')

    return feature_df

def time_series_to_extracted_features_std(time_series_df, aggregation_level, time_period):
    data_processed = pd.DataFrame()

    # List of non-time series columns
    nonnumeric_col = ['Student ID', 'Modality', 'Start Date', 'End Date', 'SurveyDate', 
                      'CESD (Sum)', 'PSS (Sum)', 'STAI (Sum)', 'PSS Score Mask', 
                      'CESD Score Mask', 'STAI Score Mask', 'Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0']
    
    for i in range(time_series_df.shape[0]):
        row = time_series_df.iloc[[i]]
        features = row.loc[:, [item for item in row.columns if item not in nonnumeric_col]].values.flatten()
        features = features.reshape(1, -1)
            # Ensure the feature length is suitable for feature extraction
        # if len(features) < 2:
        #     print("Skipping feature extraction due to insufficient data length.")
        #     continue

        # Creating a dictionary to hold calculated features
        extracted_features = {
            'mean': np.mean(features),
            'mode': stats.mode(features)[0][0],  # Mode might return multiple values; we take the first
            'median': np.median(features),
            'std': np.std(features, ddof=1),  # Sample standard deviation
            'variance': np.var(features, ddof=1),  # Sample variance
            '25%': np.percentile(features, 25),
            '50%': np.percentile(features, 50),  # Same as median
            '75%': np.percentile(features, 75),
            'range': np.max(features) - np.min(features),
            'IQR': np.percentile(features, 75) - np.percentile(features, 25),
            'sum': np.sum(features),
            'unique_values': len(np.unique(features)),
            'min': np.min(features),
            'max': np.max(features),
            'RMS': np.sqrt(np.mean(np.square(features))),
            'entropy': stats.entropy(np.histogram(features, bins=10)[0])
        }
        
        # Convert dictionary to a DataFrame
        feature_df = pd.DataFrame([extracted_features])

        data_processed = pd.concat([data_processed, feature_df], ignore_index=True)
        print(data_processed)

    # data_processed = data_processed.dropna(how='all')
    data_processed = data_processed.reset_index(drop=True)

    # Concatenating non-time series data with extracted features
    part1_columns = ['Student ID', 'Modality', 'Start Date', 'End Date']
    part2_columns = ['CESD (Sum)', 'PSS (Sum)', 'STAI (Sum)', 
                     'PSS Score Mask', 'CESD Score Mask', 'STAI Score Mask', 'SurveyDate']
    feature_df = pd.concat([
        time_series_df[part1_columns].reset_index(drop=True), 
        data_processed, 
        time_series_df[part2_columns].reset_index(drop=True)
    ], axis=1)

    # Defining column names explicitly
    column_names = part1_columns + list(data_processed.columns) + part2_columns
    feature_df.columns = column_names

    output_file_name = f'extracted_features_std_{aggregation_level}_{time_period}.csv'
    feature_df.to_csv(output_file_name, index=False)
    print(f'Saved extracted features to {output_file_name}')

    return feature_df

# Call the function
aggregation_level='8H'
time_period='W-MON'
raw_data_path="/home/rlopez2/fitbit/Raw Data/"

all_student_weekly_data, all_student_survey_data = processing_excel_to_dict(raw_data_path, aggregation_level, time_period)
merged_df=merge_survey_mod_data(all_student_weekly_data,all_student_survey_data)
apply_mask(merged_df)
time_series_df = all_modalities_filter(merged_df)
time_series_to_extracted_features(time_series_df,aggregation_level,time_period)

all_student_weekly_data_std, all_student_survey_data_std = processing_excel_to_dict_std(raw_data_path, aggregation_level, time_period)
merged_df_std=merge_survey_mod_data(all_student_weekly_data_std,all_student_survey_data_std)
apply_mask(merged_df_std)
time_series_df_std = all_modalities_filter(merged_df_std)
time_series_to_extracted_features_std(time_series_df_std,aggregation_level,time_period)
