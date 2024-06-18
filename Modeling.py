# Standard library imports
import os
import random
import warnings
from datetime import timedelta
from math import floor

# Third-party imports for data handling and visualization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# TensorFlow and related libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling1D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Accuracy
from tensorflow_addons.metrics import F1Score

# Machine learning and optimization libraries
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from keras.layers import Conv1D, Flatten, MaxPooling1D
from keras.metrics import BinaryAccuracy, Precision, Recall
from keras.optimizers import Adam as KerasAdam
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import auc
from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from xgboost import XGBClassifier

# Optimization and experiment tracking tools
import optuna
from optuna.exceptions import TrialPruned
import joblib
import wandb
import ast
from itertools import compress, combinations


# Set random seeds for reproducibility
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

# Ignore all warnings
warnings.filterwarnings("ignore")

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
    # If no GPU is found, use CPU instead
    print("No GPU found, using CPU instead.")


def create_experiment_folder(base_dir):
    """
    Function to create a new experiment folder within a specified base directory. 
    If the base directory does not exist, it will be created. The function generates 
    a unique folder name for each experiment by incrementing a number until a non-existing 
    folder name is found.

    Parameters:
    base_dir (str): The base directory where experiment folders will be created.

    Returns:
    str: The path of the newly created experiment folder.
    int: The experiment number used for the folder name.
    """

    # Check if the base directory exists, if not, create it
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    exp_num = 1  # Initialize experiment number
    while True:
        # Generate a new folder name based on the experiment number
        new_folder_name = f"experiment_{exp_num}"
        new_folder_path = os.path.join(base_dir, new_folder_name)
        
        # Check if the new folder already exists
        if not os.path.exists(new_folder_path):
            # If it doesn't exist, create the new folder
            os.makedirs(new_folder_path)
            print(f"Folder '{new_folder_name}' created at {new_folder_path}")
            # Return the path of the newly created folder and the experiment number
            return new_folder_path, exp_num 
        else:
            # If the folder exists, increment the experiment number and try again
            exp_num += 1

def f1_score(y_true, y_pred):
    # Ensure y_true and y_pred are NumPy arrays for element-wise operation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculating True Positives
    TP = np.sum((y_true == 1) & (y_pred == 1))

    # Calculating False Positives
    FP = np.sum((y_true == 0) & (y_pred == 1))

    # Calculating False Negatives
    FN = np.sum((y_true == 1) & (y_pred == 0))

    # Precision and Recall calculation
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # F1 Score calculation
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1


def train_test_processing(file_path,split_type, train_ratio, test_ratio, experiment_folder_path):
    """
    This function processes a given dataset from a CSV file. It reads the data, 
    adds binary labels based on specific criteria, and splits the dataset into 
    training, testing, sets based on unique student IDs. 
    NOTE: This train test split is only utilized for training the hyperparameters, not for results
    It also visualizes the distribution of labels in each set. 
    The output is the sizes of each dataset and the saved plots of label distributions.

    Parameters:
    file_path (str): The file path to the CSV file containing the dataset.
    """

    # Read the Excel file into a pandas dataframe
    df = pd.read_csv(file_path)
    df = df.dropna()

    # Add binary labels based on the provided criteria
    df['CESD (Label)'] = np.where(df['CESD (Sum)'].isnull(), np.nan, (df['CESD (Sum)'] > 10).astype(int))
    df['PSS (Label)'] = np.where(df['PSS (Sum)'].isnull(), np.nan, (df['PSS (Sum)'] > 6).astype(int))
    df['STAI (Label)'] = np.where(df['STAI (Sum)'].isnull(), np.nan, (df['STAI (Sum)'] > 40).astype(int))
    # Group by 'Student ID' and 'Start Date', and count unique modalities for each group
    modality_count = df.groupby(['Student ID', 'Start Date'])['Modality'].nunique()
    

    # Identify the groups that have all five modalities
    complete_modality_groups = modality_count[modality_count == 5].index.tolist()
    # print('groups',complete_modality_groups)

    # Filter the DataFrame to keep only rows from these complete groups
    df = df[df.set_index(['Student ID', 'Start Date']).index.isin(complete_modality_groups)].reset_index(drop=True)

    

    if split_type == 'between':
        

        # Create a dictionary to keep track of assigned sets for each student ID
        assigned_sets = {}

        # Count total unique students
        total_students = df['Student ID'].nunique()

        train_size = int(train_ratio * total_students)
        test_size = total_students - train_size 
        # Shuffle the unique student IDs randomly
        random.seed(0)
        unique_student_ids = df['Student ID'].unique()
        random.shuffle(unique_student_ids)

        # Assign student IDs to Train, Test, and Val sets
        train_set, test_set, val_set = [], [], []

        for student_id in unique_student_ids:
            if student_id not in assigned_sets:
                if len(train_set) < train_size:
                    train_set.append(student_id)
                    assigned_sets[student_id] = 'Train'
                elif len(test_set) < test_size:
                    test_set.append(student_id)
                    assigned_sets[student_id] = 'Test'
                else:
                    continue

        # Retrieve rows from the original DataFrame based on assigned sets
        train_df = df[df['Student ID'].isin(train_set)]
        test_df = df[df['Student ID'].isin(test_set)]
        
    elif split_type == 'random':
        # Count total unique students
        total_students = len(df)

        train_size = int(train_ratio * total_students)
        test_size = total_students - train_size 

        # Split the data into train and test_val sets
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=0)


    elif split_type == 'time dependent':
        train_dfs = []  # List to store individual training DataFrames
        test_dfs = []   # List to store individual testing DataFrames
        df.sort_values(['Student ID', 'Start Date'], inplace=True)

        for student_id in df['Student ID'].unique():
            student_data = df[df['Student ID'] == student_id]

            # Convert 'Start Date' to datetime if it's not already
            student_data['Start Date'] = pd.to_datetime(student_data['Start Date'])

            # Find the date 6 weeks after the first date
            six_weeks_later = student_data['Start Date'].min() + timedelta(weeks=5)

            # Split the data
            train_data = student_data[student_data['Start Date'] <= six_weeks_later]
            test_data = student_data[student_data['Start Date'] > six_weeks_later]

            # Append to your lists
            train_dfs.append(train_data)
            test_dfs.append(test_data)


        # Concatenate lists of DataFrames to form final training and testing sets
        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)

        # Print sizes of both datasets
        print(f"Training Dataset Size: {len(train_df)}")
        print(f"Testing Dataset Size: {len(test_df)}")
        
    elif split_type == 'health':
        # Identify rows that have all zeros in the specified columns
        condition = (df['PSS Score Mask'] == 0) & (df['CESD Score Mask'] == 0) & (df['STAI Score Mask'] == 0)
        eligible_for_test = df[condition]
        df= df[condition]
        
        # Determine the number of rows for the test set as 20% of the entire DataFrame
        test_size = int(len(df) * test_ratio)  # 20% of the entire DataFrame
        
        # If eligible rows are less than required test size, adjust the test size or handle accordingly
        if len(eligible_for_test) < test_size:
            print(f"Not enough eligible rows for test set based on condition. Using all {len(eligible_for_test)} eligible rows for the test set.")
            test_df = eligible_for_test.copy()  # Use all eligible rows for the test set
        else:
            # Randomly select rows from eligible_for_test for the test set, considering the test_size
            test_df = eligible_for_test.sample(n=test_size, random_state=0)

        # Use the index to filter out the test rows from the original DataFrame for the train set
        train_df = df.drop(test_df.index)

        # Print sizes of both datasets
        print(f"Training Dataset Size: {len(train_df)}")
        print(f"Testing Dataset Size: {len(test_df)}")

        
    return df,train_df,test_df

def objective(trial, model_name, X_train, y_train, scoring, cv):
    # Define hyperparameters for Random Forest
    if model_name == 'RandomForest':
        n_estimators = trial.suggest_int('n_estimators', 10, 200)
        max_depth = trial.suggest_int('max_depth', 5, 150)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
        max_features_options = ['sqrt', 'log2', None]  # 'auto' is equivalent to 'sqrt'
        max_features = trial.suggest_categorical('max_features', max_features_options)
        clf = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            min_samples_split=min_samples_split, 
            min_samples_leaf=min_samples_leaf, 
            max_features=max_features
        )

    # Define hyperparameters for Support Vector Machine
    elif model_name == 'SVM':
        C = trial.suggest_float('C', 0.1, 100)
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        clf = SVC(C=C, gamma=gamma, kernel=kernel)

    # Define hyperparameters for Decision Tree
    elif model_name == 'DecisionTree':
        max_depth = trial.suggest_int('max_depth', 5, 100)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
        max_features = trial.suggest_categorical('max_features', [None, 'sqrt', 'log2'])
        clf = DecisionTreeClassifier(
            max_depth=max_depth, 
            min_samples_split=min_samples_split, 
            min_samples_leaf=min_samples_leaf, 
            max_features=max_features
        )

    # Define hyperparameters for Logistic Regression
    elif model_name == 'LogisticRegression':
        C = trial.suggest_float('C', 0.01, 100)

        # Define valid combinations of solver and penalty, using None for no regularization
        valid_combinations = [
            ('newton-cg', 'l2'), ('newton-cg', None),
            ('lbfgs', 'l2'), ('lbfgs', None),
            ('liblinear', 'l1'), ('liblinear', 'l2'),
            ('sag', 'l2'), ('sag', None),
            ('saga', 'l1'), ('saga', 'l2'), ('saga', 'elasticnet'), ('saga', None)
        ]

        # Choose a combination
        chosen_combination = trial.suggest_categorical('solver_penalty_combination', valid_combinations)
        solver, penalty = chosen_combination

        if penalty == 'elasticnet':
            l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
            clf = LogisticRegression(C=C, penalty=penalty, l1_ratio=l1_ratio, solver=solver, max_iter=10000)
        elif penalty is None:
            # Correctly handle the case where no penalty should be applied
            clf = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=10000)
        else:
            clf = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=10000)

    # Define hyperparameters for K-Nearest Neighbors
    elif model_name == 'KNN':
        n_neighbors = trial.suggest_int('n_neighbors', 1, 20)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
        p = trial.suggest_int('p', 1, 2)
        clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, p=p)

    # Define hyperparameters for XGBoost
    elif model_name == 'XGBoost':
        n_estimators = trial.suggest_int('n_estimators', 50, 100)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.000001, 0.5)
        subsample = trial.suggest_float('subsample', 0.5, 1)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1)
        clf = XGBClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            learning_rate=learning_rate, 
            subsample=subsample, 
            colsample_bytree=colsample_bytree
        )

    # Define hyperparameters for AdaBoost
    elif model_name == 'AdaBoost':
        n_estimators = trial.suggest_int('n_estimators', 25, 100)
        learning_rate = trial.suggest_float('learning_rate', 0.000001, 5)
        algorithm = trial.suggest_categorical('algorithm', ['SAMME', 'SAMME.R'])
        clf = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, algorithm=algorithm)

    # Raise an error if the model is not implemented
    else:
        raise NotImplementedError()

    # Perform cross-validation and return the mean score
    score = cross_val_score(clf, X_train, y_train, scoring=scoring, cv=cv)
    return np.mean(score)

def cv_data_preparation(df, student_id):  
    """
    Function to prepare data for cross-validation by separating the data for a specific student 
    (test set) from the rest of the data (training set).

    Parameters:
    df (DataFrame): The entire dataset containing all students' data.
    student_id (int or str): The ID of the student whose data will be used as the test set.

    Returns:
    DataFrame: Training set containing data of all students except the specified student.
    DataFrame: Test set containing data of only the specified student.
    """

    # Extract the test set by filtering the DataFrame for the specified student ID
    test_df = df[df['StudentID'] == student_id]

    # Extract the training set by filtering the DataFrame for all other student IDs
    train_df = df[df['StudentID'] != student_id]

    # Return the training and test sets
    return train_df, test_df

def initialize_models():
    """
    Initialize the machine learning models.
    
    Returns:
    - models: Dictionary of model names and their corresponding instances
    """
    return {
        'RandomForest': RandomForestClassifier(),
        'SVM': SVC(),
        'DecisionTree': DecisionTreeClassifier(),
        'LogisticRegression': LogisticRegression(),
        'KNN': KNeighborsClassifier(),
        'XGBoost': XGBClassifier(),
        'AdaBoost': AdaBoostClassifier()
    }

def initialize_columns_and_modalities(df):
    """
    Initialize the feature columns, target columns, and modalities.
    
    Parameters:
    - df: DataFrame containing the data
    
    Returns:
    - feature_columns: List of feature columns
    - target_columns: List of target columns
    - modalities: List of unique modalities
    """
    feature_columns = ['mean_x', 'mode_x', 'median_x', 'std_x', 'variance_x', '25%_x', '50%_x', '75%_x', 'range_x', 'IQR_x', 'sum_x', 'unique_values_x', 'min_x', 'max_x', 'RMS_x', 'entropy_x', 'mean_y', 'mode_y', 'median_y', 'std_y', 'variance_y', '25%_y', '50%_y', '75%_y', 'range_y', 'IQR_y', 'sum_y', 'unique_values_y', 'min_y', 'max_y', 'RMS_y', 'entropy_y']
    target_columns = ['CESD (Label)', 'STAI (Label)', 'PSS (Label)']
    modalities = df['Modality'].unique()
    return feature_columns, target_columns, modalities

def initialize_resampling_technique(sampling_technique):
    """
    Initialize the resampling technique.
    
    Parameters:
    - sampling_technique: Resampling technique to use
    
    Returns:
    - resampling_technique: Instance of the resampling technique
    """
    resampling_options = {
        "undersampling": RandomUnderSampler(random_state=0),
        "no_sampling": None,
        "oversampling": RandomOverSampler(random_state=0),
        "SMOTE": SMOTE(random_state=0)
    }
    return resampling_options[sampling_technique]

def prepare_cross_validation(CV, df):
    """
    Prepare for custom cross-validation if enabled.
    
    Parameters:
    - CV: Boolean indicating whether to use cross-validation
    - df: DataFrame containing the data
    
    Returns:
    - student_ids: List of unique student IDs
    """
    if CV:
        return df['Student ID'].unique()
    return None

def prepare_modality_subsets(train_df, test_df, modality):
    """
    Prepare subsets of the data for the given modality.
    
    Parameters:
    - train_df: DataFrame containing the training data
    - test_df: DataFrame containing the test data
    - modality: Modality to filter the data
    
    Returns:
    - subset_train_df: Subset of the training data for the given modality
    - subset_test_df: Subset of the test data for the given modality
    """
    subset_train_df = train_df[train_df['Modality'] == modality].dropna()
    subset_test_df = test_df[test_df['Modality'] == modality].dropna()
    return subset_train_df, subset_test_df

def prepare_train_test_data(subset_train_df, subset_test_df, target_column, feature_columns):
    """
    Prepare the training and test data for the given target column and feature columns.
    
    Parameters:
    - subset_train_df: Subset of the training data
    - subset_test_df: Subset of the test data
    - target_column: Target column for the model
    - feature_columns: List of feature columns
    
    Returns:
    - X_train: Training features
    - y_train: Training labels
    - X_test: Test features
    - y_test: Test labels
    """
    y_train = subset_train_df[target_column]
    y_test = subset_test_df[target_column]
    X_train = subset_train_df[feature_columns]
    X_test = subset_test_df[feature_columns]
    return X_train, y_train, X_test, y_test

def standardize_data(X_train, X_test, feature_columns):
    """
    Standardize the training and test data.
    
    Parameters:
    - X_train: Training features
    - X_test: Test features
    - feature_columns: List of feature columns
    
    Returns:
    - X_train_scaled: Scaled training features
    - X_test_scaled: Scaled test features
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns)
    return X_train_scaled, X_test_scaled

def resample_data(X_train_scaled, y_train, X_test_scaled, y_test, resampling_technique, test_resampling, n_components):
    """
    Resample the training and test data.
    
    Parameters:
    - X_train_scaled: Scaled training features
    - y_train: Training labels
    - X_test_scaled: Scaled test features
    - y_test: Test labels
    - resampling_technique: Resampling technique to use
    - test_resampling: Boolean indicating whether to resample the test set
    - n_components: Number of components for PCA (None if PCA is not used)
    
    Returns:
    - X_train_resampled: Resampled training features
    - y_train_resampled: Resampled training labels
    - X_test_resampled: Resampled test features
    - y_test_resampled: Resampled test labels
    """
    if n_components is None:
        if resampling_technique is None:
            X_train_resampled, y_train_resampled = X_train_scaled, y_train
        else:
            resampler = resampling_technique.fit(X_train_scaled, y_train)
            X_train_resampled, y_train_resampled = resampler.fit_resample(X_train_scaled, y_train)
        
        if test_resampling:
            X_test_resampled, y_test_resampled = RandomOverSampler(random_state=0).fit_resample(X_test_scaled, y_test)
        else:
            X_test_resampled, y_test_resampled = X_test_scaled, y_test
    else:
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        if resampling_technique is None:
            X_train_resampled, y_train_resampled = X_train_pca, y_train
        else:
            resampler = resampling_technique.fit(X_train_pca, y_train)
            X_train_resampled, y_train_resampled = resampler.fit_resample(X_train_pca, y_train)
        
        if test_resampling:
            X_test_resampled, y_test_resampled = RandomOverSampler(random_state=0).fit_resample(X_test_pca, y_test)
        else:
            X_test_resampled, y_test_resampled = X_test_pca, y_test
    
    return X_train_resampled, y_train_resampled, X_test_resampled, y_test_resampled

def perform_feature_selection(feature_selection, X_train_resampled, y_train_resampled, X_test_resampled, y_test_resampled, feature_columns, experiment_folder_path, modality, target_column):
    """
    Perform feature selection using Recursive Feature Elimination (RFE) if enabled.
    
    Parameters:
    - feature_selection: Boolean indicating whether to perform feature selection
    - X_train_resampled: Resampled training features
    - y_train_resampled: Resampled training labels
    - X_test_resampled: Resampled test features
    - y_test_resampled: Resampled test labels
    - feature_columns: List of feature columns
    - experiment_folder_path: Path to the experiment folder
    - modality: Modality being processed
    - target_column: Target column for the model
    
    Returns:
    - selected_features: List of selected features
    """
    if feature_selection:
        X_train_resampled_df = pd.DataFrame(X_train_resampled, columns=feature_columns)
        y_train_resampled_df = pd.Series(y_train_resampled, name='Target')
        X_test_resampled_df = pd.DataFrame(X_test_resampled, columns=feature_columns)
        y_test_resampled_df = pd.Series(y_test_resampled, name='Target')
        
        data_for_correlation = pd.concat([X_train_resampled_df, y_train_resampled_df], axis=1)
        correlation_matrix = data_for_correlation.corr()
        
        plt.figure(figsize=(100, 100))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title(f"Correlation Matrix for {modality} - {target_column}")
        
        filename = f"{modality}_{target_column}_correlation_matrix.png"
        path_to_save = os.path.join(experiment_folder_path, filename)
        plt.savefig(path_to_save)
        plt.close()
        print(f"Correlation matrix plot saved to: {path_to_save}")
        
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        cv_strategy = StratifiedKFold(5)
        selector = RFECV(estimator, step=1, cv=cv_strategy, scoring='accuracy')
        selector = selector.fit(X_train_resampled_df, y_train_resampled_df)
        
        rfe_selected_features = [feature for feature, select in zip(feature_columns, selector.support_) if select]
        X_train_selected = X_train_resampled_df.loc[:, rfe_selected_features]
        X_test_selected = X_test_resampled_df.loc[:, rfe_selected_features]
        selected_features = X_train_selected.columns.tolist()
        print("Optimally selected features by RFECV:", selected_features)
    else:
        X_train_selected = X_train_resampled
        X_test_selected = X_test_resampled
        selected_features = X_train_resampled.columns.tolist()
    
    return selected_features

def optimize_hyperparameters(study_name, model_name, X_train_selected, y_train_resampled, trial_ct):
    """
    Optimize hyperparameters using Optuna.
    
    Parameters:
    - study_name: Name of the Optuna study
    - model_name: Name of the machine learning model
    - X_train_selected: Selected training features
    - y_train_resampled: Resampled training labels
    - trial_ct: Number of trials for hyperparameter optimization
    
    Returns:
    - best_params: Best hyperparameters found by Optuna
    - best_accuracy_per_model: Best accuracy achieved by the model
    """
    study = optuna.create_study(direction='maximize', study_name=study_name, storage=f"sqlite:///{study_name}.db", load_if_exists=True)
    study.optimize(lambda trial: objective(trial, model_name, X_train_selected, y_train_resampled, 'balanced_accuracy', 5), n_trials=trial_ct)
    
    best_params = study.best_params
    if 'solver_penalty_combination' in best_params:
        solver, penalty = best_params['solver_penalty_combination']
        del best_params['solver_penalty_combination']
        best_params['solver'] = solver
        best_params['penalty'] = penalty
    
    model.set_params(**best_params)
    joblib.dump(study, f"{experiment_folder_path}/{study_name}.pkl")
    
    best_accuracy_per_model = 0
    
    return best_params, best_accuracy_per_model

def prepare_cross_validation_data(df, modality, student_ids, selected_features, n_components, resampling_technique, test_resampling):
    """
    Prepare data for cross-validation.
    
    Parameters:
    - df: DataFrame containing the data
    - modality: Modality being processed
    - student_ids: List of unique student IDs
    - selected_features: List of selected features
    - n_components: Number of components for PCA (None if PCA is not used)
    - resampling_technique: Resampling technique to use
    - test_resampling: Boolean indicating whether to resample the test set
    
    Returns:
    - X_train_resampled_cv: Resampled training features for cross-validation
    - y_train_resampled_cv: Resampled training labels for cross-validation
    - X_test_resampled_cv: Resampled test features for cross-validation
    - y_test_resampled_cv: Resampled test labels for cross-validation
    """
    test_student_id = np.random.choice(student_ids)
    subset_train_df = df[(df['Modality'] == modality) & (df['Student ID'] != test_student_id)].dropna()
    subset_test_df = df[(df['Modality'] == modality) & (df['Student ID'] == test_student_id)].dropna()
    
    y_train = subset_train_df[target_column]
    y_test = subset_test_df[target_column]
    
    X_train_selected = subset_train_df[selected_features]
    X_test_selected = subset_test_df[selected_features]
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    if n_components is None:
        if resampling_technique is None:
            X_train_resampled_cv, y_train_resampled_cv = X_train_scaled, y_train
        else:
            resampler = resampling_technique.fit(X_train_scaled, y_train)
            X_train_resampled_cv, y_train_resampled_cv = resampler.fit_resample(X_train_scaled, y_train)
        
        if test_resampling:
            X_test_resampled_cv, y_test_resampled_cv = RandomOverSampler(random_state=0).fit_resample(X_test_scaled, y_test)
        else:
            X_test_resampled_cv, y_test_resampled_cv = X_test_scaled, y_test
    else:
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        if resampling_technique is None:
            X_train_resampled_cv, y_train_resampled_cv = X_train_pca, y_train
        else:
            resampler = resampling_technique.fit(X_train_pca, y_train)
            X_train_resampled_cv, y_train_resampled_cv = resampler.fit_resample(X_train_pca, y_train)
        
        if test_resampling:
            X_test_resampled_cv, y_test_resampled_cv = RandomOverSampler(random_state=0).fit_resample(X_test_pca, y_test)
        else:
            X_test_resampled_cv, y_test_resampled_cv = X_test_pca, y_test
    
    return X_train_resampled_cv, y_train_resampled_cv, X_test_resampled_cv, y_test_resampled_cv

def make_predictions(model, X_test_resampled_cv):
    """
    Make predictions using the trained model.
    
    Parameters:
    - model: Trained machine learning model
    - X_test_resampled_cv: Resampled test features for cross-validation
    
    Returns:
    - y_pred: Predicted labels
    - y_proba: Predicted probabilities
    """
    y_pred = model.predict(X_test_resampled_cv)
    y_proba = model.predict_proba(X_test_resampled_cv)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test_resampled_cv)
    return y_pred, y_proba

def update_results(y_test_resampled_cv, y_pred, y_proba, y_true_all, y_proba_all):
    """
    Update the true and predicted results.
    
    Parameters:
    - y_test_resampled_cv: True test labels
    - y_pred: Predicted labels
    - y_proba: Predicted probabilities
    - y_true_all: List to store all true labels
    - y_proba_all: List to store all predicted probabilities
    """
    y_true_all.extend(y_test_resampled_cv.tolist())
    y_proba_all.extend(y_proba.tolist())

def calculate_metrics(y_test_resampled_cv, y_pred, y_proba):
    """
    Calculate evaluation metrics for the model.
    
    Parameters:
    - y_test_resampled_cv: True test labels
    - y_pred: Predicted labels
    - y_proba: Predicted probabilities
    
    Returns:
    - cm: Confusion matrix
    - metrics: Dictionary of evaluation metrics
    """
    f1 = f1_score(y_test_resampled_cv, y_pred)
    accuracy = accuracy_score(y_test_resampled_cv, y_pred)
    balanced_acc = balanced_accuracy_score(y_test_resampled_cv, y_pred)
    cm = confusion_matrix(y_test_resampled_cv, y_pred)
    
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) else 0
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) else 0
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) else 0
    sensitivity = recall
    
    metrics = {
        'F1_Score': f1,
        'Accuracy': accuracy,
        'Balanced_Accuracy': balanced_acc,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'Sensitivity': sensitivity
    }
    
    return cm, metrics

def update_best_accuracy(best_accuracy_per_model, accuracy, cm, experiment_folder_path, target_column, model_name, modality):
    """
    Update the best accuracy and confusion matrix if the current accuracy is higher.
    
    Parameters:
    - best_accuracy_per_model: Best accuracy achieved by the model
    - accuracy: Current accuracy
    - cm: Confusion matrix
    - experiment_folder_path: Path to the experiment folder
    - target_column: Target column for the model
    - model_name: Name of the machine learning model
    - modality: Modality being processed
    
    Returns:
    - best_accuracy_per_model: Updated best accuracy
    - best_cm: Updated confusion matrix
    """
    if accuracy > best_accuracy_per_model:
        best_accuracy_per_model = accuracy
        best_cm = cm
        plot_confusion_matrix(experiment_folder_path, cm, target_column, model_name, modality, log_to_wandb=True)
    
    return best_accuracy_per_model, best_cm

def log_results(modality, target_column, model_name, num_iterations, metrics, best_params, best_accuracy_per_model, selected_features):
    """
    Log the results of the model training and evaluation.
    
    Parameters:
    - modality: Modality being processed
    - target_column: Target column for the model
    - model_name: Name of the machine learning model
    - num_iterations: Number of iterations for cross-validation
    - metrics: Dictionary of evaluation metrics
    - best_params: Best hyperparameters found by Optuna
    - best_accuracy_per_model: Best accuracy achieved by the model
    - selected_features: List of selected features
    
    Returns:
    - log_data: Dictionary of logged data
    """
    log_data = {
        'Modality': modality,
        'Target_Column': target_column,
        'Model': model_name,
        'Iteration_Count': num_iterations,
        'Accuracy': metrics['Accuracy'],
        'Balanced_Accuracy': metrics['Balanced_Accuracy'],
        'F1_Score': metrics['F1_Score'],
        'Precision': metrics['Precision'],
        'Recall': metrics['Recall'],
        'Sensitivity': metrics['Sensitivity'],
        'Specificity': metrics['Specificity'],
        'Best_Params': best_params,
        'Best_Score_In_Iteration': best_accuracy_per_model,
        'Features_Selected': selected_features
    }
    
    return log_data

def ml_unimodels(exp_num, experiment_folder_path, df, train_df, test_df, 
                 num_iterations, sampling_technique, test_resampling, n_components, 
                 feature_selection, trial_ct, CV, aggregation_level):
    """
    Function to train and evaluate machine learning models for health informatics purposes.

    Parameters:
    - exp_num: Experiment number
    - experiment_folder_path: Path to the experiment folder
    - df: DataFrame containing the data
    - train_df: DataFrame containing the training data
    - test_df: DataFrame containing the test data
    - num_iterations: Number of iterations for cross-validation
    - sampling_technique: Resampling technique to use
    - test_resampling: Boolean indicating whether to resample the test set
    - n_components: Number of components for PCA (None if PCA is not used)
    - feature_selection: Boolean indicating whether to perform feature selection
    - trial_ct: Number of trials for hyperparameter optimization
    - CV: Boolean indicating whether to use cross-validation
    - aggregation_level: Aggregation level of the data

    Returns:
    - results: Dictionary containing the results of the experiments
    """
    
    results = {}
    y_true_all = []
    y_proba_all = []
    
    models = initialize_models()
    feature_columns, target_columns, modalities = initialize_columns_and_modalities(df)
    resampling_technique = initialize_resampling_technique(sampling_technique)

    student_ids = prepare_cross_validation(CV, df)
    
    for modality in modalities:
        subset_train_df, subset_test_df = prepare_modality_subsets(train_df, test_df, modality)

        for target_column in target_columns:
            X_train, y_train, X_test, y_test = prepare_train_test_data(subset_train_df, subset_test_df, target_column, feature_columns)
            X_train_scaled, X_test_scaled = standardize_data(X_train, X_test, feature_columns)
            X_train_resampled, y_train_resampled, X_test_resampled, y_test_resampled = resample_data(X_train_scaled, y_train, X_test_scaled, y_test, resampling_technique, test_resampling, n_components)
            
            for model_name, model in models.items():
                selected_features = perform_feature_selection(feature_selection, X_train_resampled, y_train_resampled, X_test_resampled, y_test_resampled, feature_columns, experiment_folder_path, modality, target_column)
                best_params, best_accuracy_per_model = optimize_hyperparameters(study_name, model_name, X_train_selected, y_train_resampled, trial_ct)
                
                for iteration in range(num_iterations):
                    if CV:
                        X_train_resampled_cv, y_train_resampled_cv, X_test_resampled_cv, y_test_resampled_cv = prepare_cross_validation_data(df, modality, student_ids, selected_features, n_components, resampling_technique, test_resampling)
                    
                    model.fit(X_train_resampled_cv, y_train_resampled_cv)
                    y_pred, y_proba = make_predictions(model, X_test_resampled_cv)
                    
                    update_results(y_test_resampled_cv, y_pred, y_proba, y_true_all, y_proba_all)
                    cm, metrics = calculate_metrics(y_test_resampled_cv, y_pred, y_proba)
                    best_accuracy_per_model, best_cm = update_best_accuracy(best_accuracy_per_model, accuracy, cm, experiment_folder_path, target_column, model_name, modality)
                    log_data = log_results(modality, target_column, model_name, num_iterations, metrics, best_params, best_accuracy_per_model, selected_features)
                    
                    key = (modality, target_column, model_name)
                    results[key] = log_data
                    wandb.log(log_data)
                    
    return results

def flatten(df):
    """
    Function to flatten a DataFrame by combining feature data from different modalities for each student and start date.
    The function processes specific columns, renames them to include the modality, and combines them into a single row.

    Parameters:
    df (DataFrame): The input DataFrame containing data from multiple modalities for each student.

    Returns:
    DataFrame: A flattened DataFrame with combined features from different modalities and labels.
    """
    
    # Define the columns to process
    filtered_columns = [
        'mean_x', 'mode_x', 'median_x', 'std_x', 'variance_x', '25%_x', '50%_x', '75%_x', 
        'range_x', 'IQR_x', 'sum_x', 'unique_values_x', 'min_x', 'max_x', 'RMS_x', 'entropy_x',
        'mean_y', 'mode_y', 'median_y', 'std_y', 'variance_y', '25%_y', '50%_y', '75%_y',
        'range_y', 'IQR_y', 'sum_y', 'unique_values_y', 'min_y', 'max_y', 'RMS_y', 'entropy_y'
    ]

    # Initialize an empty DataFrame to hold the flattened data
    flattened_df = pd.DataFrame()

    # Group the data by 'Student ID' and 'Start Date'
    for (student_id, start_date), group_df in df.groupby(['Student ID', 'Start Date']):
        modalities_data = {}  # Dictionary to hold data from all modalities

        # Group the data by 'Modality'
        for modality, modality_group in group_df.groupby('Modality'):
            # Filter the columns that exist in the current group
            existing_filtered_columns = [col for col in filtered_columns if col in modality_group.columns]
            scaled_features = modality_group[existing_filtered_columns]

            # Rename the columns to include the modality
            rename_dict = {col: f"{modality}_{col}" for col in existing_filtered_columns}
            scaled_features_renamed = scaled_features.rename(columns=rename_dict)

            # Update the dictionary with each modality's data
            modalities_data.update(scaled_features_renamed.iloc[0].to_dict())

        # Combine all the features from different modalities into a single row
        combined_features = pd.Series(modalities_data)
        combined_features['Student ID'] = student_id
        combined_features['Start Date'] = start_date

        # Create a DataFrame from the series and append it to the flattened_df
        combined_features_df = pd.DataFrame([combined_features])
        flattened_df = pd.concat([flattened_df, combined_features_df], ignore_index=True)

    # Perform the merge to include labels
    merged_df = pd.merge(flattened_df, df[['Student ID', 'Start Date', 'CESD (Label)', 'PSS (Label)', 'STAI (Label)']],
                         on=['Student ID', 'Start Date'], how='left')

    return merged_df

def ml_multimodel(exp_num, experiment_folder_path, df, train_df, test_df, num_iterations, sampling_technique, test_resampling, n_components, feature_selection, trial_ct, CV, aggregation_level):
    """
    Function to train and evaluate multi-modality machine learning models for health informatics purposes.

    Parameters:
    - exp_num: Experiment number
    - experiment_folder_path: Path to the experiment folder
    - df: DataFrame containing the data
    - train_df: DataFrame containing the training data
    - test_df: DataFrame containing the test data
    - num_iterations: Number of iterations for cross-validation
    - sampling_technique: Resampling technique to use
    - test_resampling: Boolean indicating whether to resample the test set
    - n_components: Number of components for PCA (None if PCA is not used)
    - feature_selection: Boolean indicating whether to perform feature selection
    - trial_ct: Number of trials for hyperparameter optimization
    - CV: Boolean indicating whether to use cross-validation
    - aggregation_level: Aggregation level of the data

    Returns:
    - results: Dictionary containing the results of the experiments
    """

    results = {}
    y_true_all = []
    y_proba_all = []

    target_columns = ['CESD (Label)', 'STAI (Label)', 'PSS (Label)']

    multi_train_df, multi_test_df, multi_df = preprocess_dataframes(train_df, test_df, df)

    models = initialize_multimodels()
    resampling_technique = initialize_resampling_technique(sampling_technique)
    feature_columns = generate_feature_columns()

    student_ids = prepare_cross_validation(CV, multi_df)

    for target_column in target_columns:
        X_train, y_train, X_test, y_test = prepare_train_test_data(multi_train_df, multi_test_df, target_column, feature_columns)
        X_train_scaled, X_test_scaled = standardize_data(X_train, X_test, feature_columns)
        X_train_resampled, y_train_resampled, X_test_resampled, y_test_resampled = resample_data(X_train_scaled, y_train, X_test_scaled, y_test, resampling_technique, test_resampling, n_components)

        for model_name, model in models.items():
            selected_features = perform_feature_selection(feature_selection, X_train_resampled, y_train_resampled, X_test_resampled, y_test_resampled, feature_columns, experiment_folder_path, 'all_modalities', target_column)
            best_params, best_accuracy_per_model = optimize_hyperparameters(f"all_modalities_{model_name}_{target_column}_{aggregation_level}", model_name, X_train_selected, y_train_resampled, trial_ct)
            
            student_ids_choices = student_ids.copy()
            num_iterations = min(num_iterations, len(student_ids))

            for iteration in range(num_iterations):
                print(f"Iteration {iteration + 1}/{num_iterations}")

                if CV:
                    X_train_resampled_cv, y_train_resampled_cv, X_test_resampled_cv, y_test_resampled_cv = prepare_cross_validation_data(multi_df, 'all_modalities', student_ids_choices, selected_features, n_components, resampling_technique, test_resampling)

                model.fit(X_train_resampled_cv, y_train_resampled_cv)

                y_pred, y_proba = make_predictions(model, X_test_resampled_cv)
                
                update_results(y_test_resampled_cv, y_pred, y_proba, y_true_all, y_proba_all)
                cm, metrics = calculate_metrics(y_test_resampled_cv, y_pred, y_proba)
                best_accuracy_per_model, best_cm = update_best_accuracy(best_accuracy_per_model, metrics['Accuracy'], cm, experiment_folder_path, target_column, model_name, 'all_modalities')
                log_data = log_results('all_modalities', target_column, model_name, num_iterations, metrics, best_params, best_accuracy_per_model, selected_features)

                key = ('all_modalities', target_column, model_name)
                results[key] = log_data
                wandb.log(log_data)

    return results

def preprocess_dataframes(train_df, test_df, df):
    """
    Preprocess the training, test, and full dataframes by flattening and removing duplicates and NaNs.

    Parameters:
    - train_df: DataFrame containing the training data
    - test_df: DataFrame containing the test data
    - df: DataFrame containing the full data

    Returns:
    - multi_train_df: Preprocessed training DataFrame
    - multi_test_df: Preprocessed test DataFrame
    - multi_df: Preprocessed full DataFrame
    """
    multi_train_df = flatten(train_df).dropna().drop_duplicates()
    multi_test_df = flatten(test_df).dropna().drop_duplicates()
    multi_df = flatten(df).dropna().drop_duplicates()

    print("Training DataFrame shape:", multi_train_df.shape)
    print("Testing DataFrame shape:", multi_test_df.shape)
    print("Combined DataFrame shape:", multi_df.shape)

    return multi_train_df, multi_test_df, multi_df

def initialize_multimodels():
    """
    Initialize the machine learning models for multi-modality training.

    Returns:
    - models: Dictionary of model names and their corresponding instances
    """
    return {
        'RandomForest': RandomForestClassifier(),
        'DecisionTree': DecisionTreeClassifier(),
        'LogisticRegression': LogisticRegression(),
        'XGBoost': XGBClassifier(),
        'AdaBoost': AdaBoostClassifier()
    }

def generate_feature_columns():
    """
    Generate the list of feature columns for multi-modality training.

    Returns:
    - feature_columns: List of feature columns
    """
    modalities = ['calories', 'distance', 'heart', 'sleep', 'step']
    metrics = ['mean_x', 'mode_x', 'median_x', 'std_x', 'variance_x', '25%_x', '50%_x', '75%_x', 'range_x',
               'IQR_x', 'sum_x', 'unique_values_x', 'min_x', 'max_x', 'RMS_x', 'entropy_x', 'mean_y', 'mode_y',
               'median_y', 'std_y', 'variance_y', '25%_y', '50%_y', '75%_y', 'range_y', 'IQR_y', 'sum_y', 
               'unique_values_y', 'min_y', 'max_y', 'RMS_y', 'entropy_y']

    return [f"{mod}_{metric}" for mod in modalities for metric in metrics]


def extract_aggregation_level(file_path):
    """
    Function to extract the aggregation level from a file path.
    The aggregation level is assumed to be a part of the filename containing 'H'.

    Parameters:
    file_path (str): The full path to the file.

    Returns:
    str: The part of the filename that contains the aggregation level.
    """
    
    # Split the filename from the path
    file_name = file_path.split('/')[-1]

    # Split the filename into parts using '_' as the delimiter
    parts = file_name.split('_')

    # Iterate through the parts to find the one containing 'H'
    for part in parts:
        if 'H' in part:
            return part  # Return the aggregation level if found

    return None  # Return None if no part containing 'H' is found

project_name = "Health_Modeling"  # Replace with your project or task name
base_dir = 'Modeling'

# Paths to the CSV files
hourly_features_mean = "/home/rlopez2/fitbit/extracted_features_1H_W-MON.csv"
hourly_features_std = "/home/rlopez2/fitbit/extracted_features_std_H_W-MON.csv"

# Testing the function to extract aggregation level from the mean features file path
aggregation_level = extract_aggregation_level(hourly_features_mean)

# Read the CSV files into DataFrames
df_mean = pd.read_csv(hourly_features_mean)
df_std = pd.read_csv(hourly_features_std)

# Drop the 'Unnamed: 0' column from both DataFrames, if it exists
df_mean = df_mean.drop(columns=['Unnamed: 0'], errors='ignore')
df_std = df_std.drop(columns=['Unnamed: 0'], errors='ignore')

# Merge the DataFrames on the specified columns
merge_columns = ['Student ID', 'Modality', 'Start Date', 'End Date', 'CESD (Sum)', 'PSS (Sum)', 'STAI (Sum)', 'PSS Score Mask', 'CESD Score Mask', 'STAI Score Mask', 'SurveyDate']
merged_df = pd.merge(df_mean, df_std, on=merge_columns, how='inner')

# Drop the '0_x' and '0_y' columns from the merged DataFrame, if they exist
merged_df = merged_df.drop(columns=['0_x', '0_y'], errors='ignore')

# Write the merged DataFrame to a new CSV file
merged_file_path = "/home/rlopez2/fitbit/merged_features.csv"
merged_df.to_csv(merged_file_path, index=False)

# Construct the path for the project's folder
project_folder = os.path.join(base_dir, project_name)

# Create an experiment folder inside the project's folder
experiment_folder_path, exp_num = create_experiment_folder(project_folder)

# Define file path and train-test split parameters
train_test_split_type = 'health'
train_rat = 0.85
test_rat = 1 - train_rat

# Process the data for training and testing
df, train_df, test_df = train_test_processing(merged_file_path, split_type=train_test_split_type, train_ratio=train_rat, test_ratio=test_rat, experiment_folder_path=experiment_folder_path)

# Define experiment parameters
num_iterations = df['Student ID'].nunique()
sampling_technique = 'no_sampling'
n_components = None
feature_selection = False
problem_type = 'classification'
test_resampling = False
epoch_ct = 100
trial_ct = 40
CV = True

# Initialize WandB for experiment tracking
wandb.init(project="fitbit-health-informatics", config={
    "experiment number": exp_num,
    "iterations": num_iterations,
    "sampling_technique": sampling_technique,
    "PCA_num_components": n_components,
    "Train_Test_Split": train_test_split_type,
    "Cross_Validation": CV,
    "Train_Ratio": train_rat,
    "Feature_Selection": feature_selection,
    "Problem_Type": problem_type,
    "Description": 'Comparison of Uni & Multi modality ML Models for Health informatic purposes',
    "Name": f"health_modeling_{exp_num}"
})
config = wandb.config

# Run machine learning experiments with unimodality models
ml_results_uni = ml_unimodels(
    exp_num,
    experiment_folder_path,
    df,
    train_df,
    test_df,
    num_iterations,
    sampling_technique,
    test_resampling,
    n_components,
    feature_selection,
    trial_ct,
    CV,
    aggregation_level
)

# Run machine learning experiments with multimodality model
ml_results_multi = ml_multimodel(
    exp_num,
    experiment_folder_path,
    df,
    train_df,
    test_df,
    num_iterations,
    sampling_technique,
    test_resampling,
    n_components,
    feature_selection,
    trial_ct,
    CV,
    aggregation_level
)

# Finish the WandB run
wandb.finish()

# Combine the results from unimodal and multimodal experiments
combined_results = {}
combined_results.update(ml_results_uni)
combined_results.update(ml_results_multi)

# Convert the combined results dictionary to a DataFrame
data = pd.DataFrame.from_dict(combined_results, orient='index').reset_index(drop=True)

# Save the DataFrame to a CSV file with a name based on the experiment number
data.to_csv(f'{experiment_folder_path}/experimental_results_{exp_num}.csv', index=False)

