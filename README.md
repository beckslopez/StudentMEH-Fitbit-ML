# StudentMEH-Fitbit-ML

# StudentMEH Dataset: Mental Health Screening via Fitbit Data Collected During COVID-19

## Authors
Rebecca Lopez, Avantika Shrestha, Kevin Hickey, Xingtong Guo, ML Tlachac, Shichao Liu, Elke A. Rundensteiner

**Correspondence:**
- Lopez, Shrestha, Hickey, and Rundensteiner are with the Data Science department of Worcester Polytechnic Institute, Worcester, MA 01609 USA. (e-mail: rlopez2, ashrestha4, khickey, rundenst; @wpi.edu).
- Liu and Guo are with the Department of Civil and Environmental Engineering of Worcester Polytechnic Institute, Worcester, MA 01609 USA. (e-mail: sliu8, xguo3; @wpi.edu).
- Tlachac is with the Department of Information Systems and Analytics and Center for Health and Behavioral Sciences at Bryant University, Smithfield, RI 01609 USA. (e-mail: mltlachac@bryant.edu).

**Acknowledgments:**
This research was supported by NSF Grant # 10374216, the Henry Luce Foundation, and the WPI Data Science Dept.

---

## Abstract
College students experience many stressors, resulting in high levels of anxiety and depression. Wearable technology provides unobtrusive sensor data that can be used for the early detection of mental illness. However, current research is limited concerning the variety of psychological instruments administered, physiological modalities, and time series parameters. In this research, we provide a comprehensive assessment of the ability to screen for depression, anxiety, and stress using a variety of Fitbit modalities in predictive machine learning models. Our findings indicate potential in psychological modalities such as calories burned and distance traveled to screen for mental illness with the F1 scores as high as 0.79 for anxiety, 0.78 for depression, and 0.77 for stress screening. This research highlights the value and potential of wearable devices to support continuous mental health monitoring, the importance of identifying optimal data aggregation levels, and selecting modalities for mental illness screening.

---

## Repository Structure
This repository contains the code and data processing scripts used in the paper titled "StudentMEH Dataset: Mental Health Screening via Fitbit Data Collected During COVID-19."

#### Files:
- `Data_Processing.py`: This script contains functions and procedures for processing the raw Fitbit data, including cleaning, aggregating, and merging with psychological survey data.
  - `update_csv_date(csv_filename, filepath)`: Updates the date and time in a CSV file based on the filename or 'Time' column.
  - `survey_score_calc(input_file)`: Calculates survey scores from an input Excel file and returns a modified DataFrame.
  - `get_Modality(file)`: Extracts a specific Modality pattern from a string.
  - `get_student_id(student)`: Extracts a student ID from a string based on predefined patterns.
  - `processing_excel_to_dict(data_folder, aggregation_level, time_period)`: Processes and merges student data from various modalities and surveys from specified directories.
  - `processing_excel_to_dict_std(data_folder, aggregation_level, time_period)`: Processes and merges student data from various modalities and surveys from specified directories, using standard deviation for resampling.
  - `merge_survey_mod_data(all_student_weekly_data, all_student_survey_data)`: Merges weekly student data with survey data using an 'asof' merge technique.
  - `apply_mask(merged_df)`: Applies a masking technique to a DataFrame by creating mask columns for missing values in specific score columns.
  - `all_modalities_filter(merged_df)`: Filters a DataFrame to select rows where each student has all five modalities recorded between the 'Start Date' and 'End Date'.
  - `time_series_to_extracted_features(time_series_df, aggregation_level, time_period)`: Function to extract statistical features from a time series DataFrame.
  - `time_series_to_extracted_features_std(time_series_df, aggregation_level, time_period)`: Function to extract statistical features from a time series DataFrame using standard deviation for resampling.


- `Modeling.py`: This script includes the machine learning models used for predicting mental health outcomes (anxiety, depression, and stress) based on the processed Fitbit data.
  - `create_experiment_folder(base_dir)`: Function to create a new experiment folder within a specified base directory.
  - `f1_score(y_true, y_pred)`: Function to calculate the F1 score.
  - `train_test_processing(file_path, split_type, train_ratio, test_ratio, experiment_folder_path)`: Function to process a given dataset from a CSV file.
  - `objective(trial, model_name, X_train, y_train, scoring, cv)`: Function to define hyperparameters for various models.
  - `cv_data_preparation(df, student_id)`: Function to prepare data for cross-validation.
  - `initialize_models()`: Function to initialize machine learning models.
  - `initialize_columns_and_modalities(df)`: Function to initialize feature columns, target columns, and modalities.
  - `initialize_resampling_technique(sampling_technique)`: Function to initialize the resampling technique.
  - `prepare_cross_validation(CV, df)`: Function to prepare for custom cross-validation.
  - `prepare_modality_subsets(train_df, test_df, modality)`: Function to prepare subsets of the data for a given modality.
  - `prepare_train_test_data(subset_train_df, subset_test_df, target_column, feature_columns)`: Function to prepare the training and test data.
  - `standardize_data(X_train, X_test, feature_columns)`: Function to standardize the training and test data.
  - `resample_data(X_train_scaled, y_train, X_test_scaled, y_test, resampling_technique, test_resampling, n_components)`: Function to resample the training and test data.
  - `perform_feature_selection(feature_selection, X_train_resampled, y_train_resampled, X_test_resampled, y_test_resampled, feature_columns, experiment_folder_path, modality, target_column)`: Function to perform feature selection.
  - `optimize_hyperparameters(study_name, model_name, X_train_selected, y_train_resampled, trial_ct)`: Function to optimize hyperparameters using Optuna.
  - `prepare_cross_validation_data(df, modality, student_ids, selected_features, n_components, resampling_technique, test_resampling)`: Function to prepare data for cross-validation.
  - `make_predictions(model, X_test_resampled_cv)`: Function to make predictions using the trained model.
  - `update_results(y_test_resampled_cv, y_pred, y_proba, y_true_all, y_proba_all)`: Function to update the true and predicted results.
  - `calculate_metrics(y_test_resampled_cv, y_pred, y_proba)`: Function to calculate evaluation metrics for the model.
  - `update_best_accuracy(best_accuracy_per_model, accuracy, cm, experiment_folder_path, target_column, model_name, modality)`: Function to update the best accuracy and confusion matrix.
  - `log_results(modality, target_column, model_name, num_iterations, metrics, best_params, best_accuracy_per_model, selected_features)`: Function to log the results of the model training and evaluation.
  - `ml_unimodels(exp_num, experiment_folder_path, df, train_df, test_df, num_iterations, sampling_technique, test_resampling, n_components, feature_selection, trial_ct, CV, aggregation_level)`: Function to train and evaluate machine learning models.
  - `flatten(df)`: Function to flatten a DataFrame by combining feature data from different modalities for each student and start date.
  - `ml_multimodel(exp_num, experiment_folder_path, df, train_df, test_df, num_iterations, sampling_technique, test_resampling, n_components, feature_selection, trial_ct, CV, aggregation_level)`: Function to train and evaluate multi-modality machine learning models.
  - `preprocess_dataframes(train_df, test_df, df)`: Function to preprocess the training, test, and full dataframes by flattening and removing duplicates and NaNs.
  - `initialize_multimodels()`: Function to initialize machine learning models for multi-modality training.
  - `generate_feature_columns()`: Function to generate the list of feature columns for multi-modality training.
  - `extract_aggregation_level(file_path)`: Function to extract the aggregation level from a file path.

---

## Acknowledgments
This research was supported by NSF Grant # 10374216, the Henry Luce Foundation, and the WPI Data Science Dept.
