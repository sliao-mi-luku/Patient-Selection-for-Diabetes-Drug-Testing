"""
Helper functions

This work was completed in Udacity's AI for Healthcare Nanodegree Program
"""

import pandas as pd
import numpy as np
import os
import tensorflow as tf
import functools


def reduce_dimension_ndc(df, ndc_df):
    """
    Reduce the dimensionality of NDC codes
    Args:
        df - pandas dataframe, input dataset
        ndc_df - pandas dataframe, drug code dataset used for mapping in generic names
    Outputs:
        df - pandas dataframe, output dataframe with joined generic drug name
    """
    df = df.copy()
    # dict to map ndc_codes to generic names
    d = dict()
    d['nan'] = np.nan
    # create the dict
    for i in range(ndc_df.shape[0]):
        code = ndc_df.iloc[i]['NDC_Code']
        name = ndc_df.iloc[i]['Non-proprietary Name']
        d[code] = name
    # create an array to store generic names in df
    arr = []
    for i in range(df.shape[0]):
        code = df.iloc[i]['ndc_code']
        name = d[str(code)]
        arr.append(name)
    # create a new column neamed generic_drug_name
    df['generic_drug_name'] = np.array(arr)
    return df


def select_first_encounter(df):
    """
    Select the first encounter for each patients
    Args:
        df - pandas dataframe, dataframe with all encounters
    Outputs:
        first_encounter_df - pandas dataframe, dataframe with only the first encounter for a given patient
    """
    # sort df by encounter_id column
    df = df.sort_values(by='encounter_id')
    # list of all first encounter ids for each patient
    first_encounter_ids = df.groupby('patient_nbr')['encounter_id'].head(1).values
    # select only the rows from df with first encounter ids
    first_encounter_df = df[df['encounter_id'].isin(first_encounter_ids)]
    return first_encounter_df


def patient_dataset_splitter(df, patient_key='patient_nbr'):
    """
    Split the data into training, validation, and test datasets
    Args:
        df - pandas dataframe, input dataset that will be split
        patient_key - string, column that is the patient id
    Outputs:
        train - pandas dataframe, the training dataset
        validation - pandas dataframe, the validation dataset
        test - pandas dataframe, the test dataset
    """
    # unique patient ids
    patients = df[patient_key].unique()
    # randomly shuffle the order of the patient ids
    np.random.shuffle(patients)
    # total number of patients
    n = len(patients)
    # data split
    # train: 60%
    train_ids = patients[:int(0.6*n)]
    # validation: 20%
    val_ids = patients[int(0.6*n):int(0.8*n)]
    # test: 20%
    test_ids = patients[int(0.8*n):]

    train = df[df[patient_key].isin(train_ids)]
    validation = df[df[patient_key].isin(val_ids)]
    test = df[df[patient_key].isin(test_ids)]
    return train, validation, test


def create_tf_categorical_feature_cols(categorical_col_list, vocab_dir='./diabetes_vocab/'):
    """
    Create categorical feature columns
    Args:
        categorical_col_list - list, categorical field list that will be transformed with TF feature column
        vocab_dir - string, the path where the vocabulary text files are located
    Output:
        output_tf_list - list of TF feature columns
    """
    output_tf_list = []
    for c in categorical_col_list:
        # path to the vocabulary file
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        # create categorical feature columns from vocab file
        cat_ftr_col = tf.feature_column.categorical_column_with_vocabulary_file(key=c,
                                                                                vocabulary_file=vocab_file_path,
                                                                                num_oov_buckets=1)
        # one-hot encoding
        cat_ftr = tf.feature_column.indicator_column(cat_ftr_col)
        # append to results
        output_tf_list.append(cat_ftr)
    return output_tf_list



def normalize_numeric_with_zscore(col, mean, std):
    """
    Normalize the numeric features
    Args:
        col - the numeric column
        mean - the mean of the column
        std - the standard deviation of the column
    Output:
        normalized feature values
    """
    return (col - mean)/std


def create_tf_numeric_feature(col, MEAN, STD, default_value=0.0):
    """
    Create numeric features
    Args:
        col - string, input numerical column name
        MEAN - the mean for the column in the training data
        STD - the standard deviation for the column in the training data
        default_value - the value that will be used for imputing the field
    Outputs:
        tf_numeric_feature - tf feature column representation of the input field
    """
    # normalization function
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)
    # create numeric features
    tf_numeric_feature = tf.feature_column.numeric_column(key=col,
                                                          default_value=default_value,
                                                          normalizer_fn=normalizer,
                                                          dtype=tf.dtypes.float32)
    return tf_numeric_feature



def get_mean_std_from_preds(diabetes_yhat):
    """
    Get the mean and standard deviation of the prediction probability
    Args:
        diabetes_yhat - TF Probability prediction object
    Outputs:
        m - the mean of the prediction probability
        s - the standard deviation of the prediction probability
    """
    # mean
    m = diabetes_yhat.mean()
    # std
    s = diabetes_yhat.stddev()
    return m, s


def get_student_binary_prediction(df, col):
    """
    Convert numeric predictions into a binary prediction.
    Binary prediction is True if the patient will stay >= 5 days in the hospital
    Args:
        df - pandas dataframe prediction output dataframe
        col - str,  probability mean prediction field
    Outputs:
        student_binary_prediction: numpy array of the binary predictions
    """
    # binary classification
    df['pred_binary'] = (df['pred_mean'] >= 5).astype(int)
    # extract the column values
    student_binary_prediction = df['pred_binary'].values
    return student_binary_prediction
