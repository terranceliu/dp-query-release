import sys
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

import pdb

"""
Process (inplace) continuous column into a set of bins with fixed size
Input:
    df (pandas.DataFrame): Input dataframe
    col (string): Column name
    bin_size (int, optional): Bin size of all bins
Returns: (int) Domain size of processed column
"""
def process_cont_bin_size(df, col, bin_size=10):
    df.loc[:, col] = df[col].astype(int) // bin_size
    df.loc[:, col] -= df.loc[:, col].min() # forces the values to start from 0
    return df[col].max() + 1


"""
Process (inplace) continuous column using user-specified bins. 
By default, all values are mapped to a single bin (NOT RECOMMENDED)
Input:
    df (pandas.DataFrame): Input dataframe
    col (string): Column name
    bin_ranges (list, optional): List defining the bin ranges
Returns: (int) Domain size of processed column

Ex: bin_ranges = [0, 100, 300, 500] corresponds to the following 4 bins:
    (-INFINITY, 0), [0, 100), [300, 500), and [500, INFINITY)
"""
def process_cont_custom_bins(df, col, bin_ranges=[]):
    if len(bin_ranges) == 0:
        warnings.warn("No bin ranges are given for attribute: {}. All values will be mapped to a single bin.".format(col))

    assert sorted(bin_ranges) == bin_ranges, "`bin_ranges` must be sorted."

    bin_ranges = bin_ranges.copy() # prevent overwriting
    bin_ranges.insert(0, -np.infty)
    bin_ranges.append(np.infty)

    # create a copy of the current column to preserve the entries of original dataframe until looping through all bins
    output = df[col].copy()

    for i in range(len(bin_ranges) - 1):
        lower_bound = float(bin_ranges[i])
        upper_bound = float(bin_ranges[i+1])
        mask = (df[col] >= lower_bound) & (df[col] < upper_bound)
        output[mask] = i

    df.loc[:, col] = output
    return len(bin_ranges) - 1


"""
Process (inplace) continuous column into a fixed number of bins of equal size
Input:
    df (pandas.DataFrame): Input dataframe
    col (string): Column name
    nbins (int, optional): Number of bins
    minimum (float, optional): Minimum bin value (by default, the minimum value in the dataset is used)
    maximum (float, optional): Maximum bin value (by default, the maximum value in the dataset is used)
Returns: (int) Domain size of processed column
"""
def process_cont_n_bins(df, col, nbins=10, minimum=None, maximum=None):
    dataset_minimum = df[col].min()
    dataset_maximum = df[col].max()

    minimum = dataset_minimum if minimum is None else minimum
    maximum = dataset_maximum if maximum is None else maximum

    assert dataset_minimum >= minimum, "Minimum value entered is outside the domain of column."
    assert dataset_maximum <= maximum, "Maximum value entered is outside the domain of column."

    bin_ranges = np.linspace(minimum, maximum, nbins + 1)
    bin_ranges = bin_ranges[1:-1] # see process_cont_custom_bins(), which assumes min and max bounds are -INFINITY and INFINITY
    return process_cont_custom_bins(df, col, list(bin_ranges))


"""
Process (inplace) discrete column into integer categories (ranging from 0 to # unique categories)
Input:
    df (pandas.DataFrame): Input dataframe
    col (string): Column name
Returns: (int) Domain size of processed column
"""
def process_categorical(df, col):
    enc = LabelEncoder()
    encoded = enc.fit_transform(df[col])
    df.loc[:, col] = encoded
    return np.unique(encoded).shape[0]


'''
Input:
    df (pandas.DataFrame): Input dataframe
    cols_categorical (list): List of categorical columns to process
    cols_continuous (list): List of continuous columns to process
    cont_options (list, optional): three valid options for continuous columns:
        `binsize`: user specifies the bin size 
        `nbins`: user specifies the number of bins (of equal length)
        `custom`: user specifies a custom set of bins
        By default, `binsize` is used for all columns.
    cont_params (list, optional): List of dictionaries specifying the keyword arguments for each continuous column option.
        If the user wishes to use the default behavior for a particular option, the user can set the parameter as either
            1) an empty dictionary {}
            2) None
        By default, no parameters are passed to any continuous column processing function
Returns:
    df (pd.DataFrame): Preprocessed dataframe
    domain (dict): Domain dictionary (maps from column name to domain size)
'''
def discretize_columns(df, cols_categorical, cols_continuous,
                       cont_options=None, cont_params=None):

    assert cont_options is not None or cont_params is None, "If no options are given for continuous columms (cont_options=None), " \
                                                            "no params should be passed either (cont_params=None)."

    if cont_options is None: # default to process_cont_bin_size
        cont_params = ["binsize" for _ in cols_continuous]

    if cont_params is None: # default behavior for all attributes (params={})
        cont_params = [{} for _ in cols_continuous]

    # Process columns
    df = df[cols_categorical + cols_continuous]
    domain = {}

    # Process categorical columns
    for col in cols_categorical:
        domain[col] = process_categorical(df, col)

    # Process continuous columns
    for col, option, params in zip(cols_continuous, cont_options, cont_params):
        if params is None:
            params = {}

        if option == "binsize":
            domain[col] = process_cont_bin_size(df, col, **params)
        elif option == "nbins":
            domain[col] = process_cont_n_bins(df, col, **params)
        elif option == "custom":
            domain[col] = process_cont_custom_bins(df, col, **params)
        else:
            sys.exit("Invalid option - {}. Must be in (binsize, custom, nbins)".format(option))

    return df, domain
