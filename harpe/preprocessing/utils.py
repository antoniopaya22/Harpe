# =================================================================
#
#                   Preprocessing - Utils
#
# author:  Antonio Paya Gonzalez
# =================================================================

# ==================> Imports
import numpy as np


# ==================> Functions
def multi_corr(col1, col2, df):
    """
    This function returns correlation between 2 given features.
    Also gives corr of the given features with "label" afetr applying log1p to it.
    """
    correlation = df[[col1, col2]].corr().iloc[0, 1]
    log_corr = df[col1].apply(np.log1p).corr(df[col2])

    print("Correlation : {}\nlog_Correlation: {}".format(correlation, log_corr))


def corr(col1, col2, df):
    """
    This function returns correlation between 2 given features
    """
    return df[[col1, col2]].corr().iloc[0, 1]


def log1p_transform(col, df):
    """
    Apply log1p on given column.
    Remove the original cola and keep log1p applied col
    """
    new_col = col + '_log1p'
    df[new_col] = df[col].apply(np.log1p)
    df.drop(col, axis=1, inplace=True)
