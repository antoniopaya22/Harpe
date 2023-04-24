# =================================================================
#
#                   Preprocessing - Transformations
#
# author:  Antonio Paya Gonzalez
# =================================================================

# ==================> Imports
import numpy as np
import pandas as pd

from preprocessing.utils import multi_corr, log1p_transform
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ==================> Functions
def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds new features to the data.

    :param data: pd.DataFrame, data to add the features.
    """
    # Adding new features
    data['network_bytes'] = data['dbytes'] + data['sbytes']
    data.drop(['attack_cat'], axis=1, inplace=True)
    return data


def remove_high_correlated_features(data: pd.DataFrame, to_drop=None) -> (pd.DataFrame, list):
    """
    Removes the features that are highly correlated.

    :param data: pd.DataFrame, data to remove the features.
    :param to_drop: list, list of features to drop.
    """
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    if to_drop is None:
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    data.drop(columns=to_drop, inplace=True)
    return data, to_drop


def apply_log1p(data: pd.DataFrame, log1p_col=None) -> pd.DataFrame:
    """
    Applies log1p to the data.

    :param data: pd.DataFrame, data to apply log1p.
    """
    col_unique_values = data.nunique()
    col = col_unique_values[col_unique_values > 200].index
    for column in col:
        print("{:-^30}".format(column))
        multi_corr(column, 'label', data)
    if log1p_col is None:
        log1p_col = ['dur', 'sload', 'dload', 'spkts', 'stcpb', 'dtcpb', 'sjit', 'djit', 'network_bytes']
    for col in log1p_col:
        log1p_transform(col, df=data)
    return data


def standardize_data(data: pd.DataFrame, cat_col=None) -> pd.DataFrame:
    """
    Standardizes the data.

    :param data: pd.DataFrame, data to standardize.
    :param cat_col: list, list of categorical columns.
    """
    if cat_col is None:
        cat_col = ['proto', 'service', 'state']
    num_col = list(set(data.columns) - set(cat_col))

    scaler = StandardScaler()
    scaler = scaler.fit(data[num_col])

    data[num_col] = scaler.transform(data[num_col])
    return data


def one_hot_encoding(data: pd.DataFrame, ohe_service=None, ohe_proto=None, ohe_state=None) -> pd.DataFrame:
    """
    One hot encodes the data.

    :param data: pd.DataFrame, data to one hot encode.
    :param ohe_service: OneHotEncoder, OneHotEncoder for the service column.
    :param ohe_proto: OneHotEncoder, OneHotEncoder for the proto column.
    :param ohe_state: OneHotEncoder, OneHotEncoder for the state column.
    """
    # Onehot encoding cat col using onehotencoder objects
    X = ohe_service.transform(data['service'].values.reshape(-1, 1))
    Xm = ohe_proto.transform(data['proto'].values.reshape(-1, 1))
    Xmm = ohe_state.transform(data['state'].values.reshape(-1, 1))

    # Adding encoding data to original data
    data = pd.concat([data,
                      pd.DataFrame(Xm.toarray(), columns=['proto_' + i for i in ohe_proto.categories_[0]]),
                      pd.DataFrame(X.toarray(), columns=['service_' + i for i in ohe_service.categories_[0]]),
                      pd.DataFrame(Xmm.toarray(), columns=['state_' + i for i in ohe_state.categories_[0]])],
                     axis=1)

    # Removing cat columns
    data.drop(['proto', 'service', 'state'], axis=1, inplace=True)
    return data
