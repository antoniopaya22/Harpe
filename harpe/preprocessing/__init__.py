# =================================================================
#
#                   Preprocessing
#
# author:  Antonio Paya Gonzalez
# =================================================================

# ==================> Imports
from preprocessing.load_data import load_data
from preprocessing.transformations import *


# ==================> Functions
def preprocess(train_path: list, test_path: list) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Preprocess the data

    :param train_path: list, list of paths to the train files.
    :param test_path: list, list of paths to the test files.
    """
    train = load_data(train_path)
    test = load_data(test_path)

    train = add_features(train)
    test = add_features(test)

    train, to_drop = remove_high_correlated_features(train)
    test, _ = remove_high_correlated_features(test, to_drop=to_drop)

    train = apply_log1p(train)
    test = apply_log1p(test)

    x_train, y_train = train.drop(columns=['label']), train['label']
    x_test, y_test = test.drop(columns=['label']), test['label']

    x_train = standardize_data(x_train)
    x_test = standardize_data(x_test)

    service_ = OneHotEncoder()
    proto_ = OneHotEncoder()
    state_ = OneHotEncoder()

    ohe_service = service_.fit(
        np.concatenate((x_train['service'].values.reshape(-1, 1), x_test['service'].values.reshape(-1, 1)), axis=0))
    ohe_proto = proto_.fit(
        np.concatenate((x_train['proto'].values.reshape(-1, 1), x_test['proto'].values.reshape(-1, 1)), axis=0))
    ohe_state = state_.fit(
        np.concatenate((x_train['state'].values.reshape(-1, 1), x_test['state'].values.reshape(-1, 1)), axis=0))

    x_train = one_hot_encoding(x_train, ohe_service, ohe_proto, ohe_state)
    x_test = one_hot_encoding(x_test, ohe_service, ohe_proto, ohe_state)

    print(len(x_train.columns))
    print(len(x_test.columns))

    return x_train, y_train, x_test, y_test
