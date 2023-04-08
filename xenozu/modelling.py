import pandas as pd
import numpy as np

def train_test_split(*to_split, target:pd.Series, val_size: float = .2, test_size: float = 0):
    """
        train test split for complex data type. Works well with multi-input keras that training data is a list of dataframe.
        target: list
            list match target and to_split index
    """
    for data in to_split:
        if isinstance(data, tuple):
            for subdata in data:
                assert subdata.shape[0] == target.shape[0]
        else:
            assert data.shape[0] == target.shape[0]

    data_size = len(to_split[0])
    val_size, test_size = int(data_size*val_size), int(data_size*test_size)
    index = np.arange(data_size)

    val_index = np.random.choice(index, val_size, replace=False)
    index = np.delete(index, val_index)

    test_index = np.random.choice(index, test_size, replace=False)
    index = np.delete(index, test_index)

    train, target_train = [], []
    validation, target_val = [], []
    test, target_test = [], []

    def access_index(data, inx): # access numpy or pandas dataframe index
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.iloc[inx]
        else:
            return data[inx]
    
    for data in to_split:
        train.append(access_index(data, index))
        validation.append(access_index(data, val_index))
        test.append(access_index(data, test_index))

        if target_train == []: # do once
            target_train.append(access_index(target, index))
            target_val.append(access_index(target, val_index))
            target_test.append(access_index(target, test_index))
    if test_size > 0:
        return train, target_train, validation, target_val, test, target_test
    return train, target_train, validation, target_val