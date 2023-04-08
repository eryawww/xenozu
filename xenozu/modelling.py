import pandas as pd
import numpy as np

def train_test_split(*to_split, target:pd.Series, val_size: float = .2, test_size: float = 0):
    """
        train test split that support complex data type. Works well with multi-input keras that training data is a list.
        target: list
            list match target and to_split index
        @returns:
            test_size > 0?
                train, target_train, val, target_val, test, target_test \n
                train, target_train, validation, target_val
    """
    def validate_params():
        for data in to_split:
            if isinstance(data, tuple):
                for subdata in data:
                    assert subdata.shape[0] == target.shape[0]
            else:
                assert data.shape[0] == target.shape[0]
    def get_split_index(val_size, test_size):
        data_size = len(to_split[0])
        val_size, test_size = int(data_size*val_size), int(data_size*test_size)
        index = np.arange(data_size)

        val_index = np.random.choice(index, val_size, replace=False)
        index = np.setdiff1d(index, val_index)

        test_index = np.random.choice(index, test_size, replace=False)
        index = np.setdiff1d(index, test_index)
        return (index, val_index, test_index)
    def access_index(data, inx): # access numpy or pandas dataframe index
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.iloc[inx]
        else:
            return data[inx]
    def append_data(train: list, validation: list, test: list, indexs: tuple, data):
        train.append(access_index(data, indexs[0]))
        validation.append(access_index(data, indexs[1]))
        test.append(access_index(data, indexs[2]))
        return train, validation, test
    def append_target(target_train:list, target_val:list, target_test:list, indexs: tuple):
        if target_train == []:
            target_train.append(access_index(target, indexs[0]))
            target_val.append(access_index(target, indexs[1]))
            target_test.append(access_index(target, indexs[2]))
        return target_train, target_val, target_test
    def _unboxing_if_needed(x:list):
        return x[0] if len(x) == 1 else x


    validate_params()
    indexs = get_split_index(val_size, test_size)

    train, target_train = [], []
    validation, target_val = [], []
    test, target_test = [], []

    for data in to_split:
        train, validation, test = append_data(train, validation, test, indexs, data)
        target_train, target_val, target_test = append_target(target_train, target_val, target_test, indexs)
    
    train, validation, test = _unboxing_if_needed(train), _unboxing_if_needed(validation), _unboxing_if_needed(test)
    target_train, target_val, target_test = _unboxing_if_needed(target_train), _unboxing_if_needed(target_val), _unboxing_if_needed(target_test)

    if test_size > 0:
        return train, target_train, validation, target_val, test, target_test
    return train, target_train, validation, target_val