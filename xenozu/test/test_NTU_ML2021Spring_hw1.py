import sys
import pandas as pd
import numpy as np

sys.path.append('../')

import cleaning
import modelling

def load_data():
    df = pd.read_csv('data/covid.train.csv')
    test = pd.read_csv('data/covid.test.csv')
    return df, test

def load_data_with_target():
    df, test = load_data()
    target = df.pop('tested_positive.2')
    return df, test, target

def test_conf_dtype():
    df, test = load_data()
    _, _, dtype = cleaning.conf_dtype(df, test, return_dtypes_list=True)
    expected_list = (
        [], 
        ['shop', 'worried_become_ill', 'worried_finances', 'ili.1', 'public_transit.1', 'felt_isolated.2', 'worried_finances.2', 'large_event.2', 'worried_become_ill.1', 'spent_time.1', 'shop.2', 'anxious', 'cli.1', 'restaurant', 'cli.2', 'felt_isolated', 'anxious.1', 'depressed', 'hh_cmnty_cli', 'restaurant.1', 'work_outside_home.2', 'worried_finances.1', 'ili', 'nohh_cmnty_cli', 'wearing_mask.2', 'depressed.2', 'spent_time.2', 'hh_cmnty_cli.2', 'worried_become_ill.2', 'cli', 'work_outside_home', 'large_event', 'wearing_mask', 'felt_isolated.1', 'public_transit.2', 'shop.1', 'work_outside_home.1', 'nohh_cmnty_cli.1', 'large_event.1', 'wearing_mask.1', 'tested_positive.1', 'ili.2', 'travel_outside_state', 'anxious.2', 'depressed.1', 'nohh_cmnty_cli.2', 'travel_outside_state.2', 'travel_outside_state.1', 'restaurant.2', 'public_transit', 'spent_time', 'tested_positive', 'hh_cmnty_cli.1'], 
        ['id', 'CA', 'NY', 'LA', 'VA', 'NE', 'MS', 'AL', 'NV', 'CO', 'IL', 'SC', 'ID', 'AR', 'OH', 'MD', 'GA', 'IA', 'AK', 'TX', 'OR', 'CT', 'AZ', 'NM', 'IN', 'WV', 'FL', 'MI', 'NC', 'WA', 'PA', 'MN', 'WI', 'MO', 'MA', 'KS', 'OK', 'KY', 'RI', 'NJ', 'UT'],
        []
        )
    for output, expected in zip(dtype, expected_list):
        assert set(output) == set(expected)

def test_train_test_split():
    for val_size, test_size in zip([.5, .2, .9], [.1, 0, .09]):
        train, _, target = load_data_with_target()
        original_len = len(train)
        if test_size > 0:
            train, target_train, val, target_val, test, target_test = modelling.train_test_split(train, target=target, val_size=val_size, test_size=test_size)
        else:
            train, target_train, val, target_val = modelling.train_test_split(train, target=target, val_size=val_size, test_size=test_size)
            test, target_test = [], []
        assert (len(train)+len(val)+len(test)) == original_len and (original_len*val_size) == len(val) and int(original_len*test_size) == len(test) and int(original_len*np.round(1-val_size-test_size, 2)) == len(train)
        assert (len(target_train)+len(target_val)+len(target_test)) == original_len and int(original_len*val_size) == len(target_val) and int(original_len*test_size) == len(target_test) and int(original_len*np.round(1-val_size-test_size, 2)) == len(target_train)