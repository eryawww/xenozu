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
    pass