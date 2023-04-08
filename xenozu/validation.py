import pandas as pd
import numpy as np
from cleaning import conf_dtype

def validate_FE_train_test(train: pd.DataFrame, test: pd.DataFrame, target: str):
    
    # assert time == [] and cat == []
    _, _, (time, continu, discrete, cat) = conf_dtype(train, test, True)
    if time != []:
        print('='*20)
        print('Non number features : ', time)
        print('dtypes : time')
    if cat != []:
        print('='*20)
        print('Non number features : ', cat)
        print('dtypes : category')
    
    # assert (set(train.columns) ^ set(test.columns)) == [target]
    if (set(train.columns) ^ set(test.columns)) == [target]:
        print('='*20)
        print('not in train : ', set(test.columns)-set(train.columns))
        print('not in test : ', set(train.columns)-set(test.columns))
