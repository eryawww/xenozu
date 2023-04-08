import pandas as pd
import numpy as np

def conf_dtype(df:pd.DataFrame, test:pd.DataFrame, return_dtypes_list: bool = False):
    """
        convert dtypes to time, int (discrete), float (continu), and categorical (others)
        return: converted dtypes
            df, test
            if return_dtypes_list: return list of columns that has these types
                return df, test, ([time], [continu], [discret], [categorical])
    """
    _df, _test = df.copy(), test.copy()
    time, continu, discret, categorical = [], [], [], []
    for col in set(_df.columns) & set(_test.columns):
        if isinstance(df[col], np.datetime64):
            time.append(col)
            continue
        try:
            if all(_df[col].map(lambda x: int(x) == x)) and all(_test[col].map(lambda x: int(x) == x)):
                _df[col], _test[col] = _df[col].astype(np.int32), _test[col].astype(np.int32)
                discret.append(col)
                continue

            all(_df[col].map(lambda x: float(x))) and all(_test[col].map(lambda x: float(x)))
            _df[col], _test[col] = _df[col].astype(np.float32), _test[col].astype(np.float32)
            continu.append(col)
            continue

        except ValueError:
            _df[col], _test[col] = _df[col].astype('category'), _test[col].astype('category')
            categorical.append(col)
    if return_dtypes_list:
        return _df, _test, (time, continu, discret, categorical)
    return _df, _test

# NOT TESTED
def unique_df(df:pd.DataFrame, test:pd.DataFrame, with_continue_vars:bool=True):
    """
        Get unique DataFrame preview. Data is filtered exclude floating point
        return
            styled DataFrame
        relative coloring, coloring based on len of each dataset (train or test)
    """
    if with_continue_vars:
        cols = df.columns
    else:
        cols = df.select_dtypes(exclude=['float', 'floating']).columns

    _df = pd.DataFrame(columns=list(cols)+['total'], index=['train', 'test'])
    _df.loc['train', 'total'] = len(df)
    _df.loc['test', 'total'] = len(test)
    for col in cols:
        _df.loc['train', col] = df[col].nunique()
        if col in test.columns:
            _df.loc['test', col] = test[col].nunique()
        else:
            _df.loc['test', col] = np.nan    
    _sorted_cols = _df.loc['train', cols].sort_values(ascending=False)
    return _df[['total']+list(_sorted_cols.index)].style.background_gradient(cmap=sns.dark_palette("seagreen", as_cmap=True), vmin=0, vmax=len(df) , subset=pd.IndexSlice['train', cols]).background_gradient(cmap=sns.dark_palette('seagreen', as_cmap=True), vmin=0, vmax=len(test), subset=pd.IndexSlice['test', cols])

# NOT TESTED
def nan_df(df:pd.DataFrame, test:pd.DataFrame):
    """
        Get unique DataFrame preview. Data is filtered exclude floating point
        return
            styled DataFrame
        relative coloring, coloring based on len of each dataset (train or test)
    """
    _df = pd.DataFrame(columns=df.columns, index=['train', 'test'])
    for col in df.columns:
        _df.loc['train', col] = df[col].isna().sum()
        if col in test.columns:
            _df.loc['test', col] = test[col].isna().sum()
        else:
            _df.loc['test', col] = np.nan    
    _sorted_cols = _df.loc['train', :].sort_values(ascending=False)
    print('='*20)
    print('NA in Train : ', df.isna().sum().sum())
    print('NA in Test  : ', test.isna().sum().sum())
    return _df[_sorted_cols.index].style.background_gradient(cmap=sns.dark_palette((20, 60, 50), input="husl", as_cmap=True), vmin=0, vmax=len(df) , subset=pd.IndexSlice['train', df.columns]).background_gradient(cmap=sns.dark_palette((20, 60, 50), input="husl", as_cmap=True), vmin=0, vmax=len(test), subset=pd.IndexSlice['test', df.columns])

def value_range(df: pd.DataFrame, test: pd.DataFrame):
    cols = df.select_dtypes(exclude=['category']).columns
    _df = pd.DataFrame(columns=cols, index=['train', 'test'])
    for col in cols:
        _df.loc['train', col] = df[col].max()-df[col].min()
        if col in test.columns:
            _df.loc['test', col] = test[col].max()-test[col].min()
        else:
            _df.loc['test', col] = np.nan
    _mean = _df.mean().mean()
    return _df.style.background_gradient(cmap=sns.dark_palette((20, 60, 50), input="husl", as_cmap=True), vmin=0, vmax=_mean, subset=pd.IndexSlice['train', cols]).background_gradient(cmap=sns.dark_palette((20, 60, 50), input="husl", as_cmap=True), vmin=0, vmax=_mean, subset=pd.IndexSlice['test', cols])