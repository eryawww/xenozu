import pandas as pd

def polynomialFeatures(train: pd.DataFrame, test: pd.DataFrame, cols: list, degree: int = 2):
    from sklearn.preprocessing import PolynomialFeatures

    trans = PolynomialFeatures(degree=degree)
    train_nw_df = pd.DataFrame(trans.fit_transform(train[cols]), columns=trans.get_feature_names(cols))
    test_nw_df = pd.DataFrame(trans.transform(test[cols]), columns=trans.get_feature_names(cols))
    train = pd.concat([train, train_nw_df.drop(cols, axis=1)], axis=1)
    test = pd.concat([test, test_nw_df.drop(cols, axis=1)], axis=1)
    return train, test

def winsorization(df: pd.DataFrame, cols: list, quantile: float = .05):
    from scipy import stats
    
    for col in cols:
        df[col] = stats.mstats.winsorize(df[col], limits=[quantile, quantile])
    return df

