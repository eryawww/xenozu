import pandas as pd
import seaborn as sns
import numpy as np

def compare_summary_stats_diff(target:pd.DataFrame, pred:pd.DataFrame):
    index = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
    cols = ['target', 'pred', 'diff']
    func = [np.mean, np.std, np.min, lambda x: np.quantile(x, .25), lambda x: np.quantile(x, .50), lambda x: np.quantile(x, .75), np.max]
    _df = pd.DataFrame(index=index, columns=cols)
    for now_index, now_func in zip(index, func):
        _df.loc[now_index, 'target'] = now_func(target)
        _df.loc[now_index, 'pred'] = now_func(pred)
        _df.loc[now_index, 'diff'] = _df.loc[now_index, 'pred']-_df.loc[now_index, 'target']

    return _df.style.background_gradient(sns.color_palette("vlag", as_cmap=True), subset='diff')

def print_score(train, target_train, val, target_val, model, loss_fn: callable, eval = None, target_eval = None):
    # TODO Add Kfold interface
    print('='*20)
    try:
        print('Train Loss : ', loss_fn(target_train, model.predict(train)))
        print('Val Loss : ', loss_fn(target_val, model.predict(val)))
        if eval is not None:
            print('Eval Loss : ', loss_fn(target_eval, model.predict(eval)))
    except ValueError:
        print('Train Loss : ', loss_fn(target_train, model.predict(train).squeeze()))
        print('Val Loss : ', loss_fn(target_val, model.predict(val).squeeze()))
        if eval is not None:
            print('Eval Loss : ', loss_fn(target_eval, model.predict(eval).squeeze()))