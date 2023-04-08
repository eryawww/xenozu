import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

def compare_transform_to_normal(df: pd.DataFrame, col: str, func=[np.log]):
    """
        Visualize different between transformation and non transformation
        @params:
            func: str | callable | tuple(func_name, callable) transformation to be applied
                str: Literal['boxcox', 'yeojohnson']
        @return:
            None
    """
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('Transformation on '+col)
    n_rows, n_cols = int(np.ceil((len(func)+1)/2)), 4
    for i in range(0, 2*(len(func)+1), 2):
        fig.add_subplot(n_rows, n_cols, i+1)
        if i == 0:
            sns.histplot(df[col], kde=True)
            plt.xlabel("without transformation")
        else:
            apply_func = func[(i-2)//2]
            if apply_func == "boxcox":
                apply_func = lambda x: stats.boxcox(x)[0]
                apply_func.__name__ = 'boxcox'
            if apply_func == 'yeojohnson':
                apply_func = lambda x: stats.yeojohnson(x)[0]
                apply_func.__name__ = 'yeojohnson'
            if isinstance(apply_func, tuple):
                sns.histplot(apply_func[1](df[col]), kde=True)
                plt.xlabel(apply_func[0])
            else:
                sns.histplot(apply_func(df[col]), kde=True)
                plt.xlabel(apply_func.__name__)
        fig.add_subplot(n_rows, n_cols, i+2)
        stats.probplot(df[col], plot=plt)
        plt.title("")
        plt.xlabel("")
    plt.tight_layout()

def show_correlation_to_variable(df: pd.DataFrame, col:str):
    """
        Visualize a beauty visualization of correlation related to one columns
    """
    corr_series = df.corr()[col].drop(col)
    _df = pd.DataFrame(index=['Value'], columns=corr_series.index)
    for col in corr_series.index:
        _df.loc['Value', col] = corr_series[col]
    sorted_index = _df.loc['Value', :].sort_values(ascending=False).index
    return _df[sorted_index].style.background_gradient(sns.diverging_palette(20, 145, n=10, center='dark', as_cmap=True), vmin=-1, vmax=1, subset=sorted_index)

def plot_tf_hist(hist, test_score:int = None):
    fig = plt.figure(figsize=(20, 7))
    nrows, ncols = int(np.ceil(len(hist.history)/2)), 2
    _counter = 1
    for metrics in hist.history.keys():
        if str(metrics).startswith('val_'): continue
        fig.add_subplot(nrows, ncols, _counter)
        
        sns.lineplot(hist.history[metrics])
        sns.regplot(x=np.arange(len(hist.history[metrics])), y=hist.history[metrics])
        plt.title(str(metrics)+' on train')
        if test_score is not None:
            plt.axhline(test_score, color='red', label='eval score')
        plt.legend()

        fig.add_subplot(nrows, ncols, _counter+1)
        sns.lineplot(hist.history['val_'+metrics])
        sns.regplot(x=np.arange(len(hist.history['val_'+metrics])), y=hist.history['val_'+metrics])
        plt.title(str(metrics)+' on val')
        if test_score is not None:
            plt.axhline(test_score, color='red', label='eval score')
        plt.legend()

        _counter += 2