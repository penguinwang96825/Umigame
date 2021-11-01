import talib
import fasttext
import pandas as pd
import numpy as np
import vectorbt as vbt
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.exceptions import NotFittedError


import os
import sys
CURRENT_DIR = os.getcwd()
sys.path.append(CURRENT_DIR)
from umigame.nlp import utils
from umigame.utils import crossover, crossunder
from umigame.nlp.utils import get_dataframes
pd.options.mode.chained_assignment = None
utils.seed_everything()


DATASET = 'twitter'
LABELING = 'tb'


class SklearnClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        # Define Hyperparams for getter-setter
        self._hyperparams = [] 

    def fit(self):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError
        
    def score(self, X, y, scoring=None):
        if scoring is None:
            scoring = lambda x, y: metrics.f1_score(x, y)
        y_pred = self.predict(X)
        return scoring(y, y_pred)

    def get_params(self, deep=True):
        params = dict()
        variables = self._hyperparams
        for key in variables:
            var_value = getattr(self, key, None)
            params[key] = var_value
        return params

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_param_names(self):
        return self._hyperparams


class FasttextClassifier(SklearnClassifier):

    def __init__(self, autotune=False, **kwargs):
        """
        Parameters
        ----------
        input             # training file path (required)
        lr                # learning rate [0.1]
        dim               # size of word vectors [100]
        ws                # size of the context window [5]
        epoch             # number of epochs [5]
        minCount          # minimal number of word occurences [1]
        minCountLabel     # minimal number of label occurences [1]
        minn              # min length of char ngram [0]
        maxn              # max length of char ngram [0]
        neg               # number of negatives sampled [5]
        wordNgrams        # max length of word ngram [1]
        loss              # loss function {ns, hs, softmax, ova} [softmax]
        bucket            # number of buckets [2000000]
        thread            # number of threads [number of cpus]
        lrUpdateRate      # change the rate of updates for the learning rate [100]
        t                 # sampling threshold [0.0001]
        label             # label prefix ['__label__']
        verbose           # verbose [2]
        pretrainedVectors # pretrained word vectors (.vec file) for supervised learning []
        """
        self.autotune = autotune
        self.kwargs = kwargs
        self.kwargs.pop('input', None)
        self.filepath_train, self.filepath_valid = self._get_temp_filepath()
        self.model = None
        
        # Define Hyperparams for getter-setter
        self._hyperparams = ["lr", "epoch", "minCount"] 

    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        self.class_labels_ = ['__label__{}'.format(lbl) for lbl in self.classes_]
        self._dump_to_csv(X, y)
        
        if self.autotune:
            self.model = fasttext.train_supervised(
                self.filepath_train, 
                autotuneValidationFile=self.filepath_valid, 
                **self.kwargs
            )
            os.system(f"rm {self.filepath_train}")
            os.system(f"rm {self.filepath_valid}")
        else:
            self.model = fasttext.train_supervised(
                self.filepath_train,  
                **self.kwargs
            )
            os.system(f"rm {self.filepath_train}")
        
        return self
    
    def predict(self, X):
        self._check_model_fitted()
        test_pred = np.array([self.model.predict(x)[0][0].split("__")[-1] for x in X], dtype=float)
        return test_pred

    def predict_proba(self, X):
        self._check_model_fitted()
        proba = [self._reformat(self.model.predict(x, k=len(self.classes_))) for x in X]
        return np.array(proba)
    
    def _get_temp_filepath(self):
        filepath_train = r"./train.csv"
        filepath_valid = r"./valid.csv"
        return filepath_train, filepath_valid
    
    def _reformat(self, res):
        lbl_prb_pairs = zip(res[0], res[1])
        sorted_lbl_prb_pairs = sorted(
            lbl_prb_pairs, key=lambda x: self.class_labels_.index(x[0]))
        return [x[1] for x in sorted_lbl_prb_pairs]
        
    def _check_model_fitted(self):
        if self.model is None:
            raise NotFittedError("This {} instance is not fitted yet.".format(
                self.__class__.__name__))
        
    def _dump_to_csv(self, X, y):
        X = np.array(X).reshape(-1, )
        y = np.array(y).reshape(-1, )
        df = pd.DataFrame({
            "text": X, 
            "label": y
        }, index=range(len(y)))
        df["label"] = '__label__' + df["label"].astype(str)
        if self.autotune:
            msk = np.random.rand(len(df)) < 0.7
            df_train = df[msk]
            df_valid = df[~msk]
            df_train.to_csv(self.filepath_train, index=None, header=None, sep='\t')
            df_valid.to_csv(self.filepath_valid, index=None, header=None, sep='\t')
        else:
            df.to_csv(self.filepath_train, index=None, header=None, sep='\t')
        
    def _show_model_params(self):
        args_obj = self.model.f.getArgs()
        for hparam in dir(args_obj):
            if not hparam.startswith('__'):
                print(f"{hparam} -> {getattr(args_obj, hparam)}")


def main():
    ##################
    ### Train Code ###
    ##################

    train_df, test_df = get_dataframes(dataset=DATASET)
    X_train, X_test = train_df['clean_text'], test_df['clean_text']
    y_train, y_test = train_df[LABELING], test_df[LABELING]

    text_clf = FasttextClassifier(
        dim=300, 
        epoch=40, 
        pretrainedVectors=r'F:\kaggle\embedding\processed\wiki-news-300d-1M-subword.vec'
    )
    text_clf.fit(X_train, y_train)

    ##########################
    ### Evaluate the Model ###
    ##########################

    y_test_proba = text_clf.predict_proba(X_test)
    y_test_pred = text_clf.predict(X_test)

    acc = metrics.accuracy_score(y_test, y_test_pred)
    f1 = metrics.f1_score(y_test, y_test_pred, average="weighted")
    p = metrics.precision_score(y_test, y_test_pred, average="weighted")
    r = metrics.recall_score(y_test, y_test_pred, average="weighted")
    print(
        f'\nBefore Aggregate: \n'
        f"\tAcc: {acc:.4f}\n"
        f"\tP: {p:.4f}\n"
        f"\tR: {r:.4f}\n"
        f"\tF1: {f1:.4f}"
    )

    test_df.loc[:, "y_prob_0"] = np.array(y_test_proba)[:, 0]
    test_df.loc[:, "y_prob_1"] = np.array(y_test_proba)[:, 1]
    test_df.loc[:, "label"] = y_test
    test_tfal_df = test_df.groupby(test_df.index).agg({
        "y_prob_0":"mean", "y_prob_1":"mean", "label":"max", 
        "open":"max", "high":"max", "low":"max", "close":"max"
    })
    y_prob_tfal = test_tfal_df[["y_prob_0", "y_prob_1"]].values
    y_pred_tfal = np.argmax(y_prob_tfal, axis=1)
    y_true_tfal = test_tfal_df["label"].values.ravel()
    test_tfal_df['trend'] = test_tfal_df['y_prob_1'].apply(lambda x: 1 if x >= 0.5 else 0)
    acc = metrics.accuracy_score(y_true_tfal, y_pred_tfal)
    f1 = metrics.f1_score(y_true_tfal, y_pred_tfal, average="weighted")
    p = metrics.precision_score(y_true_tfal, y_pred_tfal, average="weighted")
    r = metrics.recall_score(y_true_tfal, y_pred_tfal, average="weighted")
    print(
        f'\nAfter Aggregate: \n'
        f"\tAcc: {acc:.4f}\n"
        f"\tP: {p:.4f}\n"
        f"\tR: {r:.4f}\n"
        f"\tF1: {f1:.4f}"
    )
    print(metrics.confusion_matrix(y_true_tfal, y_pred_tfal))

    ###################
    ### Backtesting ###
    ###################

    def get_entry_and_exit(trigger):
        if trigger == 'sma':
            fast_ma = talib.SMA(test_tfal_df['close'], timeperiod=10)
            slow_ma = talib.SMA(test_tfal_df['close'], timeperiod=20)
            entries = crossover(fast_ma, slow_ma)
            exits = crossunder(fast_ma, slow_ma)
        
        elif trigger == 'wma':
            fast_ma = talib.WMA(test_tfal_df['close'], timeperiod=10)
            slow_ma = talib.WMA(test_tfal_df['close'], timeperiod=20)
            entries = crossover(fast_ma, slow_ma)
            exits = crossunder(fast_ma, slow_ma)

        elif trigger == 'ema':
            fast_ma = talib.EMA(test_tfal_df['close'], timeperiod=10)
            slow_ma = talib.EMA(test_tfal_df['close'], timeperiod=20)
            entries = crossover(fast_ma, slow_ma)
            exits = crossunder(fast_ma, slow_ma)

        return entries, exits

    plt.figure(figsize=(15, 3))
    perf = pd.DataFrame()
    for trigger in ['sma', 'ema', 'wma']:
        entries, exits = get_entry_and_exit(trigger)

        pf_kwargs = dict(
            size=np.inf, 
            fees=0.001, 
            slippage=0.001, 
            freq='1D', 
            init_cash=10000, 
            sl_stop=0.5, 
            tp_stop=1.5
        )
        pf_trigger_without_filter = vbt.Portfolio.from_signals(test_tfal_df['close'], entries, exits, **pf_kwargs)
        trigger_without_filter_returns = (pf_trigger_without_filter.daily_returns() + 1).cumprod()
        benchmark_returns = (pf_trigger_without_filter.benchmark_returns() + 1).cumprod()

        new_entries = entries & (test_tfal_df['trend']==0)
        new_exits = exits
        pf_trigger_with_filter = vbt.Portfolio.from_signals(test_tfal_df['close'], new_entries, new_exits, **pf_kwargs)
        trigger_with_filter_returns = (pf_trigger_with_filter.daily_returns() + 1).cumprod()

        perf = pd.concat([
            perf, 
            pd.DataFrame(pf_trigger_without_filter.stats(), columns=[f'{trigger}']), 
            pd.DataFrame(pf_trigger_with_filter.stats(), columns=[f'{trigger}-with-filter'])
        ], axis=1)

        plt.plot(trigger_without_filter_returns, label=f'{trigger}')
        plt.plot(trigger_with_filter_returns, label=f'{trigger} with filter')

    plt.plot(benchmark_returns, label='buy and hold')
    plt.legend()
    plt.grid()
    plt.show()

    print(perf)


if __name__ == '__main__':
    main()