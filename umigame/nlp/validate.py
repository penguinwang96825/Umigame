import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold


class AdversarialValidator:
    """
    Adversarial Validator for textual data.
    Creating a better validation set when test examples differ from training examples.

    Parameters
    ----------
    X_train: List[str]
    X_test: List[str]

    Examples
    --------
    >>> from sklearn.feature_extraction.text import CountVectorizer
    >>> from sklearn.feature_extraction.text import TfidfTransformer
    >>> from sklearn.calibration import CalibratedClassifierCV
    >>> from sklearn.linear_model import SGDClassifier
    >>> from sklearn.pipeline import Pipeline

    >>> pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier(loss='hinge')),
        ])
    >>> estimator = CalibratedClassifierCV(pipeline, cv=5, method='sigmoid')
    >>> validator = AdversarialValidator(n_splits=10, train_size=0.8)
    >>> validator.validate(train_df["text"], test_df["text"], estimator)
    >>> train_idx, valid_idx = validator.get_index()
    """
    def __init__(self, n_splits=10, train_size=0.7):
        self.n_splits = n_splits
        self.train_size = train_size
    
    def validate(self, X_train, X_test, estimator):
        self._prepare(X_train, X_test)
        oof = np.zeros(len(self.X))
        folds = KFold(n_splits=self.n_splits, shuffle=True, random_state=914)
        for fold, (train_idx, valid_idx) in enumerate(folds.split(self.X, self.y)):
            train_feature, train_label = self.X[train_idx], self.y[train_idx]
            valid_feature, valid_label = self.X[valid_idx], self.y[valid_idx]
            estimator.fit(train_feature, train_label)
            oof[valid_idx] = estimator.predict_proba(self.X[valid_idx])[:, 1]
            valid_rocauc = metrics.roc_auc_score(valid_label, oof[valid_idx])
            print(f"Fold [{fold}]: ROC-AUC = {valid_rocauc:.4f}")
        print(f"\nOverall ROC-AUC = {metrics.roc_auc_score(self.y, oof):.4f}")
        df = pd.DataFrame({
            "test_proba": oof, 
            "train_proba": 1 - oof
        }).iloc[:len(self.y_train)]
        self.train_idx = df.sort_values(by="test_proba").index[:int(len(self.y_train)*self.train_size)]
        self.valid_idx = df.sort_values(by="test_proba").index[int(len(self.y_train)*self.train_size):]
        
    def get_index(self):
        if not hasattr(self, "train_idx"):
            raise NotRunError("Validator has not been run...")
        return self.train_idx, self.valid_idx
    
    def _prepare(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train, self.y_test = self._label()
        self.X = np.concatenate((self.X_train, self.X_test), axis=0)
        self.y = np.concatenate((self.y_train, self.y_test), axis=0)

    def _label(self):
        y_train = np.ones(shape=(len(self.X_train)))
        y_test = np.zeros(shape=(len(self.X_test)))
        return y_train, y_test


class NotRunError(ValueError):
    pass