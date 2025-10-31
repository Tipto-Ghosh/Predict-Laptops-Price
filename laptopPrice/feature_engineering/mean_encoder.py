# laptopPrice/feature_engineering/mean_encoder.py
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class MeanEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features=None):
        self.categorical_features = categorical_features
        self.encoding_maps = {}

    def fit(self, X, y):
        X = X.copy()
        y = pd.Series(y).reset_index(drop=True)

        # Auto-detect categorical features if not provided
        if self.categorical_features is None:
            self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Create mean encoding maps
        for col in self.categorical_features:
            df = pd.concat([X[col].reset_index(drop=True), y], axis=1)
            df.columns = [col, 'target']
            self.encoding_maps[col] = df.groupby(col)['target'].mean().to_dict()

        return self

    def transform(self, X):
        X = X.copy()
        for col, mapping in self.encoding_maps.items():
            if col in X.columns:
                X[col] = X[col].map(mapping)
                # handle unseen categories
                X[col].fillna(sum(mapping.values()) / len(mapping), inplace=True)
        return X
