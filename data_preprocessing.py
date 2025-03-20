import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def preprocess_data(df):
    df.fillna(df.median(), inplace=True)
    imputer = KNNImputer(n_neighbors=5)
    df.iloc[:, :] = imputer.fit_transform(df)
    return df
