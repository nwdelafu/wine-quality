# Import desired Packages
from sklearn.base import TransformerMixin
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import skew, norm, probplot, boxcox, f_oneway
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style('whitegrid')

red_df = pd.read_csv('../input/winequality-red.csv', sep=';')
white_df = pd.read_csv('../input/winequality-white.csv', sep=';')
red_df['wine_color'] = 'red'
white_df['wine_color'] = 'white'
white_df['label'] = white_df['quality'].apply(
    lambda x: 1 if x <= 5 else 2 if x <= 7 else 3)
red_df['label'] = red_df['quality'].apply(
    lambda x: 1 if x <= 5 else 2 if x <= 7 else 3)

wine = pd.concat([red_df, white_df], axis=0)  # Combing

#shuffle data for randomization of data points
wine = wine.sample(frac=1, random_state=77).reset_index(drop=True)


class null_cleaner(TransformerMixin):

    def __init__(self):
        """
        fills missing values:
        -If the column is dtype object they are imputed with the most frequent
         value within the column
        -The other columns with data types are imputed with the mean
         of the corresponding column
        """

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
                              index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

wine = null_cleaner().fit_transform(wine)
