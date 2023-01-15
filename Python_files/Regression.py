#%%
import numpy as np
import pandas as pd
import copy as cp
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sa


from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")
#%%
df = pd.read_csv("../Database/cleansed_dataframe.csv")
df = df.drop([df.columns[0], "TIME_STAMP"], axis=1)
#%%
# GENDERのダミー化
df.replace({'GENDER': {'M': 0, 'F': 1}}, inplace=True)

# OCCUPATIONのone-hot-encoding
occupation_ohe = pd.get_dummies(df['OCCUPATION'])
df = pd.concat([df, occupation_ohe], axis=1)
df.drop(['OCCUPATION', 'others'], axis=1, inplace=True)

# CEFRの除外
df.drop('CEFR', axis=1, inplace=True)

# B_なし、A_なし　の除外
df.drop(['A_なし', 'B_なし'], axis=1, inplace=True)
#%%
# READINGを目的変数とした重回帰
linear_regression = LinearRegression()

X = df.drop('READING', axis=1)
y = df.READING
model = sa.OLS(y,X)
result = model.fit()
result.summary()
#%%
f = open('../Statistic_results/a.txt', 'w')
f.write(str(result.summary()))
f.close()
#%%
def multiple_linear_regression(dataframe, dependent_name):
    linear_regression = LinearRegression()
    X = dataframe.drop(dependent_name, axis=1)
    y = dataframe[dependent_name]
    model = sa.OLS(y, X)
    result = model.fit()
    f = open('../Statistic_results/{}.txt'.format(dependent_name), 'w')
    f.write(str(result.summary()))
    f.close()
#%%
multiple_linear_regression(df, "READING")
#%%
