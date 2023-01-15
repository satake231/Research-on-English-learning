#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import gc

# scikit-learn
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


# LightGBM
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")
#%%
df = pd.read_csv("../Database/cleansed_dataframe.csv")
df = df.drop(df.columns[0], axis=1)
# TIME_STAMPをID化
for i in range(len(df["TIME_STAMP"])):
    df["TIME_STAMP"][i] = i
df["TIME_STAMP"] = df['TIME_STAMP'].astype('int8')
#%%
# show correlation
sns.heatmap(df.iloc[:,1:14].corr(),annot=True,cmap='bwr',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(12,10)
plt.savefig('../Statistic_results/corr_heatmap.png', format='png')
plt.show()
# READINGとWRITING, LISTENINGとSPEAKING に弱い相関
# READINGとSPEAKING は相関的に離れているが、他二つはどの分野とも結びつきの差が激しくない
# EXPERIENCE EXCEPT SCHOOL はどの指標とも超弱い相関(正確には無相関)がある
# EXPOSURE は弱いが相関があると見てよさそう
# AVG_JUNIOR_HIGHとAVG_HIGHには弱い相関