#%%
import numpy as np
import pandas as pd
import statsmodels.api as sa

import warnings
warnings.filterwarnings("ignore")
#%%
df = pd.read_csv("../Database/cleansed_dataframe.csv")
df = df.drop([df.columns[0], "TIME_STAMP"], axis=1)
#%%
i = 0
#%%
i += 1
df[df.columns[i]].value_counts()
#%%
i += 1
df[df.columns[i]].value_counts()
#%%
i += 1
df[df.columns[i]].value_counts()
#%%
i += 1
df[df.columns[i]].value_counts()
#%%
i += 1
df[df.columns[i]].value_counts()
#%%
i += 1
df[df.columns[i]].value_counts()
#%%
i += 1
df[df.columns[i]].value_counts()
#%%
i += 1
df[df.columns[i]].value_counts()
#%%
i += 1
df[df.columns[i]].value_counts()
#%%
i += 1
df[df.columns[i]].value_counts()
#%%
i += 1
df[df.columns[i]].value_counts()
#%%
i += 1
df[df.columns[i]].value_counts()
#%%
i += 1
df[df.columns[i]].value_counts()
#%%
i += 1
df[df.columns[i]].value_counts()
#%%
i += 1
df[df.columns[i]].value_counts()
#%%
