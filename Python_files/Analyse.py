#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%%
df = pd.read_csv("../Database/cleansed_dataframe.csv")
df = df.drop(df.columns[0], axis=1)
#%%
sns.heatmap(df.iloc[:,1:14].corr(),annot=True,cmap='bwr',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.savefig('../Statistic_results/corr_heatmap.png', format='png')
plt.show()
#%%
