#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%%
df = pd.read_csv("../Database/cleansed_dataframe.csv")
df = df.drop(df.columns[0], axis=1)
#%%
# show correlation
sns.heatmap(df.iloc[:,1:14].corr(),annot=True,cmap='bwr',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(12,10)
plt.savefig('../Statistic_results/corr_heatmap.png', format='png')
plt.show()
# READINGとWRITING, LISTENINGとSPEAKING に弱い相関
# READINGとSPEAKING は相関的に離れているが、他二つはどの分野とも結びつきの差が激しくない
# EXPERIENCE EXCEPT SCHOOL はどの指標とも弱い相関(正確には無相関)がある
# EXPOSURE は弱いが相関があると見てよさそう
#%%
