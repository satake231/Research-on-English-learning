#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%%
df = pd.read_excel("../Database/new_dataframe.xlsx")
df = df.drop(df.columns[0], axis=1)
#%%
def make_number_compare_graph(df, name, hue):
    f,ax=plt.subplots(1,figsize=(18,8), facecolor='gray')
    sns.countplot(x=name,hue=hue,data=df,ax=ax)
    ax.set_title(name + 'vs' + hue)
    plt.savefig('../Compare_images/graph_'+name+'_vs_'+hue+'_.png', format='png')
    plt.show()
#%%
def make_ratio_compare_graph(df, name, hue):
    tmp = (
        df.groupby(hue)[name]
        .value_counts(normalize=True)
        .rename("percentage")
        .reset_index()
    )
    ax = sns.barplot(x=tmp[name], y =tmp["percentage"], hue=tmp[hue])
    ax.grid()
    plt.savefig('../Compare_images/graph_'+name+'_vs_'+hue+'_.png', format='png')
    plt.show()
#%%
def make_violinplot(df, name, hue):
    import copy as cp
    df_exam = cp.deepcopy(df)
    df_exam["to_" + name] = 0
    sns.violinplot(data=df_exam, x="to_" + name, y=name, hue=hue)
    sns.stripplot(df_exam[name], color='red', size=0.2)
    plt.savefig("../Compare_images/graph_"+name+"_.png", format="png")
    plt.show()
#%%
# GENDER
make_number_compare_graph(df, "OCCUPATION", "GENDER") # 性別の分布に偏りはあるが、極端な分布ではないとわかる

make_ratio_compare_graph(df, "READING", "GENDER")
make_ratio_compare_graph(df, "LISTENING", "GENDER")
make_ratio_compare_graph(df, "WRITING", "GENDER")
make_ratio_compare_graph(df, "SPEAKING", "GENDER")
# 男性のほうが、英語に自信を持ちやすい（得意というわけではない）

make_ratio_compare_graph(df, "CEFR", "GENDER")
# 後述するが、あまりCEFRスコアが信憑性ない可能性大

make_ratio_compare_graph(df, "EXPERIENCE EXCEPT SCHOOL", "GENDER")

make_ratio_compare_graph(df, "ENGLISH OF FAMILY", "GENDER")

make_violinplot(df, "AVG_JUNIOR_HIGH", "GENDER")

make_violinplot(df, "AVG_HIGH", "GENDER")

make_ratio_compare_graph(df, "USEFUL", "GENDER")

make_ratio_compare_graph(df, "EXPOSURE", "GENDER")
#%%
# READING
make_ratio_compare_graph(df, "LISTENING", "READING")
make_ratio_compare_graph(df, "WRITING", "READING")
make_ratio_compare_graph(df, "SPEAKING", "READING")

make_ratio_compare_graph(df, "CEFR", "READING")
# 後述するが、あまりCEFRスコアが信憑性ない可能性大

make_ratio_compare_graph(df, "EXPERIENCE EXCEPT SCHOOL", "READING")

make_ratio_compare_graph(df, "ENGLISH OF FAMILY", "READING")

make_violinplot(df, "AVG_JUNIOR_HIGH", "READING")

make_violinplot(df, "AVG_HIGH", "READING")

make_ratio_compare_graph(df, "USEFUL", "READING")

make_ratio_compare_graph(df, "EXPOSURE", "READING")
#%%
# LISTENING
make_ratio_compare_graph(df, "WRITING", "LISTENING")
make_ratio_compare_graph(df, "SPEAKING", "LISTENING")

make_ratio_compare_graph(df, "CEFR", "LISTENING")
# 後述するが、あまりCEFRスコアが信憑性ない可能性大

make_ratio_compare_graph(df, "EXPERIENCE EXCEPT SCHOOL", "LISTENING")

make_ratio_compare_graph(df, "ENGLISH OF FAMILY", "LISTENING")

make_violinplot(df, "AVG_JUNIOR_HIGH", "LISTENING")

make_violinplot(df, "AVG_HIGH", "LISTENING")

make_ratio_compare_graph(df, "USEFUL", "LISTENING")

make_ratio_compare_graph(df, "EXPOSURE", "LISTENING")
#%%
# WRITING
make_ratio_compare_graph(df, "SPEAKING", "WRITING")

make_ratio_compare_graph(df, "CEFR", "WRITING")
# 後述するが、あまりCEFRスコアが信憑性ない可能性大

make_ratio_compare_graph(df, "EXPERIENCE EXCEPT SCHOOL", "WRITING")

make_ratio_compare_graph(df, "ENGLISH OF FAMILY", "WRITING")

make_violinplot(df, "AVG_JUNIOR_HIGH", "WRITING")

make_violinplot(df, "AVG_HIGH", "WRITING")

make_ratio_compare_graph(df, "USEFUL", "WRITING")

make_ratio_compare_graph(df, "EXPOSURE", "WRITING")
#%%
# SPEAKING
make_ratio_compare_graph(df, "CEFR", "SPEAKING")
# 後述するが、あまりCEFRスコアが信憑性ない可能性大

make_ratio_compare_graph(df, "EXPERIENCE EXCEPT SCHOOL", "SPEAKING")

make_ratio_compare_graph(df, "ENGLISH OF FAMILY", "SPEAKING")

make_violinplot(df, "AVG_JUNIOR_HIGH", "SPEAKING")

make_violinplot(df, "AVG_HIGH", "SPEAKING")

make_ratio_compare_graph(df, "USEFUL", "SPEAKING")

make_ratio_compare_graph(df, "EXPOSURE", "SPEAKING")
#%%
# CEFR
make_ratio_compare_graph(df, "EXPERIENCE EXCEPT SCHOOL", "CEFR")

make_ratio_compare_graph(df, "ENGLISH OF FAMILY", "CEFR")

make_violinplot(df, "AVG_JUNIOR_HIGH", "CEFR")

make_violinplot(df, "AVG_HIGH", "CEFR")

make_ratio_compare_graph(df, "USEFUL", "CEFR")

make_ratio_compare_graph(df, "EXPOSURE", "CEFR")
#%%
