import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%%
df = pd.read_excel("../Database/new_dataframe.xlsx")
df = df.drop(df.columns[0], axis=1)
#%%
# BEFORE ELEMENTARY
for i in range(len(df["BEFORE ELEMENTARY"])):
    if df["BEFORE ELEMENTARY"][i] == "全く":
        df['BEFORE ELEMENTARY'][i] = "なし"

# 複数回答データを','で分割して、１つのリストにまとめる
ans = [df['BEFORE ELEMENTARY'][i].split(';') for i in range(len(df["BEFORE ELEMENTARY"]))]

# リストのflattening
ans_ = []
for s in ans:
    ans_.extend(s)

# 重複を除きユニークな回答選択肢を取得
ans_ = np.unique(ans_)

dummy = []
for j in range(len(ans_)):
    dummy.append([ans_[j] in df['BEFORE ELEMENTARY'][i] for i in range(df.shape[0])])

dummy = pd.DataFrame(dummy, index=ans_).T

dummy["海外での生活"] = dummy["海外での生活"] + dummy["アメリカへ行っていた"]
dummy.drop("アメリカへ行っていた", axis=1, inplace=True)

dummy["塾"] = dummy["塾"] + dummy["公文(リスニング、リーディング)"]
dummy.drop("公文(リスニング、リーディング)", axis=1, inplace=True)

dummy = dummy.rename(columns={'幼稚園で英語に触れていた': 'B_幼稚園', '海外での生活': 'B_海外', 'なし': 'B_なし',
                              'アニメ/映画等': 'B_アニメ/映画等', '塾': 'B_塾', '絵本等': 'B_絵本等',
                              '英検/英検Jr.': 'B_英検/英検Jr.', '音楽': 'B_音楽', '英会話': 'B_英会話'})
df = pd.concat([df, dummy*1], axis=1)
df.drop("BEFORE ELEMENTARY", axis=1, inplace=True)
#%%
# AFTER ELEMENTARY
# 複数回答データを','で分割して、１つのリストにまとめる
ans = [df['AFTER ELEMENTARY'][i].split(';') for i in range(len(df["AFTER ELEMENTARY"]))]

# リストのflattening
ans_ = []
for s in ans:
    ans_.extend(s)

# 重複を除きユニークな回答選択肢を取得
ans_ = np.unique(ans_)
#%%
dummy = []
for j in range(len(ans_)):
    dummy.append([ans_[j] in df['AFTER ELEMENTARY'][i] for i in range(df.shape[0])])

dummy = pd.DataFrame(dummy, index=ans_).T
#%%
dummy["日常生活"] = dummy["日常生活"] + dummy["lived in US"]
dummy.drop("lived in US", axis=1, inplace=True)

dummy["アニメ/映画等"] = dummy["アニメ/映画等"] + dummy["Eminem"]
dummy.drop("Eminem", axis=1, inplace=True)
#%%
dummy = dummy.rename(columns={'日常生活': 'A_海外', 'なし': 'A_なし',
                              'アニメ/映画等': 'A_アニメ/映画等', '塾': 'A_塾', '学校': 'A_学校',
                              '英検/英検Jr.': 'A_英検/英検Jr.', '本':'A_本', '音楽': 'A_音楽','英会話': 'A_英会話'})
df = pd.concat([df, dummy*1], axis=1)
df.drop("AFTER ELEMENTARY", axis=1, inplace=True)
#%%
df.to_csv("../Database/cleansed_dataframe.csv")
#%%
