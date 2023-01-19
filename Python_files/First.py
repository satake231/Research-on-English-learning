#%%
import numpy as np
import pandas as pd

df = pd.read_csv("../Database/-4.csv", sep=",")

df = df.reset_index()
lists = df.iloc[0]
df.columns = lists
df = df.drop(0)
df = df.reset_index(drop=True)
df.to_excel("new_-4.xlsx")
#%%
name_lists = ["TIME_STAMP", "GENDER", "OCCUPATION", "READING", "WRITING", "LISTENING", "SPEAKING", "BEFORE ELEMENTARY",
              "AFTER ELEMENTARY", "CEFR", "EXPERIENCE EXCEPT SCHOOL", "ENGLISH OF FAMILY", "AVG_JUNIOR_HIGH",
              "AVG_HIGH", "USEFUL", "EXPOSURE"]

df.columns = name_lists
#%%
# GENDER, OCCUPATION -> clear
for i in range(len(df["GENDER"])):
    if df["GENDER"][i] == "female 女性":
        df["GENDER"][i] = "F"
    elif df["GENDER"][i] == "male 男性":
        df["GENDER"][i] = "M"
    else:
        df["GENDER"][i] = "X"

for i in range(len(df["OCCUPATION"])):
    if df["OCCUPATION"][i] == "university student 大学生":
        df["OCCUPATION"][i] = "university"
    elif df["OCCUPATION"][i] == "high school student 高校生":
        df["OCCUPATION"][i] = "high school"
    elif df["OCCUPATION"][i] == "professional student 専門学生":
        df["OCCUPATION"][i] = "professional student"
    else:
        df["OCCUPATION"][i] = "others"
#%%
# 4skills -> clear
for i in range(4):
    for j in range(len(df.index)):
        if df[df.columns[3 + i]][j] == "1(苦手）":
            df[df.columns[3 + i]][j] = 1
        elif df[df.columns[3 + i]][j] == "5(得意）":
            df[df.columns[3 + i]][j] = 5
#%%
df = df.astype({"READING": "int8", "WRITING": "int8", "LISTENING": "int8", "SPEAKING": "int8"})
#%%
# before after -> not clear
for i in range(2):
    for j in range(len(df.index)):
        if "ない" in df[df.columns[7 + i]][j] or \
                "ありません" in df[df.columns[7 + i]][j] or \
                "無し" in df[df.columns[7 + i]][j] or \
                "無い" in df[df.columns[7 + i]][j] or \
                "none" in df[df.columns[7 + i]][j] or \
                "なかった" in df[df.columns[7 + i]][j] or \
                "didn't" in df[df.columns[7 + i]][j] or \
                "なし" in df[df.columns[7 + i]][j]:
            df[df.columns[7 + i]][j] = "なし"
#%%
# CEFR -> clear
for i in range(len(df.index)):
    if df[df.columns[9]][i] == "a1":
        df[df.columns[9]][i] = "A1"
    elif df[df.columns[9]][i] == "a１":
        df[df.columns[9]][i] = "A1"
    elif df[df.columns[9]][i] == "A１":
        df[df.columns[9]][i] = "A1"
    elif df[df.columns[9]][i] == "a2":
        df[df.columns[9]][i] = "A2"
    elif df[df.columns[9]][i] == "a２":
        df[df.columns[9]][i] = "A2"
    elif df[df.columns[9]][i] == "A2":
        df[df.columns[9]][i] = "A2"
    elif df[df.columns[9]][i] == "b1":
        df[df.columns[9]][i] = "B1"
    elif df[df.columns[9]][i] == "b１":
        df[df.columns[9]][i] = "B1"
    elif df[df.columns[9]][i] == "B１":
        df[df.columns[9]][i] = "B1"
    elif df[df.columns[9]][i] == "b2":
        df[df.columns[9]][i] = "B2"
    elif df[df.columns[9]][i] == "b２":
        df[df.columns[9]][i] = "B2"
    elif df[df.columns[9]][i] == "B２":
        df[df.columns[9]][i] = "B2"
    elif df[df.columns[9]][i] == "c1":
        df[df.columns[9]][i] = "C1"
    elif df[df.columns[9]][i] == "c１":
        df[df.columns[9]][i] = "C1"
    elif df[df.columns[9]][i] == "C１":
        df[df.columns[9]][i] = "C1"
    elif df[df.columns[9]][i] == "c2":
        df[df.columns[9]][i] = "C2"
    elif df[df.columns[9]][i] == "c２":
        df[df.columns[9]][i] = "C2"
    elif df[df.columns[9]][i] == "C2":
        df[df.columns[9]][i] = "C2"
    elif df[df.columns[9]][i] == "C1（IELTS）":
        df[df.columns[9]][i] = "C1"
    elif "B1" in df[df.columns[9]][i]:
        df[df.columns[9]][i] = "B1"
    elif "英検準2級" in df[df.columns[9]][i]:
        df[df.columns[9]][i] = "A2"
    elif df[df.columns[9]][i] == "A1":
        df[df.columns[9]][i] = "A1"
    elif df[df.columns[9]][i] == "A2":
        df[df.columns[9]][i] = "A2"
    elif df[df.columns[9]][i] == "B1":
        df[df.columns[9]][i] = "B1"
    elif df[df.columns[9]][i] == "B2":
        df[df.columns[9]][i] = "B2"
    elif df[df.columns[9]][i] == "C1":
        df[df.columns[9]][i] = "C1"
    elif df[df.columns[9]][i] == "C2":
        df[df.columns[9]][i] = "C2"
    else:
        df[df.columns[9]][i] = np.nan
#%%
df.to_excel("../Database/pre_dataframe.xlsx")
#%%
for i in range(len(df.index)):
    if df[df.columns[10]][i] == "毎日":
        df[df.columns[10]][i] = "5"
    elif df[df.columns[10]][i] == "週に4~5回":
        df[df.columns[10]][i] = "4"
    elif df[df.columns[10]][i] == "週に2~3回":
        df[df.columns[10]][i] = "3"
    elif df[df.columns[10]][i] == "月に3~4回":
        df[df.columns[10]][i] = "2"
    elif df[df.columns[10]][i] == "ほとんどない":
        df[df.columns[10]][i] = "1"
# intにtype変更したい
#%%
for i in range(len(df.index)):
    if df[df.columns[11]][i] == "海外でも不自由なく生活できる程度":
        df[df.columns[11]][i] = "5"
    elif df[df.columns[11]][i] == "日常会話が可能":
        df[df.columns[11]][i] = "4"
    elif df[df.columns[11]][i] == "簡単な会話なら可能":
        df[df.columns[11]][i] = "3"
    elif df[df.columns[11]][i] == "ほとんど話せないが単語のみなら可能":
        df[df.columns[11]][i] = "2"
    elif df[df.columns[11]][i] == "全く話せない":
        df[df.columns[11]][i] = "1"
# int にtype変更したい
#%%
for i in range(len(df.index)):
    if df[df.columns[14]][i] == "とても役に立つと思う":
        df[df.columns[14]][i] = "4"
    elif df[df.columns[14]][i] == "どちらかというと役に立つと思う":
        df[df.columns[14]][i] = "3"
    elif df[df.columns[14]][i] == "どちらかというと役に立たないと思う":
        df[df.columns[14]][i] = "2"
    elif df[df.columns[14]][i] == "全く役に立たないと思う":
        df[df.columns[14]][i] = "1"
# int にtype変更したい
#%%
for i in range(len(df.index)):
    if df[df.columns[15]][i] == "とても自主性を持ってきた":
        df[df.columns[15]][i] = "4"
    elif df[df.columns[15]][i] == "どちらかというと自主性を持ってきた":
        df[df.columns[15]][i] = "3"
    elif df[df.columns[15]][i] == "どちらかというと自主性を持ってこなかった":
        df[df.columns[15]][i] = "2"
    elif df[df.columns[15]][i] == "全く自主性を持ってこなかった":
        df[df.columns[15]][i] = "1"
#%%
df[df.columns[12]] = df[df.columns[12]].replace('(.*)時間(.*)', r'\1\2', regex=True)
#%%
df[df.columns[13]] = df[df.columns[13]].replace('(.*)時間(.*)', r'\1\2', regex=True)
#%%
df.to_excel("dataframe.xlsx")
#%%
