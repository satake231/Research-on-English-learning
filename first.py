#%%
import pandas as pd

df = pd.read_csv("-2.csv", sep=",")

df = df.reset_index()
lists = df.iloc[0]
df.columns = lists
df = df.drop(0)
df = df.reset_index(drop=True)
#%%
name_lists = ["TIME_STAMP", "GENDER", "OCCUPATION", "READING", "WRITING", "LISTENING", "SPEAKING", "BEFORE ELEMENTARY",
              "AFTER ELEMENTARY", "CEFR", "EXPERIENCE EXCEPT SCHOOL", "ENGLISH OF FAMILY", "AVG_JUNIOR_HIGH",
              "AVG_HIGH", "USEFUL", "EXPOSURE"]

df.columns = name_lists
#%%
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
for i in range(4):
    for j in range(len(df.index)):
        if df[df.columns[3 + i]][j] == "1(苦手）":
            df[df.columns[3 + i]][j] = 1
        elif df[df.columns[3 + i]][j] == "5(得意）":
            df[df.columns[3 + i]][j] = 5
#%%
