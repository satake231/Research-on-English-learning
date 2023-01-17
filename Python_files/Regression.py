#%%
import pandas as pd
import statsmodels.api as sa

import warnings
warnings.filterwarnings("ignore")
#%%
df = pd.read_csv("../Database/cleansed_dataframe.csv")
df = df.drop([df.columns[0], "TIME_STAMP"], axis=1)

# GENDERのダミー化
df.replace({'GENDER': {'M': 0, 'F': 1}}, inplace=True)

# OCCUPATIONのone-hot-encoding
occupation_ohe = pd.get_dummies(df['OCCUPATION'])
df = pd.concat([df, occupation_ohe], axis=1)
df.drop(['OCCUPATION', 'others'], axis=1, inplace=True)

df.drop(['A_なし', 'B_なし'], axis=1, inplace=True)
#%%
df.columns = ['GENDER', 'READING', 'WRITING', 'LISTENING', 'SPEAKING', 'CEFR', 'EXPERIENCE EXCEPT SCHOOL',
              'ENGLISH OF FAMILY', 'AVG_JUNIOR_HIGH', 'AVG_HIGH', 'USEFUL', 'EXPOSURE', 'B_アニメ_映画等',
              'B_塾', 'B_幼稚園', 'B_海外', 'B_絵本等', 'B_英会話', 'B_英検_英検Jr', 'B_音楽', 'A_アニメ_映画等',
              'A_塾', 'A_学校', 'A_海外', 'A_音楽', 'A_英会話', 'A_英検_英検Jr', 'high school', 'professional student',
              'university']
#%%
# 重回帰用の関数作成
def multiple_linear_regression(dataframe, dependent_name):
    X = dataframe.drop(dependent_name, axis=1)
    y = dataframe[dependent_name]
    model = sa.OLS(y, X)
    result = model.fit()
    f = open('../Statistic_results/{}.txt'.format(dependent_name), 'w')
    f.write(str(result.summary()))
    f.close()
#%%
for i in range(len(df.columns)):
    if i != 5:
        tmp = df.drop('CEFR', axis=1)
        multiple_linear_regression(tmp, df.columns[i])
    else:
        tmp = df[df["CEFR"].notna()]
        for i in range(len(tmp.CEFR)):
            if tmp.CEFR.iloc[i] == 'A1':
                tmp.CEFR.iloc[i] = 1
            elif tmp.CEFR.iloc[i] == 'A2':
                tmp.CEFR.iloc[i] = 2
            elif tmp.CEFR.iloc[i] == 'B1':
                tmp.CEFR.iloc[i] = 3
            elif tmp.CEFR.iloc[i] == 'B2':
                tmp.CEFR.iloc[i] = 4
            elif tmp.CEFR.iloc[i] == 'C1':
                tmp.CEFR.iloc[i] = 5
            elif tmp.CEFR.iloc[i] == 'C2':
                tmp.CEFR.iloc[i] = 6
        tmp.CEFR = tmp.CEFR.astype('int8')
        i = 5
        multiple_linear_regression(tmp, df.columns[i])
#%%
