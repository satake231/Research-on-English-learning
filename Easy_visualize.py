#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%%

df = pd.read_csv("-4.csv", sep=",")

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
                "didin't" in df[df.columns[7 + i]][j] or \
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
def make_barplot(df, name):
    labels = df[name].value_counts().index.sort_values()
    metrics = df[name].value_counts().sort_index()
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()

    rect = ax.bar(x, metrics, width, color=["red", "blue", 'yellow', 'green', 'purple', 'orange', 'black', 'pink'])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rect)
    plt.title(name)
    plt.savefig("Images/graph_"+name+"_.png", format="png")
    plt.show()
#%%
make_barplot(df, df.columns[1])
#%%
make_barplot(df, df.columns[2])
#%%
make_barplot(df, df.columns[3])
#%%
make_barplot(df, df.columns[4])
#%%
make_barplot(df, df.columns[5])
#%%
make_barplot(df, df.columns[6])
#%%
make_barplot(df, df.columns[9])
#%%
make_barplot(df, df.columns[10])
#%%
make_barplot(df, df.columns[11])
#%%
make_barplot(df, df.columns[14])
#%%
make_barplot(df, df.columns[15])
#%%
df[df.columns[12]] = df[df.columns[12]].replace('(.*)時間(.*)', r'\1\2', regex=True)
#%%
df[df.columns[13]] = df[df.columns[13]].replace('(.*)時間(.*)', r'\1\2', regex=True)
#%%
df = df.astype({"EXPERIENCE EXCEPT SCHOOL": "int8", "ENGLISH OF FAMILY": "int8", "USEFUL": "int8", "EXPOSURE": "int8"})
#%%
df.to_excel("dataframe.xlsx")
#%%
# 5~6 などの表現の場合は中央値を、「覚えていない」などは欠損値とし、手動で変更を行った
# CFER以外の列においては、欠損値の発生した行が一行しか無かったために、そのデータは扱わないものとした。
df = pd.read_excel("dataframe.xlsx")
#%%
df = df.astype({"AVG_JUNIOR_HIGH": "float32", "AVG_HIGH": "float32"})
#%%
df.to_excel("new_dataframe.xlsx")

#%%
plt.hist(df["AVG_HIGH"], bins=100)
plt.savefig("Images/graph_"+"AVG_HIGH"+"_.png", format="png")
plt.show()
#%%
plt.hist(df["AVG_JUNIOR_HIGH"], bins=70)
plt.savefig("Images/graph_"+"AVG_JUNIOR_HIGH"+"_.png", format="png")
plt.show()
#%%
import numpy as np
import pandas as pd
import scipy.stats as st
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.transforms as ts

target = 'AVG_JUNIOR_HIGH'           # データフレームのなかでプロット対象とする列
x_min, x_max = 0, 20  # プロットする点数範囲（下限と上限）
j = 5                   # Y軸（度数）刻み幅
k = 2                  # 区間の幅
bins = 100               # 区間の数　(x_max-x_min)/k  (100-40)/5->12

# ここからグラフ描画処理
plt.figure(dpi=96)
plt.xlim(x_min,x_max)
d = 0.001

# (1) 統計処理
n   = len(df[target])         # 標本の大きさ
mu  = df[target].mean()       # 平均
sig = df[target].std(ddof=0)  # 標準偏差：ddof(自由度)=0
print(f'■ 平均：{mu:.1f}点、標準偏差：{sig:.1f}点')
ci1, ci2 = (None, None)

# 正規性の検定（有意水準5%）と母平均の95%信頼区間
_, p = st.shapiro(df[target])
if p >= 0.05 :
    print(f'  - p={p:.2f} ( p>=0.05 ) であり母集団には正規性があると言える')
    U2 = df[target].var(ddof=1)  # 母集団の分散推定値（不偏分散）
    DF = n-1                     # 自由度
    SE = math.sqrt(U2/n)         # 標準誤差
    ci1,ci2 = st.t.interval( alpha=0.95, loc=mu, scale=SE, df=DF )
    print(f'  - 母平均の95%信頼区間CI = [{ci1:.2f} , {ci2:.2f}]')
else:
    print(f'  ※ p={p:.2f} ( p<0.05 ) であり母集団には正規性があるとは言えない')

# (2) ヒストグラムの描画
hist_data = plt.hist(df[target], bins=bins, color='tab:cyan', range=(x_min, x_max), rwidth=0.9)
plt.gca().set_xticks(np.arange(x_min,x_max-k+d, k))

# (3) 正規分布を仮定した近似曲線
sig = df[target].std(ddof=1)  # 不偏標準偏差：ddof(自由度)=1
nx = np.linspace(x_min, x_max+d, 150) # 150分割
ny = st.norm.pdf(nx,mu,sig) * k * len(df[target])
plt.plot( nx , ny, color='tab:blue', linewidth=1.5, linestyle='--')

# (4) X軸 目盛・ラベル設定
plt.xlabel('学習時間',fontsize=12)
plt.gca().set_xticks(np.arange(x_min,x_max+d, k))

# (5) Y軸 目盛・ラベル設定
y_max = max(hist_data[0].max(), st.norm.pdf(mu,mu,sig) * k * len(df[target]))
y_max = int(((y_max//j)+1)*j) # 最大度数よりも大きい j の最小倍数
plt.ylim(0,y_max)
plt.gca().set_yticks( range(0,y_max+1,j) )
plt.ylabel('人数',fontsize=12)

# (6) 平均点と標準偏差のテキスト出力
tx = 0.03 # 文字出力位置調整用
ty = 0.91 # 文字出力位置調整用
tt = 0.08 # 文字出力位置調整用
tp = dict( horizontalalignment='left',verticalalignment='bottom',
           transform=plt.gca().transAxes, fontsize=11 )
plt.text( tx, ty, f'平均時間 {mu:.2f}', **tp)
plt.text( tx, ty-tt, f'標準偏差 {sig:.2f}', **tp)
plt.vlines( mu, 0, y_max, color='black', linewidth=1 )
#%%
#%%
def make_violinplot(df, name, hue):
    import copy as cp
    df_exam = cp.deepcopy(df)
    df_exam["Exam"] = 0
    sns.violinplot(data=df_exam, split=True, x="Exam", y=name, hue=hue)
    sns.stripplot(df_exam[name], color='red', size=0.2)
    plt.savefig("graph_"+name+"_.png", format="png")
    plt.show()