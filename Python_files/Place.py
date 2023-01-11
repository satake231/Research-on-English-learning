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