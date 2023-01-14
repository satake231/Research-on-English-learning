#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import gc

# scikit-learn
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


# LightGBM
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")
#%%
df = pd.read_csv("../Database/cleansed_dataframe.csv")
df = df.drop(df.columns[0], axis=1)
# TIME_STAMPをID化
for i in range(len(df["TIME_STAMP"])):
    df["TIME_STAMP"][i] = i
df["TIME_STAMP"] = df['TIME_STAMP'].astype('int8')
# GENDERを0, 1化
for i in range(len(df["GENDER"])):
    if df["GENDER"][i] == "M":
        df["GENDER"][i] = 0
    elif df["GENDER"][i] == "F":
        df["GENDER"][i] = 1
df['GENDER'] = df['GENDER'].astype('int8')
# 所属をone-hot-encoding
occupation_one = pd.get_dummies(df["OCCUPATION"])
df = pd.concat([df, occupation_one], axis=1)

df.drop('OCCUPATION', axis=1, inplace=True)
df['high school'] = df["high school"].astype("int8")
df['others'] = df['others'].astype('int8')
df['professional student'] = df['professional student'].astype('int8')
df['university'] = df['university'].astype('int8')
#%%
# CEFR_columns を数値データに変更
# A1 -> 1
# A2 -> 2
# .......
# C2 -> 6
for i in range(len(df["CEFR"])):
    if df["CEFR"][i] == "A1":
        df["CEFR"][i] = 1
    elif df["CEFR"][i] == "A2":
        df["CEFR"][i] = 2
    elif df["CEFR"][i] == "B1":
        df["CEFR"][i] = 3
    elif df["CEFR"][i] == "B2":
        df["CEFR"][i] = 4
    elif df["CEFR"][i] == "C1":
        df["CEFR"][i] = 5
    elif df["CEFR"][i] == "C2":
        df["CEFR"][i] = 6
    else:
        df["CEFR"][i] = 0
df['CEFR'] = df["CEFR"].astype("int64")
df["CEFR"].replace(0, np.nan, inplace=True)
#%%
# show correlation
sns.heatmap(df.iloc[:,1:14].corr(),annot=True,cmap='bwr',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(12,10)
plt.savefig('../Statistic_results/corr_heatmap.png', format='png')
plt.show()
# READINGとWRITING, LISTENINGとSPEAKING に弱い相関
# READINGとSPEAKING は相関的に離れているが、他二つはどの分野とも結びつきの差が激しくない
# EXPERIENCE EXCEPT SCHOOL はどの指標とも超弱い相関(正確には無相関)がある
# EXPOSURE は弱いが相関があると見てよさそう
# AVG_JUNIOR_HIGHとAVG_HIGHには弱い相関
#%%
# CEFR欠損値補完
#%%
# index降り直しのコードを書く
train = df[df["CEFR"].notna()].reset_index(drop=True)
test = df[df["CEFR"].isnull()].reset_index(drop=True)
#%%
# メモリ削減の関数作成
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns: # 列ごとに最大値と最小値を参照して最もデータ量の少ない型に変換する
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            pass

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
#%%
# メモリ削減関数の実行
train = reduce_mem_usage(train)
#%%
# 訓練データとテストエータを用途毎に分割して保存
X = train.drop(columns=["TIME_STAMP", "CEFR"]).values
y = train["CEFR"].values
X_test = test.drop(columns=['TIME_STAMP', 'CEFR']).values
#%%
# ランダムフォレスト
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)
rfc = RandomForestClassifier(max_depth=10, min_samples_leaf=1, n_estimators=100, n_jobs=-1, random_state=42)
rfc.fit(X_train, y_train)
#%%
print('Train Score: {}'.format(round(rfc.score(X_train, y_train), 3)))
print(' Test Score: {}'.format(round(rfc.score(X_valid, y_valid), 3)))
#%%
param_grid = {'max_depth': [3, 5, 7, 9, 11, 13],
              'min_samples_leaf': [1, 2, 3, 4, 5, 6]}

for max_depth in param_grid['max_depth']:
    for min_samples_leaf in param_grid['min_samples_leaf']:
        rfc_grid = RandomForestClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                          n_estimators=100, n_jobs=-1, random_state=42)
        rfc_grid.fit(X_train, y_train)
        print('max_depth: {}, min_samples_leaf: {}'.format(max_depth, min_samples_leaf))
        print('    Train Score: {}, Test Score: {}'.format(round(rfc_grid.score(X_train, y_train), 3),
                                                           round(rfc_grid.score(X_valid, y_valid), 3)))
#%%
rfc_gs = GridSearchCV(RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42), param_grid, cv=2)
rfc_gs.fit(X, y)

print('Best Parameters: {}'.format(rfc_gs.best_params_))
print('CV Score: {}'.format(round(rfc_gs.best_score_, 3)))
# CVscore 0.484
# -> 補完はあきらめることにした。CEFRなしと欠損なしCEFRのみの二種類でコード建築の方向で調整
#%%
# それぞれのバリデーションでどのindexを使うかを記載したリストを作成
cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=123).split(x_train, y_train))

# indexの確認：fold=0のtrainデータ
print("index(train):", cv[0][0])

# indexの確認：fold=0のvalidデータ
print("index(valid):", cv[0][1])
#%%
# 0fold目のindexのリスト取得
nfold = 0
idx_tr, idx_va = cv[nfold][0], cv[nfold][1]

# 0fold目のindexを基に、学習データと検証データに分離
x_tr, y_tr, id_tr = x_train.loc[idx_tr, :], y_train[idx_tr], id_train.loc[idx_tr, :]
x_va, y_va, id_va = x_train.loc[idx_va, :], y_train[idx_va], id_train.loc[idx_va, :]
print(x_tr.shape, y_tr.shape, id_tr.shape)
print(x_va.shape, y_va.shape, id_va.shape)
#%%
# モデル学習
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc_mu',
    'learning_rate': 0.05,
    'num_leaves': 32,
    'n_estimators': 100000,
    "random_state": 123,
    "importance_type": "gain",
}

# モデルの学習
model = lgb.LGBMClassifier(**params)
model.fit(x_tr,
          y_tr,
          eval_set=[(x_tr, y_tr), (x_va, y_va)],
          early_stopping_rounds=100,
          verbose=100
          )

# モデルの保存
with open("model_lgb_fold0.pickle", "wb") as f:
    pickle.dump(model, f, protocol=4)
#%%
# 学習データの推論値取得とROC計算
y_tr_pred = model.predict_proba(x_tr)[:,1]
metric_tr = roc_auc_score(y_tr, y_tr_pred)

# 検証データの推論値取得とROC計算
y_va_pred = model.predict_proba(x_va)[:,1]
metric_va = roc_auc_score(y_va, y_va_pred)

# 評価値を入れる変数の作成（最初のfoldのときのみ）
metrics = []

# 評価値を格納
metrics.append([nfold, metric_tr, metric_va])

# 結果の表示
print("[auc] tr:{:.4f}, va:{:.4f}".format(metric_tr, metric_va))
#%%
# oofの予測値を入れる変数の作成
train_oof = np.zeros(len(x_train))

# validデータのindexに予測値を格納
train_oof[idx_va] = y_va_pred
#%%
# 重要度の取得
imp_fold = pd.DataFrame({"col":x_train.columns, "imp":model.feature_importances_, "nfold":nfold})
# 確認（重要度の上位10個）
display(imp_fold.sort_values("imp", ascending=False)[:10])

# 重要度を格納する5fold用データフレームの作成
imp = pd.DataFrame()
# imp_foldを5fold用データフレームに結合
imp = pd.concat([imp, imp_fold])
#%%
# リスト型をarray型に変換
metrics = np.array(metrics)
print(metrics)

# 学習/検証データの評価値の平均値と標準偏差を算出
print("[cv] tr:{:.4f}+-{:.4f}, va:{:.4f}+-{:.4f}".format(
    metrics[:,1].mean(), metrics[:,1].std(),
    metrics[:,2].mean(), metrics[:,2].std(),
))

# oofの評価値を算出
print("[oof] {:.4f}".format(
    roc_auc_score(y_train, train_oof)
))
#%%
train_oof = pd.concat([
    id_train,
    pd.DataFrame({"true": y_train, "pred": train_oof}),
], axis=1)
train_oof.head()
#%%
imp = imp.groupby("col")["imp"].agg(["mean", "std"]).reset_index(drop=False)
imp.columns = ["col", "imp", "imp_std"]
imp.head()
#%%
def train_lgb(input_x,
              input_y,
              input_id,
              params,
              list_nfold=[0,1,2,3,4],
              n_splits=5,
              ):
    train_oof = np.zeros(len(input_x)) # 入力と同じ長さのゼロ配列を作成
    metrics = [] # 空の配列を作成
    imp = pd.DataFrame() # 空のデータフレームを作成

    # cross-validation
    # KFoldでCVするためのリストを作成
    cv = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123).split(input_x, input_y))
    for nfold in list_nfold:
        print("-"*20, nfold, "-"*20) # 枠線の上側作成

        # make dataset
        idx_tr, idx_va = cv[nfold][0], cv[nfold][1]
        x_tr, y_tr, id_tr = input_x.loc[idx_tr, :], input_y[idx_tr], input_id.loc[idx_tr, :]
        x_va, y_va, id_va = input_x.loc[idx_va, :], input_y[idx_va], input_id.loc[idx_va, :]
        print(x_tr.shape, x_va.shape)

        # train
        model = lgb.LGBMClassifier(**params)
        model.fit(x_tr,
                  y_tr,
                  eval_set=[(x_tr, y_tr), (x_va, y_va)],
                  early_stopping_rounds=100,
                  verbose=100
                  )
        fname_lgb = "model_lgb_fold{}.pickle".format(nfold)
        with open(fname_lgb, "wb") as f:
            pickle.dump(model, f, protocol=4)

        # evaluate
        y_tr_pred = model.predict_proba(x_tr)[:,1]
        y_va_pred = model.predict_proba(x_va)[:,1]
        metric_tr = roc_auc_score(y_tr, y_tr_pred)
        metric_va = roc_auc_score(y_va, y_va_pred)
        metrics.append([nfold, metric_tr, metric_va])
        print("[auc] tr:{:.4f}, va:{:.4f}".format(metric_tr, metric_va))

        # oof
        train_oof[idx_va] = y_va_pred

        # imp
        _imp = pd.DataFrame({"col":input_x.columns, "imp":model.feature_importances_, "nfold":nfold})
        imp = pd.concat([imp, _imp])

    print("-"*20, "result", "-"*20)
    # metric
    metrics = np.array(metrics)
    print(metrics)
    print("[cv] tr:{:.4f}+-{:.4f}, va:{:.4f}+-{:.4f}".format(
        metrics[:,1].mean(), metrics[:,1].std(),
        metrics[:,2].mean(), metrics[:,2].std(),
    ))
    print("[oof] {:.4f}".format(
        roc_auc_score(input_y, train_oof)
    ))

    # oof
    train_oof = pd.concat([
        input_id,
        pd.DataFrame({"pred":train_oof})
    ], axis=1)

    # importance
    imp = imp.groupby("col")["imp"].agg(["mean", "std"]).reset_index(drop=False)
    imp.columns = ["col", "imp", "imp_std"]

    return train_oof, imp, metrics
#%%
# ハイパーパラメータの設定
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 32,
    'n_estimators': 100000,
    "random_state": 123,
    "importance_type": "gain",
}

# 学習の実行
train_oof, imp, metrics = train_lgb(x_train,
                                    y_train,
                                    id_train,
                                    params,
                                    list_nfold=[0,1,2,3,4],
                                    n_splits=5,
                                    )
#%%
imp.sort_values("imp", ascending=False)[:10]
#%%
# ファイルの読み込み
test = reduce_mem_usage(test)

# データセットの作成
x_test = test.drop(columns=["SK_ID_CURR" ])
id_test = test[["SK_ID_CURR"]]

# カテゴリ変数をcategory型に変換
for col in x_test.columns:
    if x_test[col].dtype=="O":
        x_test[col] = x_test[col].astype("category")
#%%
def predict_lgb(input_x,
                input_id,
                list_nfold,
                ):
    pred = np.zeros((len(input_x), len(list_nfold))) #予測値を格納するための変数を作成
    for nfold in list_nfold:
        print("-"*20, nfold, "-"*20)
        fname_lgb = "model_lgb_fold{}.pickle".format(nfold)
        with open(fname_lgb, "rb") as f: #nfold目の、pickle保存していたモデルを呼び出し
            model = pickle.load(f)
        pred[:, nfold] = model.predict_proba(input_x)[:,1] #モデルの予測値を代入

    pred = pd.concat([
        input_id,
        pd.DataFrame({"pred": pred.mean(axis=1)}), #予測値の平均を算出し、整形
    ], axis=1)

    print("Done.")

    return pred
#%%
test_pred = predict_lgb(x_test,
                        id_test,
                        list_nfold=[0,1,2,3,4],
                        )