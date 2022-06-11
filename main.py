import streamlit as st

st.title("keiba AI")

## import & def func
import re
# import os                                   # ファイルディレクトリ操作
# import csv
import time
# import json
import datetime
# import random
import requests

# from IPython.core.display import display
from bs4 import BeautifulSoup
from tqdm import tqdm

from itertools import *
from collections import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

# import japanize_matplotlib

# 機械学習
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, QuantileTransformer, StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error
from sklearn import preprocessing
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import  KNeighborsClassifier, KNeighborsRegressor
# from sklearn import svm
import lightgbm as lgbm
# import xgboost as xgb


import warnings
warnings.filterwarnings('ignore')

import pickle
import dill as pickle

from scipy.misc import derivative

# import os 
# print(os.getcwd())
# print(os.listdir())

class Name:
    def __init__(self):
        self.name_dict = {}
        self.notation = {}
        self.race_name = {}
    def __getitem__(self, id):
        return self.name_dict[id]
    def __setitem__(self, id, name):
        path_list = re.findall("\w+", id)
        if len(path_list)==2 or (len(path_list)==3 and path_list[1]=="ped"):
            pass
        else:
            # print(id)
            pass

        if id in self.name_dict:
            self.notation[id].add(name)
        else:
            self.name_dict[id] = name 
            self.notation[id] = set([name])
    def get_multi_notations(self):
        return {id: value for id, value in nm.notation.items() if len(value)>=2}

def preprocess_result(_df):
    # いらない columns 消去
    _df = _df.drop(columns=["ﾀｲﾑ指数", "調教ﾀｲﾑ", "厩舎ｺﾒﾝﾄ", "備考"])
    _df = _df.drop(columns=["着差"])

    _df = _df[~_df["着順"].isin(["中","除","取","失"])]
    _df["着順"] = _df["着順"].astype(str).map(lambda x: x[:x.index("(")] if "(" in x else x).astype(float)
    assert set(_df["着順"].fillna(1)) <= set(range(1,19))

    _df["枠番"] = _df["枠番"].astype(float)
    assert set(_df["枠番"].fillna(1)) <= set(range(1,9))

    _df["馬番"] = _df["馬番"].astype(float)
    assert set(_df["馬番"].fillna(1)) <= set(range(1,19))

    _df["性"] = _df["性齢"].str[0].astype("category")
    assert set(_df["性"]) <= set(["牡", "牝", "セ"])
    _df["齢"] = _df["性齢"].str[1:].astype(float)
    assert set(_df["齢"]) <= set(range(2,20))
    _df = _df.drop(columns=["性齢"])

    _df["斤量"] = _df["斤量"].astype(float)
    assert all(45<=x<=65 for x in _df["斤量"])

    def f(x):
        if str(x) == "nan":
            return np.nan
        return int(x[0])*60 + float(x[2:])
    _df["タイム"] = _df["タイム"].map(f)
    assert all(50<=x<=400 for x in _df["タイム"].fillna(100))

    _df["単勝"] = _df["単勝"].astype(float)
    assert all(1<=x<=10000 for x in _df["単勝"].fillna(1))

    _df["人気"] = _df["人気"].astype(float)
    assert set(_df["人気"].fillna(1)) <= set(range(1,19))

    def f(x):
        if x in {"", "nan"}:
            return "nan(nan)"
        x = x.replace("前計不", "nan").replace("計不", "nan")
        return x
    _df["馬体重"] = _df["馬体重"].astype(str).map(f)

    _df["体重"] = _df["馬体重"].map(lambda x: x[:3]).astype(float)
    assert all(300<=x<=700 for x in _df["体重"].fillna(400))
    _df["増減"] = _df["馬体重"].map(lambda x: x[4:-1]).astype(float)
    assert all(-100<=x<=100 for x in _df["増減"].fillna(0))
    _df = _df.drop(columns=["馬体重"])

    _df["東西"] = _df["調教師"].map(lambda x: x[1:2]).astype("category")
    assert set(_df["東西"]) <= set("東西地外")

    _df["調教師"] = _df["調教師"].map(lambda x: x[3:])

    _df["賞金(万円)"] = _df["賞金(万円)"].fillna("0").map(lambda x: x.replace(",","")).astype(float)
    _df = _df.rename(columns={'賞金(万円)': '賞金'})
    assert all(0<=x<=40000 for x in set(_df["賞金"]))

    _df["通過"] = _df["通過"].fillna("nan").map(lambda x: np.array(x.split("-"), dtype=float) )
    # assert all( type(x) is type(np.array([])) for x in set(_df["通過"]) )


    # ID
    for col_name, col_id in (("馬名", "horse_id"), ("騎手", "jockey_id"), ("調教師", "trainer_id"), ("馬主", "owner_id")):
        _df[col_id] = _df[col_id].astype(str).map(lambda x: x.replace("/result/recent", "")).replace({"nan": np.nan})
        for name, id in _df[[col_name, col_id]].values:
            if id == id:
                nm[id] = name
        _df = _df.drop(columns=[col_name])
        _df[col_id] = _df[col_id].astype("category")



    # for col, num_null in _df.isnull().sum().items():
    #     if num_null:
    #         print(col, num_null)
    #         display(_df[_df[col].isnull()])



    return _df



def preprocess_race(_df):
    _df = _df.drop(columns=["過去レースurl", "レース総称"])

    _df["芝ダ"] = _df["コース"].map(lambda x: x[0]).astype("category")
    _df = _df[~_df["芝ダ"].isin(["障"])]
    assert set(_df["芝ダ"]) <= set(["ダ", "芝"])

    _df["回り"] = _df["コース"].map(lambda x: x[1:-5]).astype("category")
    assert set(_df["回り"]) <= {'右', '右内2周', '右外', '右外-内', '左', '左外', '直線'}

    _df["距離"] = _df["コース"].map(lambda x: x[-5:-1]).astype(float)
    assert all(1000<=x<=4000 for x in set(_df["距離"]))

    _df = _df.drop(columns=["コース"])

    _df["天気"] = _df["天気"].astype("category")
    assert all(x in {'小雨', '小雪', '晴', '曇', '雨', '雪'} or np.isnan(x) for x in _df["天気"])

    _df["馬場"] = _df["馬場"].astype("category")
    assert all(x in {'不良', '稍重', '良', '重'} or np.isnan(x) for x in _df["馬場"])

    _df["時間"] = _df["時間"].map(lambda x: x[0:2]).astype(float) *60 + _df["時間"].map(lambda x: x[3:5]).astype(float)
    assert all(0<=x<=24*60 for x in set(_df["時間"].value_counts().index))

    _df["日付"] = _df["日付"].fillna("2002年1月2日").map(lambda x: datetime.datetime.strptime(x, "%Y年%m月%d日"))

    _df["回"] = _df["開催"].map(lambda x: x[0]).astype(float)
    assert set(_df["回"]) <= set(range(1,7))
    _df["開催地"] = _df["開催"].map(lambda x: x[2:4]).astype("category")
    assert set(_df["開催地"]) <= {'中京', '中山', '京都', '函館', '小倉', '新潟', '札幌', '東京', '福島', '阪神'}
    _df["日目"] = _df["開催"].map(lambda x: int(x[4:-2] ))
    assert set(_df["日目"]) <= set(range(1,13))
    _df = _df.drop(columns=["開催"])

    def f(x):
        if x[2] == "以":
            return f"{x[0]}+"
            i = 4
        return x[0]
    _df["年齢制限"] = _df["条件"].map(f).astype("category")

    def f(x):
        if "(" in x:
            cl=x[x.index("(")+1:-1]
            if cl not in "LG":
                cl=cl[-2:]
            if cl=="G":
                cl="G3"
            return cl
        return ""
    _df["クラス"] = _df["レース名"].map(f)
    def f(x):
        if "以上" in x:
            return x[4:6]
        return x[2:4]
    _df["クラス"] = np.where(
        _df["クラス"]=="", 
        _df["条件"].map(f).replace({"オー":"OP", "10": "2勝", "16": "3勝", "50": "1勝"}), 
        _df["クラス"]
        )
    _df["クラス"] = _df["クラス"].astype("category")
    _df = _df.drop(columns=["条件"])


    for col_name, col_id in (("レース名", "race_id"), ):
        _df[col_id] = _df[col_id].map(lambda x: x.replace("/result/recent", ""))
        for name, id in _df[[col_name, col_id]].values:
            nm[id] = name
        _df[col_id] = _df[col_id].astype("category")
        _df = _df.drop(columns=[col_name])


    _df["タイプ"] = _df["タイプ"].astype(str).map(lambda x: x.replace("・","").replace("牡牝",""))
    for col in ['混', '特指', '定量', '国際', '別定', 'ハンデ', '指', '馬齢', '牝', '見習騎手', '九州産馬']:
        _df[col] = np.where( _df["タイプ"].map(lambda x: col in x), 1, 0)
    _df = _df.drop(columns=["タイプ"])



    _df["ラップ"] = _df["ラップ"].astype(str).map(lambda x: x.split(" - "))
    _df["race_テン"] = _df["ペース"].fillna("(nan-nan)").map(lambda x: x[x.index("(")+1:x.index(")")].split("-")[0]).astype(float)
    _df["race_上り"] = _df["ペース"].fillna("(nan - nan)").map(lambda x: x[x.index("(")+1:x.index(")")].split("-")[1]).astype(float)
    # _df["ペース"] = _df["ペース"].map(lambda x: x[:x.index("(")].split(" - "))
    _df = _df.drop(columns=[f"{i}コーナー" for i in range(1,5)]+["ラップ"]+["ペース"])

    return _df

def preprocess_pay(_df):
    return _df

def preprocess_prof(_df):

    _df = _df.drop(columns=["募集情報"], errors="ignore")

    _df["生年月日"] = _df["生年月日"].map(lambda x: datetime.datetime.strptime(x, "%Y年%m月%d日"))
    _df["産地"] = _df["産地"].astype("category")
    _df["調教師"]  = _df["調教師"].fillna("()")
    _df["調教場所"] = _df["調教師"].map(lambda x: x[x.index("(")+1:x.index(")")]).astype("category")
    _df["調教師"] = _df["調教師"].map(lambda x: x[:x.index("(")]).astype("category")

    _df["セリ取引価格"] = _df["セリ取引価格"].replace({"-": "0万円(nan年nan)", np.nan: "nan万円(nan年nan)"})
    def f(x):
        return x[x.index("年")+1: -1]
    _df["セリ取引場所"] = _df["セリ取引価格"].map(f).astype("category")

    def f(x):
        return x[x.index("(")+1: x.index("年")]
    _df["セリ取引年"] = _df["セリ取引価格"].map(f).astype(float)

    def f(x):
        return x[:x.index("万円")].replace("億", "").replace(",", "")
    _df["セリ取引価格"] = _df["セリ取引価格"].map(f).astype(float)


    def f(x):
        try:
            return x.replace("/result/recent", "")
        except:
            return np.nan

    for col_name, col_id in (("調教師", "trainer_id"), ("馬主", "owner_id"), ("生産者", "breeder_id"), ("母母_name", "母母_ped_id"), ("母父_name", "母父_ped_id"), ("母_name", "母_ped_id"), ("父母_name", "父母_ped_id"), ("父父_name", "父父_ped_id"), ("父_name", "父_ped_id")):
        _df[col_id] = _df[col_id].map(f)
        for name, id in _df[[col_name, col_id]].values:
            if id==id:
                nm[id] = name
        _df[col_id] = _df[col_id].astype("category")
        _df = _df.drop(columns=[col_name])
    
    _df["horse_id"] = _df["horse_id"].astype("category")
    _df = _df.drop(columns=["主な勝鞍", "近親馬", "通算成績", "獲得賞金", "horse_name"])


    return _df

def preprocess_form(_df):

    # いらない columns 消去
    _df = _df.drop(columns=["映像", "馬場指数", "ﾀｲﾑ指数", "厩舎ｺﾒﾝﾄ", "備考", "勝ち馬(2着馬)"], errors="ignore")
    _df["日付"] = _df["日付"].fillna("2022/4/16").map(lambda x: datetime.datetime.strptime(x, "%Y/%m/%d"))

    def f(x):
        if x[0].isdecimal():
            return x[0]
        return "nan"
    _df["回"] = _df["開催"].fillna("nan").map(f).astype(float)
    assert  all(True if x in set(range(1,9)) else print(x) for x in _df["回"].fillna(1))

    def f(x):
        if x[0].isdecimal():
            s = 1
        else:
            s = 0
        if x[-1].isdecimal():
            if x[-2].isdecimal():
                return x[s:-2]
            else:
                return x[s:-1]
        return x[s:]
    _df["開催地"] = _df["開催"].fillna("nan").map(f).replace({"nan", np.nan}).astype("category")

    def f(x):
        if x[-1].isdecimal():
            if x[-2].isdecimal():
                return x[-2:]
            else:
                return x[-1:]
        return "nan"
    _df["日目"] = _df["開催"].fillna("nan").map(f).astype(float)
    assert set(_df["日目"].fillna(1)) <= set(range(1,13))

    _df = _df.drop(columns=["開催"])


    _df["天気"] = _df["天気"].astype("category")
    assert set(_df["天気"]) <= {'小雨', '小雪', '晴', '曇', '雨', '雪', np.nan}

    _df["R"] = _df["R"].astype(float).replace({0: np.nan})
    assert all(True if x in set(range(1,14)) else print(x) for x in _df["R"].fillna(1))

    def f(x):
        if "未勝利" in x:
            return "未勝"
        if "新馬" in x:
            return "新馬"
        if ("500万下" in x) or ("1勝クラス" in x):
            return "1勝"
        if ("1000万下" in x) or ("2勝クラス" in x):
            return "1勝"
        if ("1600万下" in x) or ("3勝クラス" in x):
            return "3勝"
        if "ファイナル" in x:
            return "Final"
        x = x.replace("一", "1").replace("二", "2").replace("三", "3").replace("イ", "1").replace("ロ", "2").replace("ハ", "3").replace("ー", "")
        for key, value in (("(G1)", "G1"), ("(G2)", "G2"), ("(G3)", "G3"), ("(L)", "L"), ("(OP)", "OP"), ("OP", "OP_"), ("(G)", "G")):
            if key in x:
                return value

        for key in ("C1", "C2", "C3", "C4", "B1", "B2", "B3", "B4", "A1", "A2", "A3", "A4"):
            if key in x:
                return key
        return "other"
    _df["クラス"] = _df["レース名"].fillna("other").map(f).astype("category")

    def f(x):
        for key, value in ([(f"{i}歳以", f"{i}+") for i in range(2,6)] + [(f"{i}歳", f"{i}") for i in range(2,6)]):
            if key in x:
                return value
        return "other"
    _df["年齢制限"] = _df["レース名"].fillna("other").map(f).astype("category")

    _df["頭数"] = _df["頭数"].astype(float)
    assert set(_df["頭数"].fillna("nan")) <= set(range(0,40)) | set(["nan"]) 


    # 海外、再考の余地あり
    _df["枠番"] = _df["枠番"].astype(float)
    assert set(_df["枠番"].fillna("nan")) <= set(range(1, 19)) | set(["nan"])
    _df["馬番"] = _df["馬番"].astype(float)
    assert set(_df["馬番"].fillna("nan")) <= set(range(0,30)) | set(["nan"])

    _df["オッズ"] = _df["オッズ"].fillna("nan").astype(str).map(lambda x: x.replace(",", "")).astype(float).map(lambda x: np.nan if 0<=x<1 else x)
    assert all(1<=x<=10000 for x in _df["オッズ"].fillna(9999) )

    _df["人気"] = _df["人気"].astype(float).map(lambda x: np.nan if x==0 else x)
    assert set(_df["人気"].fillna("nan")) <= set(range(1,40)) | set(["nan"])

    def f(x):
        if x[-2:] in ("降)", "再)"):
            return x[:-3]
        return x
    _df["着順"] = _df["着順"].astype(str).replace({x: np.nan for x in "取中除失"}).fillna("nan").map(f).astype(float).replace({0:np.nan})
    assert set(_df["着順"].fillna("nan")) <= set(range(1,30)) | set(["nan"])

    _df["斤量"] = _df["斤量"].fillna("nan").astype(float)
    assert all(40<=x<=75 for x in set(_df["斤量"].fillna(50)))

    def f(x):
        if x[0] in "芝ダ障":
            return x[0]
        else:
            return np.nan
    _df["芝ダ"] = _df["距離"].astype(str).map(f).astype("category")

    def f(x):
        if x[0] in "芝ダ障":
            if x[1:]:
                return x[1:]
            else:
                return np.nan
        else:
            return x
    _df["距離"] = _df["距離"].astype(str).map(f).astype(float)
    assert all(True if 800<=x<=10000 else print(x) for x in set(_df["距離"].fillna(1600)))

    def f(x):
        if x[0] in "芝ダ障":
            return x[0]
        else:
            return np.nan
    _df["馬場"] = _df["馬場"].astype("category")
    assert set(_df["馬場"].fillna("良")) <= set("良稍重不")

    def f(x):
        x = x.replace(":", ".")
        if len(x)==6:
            time = int(x[0])*60 + float(x[3:])
        elif x.count(".") == 2:
            time = int(x[:x.index(".")])*60 + float(x[x.index(".")+1:])
        else:
            return np.nan
        
        if time < 20:
            return  np.nan
        return time

    _df["タイム"] = _df["タイム"].fillna("0:00.0").map(f)
    assert all(True if 30<=x<=700 else print(x) for x in set(_df["タイム"].fillna(100)))

    _df["着差"] = np.where( (_df["着差"].astype(float)>=0)|(_df["着差"].isnull()), _df["着差"].astype(float), 0)
    assert all(True if 0<=x<=150 else print(x) for x in set(_df["着差"].fillna(1)))
    
    _df["通過"] = _df["通過"].fillna("nan").map(lambda x: np.array(x.split("-"), dtype=float) )


    def f(x):
        i = x.find("-")
        if i == -1:
            return np.nan
        time = float(x[:i])
        if time < 10:
            return np.nan
        return time
    _df["race_テン"] = _df["ペース"].astype(str).map(f)
    assert all(True if 10<=x<=150 else print(x) for x in set(_df["race_テン"].fillna(32)))

    def f(x):
        i = x.find("-")
        if i == -1:
            return np.nan
        return x[i+1:]
    _df["race_上り"] = _df["ペース"].astype(str).map(f).astype(float).replace({0: np.nan})
    assert all(True if 25<=x<=50 else print(x) for x in set(_df["race_上り"].fillna(32)))
    _df = _df.drop(columns="ペース")


    _df["上り"] = _df["上り"].astype(float)
    # assert all(31<=x<=50 for x in set(_df["上り"].fillna(32)))


    ### 入力ミスの修正
    _df["馬体重"] = _df["馬体重"].replace({"-1(-454)": "454(+1)"})
    ###
    def f(x):
        i = x.find("(")
        if i == -1:
            return np.nan
        return x[4:-1]
    _df["増減"] = _df["馬体重"].fillna("nan").map(f).astype(float)
    assert all(-100<=x<=100 for x in _df["増減"].fillna(0))
    def f(x):
        return x[:3]
    _df["馬体重"] = _df["馬体重"].fillna("nan").replace({"計不":"nan"}).map(f).astype(float)
    assert all(300<=x<=700 for x in _df["馬体重"].fillna(400))

    _df["賞金"] = _df["賞金"].fillna("0").replace({"": "0"}).map(lambda x: x.replace(",","")).astype(float)
    assert all(0<=x<=40000 for x in set(_df["賞金"]))


    # ID
    for col_name, col_id in (("レース名", "race_id"), ("騎手", "jockey_id")):
        _df[col_id] = _df[col_id].astype(str).map(lambda x: x.replace("/result/recent", "")).astype("category")
        for name, id in _df[[col_name, col_id]].values:
            nm[id] = name
        _df = _df.drop(columns=[col_name])


    return _df

def preprocess_v2(df_form):
    df_form["斤量比"] = df_form["斤量"] / df_form["馬体重"]
    df_form["増減比"] = df_form["増減"] / df_form["馬体重"]

    df_form["標準タイム"] = df_form["タイム"] / df_form["距離"]
    df_form["標準着差"] = df_form["着差"] / df_form["距離"]
    df_form.loc[df_form["race_テン"]<20, "race_テン"] = np.nan
    df_form.loc[df_form["race_テン"]>90, "race_テン"] /= 3
    df_form["race_上り差"] = df_form["race_上り"] - df_form["上り"]

    df_form["通過平均"] = df_form["通過"].map(lambda x: x.mean())
    df_form["標準通過平均"] = df_form["通過平均"] / df_form["頭数"]
    df_form.loc[df_form["標準通過平均"]>1, "標準通過平均"] = np.nan

    return df_form

def df_astype(_df_race, _df_result, _df_prof, _df_form):
    category_cols = ["天気", "馬場", "race_id", "jockey_id", "horse_id", "開催地", "年齢制限", "芝ダ", "クラス"]
    _df_form = _df_form.astype({col:"category" for col in category_cols})
    _df_form["日付"] = _df_form["日付"].map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))

    category_cols = ["horse_id", "jockey_id", "jockey_id", "owner_id", "race_id", "trainer_id", "性", "東西"]
    _df_result = _df_result.astype({col:"category" for col in category_cols})

    category_cols = ["race_id", "天気", "馬場", "芝ダ", "日目", "回り", "開催地", "クラス", "年齢制限"]
    _df_race = _df_race.astype({col:"category" for col in category_cols})
    _df_race["日付"] = _df_race["日付"].map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))
    category_cols = ["horse_id", "trainer_id", "owner_id", "breeder_id", "産地", "父_ped_id", "父父_ped_id", "父母_ped_id", "母_ped_id", "母父_ped_id", "母母_ped_id", "調教場所", "セリ取引場所"]
    
    _df_prof = _df_prof.astype({col:"category" for col in category_cols})
    _df_prof["生年月日"] = _df_prof["生年月日"].map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))
    return _df_race, _df_result, _df_prof, _df_form

def get_shift_form(_df_form, n_shift=3):
    ## shift
    _df_form["key"] = _df_form["horse_id"]
    df_list = [_df_form.groupby(["key"]).shift(0)]
    for i in range(1, n_shift+1):
        df_shift = _df_form.groupby(["key"]).shift(-i).drop(columns="horse_id").add_suffix(f'_{i}')
        df_list.append(df_shift)

    ## agg
    df_agg = pd.DataFrame()
    _df_form["日付_float"] = (-_df_form["日付"].map(lambda x: x.timestamp())/(60*60*24)).astype(int)

    _df_form["1"] = 1
    _df_form["オッズ_log10"] = np.log10(_df_form["オッズ"])
    _df_form["優勝"] = (_df_form["着順"]==1)
    _df_form["連対"] = (_df_form["着順"]<=2)
    _df_form["複勝"] = (_df_form["着順"]<=3)

    ## 不良馬場での成績
    

    ## 同開催地での成績

    _df_form_groupby = _df_form.groupby(["horse_id"])
    df_agg["出走間隔"] = _df_form_groupby["日付_float"].diff().shift(-1)

    _df_form_groupby = _df_form.iloc[::-1].groupby(["horse_id"])
    _df_form["出走回数"] = _df_form_groupby["1"].cumsum()
    df_agg["出走回数"] = _df_form["出走回数"]

    # 累積和
    cumsum_cols = ["賞金", "優勝", "連対", "複勝"]
    for col in cumsum_cols:
        _df_form[f"{col}_sum"] = _df_form_groupby[col].cumsum()
    # 累積平均
    average_cols = ["賞金", "優勝", "連対", "複勝"]
    for col in average_cols:
        _df_form[f"{col}_ave"] = _df_form_groupby[col].cumsum() / df_agg["出走回数"]
    ## leak_colを-1シフト
    leak_cols = []
    leak_cols += [f"{col}_sum" for col in cumsum_cols]
    leak_cols += [f"{col}_ave" for col in average_cols]
    _df_form_groupby = _df_form.groupby(["horse_id"])
    for col in leak_cols:
        df_agg[col] = _df_form.groupby(["horse_id"])[col].shift(-1).fillna(0)

    df_list.append(df_agg)

    return pd.concat(df_list, axis=1)

def get_shift_form_v2(_df_form, n_shift=3):
    ## shift
    _df_form["key"] = _df_form["horse_id"]
    df_list = [_df_form.groupby(["key"]).shift(0)]
    for i in range(1, n_shift+1):
        df_shift = _df_form.groupby(["key"]).shift(-i).drop(columns="horse_id").add_suffix(f'_{i}')
        df_list.append(df_shift)

    ## agg
    df_agg = pd.DataFrame()
    _df_form["日付_float"] = (-_df_form["日付"].map(lambda x: x.timestamp())/(60*60*24)).astype(int)


    ## 出走間隔
    _df_form_groupby = _df_form.groupby(["horse_id"])
    df_agg["出走間隔"] = _df_form_groupby["日付_float"].diff().shift(-1)

    conditions =  [
                ["", np.ones(_df_form.shape[0]).astype(bool)], 
                ["馬場稍~不_", _df_form["馬場"]!="良"],
                #    ["S_", _df_form["距離"]<=1599],
                #    ["M_", (_df_form["距離"]>=1600) & (_df_form["距離"]<=1899)],
                #    ["I_", (_df_form["距離"]>=1900) & (_df_form["距離"]<=2100)],
                #    ["L_", (_df_form["距離"]>=2101) & (_df_form["距離"]<=2700)],
                #    ["E_", (_df_form["距離"]>=2701)],
                ]
    
    # conditions += [
    #                 [f"{kaisaichi}_", _df_form["開催地"]==kaisaichi] 
    #                             for kaisaichi in ["東京", "中山", "京都", "阪神"]
    #                ]

    for key, condition in conditions:

        _df_form0 = _df_form.copy()
        _df_form0["1"] = 1 * condition
        _df_form0["優勝"] = (_df_form["着順"]==1) * condition
        _df_form0["連対"] = (_df_form["着順"]<=2) * condition
        _df_form0["複勝"] = (_df_form["着順"]<=3) * condition
        _df_form0["賞金"] = _df_form["賞金"] * condition

        _df_form0_groupby = _df_form0.iloc[::-1].groupby(["horse_id"])
        _df_form0["出走回数"] = _df_form0_groupby["1"].cumsum()
        df_agg[f"{key}出走回数"] = _df_form0["出走回数"]

        # 累積和
        cumsum_cols = ["賞金", "優勝", "連対", "複勝"]
        for col in cumsum_cols:
            _df_form0[f"{col}_sum"] = _df_form0_groupby[col].cumsum()
        # 累積平均
        average_cols = ["賞金", "優勝", "連対", "複勝", "標準着差", "標準通過平均", "標準タイム", "オッズ", "上り"]
        for col in average_cols:
            _df_form0[f"{col}_ave"] = (_df_form0_groupby[col].cumsum() / df_agg[f"{key}出走回数"]).replace({np.inf: np.nan})
        ## leak_colを-1シフト
        leak_cols = []
        leak_cols += [f"{col}_sum" for col in cumsum_cols]
        leak_cols += [f"{col}_ave" for col in average_cols]
        _df_form0_groupby = _df_form0.groupby(["horse_id"])
        for col in leak_cols:
            df_agg[f"{key}{col}"] = _df_form0.groupby(["horse_id"])[col].shift(-1).fillna(np.nan)

    df_list.append(df_agg)

    return pd.concat(df_list, axis=1)

def df_concat(_df_result, _df_race, _df_prof, _df_form):
    _df_result = pd.merge(_df_result, _df_race, how='left', on="race_id")
    _df_result = pd.merge(_df_result, _df_prof.drop(columns=["trainer_id", "owner_id"]), how='left', on="horse_id")
    drop_colls = ["着順", "枠番", "馬番", "斤量", "タイム", "通過", "上り", "オッズ", "人気", "賞金", "jockey_id", "馬体重", "増減", "R", "天気", "馬場", "日付", "芝ダ", "距離", "回", "開催地", "日目", "年齢制限", "クラス", "race_テン", "race_上り"]
    _df_result = pd.merge(_df_result, _df_form.drop(columns=drop_colls), how='left', on=["race_id", "horse_id"])
    return _df_result

def get_pp2(_df):
    _df["乗り替わり"] = (_df["jockey_id"].astype(str) != _df["jockey_id_1"].astype(str)).value_counts()
    _df["距離差"] = (_df["距離"] - _df["距離_1"])
    _df["距離比"] = _df["距離"] / _df["距離_1"]

    _df["年齢flaot"] = (_df["日付"] - _df["生年月日"]).map(lambda x: x.days)/365

    for i in range(1, n_shift+1):
        _df[f"オッズ_log_{i}"] = _df[f"オッズ_{i}"]
    for col in ["オッズ", "オッズ_log", "人気", "着差", "着順", "標準着差", "標準タイム", "上り", "通過平均", "race_上り差"]:
        _df[f"{col}_{n_shift}_ave"] = np.nanmean(_df[[f"{col}_{i}" for i in range(1, n_shift+1)]], axis=1)
        _df[f"{col}_{n_shift}_ave"] = np.mean(_df[[f"{col}_{i}" for i in range(1, n_shift+1)]], axis=1)
    return _df

def encoded(_df, upload=False, download=False):
    if download:
#         %cd /content/drive/My Drive/Horse Racing/models
        category_cols = pickle.load(open("category_cols.sav", 'rb'))
        print("GET OrdinalEncoder, 'oe.sav' !!")
        oe = pickle.load(open("oe.sav", 'rb'))
        _df[category_cols] = oe.transform(_df[category_cols].values)
        _df[category_cols] = _df[category_cols].astype('category')
    else:
        category_cols = [col for col in _df.columns if _df[col].dtype.name == "category"]
        oe = preprocessing.OrdinalEncoder(
                handle_unknown = 'use_encoded_value',
                unknown_value = np.nan,
        )
        _df[category_cols] = oe.fit_transform(_df[category_cols].values)
        _df[category_cols] = _df[category_cols].astype('category')

    if upload:
#         %cd /content/drive/My Drive/Horse Racing/models
        pickle.dump(category_cols, open(f'category_cols.sav', 'wb'))
        print("LOAD OrdinalEncoder, 'oe.sav' !!")
        pickle.dump(oe, open(f'oe.sav', 'wb'))

    return _df

# My LightGBM

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    f = np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)
    return f

def focal_loss(x: np.ndarray, t: np.ndarray, gamma: float=1):
    """
    Params::
        x:
            予測値. 規格化されていない値であること. 1次元
            >>> x
            array([0.65634349, 0.8510698 , 0.61597224])
            multi classのとき. 例えば 3 class の場合は下記のような値になっていること.
            >>> x
            array([
                [0.65634349, 0.8510698 , 0.61597224],
                [0.58012161, 0.79659195, 0.39168051]
            ])
        t: (n_class, ) 1次元配列
    """
    t = t.astype(np.int32)
    if len(x.shape) > 1:
        x = softmax(x)
        t = np.identity(x.shape[1])[t]
        return -1 * t * (1 - x)**gamma * np.log(x)        
    else:
        x = sigmoid(x)
        x[t == 0] = 1 - x[t == 0] # 0ラベル箇所は確率を反転する
        return -1 * (1 - x)**gamma * np.log(x)

def focal_loss_grad(x: np.ndarray, t: np.ndarray, gamma: float=1):
    """
    内部に softmax を含む関数については derivative では計算が安定しない.
    """
    t = t.astype(np.int32)
    if len(x.shape) > 1:
        x = softmax(x) # softmax で規格化
        # 正解列を抜き出し
        xK = x[np.arange(t.shape[0]).reshape(-1, 1), t.reshape(-1, 1)]
        xK = np.tile(xK, (1, x.shape[1]))
        # x1 は 不正解列に -1 をかけて、さらに正解列はそこから1を足す操作
        x1 = x.copy()
        x1 = -1 * x1
        x1[np.arange(t.shape[0]).reshape(-1, 1), t.reshape(-1, 1)] = x1[np.arange(t.shape[0]).reshape(-1, 1), t.reshape(-1, 1)] + 1
        dfdy = gamma * (1 - xK) ** (gamma-1) * np.log(xK) - ((1 - xK) ** gamma / xK)
        dydx = xK * x1
        grad = dfdy * dydx

        dfdydx = dydx * (2 * gamma * (1 - xK) ** (gamma - 1) / xK - gamma * (gamma - 1) * np.log(xK) * (1 - xK) ** (gamma - 2) + (1 - xK) ** gamma * (xK ** -2))
        dydxdx = dydx * (1 - 2 * x)
        hess = dfdy * dydxdx + dydx * dfdydx
    else:
        grad = derivative(lambda _x: focal_loss(_x, t, gamma=gamma), x, n=1, dx=1e-6)
        hess = derivative(lambda _x: focal_loss(_x, t, gamma=gamma), x, n=2, dx=1e-6)

    return grad, hess

def lgb_custom_objective(y_pred: np.ndarray, data: lgbm.Dataset, func_loss, is_lgbdataset: bool=True):
    """
    lightGBMのcustomized objectiveの共通関数
    Params::
        y_pred:
            予測値. multi classの場合は、n_sample * n_class の長さになったいる
            値は、array([0データ目0ラベルの予測値, ..., Nデータ目0ラベルの予測値, 0データ目1ラベルの予測値, ..., ])
        data:
            train_set に set した値
        func_loss:
            y_pred, y_true を入力に持ち、y_pred と同じ shape を持つ return をする
        is_lgbdataset:
            lgb.dataset でなかった場合は入力が逆転するので気をつける
    """
    if is_lgbdataset == False:
        y_true = y_pred.copy()
        y_pred = data
    else:
        y_true = data.label
    if y_pred.shape[0] != y_true.shape[0]:
        # multi class の場合
        n_class = int(y_pred.shape[0] / y_true.shape[0])
        y_pred = y_pred.reshape(n_class, -1).T
    grad, hess = func_loss(y_pred, y_true)
    return grad.T.reshape(-1), hess.T.reshape(-1)

def lgb_custom_eval(y_pred: np.ndarray, data: lgbm.Dataset, func_loss, func_name: str, is_higher_better: bool, is_lgbdataset: bool=True):
    """
    lightGBMのcustomized objectiveの共通関数
    Params::
        y_pred:
            予測値. multi classの場合は、n_sample * n_class の長さになったいる
            値は、array([0データ目0ラベルの予測値, ..., Nデータ目0ラベルの予測値, 0データ目1ラベルの予測値, ..., ])
        data:
            train_set に set した値
        func_loss:
            y_pred, y_true を入力に持ち、grad, hess を return する関数
    """
    if is_lgbdataset == False:
        y_true = y_pred.copy()
        y_pred = data
    else:
        y_true  = data.label
    n_class = 1
    if y_pred.shape[0] != y_true.shape[0]:
        # multi class の場合
        n_class = int(y_pred.shape[0] / y_true.shape[0])
        y_pred = y_pred.reshape(n_class, -1).T
    value = func_loss(y_pred, y_true)
    return func_name, np.sum(value), is_higher_better

f   = lambda x, y: focal_loss(x, y, gamma=1.0)
f_g = lambda x, y: focal_loss_grad(x, y, gamma=1.0)

class MyLGBMClassifier(lgbm.LGBMClassifier):
    """
    custom objective を想定して値を規格化できるように自作classを定義する
    """
    def predict_proba(self, X, *argv, **kwargs):
        proba = super().predict_proba(X, *argv, **kwargs)
        if len(proba.shape) == 2:
            proba = softmax(proba)
        else:
            proba = sigmoid(proba)
            proba[:, 0] = 1 - proba[:, 1]
        return proba

def model_fit(model_name, X_train, y_train, X_valid, y_valid, **params):
    """
    return score(float), model, pred(arr)
    """
    is_binary = all(yi in {1,0} for yi in y_valid)
    if is_binary:
        if model_name == "LightGBM":
            model = lgbm.LGBMClassifier(
                                **params,
                                n_estimators=10000, 
                                silent=1
                                )        
            model.fit(
                X_train, y_train, 
                eval_set=[(X_valid, y_valid)],  
                early_stopping_rounds=50,
                eval_metric='auc',
                feature_name='auto', 
                categorical_feature = 'auto',
                verbose=False,
                )

            pred = model.predict_proba(X_valid, num_iteration=model.best_iteration_)[:, 1]
            score = roc_auc_score(y_valid, pred)
        elif model_name == "MyLightGBM":
            model = MyLGBMClassifier(
                                **params, 
                                n_estimators=10000, 
                                silent=1,
                                objective=(lambda x,y: lgb_custom_objective(x, y, f_g, is_lgbdataset=False)),
                                )
            model.fit(
                X_train, y_train, 
                eval_set=[(X_valid, y_valid)],  
                early_stopping_rounds=50,
                eval_metric='auc',
                feature_name='auto', 
                categorical_feature = 'auto',
                verbose=False,
                )

            pred = model.predict_proba(X_valid, num_iteration=model.best_iteration_)[:, 1]
            score = roc_auc_score(y_valid, pred)
        elif model_name == "CatBoost":
            model = CatBoostClassifier(
                    **params, 
                    early_stopping_rounds=50,
                    n_estimators=10000, 
                    silent=1,
                    )
            train_pool = Pool(X_train, y_train)
            valid_pool = Pool(X_valid, y_valid)

            model.fit(train_pool, eval_set=valid_pool)
            pred = model.predict_proba(X_valid, num_iteration=model.best_iteration_)[:, 1]
            # model.predict(X_test)
            score = roc_auc_score(y_valid, pred)
        else:
            print("is_not_exist")
    else:
        if model_name == "MyLightGBM":
            model_name = "LightGBM"
        if model_name == "LightGBM":
            model = lgbm.LGBMRegressor(
                            objective='regression',
                            **params,
                            n_estimators=10000, 
                            silent=1
                        ) 
            model.fit(
                X_train, y_train, 
                eval_set=[(X_valid, y_valid)],  
                early_stopping_rounds=50,
                # eval_metric='rmse',
                feature_name='auto', 
                categorical_feature = 'auto',
                verbose=False,
                )
            pred = model.predict(X_valid, num_iteration=model.best_iteration_)
            score = mean_squared_error(y_valid, pred)
        else:
            print("is_not_exist")

    return score, model, pred

def arr_dev(df_valid_, TARGET_pred, order="infer"):
    mean = np.array(df_valid_[["race_id", TARGET_pred]].groupby("race_id").mean().loc[df_valid_["race_id"], TARGET_pred])
    std = np.array(df_valid_[["race_id", TARGET_pred]].groupby("race_id").std().loc[df_valid_["race_id"], TARGET_pred])
    
    arr = 50 + (df_valid_[TARGET_pred] - mean) / std * 10

    reverse = False
    if order == "infer":
        if np.corrcoef(arr, df_valid_["着順"])[0,1]>0:
            arr = 100 - arr
            reverse = True

    if order == "reverse":
        arr = 100 - arr
        reverse = True
    return arr, reverse

def get_new_race_csv(race_id_list):
    race_list = []
    result_list = []
    def add_new_race_info(race_id):
        url=f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}&rf=race_submenu"
        res = requests.get(url)
        res.encoding = "EUC-JP"
        soup = BeautifulSoup(res.text, "html.parser")

        # race
        race_soup = soup.find("div", attrs={"class": "RaceList_NameBox"})
        d = {}
        d["race_id"] = f"/race/{race_id}/"
        d["レース名"] = race_soup.find("div", class_="RaceName").text.split()[0]    
        grade = ""
        for key, g in [(1,"G1"), (2,"G2"), (3,"G3"), (5,"OP"), (15, "L"), (16, "3勝"), (17, "2勝"), (18, "1勝")]:
            if soup.find("span", class_=f"Icon_GradeType Icon_GradeType{key}"):
                grade = g
        if g:
            d["レース名"] += f"({grade})"
        d["R"] = race_soup.find("span", class_="RaceNum").text[:-1]
        l = race_soup.find("div", class_="RaceData01").text.replace(" ", "").replace("\n", "").split("/")
        if len(l)==2:
            l += [":晴", ":良"]

        d["コース"] = l[1][0] + l[1][l[1].index("(")+1: l[1].index(")")] + l[1][1: l[1].index("(")]
        d["天気"] = l[2][l[2].index(":")+1:]
        d["馬場"] = l[3][l[3].index(":")+1:].replace("稍", "稍重").replace("不", "不良")
        d["時間"] = l[0][:l[0].index("発")]
        d["日付"] = np.nan
        l = race_soup.find("div", class_="RaceData02").text.split()
        d["開催"] = l[0] + l[1] + l[2]
        d["条件"] = l[3].replace("サラ系", "") + l[4]
        d["タイプ"] = "(".join(l[5:-2])
        d["過去レースurl"] = np.nan
        d["レース総称"] = race_soup.find("div", class_="RaceName").text.split()[0]
        for col in [f"{i+1}コーナー" for i in range(4)] + ["ラップ", "ペース"]:
            d[col] = np.nan
        d["ラップ"] = d["ペース"] = "(nan-nan)"
        race_list.append(d)

        ## result
        shutuba_soups = soup.find("div", class_="RaceTableArea").find("table", class_="Shutuba_Table RaceTable01 ShutubaTable").find_all("tr")
        columns = [i.text.split()[0] for i in shutuba_soups[0].find_all("th")] + ["-"] + [f"{type_}_id" for type_ in ("horse", "jockey", "trainer")]
        shutuba_data = []
        # id_manage.add("race", race_id, race_soup.find("h1").text, True)
        data = [line for line in shutuba_soups[2:]]

        for line in data:
            shutuba_data.append([x.text.replace("\n", "") for x in line.find_all("td")]+[None]*3)   # １行データを加える
            for text in line.find_all("a"):
                id = text.get("href")
                name = text.get("title")
                if id:
                    for i, type_ in enumerate(("horse", "jockey", "trainer")):
                        if f"/{type_}/" in id:
                            shutuba_data[-1][13+i] = id.replace("https://db.netkeiba.com", "").replace("/result/recent", "")
                            # id_manage.add(type_, id, name, False)
                            # id_data.append([type_, id, name, False])
        _df = pd.DataFrame(data=shutuba_data, columns=columns)\
                    .drop(columns=["印", "お気に入り馬", "-", "人気", "更新"])\
                    .rename(columns={"枠": "枠番", "馬体重(増減)": "馬体重"})
        _df['race_id'] = f"/race/{race_id}/"

        _df["馬体重"] = _df["馬体重"].fillna("計不").replace({"nan":"計不"})

        # 事後
        _df[['着順','タイム','着差','ﾀｲﾑ指数','通過','上り','厩舎ｺﾒﾝﾄ','備考','賞金(万円)']] = np.nan
        # 1週間前
        # _df[["枠番", "馬番"]] = np.nan
        # 変動
        _df[['単勝','人気',]] = np.nan
        # _df['単勝'] = [190.8,111.6,182.3,3.7,20.4,1.5,60.2,58.7,47.5,27.4,74.5,298.8,45.5,9.2,92.9,145.9]
        # _df['人気'] = [15,12,14,2,4,1,9,8,7,5,10,16,6,3,11,13]
        # 入手可能
        _df["調教師"] = _df["厩舎"].map(lambda x: x.replace("美浦", "[東]").replace("栗東", "[西]").replace("地方", "[地]"))
        _df[['調教ﾀｲﾑ', '馬主', 'owner_id']] = np.nan

        _df = _df.drop(columns=["厩舎"])
        result_list.append(_df)
    
    n = len(race_id_list)
    xxx = st.text(f"scrape {0}/{n} races")
    bar = st.progress(0.0)
    for i, race_id in enumerate(race_id_list):
        add_new_race_info(race_id)
        xxx.text(f"scrape {i+1}/{n} races")
        bar.progress((i+1)/n)
    df_race = pd.DataFrame(race_list).replace({"": np.nan, "--": np.nan})
    df_result = pd.concat(result_list, axis=0).replace({"": np.nan, "--": np.nan})
    xxx = st.empty()
    bar = st.empty()
    return df_race, df_result

def get_new_horse_csv(df_result):
    prof_list = []
    form_list = []
    def add_new_horse_info(horse_id, race_id):
        time.sleep(0.5)
        url=f"https://db.netkeiba.com{horse_id}"
        res = requests.get(url)
        res.encoding="EUC-JP"
        soup = BeautifulSoup(res.text, "html.parser")

        # [dict] profile, ped
        d = {}
        d["horse_id"] = horse_id
        d["horse_name"] = soup.find("div", attrs={"class": "horse_title"}).find("h1").text
        for line in soup.find("table", attrs={"class": "db_prof_table", "summary": "のプロフィール"}).find_all("tr"):
            d[line.th.text] = ''.join(line.td.text.split())
            for text in line.find_all("a"):
                id = text.get("href")
                name = text.get("title")
                if id:
                    for i, type_ in enumerate(("trainer", "owner", "breeder")):
                        if f"/{type_}/" in id:
                            d[f"{type_}_id"] = id
        ped_soups = soup.find("table", attrs={"class": "blood_table"}).find_all("td")
        for line, relative in zip(ped_soups, ("父", "父父", "父母", "母", "母父", "母母")):
            name = "".join(line.text.split())
            ped_id = line.find("a").get("href")
            d[f"{relative}_ped_id"] = ped_id
            d[f"{relative}_name"] = name
        prof_list.append(d)

        # [df] form
        tabel_data = soup.find("table", attrs={"class": "db_h_race_results nk_tb_common"})
        if tabel_data:
            form_soups = tabel_data.find_all("tr")
            columns = [column.text for column in form_soups[0].find_all("th")] + [f"{type_}_id" for type_ in ("race", "jockey", "horse")]
            form_data = []
            data = [line for line in form_soups[1:]]
            for line in data:
                form_data.append(["".join(x.text.split()) for x in line.find_all("td")]+[None]*2+[horse_id])
                for text in line.find_all("a"):
                    id = text.get("href")
                    name = text.get("title")
                    if id:
                        for i, type_ in enumerate(("race", "jockey")):
                            if name and (f"/{type_}/" in id) and (f"/{type_}/movie" not in id):
                                form_data[-1][28+i] = id
        else:
            form_data = [[]]
            columns = []

        d = {
            '着差': np.nan,
            'race_id': race_id,
            'horse_id': horse_id
        }
        form_list.append(
            pd.concat([pd.DataFrame([d]), pd.DataFrame(data=form_data, columns=columns)], axis=0).reset_index()
        )
    n = df_result.shape[0]
    xxx = st.text(f"scrape {n} horses")
    bar = st.progress(0.0)
    for i, (horse_id, race_id) in enumerate(df_result[["horse_id", "race_id"]].values):
        add_new_horse_info(horse_id, race_id)
        xxx.text(f"scrape {i+1}/{n} horses")
        bar.progress((i+1)/n)
    df_prof = pd.DataFrame(prof_list).replace({"": np.nan, "--": np.nan})
    df_form = pd.concat(form_list, axis=0).replace({"": np.nan, "--": np.nan})
    xxx = st.empty()
    bar = st.empty()
    return df_prof, df_form

def encoded(_df, upload=False, download=False):
    if download:
#         %cd /content/drive/My Drive/Horse Racing/models
        category_cols = pickle.load(open("category_cols.sav", 'rb'))
        print("GET OrdinalEncoder, 'oe.sav' !!")
        oe = pickle.load(open("oe.sav", 'rb'))
        _df[category_cols] = oe.transform(_df[category_cols].values)
        _df[category_cols] = _df[category_cols].astype('category')
    else:
        category_cols = [col for col in _df.columns if _df[col].dtype.name == "category"]
        oe = preprocessing.OrdinalEncoder()
        _df[category_cols] = oe.fit_transform(_df[category_cols].values)
        _df[category_cols] = _df[category_cols].astype('category')

    if upload:
#         %cd /content/drive/My Drive/Horse Racing/models
        pickle.dump(category_cols, open(f'category_cols.sav', 'wb'))
        print("LOAD OrdinalEncoder, 'oe.sav' !!")
        pickle.dump(oe, open(f'oe.sav', 'wb'))

    return _df

def get_held_races(date):
    url=f"https://yoso.netkeiba.com/?pid=race_list&kaisai_date={date}"
    res = requests.get(url)
    res.encoding = "EUC-JP"
    soup = BeautifulSoup(res.text, "html.parser")
    soups = soup.find_all("div", attrs={"class": "RaceList"})#.find("div", attrs={"class": "Jyo"})
    data = []
    for soup_i in soups:
        d = {}    
        headder1 = soup_i.find(class_="Jyo").text
        # headder2 = soup_i.find("ul", class_="JyoData")
        d["開催地"] = headder1[:2]
        d["回"] = headder1[headder1.index("(")+1:headder1.index("回")]
        d["日目"] = headder1[headder1.index("回")+1:headder1.index("日目")]

        for race_i in soup_i.find("div", class_="RaceList_Main").find_all("li"):
            text = race_i.text.replace("\n","")
            R = int(text[:text.index("R")])
            url = race_i.find("a").get("href")
            name = text[text.index("R")+1: text.find(":")-2]
            race_id = re.sub(r"\D", "", url)

            d[R] = {}
            d[R]["R"] = R
            d[R]["name"] = name
            d[R]["race_id"] = race_id
        data.append(d)
    return data


    

## 日付を選ぶ
if "data" not in st.session_state:
    today = datetime.date.today().isocalendar()
    year = today[0]
    week = today[1]
    st.write("曜日の選択")
    sat = st.checkbox(f'土')
    sun = st.checkbox('日')

    date_list = []

    click1 = st.button("OK")

    data = []

    if click1:
        if sat:
            date_list.append(
                str(datetime.date.fromisocalendar(year, week, 6)).replace("-", "")
            )
        if sun:
            date_list.append(
                str(datetime.date.fromisocalendar(year, week, 7)).replace("-", "")
            )
        st.session_state["data"] = []
        for date in date_list:
            st.session_state["data"] += get_held_races(date)
        
        st.button("次へ")

# 開催を選ぶ
elif "data2" not in st.session_state:
    st.write("開催地の選択")
    opt = [f'{data_i["開催地"]}{data_i["回"]}回{data_i["日目"]}日目' for data_i in st.session_state["data"]]
   
    d = {opt_i: i for i, opt_i in enumerate(opt)}
    options = st.selectbox('',opt)

    click2 = st.button("OK")
    if click2:
        st.session_state["data2"] = st.session_state["data"][d[options]]
        st.button("次へ")
        

# レースを選ぶ
elif "id_list" not in st.session_state:

    races = [i for i in st.session_state["data2"].keys() if isinstance(i, int)]
    opt = [f'{i}R {st.session_state["data2"][i]["name"]}' for i in races]

    if len(races)==12:
        options = st.multiselect('レースの選択', opt, [opt_i for opt_i in opt if "11R" in opt_i])
    else:
        options = st.multiselect('レースの選択',opt)
    d = {opt_i: i+1 for i, opt_i in enumerate(opt)}
    click3 = st.button("OK")

    if click3:
        st.session_state["id_list"] = []
        st.session_state["id2name"] = {}
        for i in options:
            R = d[i]
            race_id = st.session_state["data2"][R]["race_id"]
            race_name = st.session_state["data2"][R]["name"]
            st.session_state["id_list"].append(race_id)
            st.session_state["id2name"][race_id] = f"{R}R {race_name}"
        st.button("次へ")


elif "df_pred_test_groupby" not in st.session_state:
    seed=42
    past_race = False
    race_id = st.session_state["id_list"]

    df_test_race, df_test_result = get_new_race_csv(race_id)
    df_test_prof, df_test_form = get_new_horse_csv(df_test_result)
    nm = pickle.load(open("nm.sav", 'rb'))

    # # 前処理
    df_test_race = preprocess_race(df_test_race)
    df_test_result = preprocess_result(df_test_result)
    df_test_prof = preprocess_prof(df_test_prof)
    df_test_form = preprocess_form(df_test_form)
    df_test_form = preprocess_v2(df_test_form)

    # SHIFT
    n_shift = 3
    df_test_form = df_test_form.reset_index()
    df_test_form_shifted = get_shift_form_v2(df_test_form.copy(), n_shift)
    # CONCAT
    df_test = df_concat(df_test_result, df_test_race, df_test_prof, df_test_form_shifted)
    # LABEL ENCODING
    df_test["年齢制限"] = df_test["年齢制限"].map(lambda x: f"{int(x[0])}{x[1:]}")
    df_test["乗り替わり"] = (df_test["jockey_id"].astype(str) != df_test["jockey_id_1"].astype(str)).value_counts()
    df_test["距離差"] = (df_test["距離"] - df_test["距離_1"])
    df_test["距離比"] = df_test["距離"] / df_test["距離_1"]
    df_test["年齢flaot"] = (df_test["日付"] - df_test["生年月日"]).map(lambda x: x.days)/365



    for i in range(1, n_shift+1):
        df_test[f"オッズ_log_{i}"] = df_test[f"オッズ_{i}"]
    for col in ["オッズ", "オッズ_log", "人気", "着差", "着順", "標準着差", "標準タイム", "上り", "通過平均", "race_上り差"]:
        df_test[f"{col}_{n_shift}_ave"] = np.nanmean(df_test[[f"{col}_{i}" for i in range(1, n_shift+1)]], axis=1)
        df_test[f"{col}_{n_shift}_ave"] = np.mean(df_test[[f"{col}_{i}" for i in range(1, n_shift+1)]], axis=1)
        
    df_test_encoded = encoded(df_test.copy(), download=True)

    with st.spinner('Load AI models'):
        models_data = pickle.load(open("models_data_v7.sav", 'rb'))

    def test(models, _df, _df_encoded, X_cols, TARGET, dev_reverse):
        print(f"        {TARGET}")

        ## set X
        X_test = _df_encoded[X_cols].values

        ## eval
        pred_test = np.zeros(_df_encoded.shape[0])
        for i, model in enumerate(models):
            if "Regressor" not in str(type(model)):
                pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)[:, 1]
            else:
                pred = model.predict(X_test, num_iteration=model.best_iteration_)
            pred_test += pred
        pred_test /= len(models)


        df_test_pred = _df[["着順", "race_id"]]
        df_test_pred[f"{TARGET}_pred"] = pred_test

        if dev_reverse:
            dev_test = arr_dev(df_test_pred, f"{TARGET}_pred", order="reverse")[0]
        else:
            dev_test = arr_dev(df_test_pred, f"{TARGET}_pred", order="foreard")[0]
        
        return pred_test, dev_test

    is_test = (df_test["race_id"] != df_test["race_id_1"]) & (df_test["race_id"] != df_test["race_id_2"]) & (df_test["race_id"] != df_test["race_id_3"])
    if past_race:
        is_test &= (df_test.index % 2 == 1)


    df_pred_test = df_test[is_test]
    df_pred_test_encoded = df_test_encoded[is_test]

    n = len(models_data)
    xxx = st.text(f"infer {0}/{n}")
    bar = st.progress(0.0)
    i = 0


    for learn_level in [1,2,3]:
        print(f"Level {learn_level}")
        for TARGET in models_data.keys():
            if models_data[TARGET]["level"] == learn_level:
                X_cols = models_data[TARGET]["X_cols"]
                models = models_data[TARGET]["models"]
                dev_reverse = models_data[TARGET]["dev_reverse"]

                pred_test, dev_test = test(models, df_pred_test, df_pred_test_encoded, X_cols, TARGET, dev_reverse)

                df_pred_test[f"{TARGET}_pred"] = pred_test
                df_pred_test[f"{TARGET}_dev"] = dev_test

                df_pred_test_encoded[f"{TARGET}_pred"] = pred_test
                df_pred_test_encoded[f"{TARGET}_dev"] = dev_test

                i += 1
                xxx.text(f"infer {i}/{n}")
                bar.progress(i/n)
                
    st.button("AI予測結果を見る")
    st.session_state["df_pred_test_groupby"] = df_pred_test.groupby("race_id")
    st.session_state["nm"] = nm


else:
    def display_pred(df_pred_test, display_cols, sort_key):

        df_target = df_pred_test[["枠番", "馬番"] + display_cols].fillna(0).astype({"馬番":int, "枠番":int})
        df_target.index = df_pred_test["horse_id"].map(lambda x: nm[x]).values
        # df_target["スコア"] = df_target[
        #     [f"{col}_dev" for col in ["勝利", "連対",  "複勝", "着差", "タイム"]]
        #     ].mean(axis=1)
        df_target = df_target.sort_values(by=sort_key, ascending=False)
        df_target = df_target.drop_duplicates()

        df_target.columns = [col.replace("_dev", "").replace("ensemble", "") for col in df_target.columns]
        display_cols = [col.replace("_dev", "").replace("ensemble", "") for col in display_cols]

        d = {}
        for name, (umaban, wakuban) in zip(df_target.index, df_target[["馬番", "枠番"]].values):
            d[name] = wakuban
            d[umaban] = wakuban

        def back_color(s):
            return [f'background-color: #333333' for i in s]

        def wakuban_color(s):

            alpha = "4D"
            color = ["", f"#ffffff{alpha}",  f"#000000{alpha}", f"#ff0000{alpha}", f"#0000ff{alpha}", f"#ffff00{alpha}",  f"#008000{alpha}", f"#ffa500{alpha}", f"#ffc0cb{alpha}"]
            return [f'background-color: {color[i]}' for i in s]

        def umaban_color(s=""):
            s = [d[i] for i in list(s)]
            alpha = "2D"
            color = ["", f"#ffffff{alpha}",  f"#000000{alpha}", f"#ff0000{alpha}", f"#0000ff{alpha}", f"#ffff00{alpha}",  f"#008000{alpha}", f"#ffa500{alpha}", f"#ffc0cb{alpha}"]
            return [f'background-color: {color[i]}' for i in s]

        def highlight(s):
            if s>50:
                color = f"#ff1493"
            elif s<=50:
                color = f"#00bfff"
            alpha = hex(int(abs(s-50))*6)[2:]
            if len(alpha) == 1:
                alpha = "0" + alpha

            return f'background-color: {color}{alpha}'

        def max_bold(s):
            is_max = (s == s.max())
            
            return ['font-weight:bold; border:solid;' if v else '' for v in is_max]
        
        used = {}
        is_use = []
        for i, name in enumerate(df_target.index):
            if name in used:
                is_use.append(False)
            else:
                is_use.append(True)
            used[name] = i
        df_target = df_target[is_use]

        race_id = df_pred_test.iloc[0].race_id
        df_target = (
                df_target.style
                    .set_precision(1)
                    .apply(back_color)
                    .apply(wakuban_color, subset=['枠番'])
                    .apply(umaban_color, subset=['馬番'])
                    .applymap(highlight, subset=display_cols)
                    .apply(max_bold, subset=display_cols)
                    .set_caption(f"{int(race_id[12:14])}回{df_pred_test.iloc[0].開催地}{int(race_id[14:16])}日目 {int(race_id[16:18])}R {str(nm[race_id]).replace('()','')} {df_pred_test.iloc[0].芝ダ} {df_pred_test.iloc[0].距離:.0f}m")    
            )
        return df_target 

    sort_key = "勝利"


    race_list = st.session_state["id2name"].values()
    nm = st.session_state["nm"]
    print(race_list)
    options = st.multiselect(
        '予測するレースを選ぶ',
        race_list,
        race_list)

    display_cols_opt = ["勝利", "連対",  "複勝", "着差", "タイム", "単勝払戻", "複勝払戻"]
    display_cols_opt = ["勝利"]

    # display_cols = st.sidebar.multiselect('表示する指標', display_cols_opt, display_cols_opt)
    display_cols = display_cols_opt
    # display_cols = [f"{col}ensemble_dev" for col in display_cols]
    display_cols = [f"{col}_dev" for col in display_cols]


    click4 = st.button("更新")

    if click4:
        for race_id, df_pred_testi in st.session_state["df_pred_test_groupby"]:
            race_name = st.session_state["id2name"][re.sub(r"\D", "", race_id)]
            if race_name in options:
                st.write(race_name)
                n = df_pred_testi.shape[0] + 1
                df_display = display_pred(df_pred_testi, display_cols, sort_key)
                st.dataframe(df_display, height=35*n)
