
import numpy as np
import pandas as pd
import datetime
import re
from sklearn.preprocessing import OrdinalEncoder
import dill



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
        return {id: value for id, value in self.notation.items() if len(value)>=2}

def preprocess_result(_df, nm):
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

def preprocess_race(_df, nm):
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


    _df["日付"] = _df["日付"].fillna("2022年6月26日").map(lambda x: datetime.datetime.strptime(x, "%Y年%m月%d日"))

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

def preprocess_prof(_df, nm):

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

def preprocess_form(_df, nm):

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
    assert set(_df["天気"].astype("str")) <= {'小雨', '小雪', '晴', '曇', '雨', '雪', "nan"}

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
    assert set(_df["馬場"].astype(str)) <= set("良稍重不")|set(["nan"])

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

def get_pp2(_df, n_shift):
    _df["年齢制限"] = _df["年齢制限"].map(lambda x: f"{int(x[0])}{x[1:]}")
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
        category_cols = dill.load(open("category_cols.sav", 'rb'))
        print("GET OrdinalEncoder, 'oe.sav' !!")
        oe = dill.load(open("oe.sav", 'rb'))
        _df[category_cols] = oe.transform(_df[category_cols].values)
        _df[category_cols] = _df[category_cols].astype('category')
    else:
        category_cols = [col for col in _df.columns if _df[col].dtype.name == "category"]
        oe = OrdinalEncoder(
                handle_unknown = 'use_encoded_value',
                unknown_value = np.nan,
        )
        _df[category_cols] = oe.fit_transform(_df[category_cols].values)
        _df[category_cols] = _df[category_cols].astype('category')

    if upload:
#         %cd /content/drive/My Drive/Horse Racing/models
        dill.dump(category_cols, open(f'category_cols.sav', 'wb'))
        print("LOAD OrdinalEncoder, 'oe.sav' !!")
        dill.dump(oe, open(f'oe.sav', 'wb'))

    return _df

