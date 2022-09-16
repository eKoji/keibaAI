import re
import os
import dill
import datetime
import numpy as np
# import pandas as pd

import requests
from bs4 import BeautifulSoup

import streamlit as st

import lightgbm as lgbm

from scipy.misc import derivative
import warnings
warnings.filterwarnings('ignore')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    f = np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)
    return f
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



# 日付を選ぶ。（レースのスクレイピングまで）
def page1():
    def next_page():
        st.session_state.date = [key for key in keys if st.session_state[key]]
        if st.session_state.select_date not in st.session_state.date:
            date = str(st.session_state.select_date).replace("-", "")
            st.session_state.date.append(date)
        if len(st.session_state.date) >= 1:
            st.session_state.page += 1
        else:
            st.warning("日付を選択して下さい")

    # 日付の選択肢を取得
    def i_day(day_0: datetime.date, i: int) -> datetime.date:
        """
        週のi番目の日を取得
        
        Parameters
        ----------
        day_0: datetime.date
            基準となる日
        i: int
            週の何日目か（1 <= i <= 7）
        
        Returns
        -------
        day: datetime.date 
            週のi番目の日
        """
        year = day_0.isocalendar()[0]
        week = day_0.isocalendar()[1]
        day = datetime.date.fromisocalendar(year, week, i)
        return day
    
    today = datetime.date.today()
    date_opts = [i_day(today, i) for i in (6,7)]

    # 日付の選択、ページ遷移
    def add_date_checkboxes(dates: list) -> list:
        """
        日付の選択肢を追加
        
        Parameters
        ----------
        dates : list[datetime.date]
            日付のリスト
        
        Return
        ------
        keys: list[str]
            keyのリスト
        """
        weekday2jp = "月火水木金土日"
        keys = []
        for date in dates:
            day_jp = weekday2jp[date.weekday()]
            date_str = str(date)
            label = f'{date_str.replace("-", "/")}({day_jp})'
            key = date_str.replace("-", "")
            st.checkbox(label=label, key=key)
            keys.append(key)
        st.date_input(label="土日以外もみる", 
                     value=date_opts[0],  
                     key="select_date")

        return keys
    st.write("日付の選択")
    with st.form(key="form"):
        keys = add_date_checkboxes(date_opts)
        st.form_submit_button(label="OK", on_click=next_page)

def page2():
    def next_page():
        st.session_state.race_id = [key for key in keys if st.session_state[key]]
        if len(st.session_state.race_id) >= 1:
            st.session_state.page += 1
        else:
            st.warning("レースを選択して下さい")

    # 開催レースデータの取得
    def get_races_held(dates: list) -> dict:
        """
        開催レースのデータ取得
        
        Parameters
        ----------
        dates: list[str]
            開催日のリスト ["20200101", "20200102"]
        
        Returns
        -------
        data: dict
            開催レースデータ 
                {
                    ("東京", "1", "5"): [{"R": 1, "name": "日本ダービー", "race_id": "11111"}, {...}, ...],
                    ...
                }
        """
        data = {}
        # 開催日ごと
        for date in dates:
            url=f"https://yoso.netkeiba.com/?pid=race_list&kaisai_date={date}"
            res = requests.get(url)
            res.encoding = "EUC-JP"
            soup = BeautifulSoup(res.text, "html.parser")
            soups = soup.find_all("div", attrs={"class": "RaceList"})#.find("div", attrs={"class": "Jyo"})
            # 競馬場ごと
            for soup_i in soups: 
                headder1 = soup_i.find(class_="Jyo").text
                place = headder1[:2]
                kai = headder1[headder1.index("(")+1:headder1.index("回")]
                nichi = headder1[headder1.index("回")+1:headder1.index("日目")]
                key = (place, kai, nichi)
                race_data = []
                # レースごと
                for race_i in soup_i.find("div", class_="RaceList_Main").find_all("li"):
                    text = race_i.text.replace("\n","")
                    R = int(text[:text.index("R")])
                    url = race_i.find("a").get("href")
                    name = text[text.index("R")+1: text.find(":")-2]
                    race_id = re.sub(r"\D", "", url)

                    race_data.append(
                        {
                            "R": R,
                            "name": name,
                            "race_id": race_id,
                        }
                    )
                data[key] = race_data
        return data

    with st.spinner("レース取得中..."):
        dates = st.session_state.date
        data_races_held = get_races_held(dates)

    # レースデータの修正
    st.session_state.data = {}
    for place in data_races_held:
        for race_data in data_races_held[place]:
            race_id = race_data["race_id"]
            st.session_state.data[race_id] = {
                "R": race_data["R"],
                "name": race_data["name"],
                "開催地": place[0],
                "回": place[1],
                "日目": place[2],
            }

    # 日付の選択、ページ遷移
    def add_race_checkboxes(data_races_held: list) -> list:
        """
        日付の選択肢を追加
        
        Parameters
        ----------
        data_races_held : list[datetime.date]
            開催レースデータ
        
        Return
        ------
        keys: list[str]
            keyのリスト
        """
        columns = st.columns(len(data_races_held))
        keys = []
        for i, place in enumerate(data_races_held):
            with columns[i]:
                st.header(f'{place[0]}{place[1]}回{place[2]}日目')
                for race_data in data_races_held[place]:
                    label = f'{race_data["R"]}R {race_data["name"]}'
                    key = race_data["race_id"]
                    st.checkbox(label=label, key=key)
                    keys.append(key)
        return keys
    st.write("レースの選択")
    with st.form(key="form2"):
        if not data_races_held:
            st.error("開催日を指定してください")
        keys = add_race_checkboxes(data_races_held)
        st.form_submit_button(label="OK", on_click=next_page)

def page3():
    def next_page():
        st.session_state.page += 1

    # レースデータのクロール
    from scrape import get_new_race_csv, get_new_horse_csv
    from prepocess import Name
    race_id = st.session_state.race_id
    df_test_race, df_test_result = get_new_race_csv(race_id)
    df_test_prof, df_test_form = get_new_horse_csv(df_test_result)
    # nm = dill.load(open("nm.sav", 'rb'))
    nm = Name()

    # 前処理
    from prepocess import preprocess_race, preprocess_result, preprocess_prof, preprocess_form, preprocess_v2, get_shift_form_v2, df_concat, get_pp2, encoded

    with st.spinner("データ処理中..."):
        df_test_race = preprocess_race(df_test_race, nm)
        df_test_result = preprocess_result(df_test_result, nm)
        df_test_prof = preprocess_prof(df_test_prof, nm)
        df_test_form = preprocess_form(df_test_form, nm)    
        df_test_form = preprocess_v2(df_test_form)
        # SHIFT
        n_shift = 3
        df_test_form = df_test_form.reset_index()
        df_test_form_shifted = get_shift_form_v2(df_test_form.copy(), n_shift)
        # CONCAT
        df_test = df_concat(df_test_result, df_test_race, df_test_prof, df_test_form_shifted)
        # LABEL ENCODING
        df_test = get_pp2(df_test, n_shift)
        df_test_encoded = encoded(df_test.copy(), download=True)
        # 利用しないデータの除去
        is_test = (df_test["race_id"] != df_test["race_id_1"]) & (df_test["race_id"] != df_test["race_id_2"]) & (df_test["race_id"] != df_test["race_id_3"])
        past_race = False
        if past_race:
            is_test &= (df_test.index % 2 == 1)
        df_pred_test = df_test[is_test]
        df_pred_test_encoded = df_test_encoded[is_test]
    
    # モデルの取り込み
    with st.spinner('モデルのロード中...'):
        models_data = {}
        for key in os.listdir("model"):
            models_data[key] = dill.load(open(f"model/{key}/metadata.sav", 'rb'))
            models_data[key]["models"] = []
            for model_path in os.listdir(f"model/{key}/models"):
                model = dill.load(open(f"model/{key}/models/{model_path}", 'rb'))
                models_data[key]["models"].append(model)


    # 推論
    n = len(models_data)
    xxx = st.text(f"infer {0}/{n}")
    bar = st.progress(0.0)
    i = 0

    def test(models, _df, _df_encoded, X_cols, TARGET, dev_reverse):
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


    for learn_level in [1,2,3]:
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
                
    st.session_state["df_pred_test_groupby"] = df_pred_test.groupby("race_id")
    st.session_state["nm"] = nm
    

    next_page()
    st.experimental_rerun()

def page4():
    def display_pred(df_pred_test, display_cols, sort_key):

        df_target = df_pred_test[["枠番", "馬番"] + display_cols].fillna(0).astype({"馬番":int, "枠番":int})
        df_target.index = df_pred_test["horse_id"].map(lambda x: nm[x]).values
        df_target["スコア"] = df_target[
            [f"{col}ensemble_dev" for col in ["勝利", "連対",  "複勝", "着差", "タイム"]]
            ].mean(axis=1)
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


        for col in df_target.columns:
            if "log" in col:
                df_target[col] = 10 * df_target[col]
            if "標準着差" in col and "pred" in col:
                df_target[col] = 10000 * df_target[col]
            if any(key in col for key in ["勝利", "連対", "複勝", "掲示板"]) and "pred" in col:
                df_target[col] = 100 * df_target[col]

        display_cols = [col for col in display_cols if "pred" not in col]

        race_id = df_pred_test.iloc[0].race_id
        df_target = (
                df_target.style
                    .set_precision(1)
                    .apply(back_color)
                    .apply(wakuban_color, subset=['枠番'])
                    .apply(umaban_color, subset=['馬番'])
                    .applymap(highlight, subset=display_cols+["スコア"])
                    .apply(max_bold, subset=display_cols+["スコア"])
                    .set_caption(f"{int(race_id[12:14])}回{df_pred_test.iloc[0].開催地}{int(race_id[14:16])}日目 {int(race_id[16:18])}R {str(nm[race_id]).replace('()','')} {df_pred_test.iloc[0].芝ダ} {df_pred_test.iloc[0].距離:.0f}m")    
            )
        return df_target

    sort_key = "スコア"
    nm = st.session_state["nm"]

    display_cols_opt = ["勝利", "連対",  "複勝", "着差", "タイム", "単勝払戻", "複勝払戻"]
    display_cols = display_cols_opt
    display_cols = [f"{col}ensemble_dev" for col in display_cols]

    for race_id, df_pred_testi in st.session_state["df_pred_test_groupby"]:
        race_name = st.session_state.data[re.sub(r"\D", "", race_id)]
        race_data = df_pred_testi.iloc[0]
        tenki = {'小雨': "☂️",
                    '小雪': "⛄️",
                    '晴': "☀️",
                    '曇': "☁️",
                    '雨': "☔︎",
                    '雪': "☃️"}
        st.write(f'{race_name["開催地"]} {race_name["R"]}R {race_name["name"]}\t{race_data.芝ダ}{race_data.距離:.0f}m {tenki[race_data.天気]} {race_data.馬場}')
        n = df_pred_testi.shape[0] + 1
        df_display = display_pred(df_pred_testi, display_cols, sort_key)
        st.dataframe(df_display, height=35*n)

if "page" not in st.session_state:
    st.session_state.page = 1

if "page" in st.session_state:
    # 日付の選択
    if st.session_state.page == 1:
        page1()

    # レースの選択
    elif st.session_state.page == 2:
        page2()

    # レースの取得・推論・表示
    elif st.session_state.page == 3:
        page3()

    elif st.session_state.page == 4:
        page4()