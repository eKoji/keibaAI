from bs4 import BeautifulSoup
import requests
import streamlit as st
import pandas as pd
import numpy as np
import time



def get_new_race_csv(race_id_list):
    race_list = []
    result_list = []
    def add_new_race_info(race_id):
        time.sleep(0.5)
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
        # print(soup.find("dd", attrs={"class": "Acctive"}))
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

    # formが空の場合に対応（新馬戦のみの選択）
    cols = set(['index', '着差', 'race_id', 'horse_id', '日付', '開催', '天気', 'R', 'レース名', '映像', '頭数', '枠番', '馬番', 'オッズ', '人気', '着順', '騎手', '斤量', '距離', '馬場', '馬場指数', 'タイム', 'ﾀｲﾑ指数', '通過', 'ペース', '上り', '馬体重', '厩舎ｺﾒﾝﾄ', '備考', '勝ち馬(2着馬)', '賞金', 'jockey_id'])
    df_form_cols = set(df_form.columns)
    for col in cols-df_form_cols:
        df_form[col] = np.nan
    return df_prof, df_form

