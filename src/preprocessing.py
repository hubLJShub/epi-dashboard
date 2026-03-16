
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# def cumulative_sum(data, epi, season_start_week=38):
#     data["season"] = np.where(data['Week'] >= season_start_week, data['Year'], data['Year'] - 1)
#     data["cusum"] = (data[epi] - data.groupby("season")[epi].transform("mean")).groupby(data["season"]).cumsum()
#     season_counts = data.groupby("season").size()
#     valid_seasons = season_counts[season_counts >= 52].index
#     data["valid_season"] = data["season"].isin(valid_seasons)
#     data["cusum"] = data["cusum"].where(data["valid_season"], np.nan)
#     return data
def cumulative_sum(data, epi, season_start_week = 24):
    data["cusum"] = (data[epi] - data.groupby("Season")[epi].transform("mean")).groupby(data["Season"]).cumsum()
    return data

# def cumulative_sum_3years(data, epi, season_start_week=38, n_years=3, covid_seasons=[2020, 2021]):
#     data["season"] = np.where(data['Week'] >= season_start_week, data['Year'], data['Year'] - 1)
#     season_counts = data.groupby("season").size()
#     valid_seasons = season_counts[season_counts >= 3].index
#     data = data[data["season"].isin(valid_seasons)]

#     if covid_seasons is not None:
#         data = data[~data["season"].isin(covid_seasons)]
    
#     season_mean = data.groupby("season")[epi].mean()
#     season_baseline = (season_mean.shift(1).rolling(window=n_years, min_periods=n_years).mean())

#     data = data.merge(season_baseline.rename("baseline"), left_on="season", right_index=True, how="left")
#     data["diff"] = data[epi] - data["baseline"]
#     data["cusum"] = (data.groupby("season")["diff"].cumsum())
#     return data
# 과거 3년 동안의 데이터의 평균을 이용해 cusum을 계산하는 함수

def cumulative_sum_3years(data, epi, season_start_week = 24, n_years = 3):
    # 시즌별 과거 3년의 평균값 저장
    season_mean = data.groupby("Season")[epi].mean()
    season_baseline = (season_mean.shift(1).rolling(window=n_years, min_periods=n_years).mean())

    # cusum 값 계산 (시즌이 시작될 때마다 0으로 reset)
    data = data.merge(season_baseline.rename("baseline"), left_on="Season", right_index=True, how="left")
    data["diff"] = data[epi] - data["baseline"]
    data["cusum"] = (data.groupby("Season")["diff"].cumsum())
    return data

def window_sample(data, itv, epi):
    scaler = MinMaxScaler()
    temp_df = {}
    if len(data) < itv:
        return {}
        
    for i in range(len(data) - itv + 1):
        temp = data.iloc[i:i+itv].copy()
        temp.reset_index(drop=True, inplace=True)
        # 기울기용 Scaled Data
        temp['N_total'] = scaler.fit_transform(temp[[epi]])
        key = i
        temp_df[key] = temp
    return temp_df

# 회귀 분석 및 누적합 평균 계산하는 함수
def window_sample_feature(df_dict):
    import numpy as np
    from sklearn.linear_model import LinearRegression

    temp_lr = {}
    for key, temp in df_dict.items():
        temp_lr[key] = {}

        # --- 전체 구간 회귀 ---
        lr = LinearRegression()
        X_data = temp.num.values.reshape(-1, 1)
        y_data = temp.N_total.values
        lr.fit(X_data, y_data)
        lr_pred = lr.predict(X_data)
        temp_lr[key]['LR'] = lr
        temp_lr[key]['LR_val'] = lr_pred

        # --- 누적합 ---
        temp_lr[key]['CS'] = temp['cusum'].values
        temp_lr[key]['CS_mean'] = np.mean(temp['cusum'].values)

    return temp_lr
# def make_df(data, window_size, epi):
#     # 1. 윈도우 생성
#     dic1 = window_sample(data, window_size, epi)
#     if not dic1:
#         return pd.DataFrame()
#     # 2. 특징 추출 (epi 컬럼명 전달)
#     dic2 = window_sample_feature(dic1, epi)
    
#     # 3. 데이터프레임 변환
#     cols = ['mean', 'slope', 'CS_mean']
#     rows = []
    
#     sorted_keys = sorted(dic1.keys())
    
#     for key in sorted_keys:
#         add = [
#             dic2[key]['mean'],
#             dic2[key]['slope'],
#             dic2[key]['CS_mean'],
#         ]
#         rows.append(add)
        
#     features = pd.DataFrame(rows, columns=cols)
#     return features
def make_df(data, itv, epi): 
    dic1 = window_sample(data, itv, epi)
    dic2 = window_sample_feature(dic1)
    
    cols = ['data_num','mean', 'slope', 'CS_mean']
    rows = [] # 데이터를 담을 리스트
    
    for key in dic1.keys():
        data_tmp = dic1[key]
        add = [
            key,
            data_tmp[epi].mean(),
            dic2[key]['LR'].coef_[0],
            dic2[key]['CS_mean'],
        ]
        rows.append(add)
        
    features = pd.DataFrame(rows, columns=cols)
    return features
# def make_raw(data, date, sample_window, epi, covid_start, covid_end):
#     data_df = data[data['Date'] <= date].copy()
    
#     date_dt = pd.to_datetime(date)
#     covid_start_dt = pd.to_datetime(covid_start)
#     covid_end_dt = pd.to_datetime(covid_end)
    
#     n = sample_window
    
#     if date_dt.year >= 2022:
#         # BP
#         data_bp = data_df[data_df['Date'] <= covid_start_dt].reset_index(drop=True)
#         # AP
#         data_ap = data_df[data_df['Date'] >= covid_end_dt].reset_index(drop=True)
        
#         data_all = pd.concat([data_bp.iloc[n-1:], data_ap.iloc[n-1:]], ignore_index=True)
        
#         df1 = make_df(data_bp, n, epi)
#         df2 = make_df(data_ap, n, epi)
        
#         df = pd.concat([df1, df2], ignore_index=True)
#     else:
#         data_df = data_df.reset_index(drop=True)
#         df = make_df(data_df, n, epi)
#         data_all = data_df.iloc[n-1:].reset_index(drop=True)
        
#     df.reset_index(drop=True, inplace=True)
#     df['data_num'] = df.index
#     data_all.reset_index(drop=True, inplace=True)

#     return df, data_all

def make_raw(data, set_type, sample_window, epi):
    if set_type == 'train':
        data_df = data[data['set']==set_type]
    elif set_type == 'test':
        test_idx = data.index[data['set'] == 'test']
        start_idx = max(0, test_idx.min() - sample_window + 1)
        data_df = data.iloc[start_idx:]
    date_dt = data_df['Date'].iloc[-1]
    n = sample_window

    # df = make_df(data_df,n,epi)
    # data_all = data_df
    if 2020 in np.unique(data['Season']):
        df = make_df(data_df,n,epi)
        data_all = data_df[n-1:].reset_index(drop=True)
    else:
        data_bp = data_df[data_df['Season']<=2020]
        data_ap = data_df[data_df['Season']>2020]
        data_all = pd.concat([data_bp[n-1:], data_ap[n-1:]], ignore_index=True)
        df1 = make_df(data_bp,n,epi)
        df2 = make_df(data_ap,n,epi)
        df = pd.concat([df1, df2])
    
    df.reset_index(drop=True, inplace=True)
    df['data_num'] = df.index

    return df, data_all