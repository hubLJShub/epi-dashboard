import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def set_season_start_week(data, epi, window_years = 3, season_years=1, half_year_weeks = 26):
    results = []

    # 처음 3년 동안의 데이터를 사용하여 threshold 설정
    WEEKS_IN_YEAR = 52
    window_weeks = window_years * WEEKS_IN_YEAR
    season_weeks = season_years * WEEKS_IN_YEAR
    first_start_date = data["Date"].min() + pd.DateOffset(years=window_years)
    start_idx = data[data["Date"] >= first_start_date].index[0]

    season_id = 1

    while True:
        season_start = data.loc[start_idx, "Date"]
        season_end = season_start + pd.Timedelta(weeks=season_weeks)

        # threshold 계산 (직전 3년)
        ref_start = season_start - pd.Timedelta(weeks=window_weeks)
        ref_data = data[(data["Date"] >= ref_start) & (data["Date"] < season_start)]
        # 데이터 부족 → 종료
        if ref_data["Date"].dt.year.nunique() < window_years:
            break

        threshold = ref_data[epi].mean()

        # 다음 시즌 설정 (1년)
        season_data = data[(data["Date"] >= season_start) & (data["Date"] < season_end)]
        # 데이터 부족 → 종료
        if season_data.empty:
            break

        # threshold 초과 = 유행 기간
        season_data = season_data.copy()
        season_data["exceed"] = season_data[epi] > threshold

        # 유행 기간 데이터만 추출
        epidemic_data = season_data[season_data["exceed"]]

        # 유행 없음 → 다음 시즌
        if epidemic_data.empty:
            next_idx = data[data["Date"] >= season_end].index
            if len(next_idx) == 0:
                break

            start_idx = next_idx[0]
            season_id += 1
            year_id +=1
            continue

        # 유행 기간 중 최대 peak 하나만 선택
        peak = epidemic_data.loc[epidemic_data[epi].idxmax()]
        peak_date = peak["Date"]

        new_start = peak_date - pd.Timedelta(weeks=26)
        year_id = data[data["Date"] >= new_start].index[0]

        # 결과 저장
        results.append({
            "season":  data.loc[year_id, 'Year'],
            "season_start": season_start,
            "threshold": threshold,
            "peak_date": peak_date,
            "start_week": data.loc[start_idx, "Week"],
            "peak_week": peak["Week"],
            "peak_epi": peak[epi],
            "epi_dates": season_data.loc[season_data["exceed"] == 1, "Date"].tolist(),
            "epi_weeks": season_data.loc[season_data["exceed"] == 1, "Week"].tolist(),
            "threshold": threshold
        })

        # 가장 큰 peak 1개 ± 6개월 → 새로운 유행 구간 설정
        epi_end = peak_date + pd.Timedelta(weeks=half_year_weeks)
        # print('epi end: ', str(epi_end))

        # 다음 시즌 시작점 업데이트
        next_idx = data[data["Date"] >= epi_end].index
        if len(next_idx) == 0:
            break

        start_idx = next_idx[0]
        season_id += 1
        year_id +=1
    results_df = pd.DataFrame(results)
    return results_df

def filter_data(data, seasons, start_week=24):
    df = data.copy()

    # 시즌 데이터만 사용
    df['Season'] = np.where(df['Week'] >= start_week, df['Year'], df['Year'] - 1)
    df = df[df['Season'].isin(seasons)]

    # train / test 구분
    season_counts = df.groupby('Season')['Week'].nunique()
    full_seasons = season_counts[season_counts >= 52].index
    df['set'] = np.where(df['Season'].isin(full_seasons), 'train', 'test')

    return df

def Observation_period(outbreak_result, year, peak_week, len_period=40):
    df_sorted = outbreak_result.sort_values(['Year', 'Week']).reset_index(drop=True)
    target_idx = df_sorted[(df_sorted['Season'] == year) & (df_sorted['Week'] == peak_week)].index
    
    if len(target_idx) == 0:
        print(f"Warning: {year}-{year+1}시즌 {peak_week}주차 데이터를 찾을 수 없습니다.")
        return None
    
    idx = target_idx[0]
    start_idx = max(0, idx - len_period+1)
    d = df_sorted.iloc[start_idx : idx + 1].copy()
    
    return d.reset_index(drop=True)

# def Module_HockeyStick(x, y, k, ty='linear'):
#     match ty:
#         case 'linear':
#             y_obs = y
#         case 'exponential':
#             eps=1e-8
#             y_obs = np.log(np.maximum(y, eps))
#         case _:
#             warnings.warn(
#                 f"[Module_HockeyStick] Unknown type='{ty}'. "
#                 "Supported types are 'linear' or 'exponential'. "
#                 "Defaulting to 'linear'.",
#                 UserWarning
#             )
#             ty = 'linear'
#             y_obs = y
    
#     hinge = np.maximum(0.0, x - k)
#     X = np.column_stack([np.ones_like(x), hinge])
#     beta, *_ = np.linalg.lstsq(X, y_obs, rcond=None)
    
#     yhat_obs = X @ beta
#     sse = float(np.sum((y_obs - yhat_obs) ** 2))

#     yhat = np.exp(yhat_obs) if ty == 'exponential' else yhat_obs
#     beta[0]=np.exp(beta[0]) if ty == 'exponential' else beta[0]
    
#     return beta, yhat, sse

def Module_HockeyStick(x, y, k, ty='linear'):
    # match-case 대신 if-elif-else 사용 (파이썬 구버전 호환)
    if ty == 'linear':
        y_obs = y
    elif ty == 'exponential':
        eps = 1e-8
        y_obs = np.log(np.maximum(y, eps))
    else:
        warnings.warn(
            f"[Module_HockeyStick] Unknown type='{ty}'. "
            "Supported types are 'linear' or 'exponential'. "
            "Defaulting to 'linear'.",
            UserWarning
        )
        ty = 'linear'
        y_obs = y
    
    hinge = np.maximum(0.0, x - k)
    X = np.column_stack([np.ones_like(x), hinge])
    beta, *_ = np.linalg.lstsq(X, y_obs, rcond=None)
    
    yhat_obs = X @ beta
    sse = float(np.sum((y_obs - yhat_obs) ** 2))

    yhat = np.exp(yhat_obs) if ty == 'exponential' else yhat_obs
    beta[0] = np.exp(beta[0]) if ty == 'exponential' else beta[0]
    
    return beta, yhat, sse

def Run_HockeyStick(d, epi, min_seg_len=4, ty='linear'):
    x = d.index.to_numpy(dtype=float)
    # x = d['Week'].to_numpy(dtype=float)
    y = d[epi].to_numpy(dtype=float)
    n = len(x)
    
    if n < 2 * min_seg_len + 1:
        return np.nan, None
    k_candidates = x[min_seg_len: n - min_seg_len]

    best = {'k': None, 'sse': np.inf, 'beta': None, 'yhat': None}
    for k in k_candidates:
        beta, yhat, sse = Module_HockeyStick(x, y, k, ty)
        if sse < best['sse']:
            best.update({'k': int(k), 'sse': sse, 'beta': beta, 'yhat': yhat})

    b0, b1 = best['beta']
    best['b0'] = float(b0)
    best['b1'] = float(b1)
    best['slope_before'] = 0.0
    best['slope_after']  = float(b1)

    return best['k'], best

def assign_season(row, base_idx=30):
    year = int(row['Year'])
    week = int(row['Week'])
    return year if week >= base_idx else year - 1

def hockey_stick_regression(data, epi, HockeyStick_type, years):
    from scipy.signal import find_peaks
    rows = []
    info_by_year = {}

    # peak 시점 찾기
    peak_period = []
    peak_date = []
    for season in years:
        season_data = data[data['Season'] == season].reset_index(drop=True)
        peaks = find_peaks(season_data[epi], distance=30)
        peak_period.append(peaks[0][0]+1)
        peak_date.append(season_data.iloc[peaks[0][0]]['Date'])
    peak_df = data[data['Date'].isin(peak_date)].sort_values('Year').reset_index(drop=True)

    for i, yr in enumerate(years):
        row = peak_df.loc[peak_df['Season'] == yr, ['Season', 'Week']].iloc[0]
        target_year = int(row['Season'])
        target_week = int(row['Week'])
    
        d = Observation_period(data, target_year, target_week, peak_period[i])

        k, info = Run_HockeyStick(d, epi, min_seg_len=4, ty=HockeyStick_type)
        if pd.isna(k):
            continue
            
        # k가 정상적인 숫자일 때만 아래 코드가 실행됩니다.
        rows.append({
            'Season':               yr,
            'peak_year':            target_year,
            'peak_week':            target_week,
            'hockey_break_year': int(d.iloc[int(k)]['Year']),
            'hockey_break_week': int(d.iloc[int(k)]['Week']),
            'b0':                   None if info is None else info['b0'],
            'b1':                   None if info is None else info['b1'],
            'SSE':                  None if info is None else info['sse'],
            'n_points':             len(d)
        })
        info_by_year[yr] = (d, info)

        # rows.append({
        #     'Season':               yr,
        #     'peak_year':            target_year,
        #     'peak_week':            target_week,
        #     # 'hockey_break_year':    int(d[d.index == k]['Year'].iloc[0]),
        #     # 'hockey_break_week':    int(d[d.index == k]['Week'].iloc[0]),
        #     'hockey_break_year': int(d.iloc[int(k)]['Year']),
        #     'hockey_break_week': int(d.iloc[int(k)]['Week']),
        #     'b0':                   None if info is None else info['b0'],
        #     'b1':                   None if info is None else info['b1'],
        #     'SSE':                  None if info is None else info['sse'],
        #     'n_points':             len(d)
        # })
        info_by_year[yr] = (d, info)
    hockey_peak_df = pd.DataFrame(rows)

    break_dates = []

    for index, row in hockey_peak_df.iterrows():
        year = row['hockey_break_year']
        week = row['hockey_break_week']
        
        match = data[(data['Year'] == year) & (data['Week'] == week)]['Date']
        if not match.empty:
            break_dates.append(pd.to_datetime(match.values[0]))
            
    return break_dates, hockey_peak_df