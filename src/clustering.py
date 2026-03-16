
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import joblib

# def K_means_clustering(df, X_scaled=None):
#     if X_scaled is None:
#         X_train = df[['slope', 'mean', 'CS_mean']].copy()
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X_train)

#     model = KMeans(random_state=727, n_init=10)
#     plt.figure()
#     visualizer = KElbowVisualizer(model, k=(2, 10), metric='silhouette', timings=True)
#     visualizer.fit(X_scaled)
#     k = visualizer.elbow_value_
#     plt.close()
    
#     kmeans = KMeans(n_clusters=k, random_state=727, n_init=10)
#     labels = kmeans.fit_predict(X_scaled)
#     df['label'] = labels
    
#     RI_means = df.groupby('label')['slope'].mean()
#     label_order = RI_means.sort_values().index
#     custom_order = {old_label: new_label for new_label, old_label in enumerate(label_order)}
#     df['label'] = df['label'].map(custom_order)
    
#     result_data = df.sort_values('data_num').reset_index(drop=True)
#     result_data['label'] = result_data['label'].astype(str)

#     return result_data, kmeans

# def find_warning_periods(result_data, warning_label=1):
#     row_start = []
#     row_end = []
    
#     result_data['label'] = result_data['label'].astype(str)
#     warning_label = str(warning_label)
    
#     if (result_data['label'] == warning_label).sum() == 0:
#         return []

#     if result_data.loc[0, 'label'] == warning_label:
#         row_start.append(result_data.index[0])

#     for i in range(1, len(result_data)):
#         if result_data.loc[i, 'label'] == warning_label:
#             if result_data.loc[i-1, 'label'] != warning_label:
#                 row_start.append(result_data.index[i])
    
#     for i in range(len(result_data)-1):
#         if result_data.loc[i+1, 'label'] != warning_label:
#             if result_data.loc[i, 'label'] == warning_label:
#                 row_end.append(result_data.index[i])
    
#     if result_data.loc[len(result_data)-1, 'label'] == warning_label:
#         row_end.append(result_data.index[-1])  

#     last_warning_index = result_data[result_data['label'] == warning_label].tail(1).index[0]
#     if len(row_start) == 0 or row_start[-1] != last_warning_index:
#         row_start.append(last_warning_index)
        
#     row_2 = []
#     min_len = min(len(row_start), len(row_end))
#     for i in range(min_len):
#         if row_end[i] - row_start[i] >= 2:
#              row_2.append(row_start[i])
#     return row_2

# def train_bootstrap_ensemble(df_train, B=300):
#     boot_ensemble = []
#     feature_cols = ['slope', 'mean', 'CS_mean']
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(df_train[feature_cols])
    
#     model = KMeans(random_state=0, n_init=10)
#     plt.figure(figsize=(8, 2))
#     visualizer = KElbowVisualizer(model, k=(2, 10), metric='silhouette', timings=True)
#     visualizer.fit(X_train_scaled)
#     k_best = visualizer.elbow_value_
#     plt.close()
    
#     print(f"Training Ensemble of {B} models...")
#     for b in range(B):
#         resampled_df = df_train.sample(n=len(df_train), replace=True, random_state=b)
#         X_resamp = scaler.transform(resampled_df[feature_cols])
        
#         kmeans = KMeans(n_clusters=k_best, random_state=b, n_init=10)
#         resampled_df['temp_label'] = kmeans.fit_predict(X_resamp)
        
#         RI_means = resampled_df.groupby('temp_label')['slope'].mean()
#         label_order = RI_means.sort_values().index
#         custom_order = {old_label: new_label for new_label, old_label in enumerate(label_order)}
        
#         boot_ensemble.append({'model': kmeans, 'order': custom_order})
        
#     return boot_ensemble, scaler

# def analyze_train_distribution(df_train, data_all_train, boot_ensemble, scaler, outbreak_season, ED_date):
#     all_iterations_ed_dates = []
#     feature_cols = ['slope', 'mean', 'CS_mean']
#     X_orig_scaled = scaler.transform(df_train[feature_cols])
#     date_df = pd.DataFrame()
    
#     for item in boot_ensemble:
#         model = item['model']
#         order = item['order']
        
#         preds = model.predict(X_orig_scaled)
#         temp_df = df_train.copy()
#         temp_df['label'] = pd.Series(preds).map(order).values
        
#         try:
#             row_2 = find_warning_periods(temp_df, warning_label=temp_df['label'].max())
#             if len(row_2) > 0:
#                 ed_nums = np.array(row_2)
#                 detection_dates_series = pd.to_datetime(data_all_train.loc[ed_nums.astype(int) + 2, 'Date'])
                
#                 if not outbreak_season:
#                     all_iterations_ed_dates.append(detection_dates_series.tolist())
#                 else:
#                     post_sept_dates = detection_dates_series[detection_dates_series.dt.month >= outbreak_season]
#                     if not post_sept_dates.empty:
#                         first_post_sept_per_year = post_sept_dates.drop_duplicates().groupby(post_sept_dates.dt.year).min()
#                         row_date = first_post_sept_per_year
#                         tmp = (pd.DataFrame({"date": row_date}).assign(year=lambda x: x["date"].dt.year.astype(str),idx=lambda x: x.groupby(x["date"].dt.year).cumcount() + 1))
#                         tmp["col"] = tmp["year"] + "_" + tmp["idx"].astype(str)
#                         row = tmp.set_index("col")["date"].to_frame().T
#                         date_df = pd.concat([date_df, row], ignore_index=True)
#                         all_iterations_ed_dates.append(first_post_sept_per_year.tolist())
#                     else:
#                         all_iterations_ed_dates.append([])
#             else:
#                 all_iterations_ed_dates.append([])
#         except Exception as e:
#             all_iterations_ed_dates.append([])
#             continue
            
#     return all_iterations_ed_dates, date_df
# # src/clustering.py

# # def analyze_train_distribution(df_train, data_all_train, boot_ensemble, scaler, outbreak_season):
# #     all_iterations_ed_dates = []
# #     feature_cols = ['slope', 'mean', 'CS_mean']
# #     X_orig_scaled = scaler.transform(df_train[feature_cols])
# #     date_df = pd.DataFrame()
    
# #     for item in boot_ensemble:
# #         model = item['model']
# #         order = item['order']
        
# #         preds = model.predict(X_orig_scaled)
# #         temp_df = df_train.copy()
# #         temp_df['label'] = pd.Series(preds).map(order).values
        
# #         try:
# #             row_2 = find_warning_periods(temp_df, warning_label=temp_df['label'].max())
            
# #             if len(row_2) > 0:
# #                 ed_nums = np.array(row_2)
# #                 # 인덱스를 실제 날짜로 변환
# #                 detection_dates_series = pd.to_datetime(data_all_train.loc[ed_nums.astype(int) + 2, 'Date'])
                
# #                 # ---------------------------------------------------------
# #                 # [수정 핵심] 단순 월 비교가 아니라 '시즌(Season)'별로 그룹핑
# #                 # ---------------------------------------------------------
# #                 if outbreak_season: # outbreak_season 설정이 있을 때 (예: 9)
# #                     # 임시 데이터프레임 생성
# #                     temp_dates = pd.DataFrame({'Date': detection_dates_series})
                    
# #                     # 시즌 계산: 9월 이상이면 현재 연도, 아니면 작년 연도
# #                     temp_dates['Season'] = np.where(
# #                         temp_dates['Date'].dt.month >= outbreak_season,
# #                         temp_dates['Date'].dt.year,
# #                         temp_dates['Date'].dt.year - 1
# #                     )
                    
# #                     # 각 시즌별로 가장 빠른(min) 날짜만 추출
# #                     first_detection_per_season = temp_dates.groupby('Season')['Date'].min()
                    
# #                     # 결과 저장을 위한 데이터프레임 구성 (시각화용)
# #                     row_date = first_detection_per_season
# #                     tmp = pd.DataFrame({"date": row_date.values})
# #                     tmp["year"] = row_date.index.astype(str) # Season을 year로 사용
# #                     tmp["idx"] = tmp.groupby("year").cumcount() + 1 # 혹시 모를 중복 방지
# #                     tmp["col"] = tmp["year"] + "_" + tmp["idx"].astype(str)
                    
# #                     row = tmp.set_index("col")["date"].to_frame().T
# #                     date_df = pd.concat([date_df, row], ignore_index=True)
                    
# #                     # 리스트에 추가
# #                     all_iterations_ed_dates.append(first_detection_per_season.tolist())
                    
# #                 else: 
# #                     # outbreak_season 설정이 없을 때 (기존 로직 유지)
# #                     all_iterations_ed_dates.append(detection_dates_series.tolist())
                    
# #             else:
# #                 all_iterations_ed_dates.append([])
                
# #         except Exception as e:
# #             all_iterations_ed_dates.append([])
# #             continue
            
# #     return all_iterations_ed_dates, date_df

# # def analyze_train_distribution(df_train, data_all_train, boot_ensemble, scaler):
# #     all_iterations_ed_dates = []
# #     feature_cols = ['slope', 'mean', 'CS_mean']
# #     X_orig_scaled = scaler.transform(df_train[feature_cols])
# #     date_df = pd.DataFrame()
    
# #     for item in boot_ensemble:
# #         model = item['model']
# #         order = item['order']
        
# #         preds = model.predict(X_orig_scaled)
# #         temp_df = df_train.copy()
# #         temp_df['label'] = pd.Series(preds).map(order).values
        
# #         try:
# #             row_2 = find_warning_periods(temp_df, warning_label=temp_df['label'].max())
            
# #             if len(row_2) > 0:
# #                 ed_nums = np.array(row_2)
                
# #                 # [수정] 인덱스에 해당하는 행 전체를 가져옴 (Date, Season 정보 포함)
# #                 # data_all_train은 make_raw에서 왔으므로 Season 컬럼을 가지고 있어야 함
# #                 detected_rows = data_all_train.loc[ed_nums.astype(int) + 2].copy()
                
# #                 # ---------------------------------------------------------
# #                 # [수정 완료] Season 컬럼을 기준으로 그룹핑 (자동 시즌 적용)
# #                 # ---------------------------------------------------------
# #                 if 'Season' in detected_rows.columns:
# #                     # 각 시즌(Season)별로 가장 빠른(min) 날짜만 추출
# #                     first_detection_per_season = detected_rows.groupby('Season')['Date'].min()
                    
# #                     # 결과 저장을 위한 데이터프레임 구성 (시각화용)
# #                     row_date = first_detection_per_season
                    
# #                     tmp = pd.DataFrame({"date": row_date.values})
# #                     tmp["year"] = row_date.index.astype(str) # Index가 Season(연도)임
# #                     tmp["idx"] = tmp.groupby("year").cumcount() + 1
# #                     tmp["col"] = tmp["year"] + "_" + tmp["idx"].astype(str)
                    
# #                     row = tmp.set_index("col")["date"].to_frame().T
# #                     date_df = pd.concat([date_df, row], ignore_index=True)
                    
# #                     # 리스트에 추가 (시즌별 최초 탐지일들)
# #                     all_iterations_ed_dates.append(first_detection_per_season.tolist())
                    
# #                 else: 
# #                     # 만약 Season 컬럼이 없다면 단순히 모든 탐지 날짜를 저장 (Fallback)
# #                     detection_dates = pd.to_datetime(detected_rows['Date'])
# #                     all_iterations_ed_dates.append(detection_dates.tolist())
                    
# #             else:
# #                 all_iterations_ed_dates.append([])
                
# #         except Exception as e:
# #             # 에러 발생 시 빈 리스트 처리
# #             all_iterations_ed_dates.append([])
# #             continue
            
# #     return all_iterations_ed_dates, date_df

# def predict_new_data_probability(df_test, data_all_test, boot_ensemble, scaler, step, warning_label=1):
#     incremental_detection_results = {}
#     incremental_prob_results = {}
#     feature_cols = ['slope', 'mean', 'CS_mean']
    
#     X_test = scaler.transform(df_test[feature_cols])
#     test_dates_objects = pd.to_datetime(data_all_test['Date'].values)

#     # 1. 전체 구간 예측
#     all_mapped_preds = []
#     detection_dates = []
    
#     for item in boot_ensemble:
#         model = item['model']
#         order = item['order']
#         raw_preds = model.predict(X_test)
#         mapped_preds = pd.Series(raw_preds).map(order).values
        
#         binary_preds = (mapped_preds == (len(order) - 1)).astype(int)
#         all_mapped_preds.append(binary_preds)
        
#         temp_df = df_test.copy().reset_index(drop=True)
#         temp_df['label'] = binary_preds
        
#         try:
#             row_2 = find_warning_periods(temp_df)
#             if len(row_2) > 0:
#                 first_idx = int(row_2[0])
#                 actual_date = test_dates_objects[first_idx+2]
#                 detection_dates.append(actual_date)
#             else:
#                 detection_dates.append(np.nan)
#         except:
#             detection_dates.append(np.nan)
            
#     prob_values = np.mean(all_mapped_preds, axis=0)
    
#     # [수정됨] prob_values 길이와 data_all_test 길이가 다름 (window size 차이)
#     # df_test의 끝 지점들에 해당하는 날짜만 가져오기 위해 뒤에서부터 길이만큼 슬라이싱
#     offset = len(test_dates_objects) - len(prob_values)
#     aligned_dates = test_dates_objects[offset:]
    
#     prob_df = pd.DataFrame({
#         'Date': aligned_dates,
#         'Warning_Probability': prob_values
#     })
#     date_df = pd.DataFrame({'Detect_date': detection_dates})

#     # 2. Step별 예측
#     max_idx = len(df_test)
#     for end_idx in range(step, max_idx + 1, step):
#         current_df = df_test.iloc[:end_idx].reset_index(drop=True)
#         current_test = scaler.transform(current_df[feature_cols])
#         current_X = current_test[:end_idx]
        
#         # 현재 구간의 마지막 날짜 (test_dates_objects는 전체이므로 offset 고려 필요)
#         # offset은 window로 인해 잘린 앞부분 길이
#         current_last_date_idx = end_idx + offset - 1
#         last_date_val = test_dates_objects[current_last_date_idx]
#         last_date_str = last_date_val.strftime('%Y-%m-%d')
        
#         step_detection_dates = []
#         all_step_preds = []
        
#         for item in boot_ensemble:
#             model = item['model']
#             order = item['order']
#             preds = model.predict(current_X)
            
#             mapped_series = pd.Series(preds).map(order)
#             binary_step_preds = (mapped_series == (len(order) - 1)).astype(int).values
#             all_step_preds.append(binary_step_preds)
            
#             temp_df = current_df.copy()
#             temp_df['label'] = binary_step_preds
            
#             try:
#                 row_2 = find_warning_periods(temp_df, warning_label=warning_label)
#                 if len(row_2) > 0:
#                     first_idx = int(row_2[0])
#                     # 여기서도 인덱스 보정 필요할 수 있으나 find_warning_periods 반환값은 temp_df 기준
#                     # temp_df 기준 idx에 offset을 더해줘야 실제 data_all_test 날짜 인덱스
#                     actual_date_idx = first_idx + offset + 2 # +2는 원본 로직 유지
#                     # 범위 체크
#                     if actual_date_idx < len(test_dates_objects):
#                         actual_date = test_dates_objects[actual_date_idx]
#                         step_detection_dates.append(actual_date)
#                     else:
#                         step_detection_dates.append(np.nan)
#                 else:
#                     step_detection_dates.append(np.nan)
#             except:
#                 step_detection_dates.append(np.nan)
        
#         incremental_detection_results[last_date_str] = step_detection_dates
#         step_prob_values = np.mean(all_step_preds, axis=0)
        
#         # 날짜 매핑 (offset 적용)
#         step_dates = test_dates_objects[offset : offset+end_idx]
        
#         step_prob_df = pd.DataFrame({
#             'Date': step_dates,
#             'Warning_Probability': step_prob_values
#         })
#         incremental_prob_results[last_date_str] = step_prob_df
        
#     iteration_results = pd.DataFrame(incremental_detection_results)
    
#     return prob_df, date_df, iteration_results, incremental_prob_results
def find_warning_periods(result_data, data_all_train, outbreak_season, warning_label=1):
    row_start = []
    row_end = []

    ## warning label이 하나도 없는 경우
    result_data['label'] = result_data['label'].astype(str)
    warning_label = str(warning_label)
    if (result_data['label'] == warning_label).sum() == 0:
        return []

    ## warning label이 존재하는 경우
    if result_data.loc[0,'label']==warning_label:
        row_start.append(result_data.index[0])

    for i in range(1, len(result_data)):
        if result_data.loc[i,'label']==warning_label:
            if result_data.loc[i-1,'label']!=warning_label:
                row_start.append(result_data.index[i])
    
    for i in range(len(result_data)-1):
        if result_data.loc[i+1,'label']!=warning_label:
            if result_data.loc[i,'label']==warning_label:
                row_end.append(result_data.index[i])
    
    if result_data.loc[len(result_data)-1,'label']==warning_label:
        row_end.append(result_data.index[-1])  

    last_warning_index = result_data[result_data['label']==warning_label].tail(1).index[0]
    if len(row_start)==0 or row_start[-1] != last_warning_index:
        row_start.append(last_warning_index)
    
    # 3주 이상 지속되었을때 추가
    row_2=[]
    for i in range(len(row_start)-1):
        if row_start[i+1]-row_start[i]>=2:
            if row_end[i]-row_start[i]>=2:
                row_2.append(row_start[i])
    
    ## 조기탐지 기간 설정
    ED_num = row_2*np.ones(len(row_2))
    detection_dates_series = data_all_train.loc[ED_num.astype(int) + 2, 'Date']
    detection_dates_series = pd.to_datetime(detection_dates_series)
    detection_week = data_all_train.loc[ED_num.astype(int) + 2, 'Week']
    filter_dates = detection_dates_series[detection_week >= outbreak_season]
    row_dates = filter_dates.drop_duplicates().groupby(filter_dates.dt.year).min()
    ED_date = row_dates.tolist()
    
    return ED_date

# K-means clustering을 수행하는 함수
def K_means_clustering(df):
    X_train = df[['slope', 'mean', 'CS_mean']].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = KMeans(random_state=727, n_init=10)
    
    visualizer = KElbowVisualizer(model, k=(2, 10), metric='silhouette', timings=True)
    visualizer.fit(X_scaled)
    visualizer.finalize()
    k = visualizer.elbow_value_
    plt.close()
    
    kmeans = KMeans(n_clusters=k, random_state=727, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    df['label'] = labels
    
    RI_means = df.groupby('label')['slope'].mean()
    label_order = RI_means.sort_values().index
    custom_order = {old_label: new_label for new_label, old_label in enumerate(label_order)}
    df['label'] = df['label'].map(custom_order)
    
    result_data = df.sort_values('data_num')
    result_data.reset_index(drop=True, inplace=True)
    result_data['label'] = result_data['label'].astype(str)

    return result_data, kmeans, k, scaler

def train_bootstrap_ensemble(df_train, scaler, feature_cols, B=300, k_best=2, types='random'):
    print(f"Training Ensemble of {B} models ({types})...")

    T = len(df_train)
    indices = np.arange(T)
    boot_ensemble = []

    if types == 'random':
        sampler = [(np.random.choice(indices, T, replace=True),) for _ in range(B)]
    elif types == 'sb':
        ### acf 코드 추가 필요
        sb = StationaryBootstrap(11, indices)
        sampler = (sample[0] for sample in sb.bootstrap(B))
    elif types == 'cbb':
        ### acf 코드 추가 필요
        cbb = CircularBlockBootstrap(12, indices)
        sampler = (sample[0] for sample in cbb.bootstrap(B))
    else:
        raise ValueError("types must be one of ['random', 'sb', 'cbb']")

    for b, boot_idx in enumerate(sampler):

        resampled_df = df_train.iloc[boot_idx].copy()
        X_resamp = scaler.transform(resampled_df[feature_cols])
        kmeans = KMeans(n_clusters=k_best, random_state=b, n_init=10)
        resampled_df['temp_label'] = kmeans.fit_predict(X_resamp)
        RI_means = resampled_df.groupby('temp_label')['slope'].mean()
        label_order = RI_means.sort_values().index
        custom_order = {old: new for new, old in enumerate(label_order)}
        boot_ensemble.append({
            'model': kmeans,
            'order': custom_order
        })

    return boot_ensemble, scaler

# 학습한 bootstrap 모델를 Train data에 적용하는 함수
def analyze_train_distribution(df_train, data_all_train, feature_cols, boot_ensemble, scaler, outbreak_season, ED_date):
    all_iterations_ed_dates = []
    X_orig_scaled = scaler.transform(df_train[feature_cols])
    date_df = pd.DataFrame()
    label_df = pd.DataFrame()
    
    for item in boot_ensemble:
        model = item['model']
        order = item['order']
        preds = model.predict(X_orig_scaled)
        temp_df = df_train.copy()
        temp_df['label'] = pd.Series(preds).map(order).values
        ED_date = find_warning_periods(temp_df, data_all_train, outbreak_season, warning_label=temp_df['label'].max())
        tmp = (pd.DataFrame({"date": ED_date}).assign(year=lambda x: x["date"].dt.year.astype(str),idx=lambda x: x.groupby(x["date"].dt.year).cumcount() + 1))
        tmp["col"] = tmp["year"] + "_" + tmp["idx"].astype(str)
        row = tmp.set_index("col")["date"].to_frame().T
        date_df = pd.concat([date_df, row], ignore_index=True)
        label_df = pd.concat([label_df, temp_df['label'].to_frame().T], ignore_index=True)
    return date_df, label_df

# 학습한 bootstrap 모델를 Test data에 적용하는 함수 (step개씩 증가)
def predict_new_data_probability(df_test, data_all_test, boot_ensemble, scaler, outbreak_season, step):
    incremental_detection_results = {}
    incremental_prob_results = {}
    feature_cols = ['slope', 'mean', 'CS_mean']

    X_test = scaler.transform(df_test[feature_cols])
    test_dates_objects = pd.to_datetime(data_all_test['Date'].values)

    ### 전체 시점에 대한 bootstrap 예측 결과
    all_mapped_preds = []
    detection_dates = []
    for item in boot_ensemble:
        model = item['model']
        order = item['order']
        
        raw_preds = model.predict(X_test)
        mapped_preds = pd.Series(raw_preds).map(order).values
        binary_preds = (mapped_preds == (len(order) - 1)).astype(int)
        all_mapped_preds.append(binary_preds)
        temp_df = df_test.copy().reset_index(drop=True)
        temp_df['label'] = binary_preds # binary로 변환하여 무조건 1이 가장 높은 숫자임
        try:
            ED_date = find_warning_periods(temp_df, data_all_test, outbreak_season, warning_label=1)
            # ED_date = ED_date.apply(lambda x: x[0])
            if len(ED_date) > 0:
                detection_dates.append(ED_date)
            else:
                detection_dates.append(np.nan)
        except:
            detection_dates.append(np.nan)
    
    prob_values = np.mean(all_mapped_preds, axis=0)
    prob_df = pd.DataFrame({
        'Date': data_all_test['Date'].values,
        'Warning_Probability': prob_values
    })
    date_df = pd.DataFrame({'Detect_date': detection_dates})
    
    ### step 단위로 데이터를 증가시키면서 각 구간별 탐지 날짜 및 확률 계산
    max_idx = len(df_test)
    for end_idx in range(step, max_idx + 1, step):
        current_df = df_test.iloc[:end_idx].reset_index(drop=True)
        current_test = scaler.transform(current_df[feature_cols])
        current_X = current_test[:end_idx]
        
        # 현재 구간의 마지막 날짜를 컬럼명으로 지정
        last_date_val = test_dates_objects[end_idx - 1]
        last_date_str = last_date_val.strftime('%Y-%m-%d')
        
        detection_dates = []
        all_step_preds = []
        
        # 탐지지점 및 시점별 확률 계산
        for item in boot_ensemble:
            model = item['model']
            order = item['order']
            
            # 예측 및 이진화 매핑
            preds = model.predict(current_X)
            temp_df = current_df.copy()
            mapped_series = pd.Series(preds).map(order)
            
            # 가장 높은 클러스터만 1로 (위험군)
            temp_df['label'] = (mapped_series == (len(order) - 1)).astype(int).values
            all_step_preds.append(temp_df['label'].values)
            try:
                ED_date = find_warning_periods(temp_df, data_all_test, outbreak_season, warning_label=1)
                if len(ED_date) > 0:
                    detection_dates.append(ED_date)
                else:
                    detection_dates.append(np.nan)
            except:
                detection_dates.append(np.nan)
        
        # 한 step이 끝날 때마다 컬럼 하나 생성
        incremental_detection_results[last_date_str] = detection_dates
        step_prob_values = np.mean(all_step_preds, axis=0)
        step_prob_df = pd.DataFrame({
            'Date': test_dates_objects[:end_idx],
            'Warning_Probability': step_prob_values
        })
        incremental_prob_results[last_date_str] = step_prob_df
        
    # 결과: 행(B개, 각 Bootstrap 결과), 열(각 시점별 탐지된 날짜)
    iteration_results = pd.DataFrame(incremental_detection_results)
    
    return prob_df, date_df, iteration_results, incremental_prob_results