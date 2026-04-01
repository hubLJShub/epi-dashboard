import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------------------------------------------------------
# [Backend Import] - 새로운 알고리즘 모듈 추가
# -----------------------------------------------------------------------------
try:
    from config import Config
    from src.preprocessing import make_raw, cumulative_sum, cumulative_sum_3years
    from src.clustering import (
        K_means_clustering, 
        find_warning_periods, # 추가됨
        train_bootstrap_ensemble, 
        analyze_train_distribution,
        predict_new_data_probability
    )
    # 새로 업데이트된 시각화 함수들로 임포트 변경
    from src.visualization import (
        # K_means_visualization, 
        early_warning_visualization, 
        early_warning_visualization_bootstrap, 
        visualization_real_time_early_detection,
        visualization_season,
        interactive_real_time_chart
    )
    # 시즌 세팅 및 하키스틱 모듈 추가
    from src.season_setting import set_season_start_week,  hockey_stick_regression, filter_data
except ImportError as e:
    st.error(f"Failed to import modules from src folder.\nError: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# [캐싱 함수] 무거운 연산 (윈도우 최적화)
# -----------------------------------------------------------------------------
@st.cache_resource
def optimize_window_size(_data, epi, hockey_date, seasons, peak_start):
    sample_window_list = np.arange(3, 25)
    score_list = []
    best_score = np.inf
    
    for sample_window in sample_window_list:
        df_train, data_all_train = make_raw(_data, 'train', sample_window, epi)
        result_data_t, kmeans, _, _ = K_means_clustering(df_train)
        warning_label_t = result_data_t['label'].max()
        ED_date = find_warning_periods(result_data_t, data_all_train, peak_start, warning_label_t)
        
        if len(ED_date) != (len(hockey_date) - 1):
            score = 1000
        else:
            diff = (pd.Series(hockey_date) - pd.Series(ED_date)).dt.days
            score = diff.abs().sum(skipna=True)
            
        if score < best_score:
            best_score = score
        score_list.append(score)
        
    best_window = score_list.index(best_score) + int(sample_window_list[0])
    return best_window, best_score

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Epidemic Early Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. Sidebar: Settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("Settings")
    
    st.header("1. Data Input")
    uploaded_file = st.file_uploader("Upload Excel File (.xlsx)", type=["xlsx", "xls"])
    
    @st.cache_data
    def load_data(file):
        if file is not None:
            return pd.read_excel(file)
        elif hasattr(Config, 'DATA_PATH') and os.path.exists(Config.DATA_PATH):
            return pd.read_excel(Config.DATA_PATH)
        else:
            return None

    raw_data = load_data(uploaded_file)
    
    if raw_data is not None:
        all_cols = raw_data.columns.tolist()
        
        # default_idx = all_cols.index(Config.EPI_COL) if hasattr(Config, 'EPI_COL') and Config.EPI_COL in all_cols else (1 if len(all_cols) > 1 else 0)
        # 🌟 대표적인 질병 키워드 사전 (대소문자 무관하게 탐지)
        disease_keywords = ['ili', 'noro', 'hfmd', 'hrsv', 'covid', 'flu', 'patient', 'cases']
        
        target_default_idx = 0
        found = False
        
        # 1순위: 엑셀 컬럼명 중에 질병 키워드가 포함되어 있는지 스캔!
        for i, col in enumerate(all_cols):
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in disease_keywords):
                target_default_idx = i
                found = True
                break
        
        # 2순위: 다 못 찾았으면 두 번째 컬럼(index 1)을 기본값으로 설정
        if not found and len(all_cols) > 1:
            target_default_idx = 1
                
        # 셀렉트박스 생성 (상태 유지를 위한 key 포함)
        target_col = st.selectbox("Target Column (EPI_COL)", all_cols, index=target_default_idx, key="target_col_select")
        # target_col = st.selectbox("Target Column (EPI_COL)", all_cols, index=default_idx, key="target_col_select")
        
        date_default_idx = 0
        if 'Date' in all_cols:          
            date_default_idx = all_cols.index('Date')
        elif 'date' in all_cols:        
            date_default_idx = all_cols.index('date')
        elif '일자' in all_cols:        
            date_default_idx = all_cols.index('일자')
        else:
            candidates = [c for c in all_cols if 'date' in str(c).lower()]
            if candidates:
                date_default_idx = all_cols.index(candidates[0])

        date_col = st.selectbox("Date Column", all_cols, index=date_default_idx, key="date_col_select")        
        temp_dates = pd.to_datetime(raw_data[date_col], errors='coerce')
        if temp_dates.notna().sum() == 0:
            st.warning("No valid date data found in the selected column.")
            st.stop()
            
        raw_data[date_col] = temp_dates
        raw_data = raw_data.dropna(subset=[date_col])
        
        if raw_data.empty:
            st.error("No valid data available.")
            st.stop()
    else:
        st.warning("No data loaded.")
        st.stop()

    # st.markdown("---")
    # st.header("2. Period Settings")

    # # 🌟 1. 위에서 고른 타겟 컬럼 이름(target_col)을 소문자로 변환해서 확인합니다.
    # target_lower = str(target_col).lower()
    
    # # 🌟 2. 질병 이름에 따라 디폴트 시작 주차를 똑똑하게 지정해 줍니다!
    # if 'noro' in target_lower:
    #     default_start = 20
    # elif 'ili' in target_lower or 'hfmd' in target_lower:
    #     default_start = 1
    # else:
    #     default_start = 1  # 혹시 모를 다른 질병을 위한 기본값
        
    # # 만약 session_state에 이전 값이 남아있어서 안 바뀌는 현상을 막기 위해 강제로 업데이트!
    # if 'prev_target' not in st.session_state or st.session_state.prev_target != target_col:
    #     st.session_state['start_week_input'] = default_start
    #     st.session_state.prev_target = target_col

    # # 🌟 3. 계산된 default_start를 입력창에 넣어줍니다.
    # manual_start_week = st.number_input(
    # "Set Season Start Week (1~52)", 
    # min_value=1, max_value=52, step=1,
    # help="Recommended: 1 for ILI/HFMD, 20~37 for Norovirus.",
    # key="start_week_input"
    # )
    
    # train_date = None
    # test_date = None
    manual_start_week = 1   
    train_date = None
    test_date = None

    st.markdown("---")
    st.header("2. Parameters")
    
    # 꼭 필요한 파라미터만 남겨둡니다.
    boot_num = st.number_input("Bootstrap Iterations", 50, 2000, 1000, step=50, key="boot_num_input")
    HockeyStick_type = st.selectbox("Hockey Stick Type", ["linear", "exponential"], key="hockey_type_select")
    
    st.markdown("---")
    run_btn = st.button("Start Analysis", type="primary")

# -----------------------------------------------------------------------------
# 3. Main Logic
# -----------------------------------------------------------------------------
st.title("Universal Respiratory Epidemic Early Detection System")

st.markdown("""
<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 25px; font-size: 18px; line-height: 1.6;">
    <strong style="font-size: 22px;">[System Description]</strong><br>
    This dashboard is designed for the early detection of infectious diseases.<br>
    Please upload your data in the sidebar on the left, configure the settings below, and click <strong>'Start Analysis'</strong>.<br><br>
    <em>For detailed instructions on the settings, please refer to <strong>Tab 1</strong> below.</em>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Manual & Settings Guide", "Dashboard Analysis", "논문요약페이지"])

# with tab1:
#     st.markdown("###  Detailed Instructions & Defaults")
    
#     st.markdown("""
#     <div style="background-color: #ffffff; padding: 25px; border: 1px solid #ddd; border-radius: 10px; font-size: 18px; line-height: 1.8;">
#         <div style="background-color: #e8f4f8; padding: 15px; border-radius: 8px; margin-bottom: 25px; border-left: 5px solid #2e86c1;">
#             <strong> Default Data Notice:</strong><br>
#             If no Excel file is uploaded, the system automatically loads the internal 
#             <strong>South Korea Influenza Surveillance Data (KDCA)</strong>.
#         </div>
#         <h3 style="color: #2e86c1; border-bottom: 2px solid #2e86c1; padding-bottom: 10px; margin-top: 0;">1. Data Input</h3>
#         <ul style="margin-bottom: 30px;">
#             <li><strong>Target Column:</strong> The column representing patient counts.<br>
#             <span style="color: #555; font-size: 16px; background-color: #f1f1f1; padding: 2px 8px; border-radius: 4px;">
#              Default: <b>ILI</b> (Influenza-like Illness)</span></li>
#             <li><strong>Date Column:</strong> The column containing date information (YYYY-MM-DD).<br>
#             <span style="color: #555; font-size: 16px; background-color: #f1f1f1; padding: 2px 8px; border-radius: 4px;">
#              Default: Automatically detected</span></li>
#         </ul>
#         <h3 style="color: #2e86c1; border-bottom: 2px solid #2e86c1; padding-bottom: 10px;">2. Period Settings</h3>
#         <ul style="margin-bottom: 30px;">
#             <li><strong>Dynamic Detection Mode (Default):</strong> The system operates in fully automatic mode by default. The algorithm dynamically detects the optimal season start week and automatically splits the historical data (Train) and the ongoing real-time data (Test).<br>
#             <span style="color: #2e86c1; font-size: 16px; background-color: #e8f4f8; padding: 2px 8px; border-radius: 4px;">
#              <b>Highly Recommended</b> for standard real-time monitoring.</span></li>
#             <li><strong>Manual Parameter Settings:</strong> By checking this option, expert users can override the automated system and manually define the <b>Train End Date</b>, <b>Test End Date</b>, and a specific <b>Season Start Date</b>.</li>
#         </ul>
#         <h3 style="color: #2e86c1; border-bottom: 2px solid #2e86c1; padding-bottom: 10px;">3. Algorithm Parameters</h3>
#         <ul>
#             <li><strong>Bootstrap Iterations (B):</strong> The number of resampling iterations for the ensemble clustering model.<br>
#             <span style="color: #555; font-size: 16px; background-color: #f1f1f1; padding: 2px 8px; border-radius: 4px;">
#              Default: <b>1000 Iterations</b> (Higher = More stable but longer computation)</span></li>
#             <li><strong>Hockey Stick Type:</strong> The regression method used to detect sudden exponential trend changes at the epidemic onset.<br>
#             <span style="color: #555; font-size: 16px; background-color: #f1f1f1; padding: 2px 8px; border-radius: 4px;">
#              Default: <b>Linear</b></span></li>
#         </ul>
#     </div>
#     """, unsafe_allow_html=True)
with tab1:
    # 🌟 대제목 (가장 크게)
    st.markdown("<h2 style='font-size: 32px; font-weight: 800; color: #2c3e50; margin-bottom: 20px;'>Dashboard Guide & Setup</h2>", unsafe_allow_html=True)
    st.markdown("""
    <style>
    div[data-testid="stExpander"] details summary p,
    div[data-testid="stExpander"] details summary span,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span {
        font-size: 24px !important; 
        font-weight: 800 !important;
        color: #2c3e50 !important; /* 대제목과 어울리는 진한 남색 */
    }
    /* 옆에 있는 꺾쇠(화살표) 아이콘도 글자 크기에 맞춰 살짝 키워줍니다 */
    [data-testid="stExpander"] summary svg {
        width: 24px !important;
        height: 24px !important;
        color: #2c3e50 !important;
    }
    </style>
    """, unsafe_allow_html=True)
with tab1:
    # 🌟 [CSS 최적화] 모든 Expander 제목 글자 크기를 24px로 크고 진하게 고정합니다!
    st.markdown("""
    <style>
    div[data-testid="stExpander"] details summary p,
    div[data-testid="stExpander"] details summary span,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span {
        font-size: 24px !important; 
        font-weight: 800 !important;
        color: #2c3e50 !important;
    }
    [data-testid="stExpander"] summary svg {
        width: 24px !important;
        height: 24px !important;
        color: #2c3e50 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    # =========================================================================
    # 🌟 [최종 수정] 0. Introduction (그림 위치 상단으로 이동 + 고화질)
    # =========================================================================
    with st.expander("0. Introduction", expanded=True):
        # 사용자 지정 비율을 유지합니다. [0.5, 2.5]
        col1, col2 = st.columns([0.5, 2.5], gap="large")
        
        with col1:
            import os
            import base64
            
            # 경로 설정
            current_dir = os.path.dirname(os.path.abspath(__file__))
            img_path = os.path.join(current_dir, "images", "intro_example.png")
            
            try:
                # 🌟 파일을 base64로 인코딩해서 HTML 안에 직접 쏘아주는 방식
                with open(img_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode()
                
                # 🌟 [핵심 해결책] 복잡한 flex, height 100% 다 지웠습니다!
                # text-align: center로 가운데 정렬만 남기고,
                # margin-top: -5px; 를 주어서 위로 바짝 끌어올렸습니다. (숫자를 -10px, -20px 등으로 조절하면 더 위로 뚫고 올라갑니다!)
                # 🌟 [핵심 해결책] margin-bottom: 30px; 를 추가하여 사진 아래에 넉넉한 빈 공간을 만듭니다!
                st.markdown(f'<div style="text-align: center; margin-top: 5px; margin-bottom: 40px;"><img src="data:image/png;base64,{encoded_string}" width="200" style="border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"></div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Image load error: {e}")

        with col2:
            # 🌟 [오른쪽: 설명 부분, 중앙 고정을 풀고 사진과 똑같은 여백을 주어 함께 움직이게 묶음!]
            st.markdown("""
            <div style="border-left: 5px solid #1f77b4; background-color: #f8fbff; padding: 20px 25px; border-radius: 0 8px 8px 0; margin-top: 5px; margin-bottom: 40px;">
                <div style="color: #333; line-height: 1.8;">
                    <p style="font-size: 22px; font-weight: 500; margin-bottom: 15px;">This dashboard is a tool that defines when an outbreak started using seasonal infectious disease data, and uses this to determine when the next outbreak will occur during an ongoing season.</p>
                    <p style="font-size: 22px; font-weight: 500; margin-bottom: 15px;">A 'season' is defined from the start to the end of a single outbreak wave, and the dashboard is equipped with an algorithm to detect this automatically.</p>
                    <p style="font-size: 22px; font-weight: 500; margin-bottom: 0;">Therefore, the real-time detection period can only be verified for the final, ongoing season.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
# 🌟 1. Setting Guide (텍스트 대신 이미지 한 장으로 깔끔하게 대체)
    with st.expander("1. Setting Guide", expanded=False):
        import os
        import base64
        
        # 경로 설정 (Setting_guide.jpg 파일이 images 폴더 안에 있어야 합니다!)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        setting_img_path = os.path.join(current_dir, "images", "Setting_guide.png")
        
        try:
            # 파일을 base64로 인코딩해서 HTML 안에 직접 쏘아주는 방식
            with open(setting_img_path, "rb") as image_file:
                # 🌟 확장자가 png이므로 data:image/png 로 맞춰줍니다!
                encoded_setting_img = base64.b64encode(image_file.read()).decode()
            # 🌟 [핵심] 기존 텍스트를 모두 지우고 이미지를 중앙에 띄웁니다.
            # max-width: 100% 를 주어 모니터 창 크기에 맞춰 이미지가 예쁘게 자동 축소/확대되도록 했습니다.
            st.markdown(f"""
                <div style="display: flex; justify-content: center; padding: 20px 0;">
                    <img src="data:image/jpeg;base64,{encoded_setting_img}" style="max-width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            # 이미지를 찾을 수 없을 때 안내 메시지
            st.error(f"Image load error: {e}")
            st.info("💡 'images' 폴더 안에 'Setting_guide.jpg' 파일이 있는지, 파일명 대소문자가 정확한지 확인해주세요!")

    # 🌟 2. Dashboard Analysis (Placeholder)
    with st.expander("2. Dashboard Analysis", expanded=False):
        st.markdown("""
        <div style="border: 2px dashed #ccc; border-radius: 8px; padding: 80px 20px; text-align: center; margin-bottom: 25px; margin-top: 15px; background-color: #fafafa;">
            <span style="font-size: 26px; color: #555; font-weight: bold;">Plot 1</span><br>
            <span style="font-size: 16px; color: #999;">(Insert your first plot code here later)</span>
        </div>
        
        <div style="border: 2px dashed #ccc; border-radius: 8px; padding: 80px 20px; text-align: center; margin-bottom: 15px; background-color: #fafafa;">
            <span style="font-size: 26px; color: #555; font-weight: bold;">Plot 2</span><br>
            <span style="font-size: 16px; color: #999;">(Insert your second plot code here later)</span>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    if run_btn:
        status = st.status("Analysis in progress...", expanded=True)
        
        try:
            proc_data = raw_data.copy()
            proc_data['Date'] = proc_data[date_col]
            if 'Year' not in proc_data.columns:
                proc_data['Year'] = proc_data['Date'].dt.year
            if 'Week' not in proc_data.columns:
                proc_data['Week'] = proc_data['Date'].dt.isocalendar().week
            # ---------------------------------------------------------------------
            # [Step 0.5] Season Dynamics & Train/Test Split
            # ---------------------------------------------------------------------
            status.write("1. Running Season Detection...")            
            # 1. 유행 기준 찾기
            season_df = set_season_start_week(proc_data, target_col)
            seasons = season_df['season'].to_list()
            
            # 2. 시각화 및 peak_start 획득 
            # 🌟 사이드바에서 입력한 manual_start_week 값을 그대로 사용합니다!
            peak_start, peak_len = visualization_season(proc_data, season_df, start_week=manual_start_week)
            
            # 3. Train / Test 분할
            data = filter_data(proc_data, seasons, start_week=peak_start)
            
            status.write("Calculating Break Points (Hockey Stick)...")
            hockey_date, hockey_df = hockey_stick_regression(data, target_col, HockeyStick_type, seasons)
            # ---------------------------------------------------------------------
            # [Step 0] Data Preprocessing (CUSUM Calculation)
            # ---------------------------------------------------------------------
            status.write("2. Preprocessing (CUSUM Calculation)...")
            
            # (이하 기존 CUSUM 계산 코드 그대로 유지)
            data['cusum'] = np.nan 

            train_df = data[data['set'] == 'train'].copy()
            train_cusum_result = cumulative_sum(train_df, target_col, season_start_week=peak_start)
            data.loc[data['set'] == 'train', 'cusum'] = train_cusum_result['cusum'].values
            
            test_cusum_result = cumulative_sum_3years(data.copy(), target_col, season_start_week=peak_start)
            data.loc[data['set'] == 'test', 'cusum'] = test_cusum_result.loc[data['set'] == 'test', 'cusum'].values
            
            data.reset_index(drop=True, inplace=True)
            data['num'] = data.index
            
            proc_data = data.copy()

            # ---------------------------------------------------------------------
            # [Step 1] Window Size Optimization & Split
            # ---------------------------------------------------------------------
            status.write("3. Optimizing Window Size & Splitting Data...")
            
            # [수정됨] if/else 조건문을 완전히 삭제하고, 무조건 자동 최적화를 실행합니다!
            best_window, best_score = optimize_window_size(proc_data, target_col, hockey_date, seasons, peak_start)
            st.toast(f"Optimal Window Size Auto-selected: {best_window} Weeks")
                
            df_train, data_all_train = make_raw(proc_data, 'train', best_window, target_col)
            df_train = df_train.dropna()

            status.write(f"Train Window Generated ({len(df_train)} samples, Window Size: {best_window})")

            # ---------------------------------------------------------------------
            # [Step 2] Model Training
            # ---------------------------------------------------------------------
            status.write(f"4. Training Baseline & Bootstrap Ensemble (B={boot_num})...")
            
            feature_col = ['slope', 'mean', 'CS_mean']
            feature_name = [r'$\beta_{\omega}$', r'$\mu_{\omega}$', r'$\widebar{S_{\omega}}$']
            
            # Baseline K-means (C0)
            result_data_t, kmeans, best_k, scaler = K_means_clustering(df_train)
            warning_label_t = result_data_t['label'].max()
            ED_date = find_warning_periods(result_data_t, data_all_train, peak_start, warning_label_t)
            
            # Bootstrap Training
            boot_ensemble, scaler = train_bootstrap_ensemble(df_train, scaler, feature_col, B=boot_num, k_best=best_k, types='random')
            
            status.write("Model Training Complete")

            # ---------------------------------------------------------------------
            # [Step 3] Distribution Analysis
            # ---------------------------------------------------------------------
            status.write("5. Analyzing Train Distribution...")
            
            date_df, label_df = analyze_train_distribution(df_train, data_all_train, feature_col, boot_ensemble, scaler, peak_start, ED_date)
            
            # ---------------------------------------------------------------------
            # 4. Visualization (Train)
            # ---------------------------------------------------------------------
            st.markdown("---")
            st.header("Analysis Report")

            # 정답지 비교를 위한 Other dates 설정
            start_epidemic = [pd.to_datetime('2017-12-03'), pd.to_datetime('2018-11-18'), pd.to_datetime('2019-11-17'), pd.to_datetime('2022-09-18'), pd.to_datetime('2023-09-17'), pd.to_datetime('2024-12-22'), pd.to_datetime('2025-10-17')]
            CPD_date= [pd.to_datetime('2017-07-23'), pd.to_datetime('2018-12-02'), pd.to_datetime('2019-12-08'), pd.to_datetime('2022-07-03'), pd.to_datetime('2023-09-17'), pd.to_datetime('2024-12-22')]
            other_dates = {'CUSUM': CPD_date, 'KDCA': start_epidemic}

            # st.subheader("1. Baseline Model Cluster Characteristics (Boxplot)")
            
            # # Boxplot 
            # fig_box, axes = plt.subplots(1, 3, figsize=(15, 5))
            # sns.boxplot(x='label', y='slope', data=result_data_t, ax=axes[0], hue='label', palette='Set1')
            # axes[0].set_title('Slope Distribution')
            # sns.boxplot(x='label', y='mean', data=result_data_t, ax=axes[1], hue='label', palette='Set1')
            # axes[1].set_title('Mean Distribution')
            # sns.boxplot(x='label', y='CS_mean', data=result_data_t, ax=axes[2], hue='label', palette='Set1')
            # axes[2].set_title('CUSUM Mean Distribution')
            # st.pyplot(fig_box, use_container_width=False)

            # # 기존 HTML 설명 박스 유지
            # st.markdown("""
            # <div style="background-color: #f9f9f9; padding: 10px; border-left: 5px solid #2e86c1; margin-bottom: 10px;">
            #     <span style="font-size: 24px;"><strong>1. Baseline Cluster Characteristics</strong></span><br>
            #     <span style="font-size: 20px;">
            #         <ul>
            #             <li>The optimal number of clusters <strong>K</strong> was selected based on the silhouette coefficient.</li>
            #             <li>Clusters were ordered according to their corresponding <strong>&beta;<sub>&omega;</sub></strong> values.</li>
            #             <li>An outbreak was detected in the third week when the cluster with the largest <strong>&beta;<sub>&omega;</sub></strong> was observed for three consecutive weeks.</li>
            #         </ul>
            #     </span>
            # </div>
            # """, unsafe_allow_html=True)
            # st.markdown("---")

            st.subheader("1. Bootstrap Early Warning Detection")
            
            # 새로운 시각화 함수 적용
            fig1 = early_warning_visualization_bootstrap(
                proc_data, data_all_train, target_col, other_dates, hockey_date, date_df, best_window
            )
            
            # 두 개의 그래프를 차례대로 Streamlit에 띄웁니다!
            # st.pyplot(fig1, use_container_width=False)
            st.plotly_chart(fig1, use_container_width=True)  # 동적 그래프 출력
            # st.markdown("<br>", unsafe_allow_html=True) # 그래프 사이 간격 살짝 띄우기
            # st.pyplot(fig2, use_container_width=False)
            # st.pyplot(fig_boot, use_container_width=False)
                
            st.markdown("""
            <div style="background-color: #f9f9f9; padding: 15px; border-left: 5px solid #2e86c1; margin-bottom: 20px;">
                <span style="font-size: 22px;"><strong>Bootstrap Early Warning Detection (Train Data)</strong></span><br>
                <span style="font-size: 16px; color: #444; line-height: 1.6;">
                    <ul style="margin-top: 10px;">
                        <li><strong style="color: #2CA02C;">Green Line (Hockey Stick)</strong>: Identifies the structural break point in the epidemic curve. This mathematical approach automatically optimizes the sliding window size by detecting the exact moment the trend shifts to exponential growth.</li>
                        <li><strong style="color: #1F77B4;">Blue Dashed Line</strong>: Represents the most robust and frequently observed early detection time across all bootstrap ensemble clustering models.</li>
                        <li><strong style="color: #85C1E9;">Blue Shaded Area</strong>: Visualizes the variability (uncertainty) of the detection time across the <i>B</i> bootstrap iterations, providing a confidence interval for the warning signal.</li>
                    </ul>
                </span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")
            
            # =====================================================================
            # [Section 2] Real-time Surveillance
            # =====================================================================
            st.subheader("2. Real-time Surveillance")
            
            if (train_date is not None) and (test_date is not None) and (test_date <= train_date):
                st.warning("Test End Date must be later than Train End Date.")
            else:
                status.write("6. Generating Test Data & Real-time Surveillance...")
                
                df_test, data_all_test = make_raw(proc_data, 'test', best_window, target_col)
                
                try:
                    if df_test.empty:
                        st.error("No valid Test data available in the selected range.")
                    else:
                        # 신규 알고리즘의 실시간 예측 로직 적용
                        prob_table, date_table, bootstrap_dates, incremental_prob_results = predict_new_data_probability(
                            df_test, data_all_test, boot_ensemble, scaler, peak_start, step=1
                        )
                        date_table['Detect_date'] = date_table['Detect_date'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
                        
                        status.write("Surveillance Complete! Generating Visualizations...")

                        # "Simulation" 단어 제거 후 Surveillance로 통일
                        # st.markdown("Real-Time Surveillance")
                        
                        # 1. 시각화 함수를 실행해서 그림과 날짜 3개를 받아옵니다.
                        fig_interactive, d_blue, d_orange, d_red = interactive_real_time_chart(
                            data_all=data_all_test, 
                            bootstrap_dates=bootstrap_dates, 
                            other_dates=other_dates, 
                            epi=target_col
                        )
                        
                        st.plotly_chart(fig_interactive, use_container_width=True)
                        st.markdown(f"""
                        <style>
                            .summary-card-container {{
                                display: flex;
                                gap: 15px;
                                margin-bottom: 30px;
                            }}
                            .summary-card {{
                                flex: 1;
                                background: white; 
                                padding: 25px 20px;
                                border-radius: 12px;
                                box-shadow: 0 4px 15px rgba(0,0,0,0.05);
                                transition: background-color 0.3s ease, transform 0.2s ease;
                                cursor: pointer; 
                            }}
                            .summary-card:hover {{
                                background-color: #f1f3f5 !important; 
                                transform: translateY(-3px); 
                            }}
                        </style>

                        <div style="margin-top: 15px; margin-bottom: 10px; padding-left: 5px;">
                            <span style="font-size: 20px; font-weight: 800; color: #333; letter-spacing: 0.5px;">Early Warning Timeline Summary</span>
                        </div>
                        
                        <div class="summary-card-container">
                            <div class="summary-card" style="border-left: 8px solid #1f77b4;">
                                <div style="font-size: 14px; color: #666; font-weight: 600; margin-bottom: 5px;">Level 1: Attention (Blue)</div>
                                <div style="font-size: 28px; color: #1f77b4; font-weight: 800; letter-spacing: 0.5px;">{d_blue}</div>
                            </div>
                            <div class="summary-card" style="border-left: 8px solid #ff7f0e;">
                                <div style="font-size: 14px; color: #666; font-weight: 600; margin-bottom: 5px;">Level 2: Caution (Orange)</div>
                                <div style="font-size: 28px; color: #ff7f0e; font-weight: 800; letter-spacing: 0.5px;">{d_orange}</div>
                            </div>
                            <div class="summary-card" style="border-left: 8px solid #d62728;">
                                <div style="font-size: 14px; color: #666; font-weight: 600; margin-bottom: 5px;">Level 3: Alert (Red)</div>
                                <div style="font-size: 28px; color: #d62728; font-weight: 800; letter-spacing: 0.5px;">{d_red}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                       
                        st.divider()

                except Exception as e:
                    st.error(f"Error during data slicing/prediction: {e}")
            
            status.update(label="Analysis Process Completed.", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Error occurred during analysis!", state="error", expanded=True)
            st.error(f"Details: {e}")
            st.exception(e)

        st.markdown("""
        <div style="background-color: #f9f9f9; padding: 15px; border-left: 5px solid #2e86c1; margin-bottom: 20px;">
            <span style="font-size: 22px;"><strong>Real-Time Surveillance Results</strong></span><br>
            <span style="font-size: 16px; color: #444; line-height: 1.6;">
                <ul style="margin-top: 10px;">
                    <li>As new data arrives week by week, we <strong>monitor in real-time</strong> how the probability of an epidemic outbreak evolves.</li>
                    <li>The <strong>probability of reaching risk</strong> is defined as the probability that a sliding window sample is assigned to a high-risk cluster.</li>
                    <li><strong>Interactive View:</strong> Use the <b>range slider</b> at the bottom of the chart to zoom into specific periods.</li>
                    <li><strong>Summary Cards:</strong> The cards below indicate the exact dates when each risk level (Attention, Caution, Alert) was first detected by the ensemble model.</li>
                </ul>
            </span>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("### Detailed Paper")
    # 기존 HTML 블록 100% 유지
    st.markdown("""
    <div style="background-color: #ffffff; padding: 25px; border: 1px solid #ddd; border-radius: 10px; font-size: 18px; line-height: 1.8;">
        <div style="background-color: #e8f4f8; padding: 15px; border-radius: 8px; margin-bottom: 25px; border-left: 5px solid #2e86c1;">
            <strong> Default Data Notice:</strong><br>
            If no Excel file is uploaded, the system automatically loads the internal 
            <strong>South Korea Influenza Surveillance Data (KDCA)</strong>.
        </div>
        <h3 style="color: #2e86c1; border-bottom: 2px solid #2e86c1; padding-bottom: 10px; margin-top: 0;">1. Data Input</h3>
        <ul style="margin-bottom: 30px;">
            <li><strong>Target Column:</strong> The column representing patient counts.<br>
            <span style="color: #555; font-size: 16px; background-color: #f1f1f1; padding: 2px 8px; border-radius: 4px;">
             Default: <b>ILI</b> (Influenza-like Illness)</span></li>
            <li><strong>Date Column:</strong> The column containing date information (YYYY-MM-DD).<br>
            <span style="color: #555; font-size: 16px; background-color: #f1f1f1; padding: 2px 8px; border-radius: 4px;">
             Default: Automatically detected (e.g., <b>date</b>, <b>일자</b>)</span></li>
        </ul>
        <h3 style="color: #2e86c1; border-bottom: 2px solid #2e86c1; padding-bottom: 10px;">2. Period Settings</h3>
        <ul style="margin-bottom: 30px;">
            <li><strong>Start Date:</strong> The beginning date for the entire analysis period.<br>
            <span style="color: #555; font-size: 16px; background-color: #f1f1f1; padding: 2px 8px; border-radius: 4px;">
             Default: <b>2017-01-01</b></span></li>
            <li><strong>Train End Date:</strong> The cutoff date for the training dataset.<br>
            <span style="color: #555; font-size: 16px; background-color: #f1f1f1; padding: 2px 8px; border-radius: 4px;">
             Default: <b>2024-09-15</b></span></li>
            <li><strong>Test End Date:</strong> The end date for the prediction simulation.<br>
            <span style="color: #555; font-size: 16px; background-color: #f1f1f1; padding: 2px 8px; border-radius: 4px;">
             Default: <b>Latest Date</b> available in data</span></li>
        </ul>
        <h3 style="color: #2e86c1; border-bottom: 2px solid #2e86c1; padding-bottom: 10px;">3. Algorithm Parameters</h3>
        <ul>
            <li><strong>Window Size:</strong> The number of weeks used to group data for trend analysis.<br>
            <span style="color: #555; font-size: 16px; background-color: #f1f1f1; padding: 2px 8px; border-radius: 4px;">
             Default: <b>12 Weeks</b></span></li>
            <li><strong>Bootstrap Iterations:</strong> The number of resampling iterations for the ensemble model.<br>
            <span style="color: #555; font-size: 16px; background-color: #f1f1f1; padding: 2px 8px; border-radius: 4px;">
             Default: <b>1000 Iterations</b> (Higher = More stable but slower)</span></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
