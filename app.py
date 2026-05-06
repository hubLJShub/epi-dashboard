import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Import the modules used across preprocessing, modeling, and visualization.
try:
    from config import Config
    from src.preprocessing import make_raw, cumulative_sum_adaptive
    from src.clustering import (
        K_means_clustering, 
        extract_seasonal_detection_dates,
        find_warning_periods,
        train_bootstrap_ensemble, 
        analyze_train_distribution,
        summarize_detection_progression
    )
    from src.visualization import (
        early_warning_visualization_bootstrap, 
        overall_period_visualization_bootstrap,
        visualization_season,
    )
    from src.season_setting import (
        set_season_start_week_adaptive,
        hockey_stick_regression,
        assign_retrospective_period,
    )
except ImportError as e:
    st.error(f"Failed to import modules from src folder.\nError: {e}")
    st.stop()

def format_alert_date(date_value, fallback="Not detected"):
    parsed = pd.to_datetime(date_value, errors='coerce')
    if pd.isna(parsed):
        return fallback
    return parsed.strftime('%Y-%m-%d')

# Choose the sliding-window size that best aligns clustering alerts with hockey-stick breakpoints.
@st.cache_resource(show_spinner=False)
def optimize_window_size(_data, epi, hockey_dates, eval_seasons, peak_start):
    sample_window_list = np.arange(3, 25)
    score_list = []
    best_score = np.inf
    eval_seasons = [int(season) for season in eval_seasons]
    hockey_by_season = {
        int(season): pd.to_datetime(date).normalize()
        for season, date in zip(eval_seasons, hockey_dates)
        if pd.notna(pd.to_datetime(date, errors='coerce'))
    }

    if not hockey_by_season:
        return int(sample_window_list[0]), np.inf
    
    for sample_window in sample_window_list:
        df_analysis, data_all_analysis = make_raw(_data, 'analysis', sample_window, epi)
        feature_cols = ['slope', 'mean', 'CS_mean']
        valid_mask = df_analysis[feature_cols].notna().all(axis=1)
        df_analysis = df_analysis.loc[valid_mask].reset_index(drop=True)
        data_all_analysis = data_all_analysis.loc[valid_mask].reset_index(drop=True)

        if df_analysis.empty or data_all_analysis.empty:
            score = 1000
        else:
            try:
                result_data_t, _, _, _ = K_means_clustering(df_analysis)
                warning_label_t = result_data_t['label'].max()
                seasonal_dates = extract_seasonal_detection_dates(
                    result_data_t,
                    data_all_analysis,
                    warning_label=warning_label_t
                )
                detected_by_season = {
                    int(season): pd.to_datetime(detect_date).normalize()
                    for season, detect_date in seasonal_dates[['Season', 'detect_date']].itertuples(index=False)
                }
                compared_seasons = [
                    season for season in eval_seasons
                    if season in hockey_by_season and season in detected_by_season
                ]
                missing_count = len(hockey_by_season) - len(compared_seasons)

                if not compared_seasons:
                    score = 1000
                else:
                    hockey_series = pd.Series([hockey_by_season[season] for season in compared_seasons])
                    detected_series = pd.Series([detected_by_season[season] for season in compared_seasons])
                    diff = (hockey_series - detected_series).dt.days
                    score = diff.abs().sum(skipna=True) + (missing_count * 1000)
            except Exception:
                score = 1000
            
        if score < best_score:
            best_score = score
        score_list.append(score)
        
    best_window = score_list.index(best_score) + int(sample_window_list[0])
    return best_window, best_score

st.set_page_config(
    page_title="Early Warning Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.title("Settings")
    
    st.header("1. Data Upload")
    uploaded_file = st.file_uploader("Upload Excel File (.xlsx)", type=["xlsx", "xls"])
    
    # Load either the uploaded file or the default local dataset.
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
        
        # Infer a sensible default target column from common disease-related column names.
        disease_keywords = ['ili', 'noro', 'hfmd', 'hrsv', 'covid', 'flu', 'patient', 'cases']
        
        target_default_idx = 0
        found = False
        
        for i, col in enumerate(all_cols):
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in disease_keywords):
                target_default_idx = i
                found = True
                break
        
        if not found and len(all_cols) > 1:
            target_default_idx = 1
                
        target_col = st.selectbox("Target Column (EPI_COL)", all_cols, index=target_default_idx, key="target_col_select")
        
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
    manual_start_week = 1

    # st.markdown("---")
    # st.header("2. Reference Dates(Optional)")

    # reference_file = st.file_uploader(
    #     "Upload Reference Date File (.xlsx)",
    #     type=["xlsx", "xls"],
    #     key="reference_date_file"
    # )
    reference_file = None
    reference_dates = None
    reference_label = None

    if reference_file is not None:
        reference_data = pd.read_excel(reference_file)
        reference_cols = reference_data.columns.tolist()

        reference_date_default_idx = 0
        if 'Date' in reference_cols:
            reference_date_default_idx = reference_cols.index('Date')
        elif 'date' in reference_cols:
            reference_date_default_idx = reference_cols.index('date')
        else:
            reference_candidates = [c for c in reference_cols if 'date' in str(c).lower()]
            if reference_candidates:
                reference_date_default_idx = reference_cols.index(reference_candidates[0])

        reference_date_col = st.selectbox(
            "Reference Date Column",
            reference_cols,
            index=reference_date_default_idx,
            key="reference_date_col_select"
        )

        reference_dates = pd.to_datetime(reference_data[reference_date_col], errors='coerce').dropna().tolist()
        reference_label = f"User Input Date ({Path(reference_file.name).stem})"

        if len(reference_dates) == 0:
            st.warning("No valid dates were found in the uploaded reference date file.")

    st.markdown("---")
    st.header("2. Parameters")
    
    boot_num = st.number_input("Repeat Runs", 50, 2000, 200, step=50, key="boot_num_input")
    HockeyStick_type = "linear"
    
    st.markdown("---")
    run_btn = st.button("Run Analysis", type="primary")

st.title("Early Warning Dashboard for Seasonal Signals")

st.markdown("""
<div style="padding: 18px 22px; margin-bottom: 12px; font-size: 18px; line-height: 1.5; color: #000000; background-color: #f0f2f6; border-radius: 10px; border-top: 1px solid #d9d9d9;">
    <div style="font-size: 22px; font-weight: 800;">[System Description]</div>
    <div style="font-size: 18px;">
        This dashboard analyzes seasonal time-series data to detect early warning signals.<br>
        It helps you:<br>
        &nbsp;&nbsp;&nbsp;&nbsp;- Identify when seasonal signal activity begins<br>
        &nbsp;&nbsp;&nbsp;&nbsp;- Track changes in recurring seasonal patterns<br>
    </div>
    <br>
    <div style="font-size: 22px; font-weight: 800;">[Analysis Steps]</div>
    <div style="font-size: 18px;">
        Upload Data &rarr; Configure Settings &rarr; Run Analysis &rarr; View Analysis Report<br>
        Upload your data and click <strong>Run Analysis</strong> to get started.
    </div>
    <div style="margin-top: 18px; font-size: 18px; font-style: italic;">
        For detailed setup instructions, refer to the <strong>'Setup Guide'</strong> tab below
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
button[data-baseweb="tab"] {
    padding-top: 0.85rem !important;
    padding-bottom: 0.85rem !important;
}
button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
    font-size: 1.7rem !important;
    font-weight: 800 !important;
    line-height: 1.2 !important;
}
button[data-baseweb="tab"][aria-selected="true"] > div[data-testid="stMarkdownContainer"] > p {
    font-size: 1.75rem !important;
    font-weight: 800 !important;
}
[data-testid="stTabs"] button[data-baseweb="tab"] {
    padding-top: 0.75rem !important;
    padding-bottom: 0.75rem !important;
}
</style>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Setup Guide", "Analysis Report"])
with tab1:
    st.markdown("<h2 style='font-size: 32px; font-weight: 800; color: #2c3e50; margin-bottom: 20px;'>Getting Started & Setup</h2>", unsafe_allow_html=True)
    st.markdown("""
    <style>
    div[data-testid="stExpander"] details summary p,
    div[data-testid="stExpander"] details summary span,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span {
        font-size: 24px !important; 
        font-weight: 800 !important;
        color: #2c3e50 !important; /* Dark navy tone that matches the main heading */
    }
    /* Slightly enlarge the chevron icon to match the heading text size */
    [data-testid="stExpander"] summary svg {
        width: 24px !important;
        height: 24px !important;
        color: #2c3e50 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    with st.expander("0. Introduction", expanded=True):
        col1, col2 = st.columns([0.7, 2.3], gap="large")
        
        with col1:
            import os
            import base64
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            img_path = os.path.join(current_dir, "images", "intro_example.png")
            
            try:
                with open(img_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode()
                st.markdown(f"""
                <div style="
                    min-height: 500px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 5px 0 40px 0;
                ">
                    <img src="data:image/png;base64,{encoded_string}" style="
                        max-width: 88%;
                        max-height: 360px;
                        object-fit: contain;
                        border-radius: 8px;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    ">
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Image load error: {e}")

        with col2:
            st.markdown("""
            <div style="min-height: 500px; border-left: 5px solid #1f77b4; background-color: #f8fbff; padding: 20px 25px; border-radius: 0 8px 8px 0; margin-top: 5px; margin-bottom: 40px; display: flex; flex-direction: column; justify-content: center;">
                <div style="color: #000000; line-height: 1.6; font-size: 20px;">
                    <div style="font-size: 28px; font-weight: 800; margin-bottom: 5px;">
                        What this dashboard does
                    </div>
                    <div style="margin-left: 18px; margin-bottom: 26px;">
                        - Detects early warning signals from seasonal time-series data<br>
                        - Identifies when signal activity begins within a season
                    </div>
                    <div style="font-size: 28px; font-weight: 800; margin-bottom: 5px;">
                        How the analysis works
                    </div>
                    <div style="margin-left: 18px; margin-bottom: 26px;">
                        - The full retrospective period is used to learn a stable baseline pattern<br>
                        - <strong>At least 1 year</strong> of data is required; <strong>4 years</strong> or more uses the standard season algorithm<br>
                    </div>
                    <div style="font-size: 28px; font-weight: 800; margin-bottom: 5px;">
                        How to set it up
                    </div>
                    <div style="margin-left: 18px;">
                        - Upload your data, configure the settings, and click <strong>Run Analysis</strong><br>
                        - Review retrospective signal detection results in the <strong>Analysis Report</strong>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    with st.expander("1. Setup Guide", expanded=False):
        import os
        import base64
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        setting_img_path = os.path.join(current_dir, "images", "Setting_guide.png")
        
        try:
            with open(setting_img_path, "rb") as image_file:
                encoded_setting_img = base64.b64encode(image_file.read()).decode()
            st.markdown(f"""
                <div style="display: flex; justify-content: center; padding: 20px 0;">
                    <img src="data:image/jpeg;base64,{encoded_setting_img}" style="max-width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Image load error: {e}")
            st.info("💡 'images' 폴더 안에 'Setting_guide.jpg' 파일이 있는지, 파일명 대소문자가 정확한지 확인해주세요!")

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
        status = st.status("Preprocessing...", expanded=True)
        
        try:
            proc_data = raw_data.copy()
            proc_data['Date'] = proc_data[date_col]
            if 'Year' not in proc_data.columns:
                proc_data['Year'] = proc_data['Date'].dt.year
            if 'Week' not in proc_data.columns:
                proc_data['Week'] = proc_data['Date'].dt.isocalendar().week
            status.update(label="Preprocessing...", state="running", expanded=True)
            season_df, season_meta = set_season_start_week_adaptive(proc_data, target_col)
            if season_meta.get('mode') == 'insufficient':
                st.error(season_meta.get('reason', 'At least 1 year of data is required.'))
                st.stop()

            if season_df.empty:
                st.error("No valid seasons were detected from the selected data.")
                st.stop()

            seasons = season_df['season'].to_list()
            visual_peak_start, peak_len = visualization_season(proc_data, season_df, start_week=manual_start_week)
            if season_meta.get('mode') == 'standard':
                peak_start = visual_peak_start
            else:
                peak_start = season_meta.get('start_week')
                if peak_start is None:
                    st.error("No valid short-history season start week was detected.")
                    st.stop()

            data, period_meta = assign_retrospective_period(
                proc_data,
                seasons,
                start_week=peak_start,
                season_starts=season_df if season_meta.get('mode') != 'standard' else None
            )
            window_eval_seasons = period_meta['window_eval_seasons']
            if window_eval_seasons:
                hockey_date, hockey_df = hockey_stick_regression(data, target_col, HockeyStick_type, window_eval_seasons)
            else:
                hockey_date, hockey_df = [], pd.DataFrame()
                st.warning("No complete season was available for window-size optimization. A default 12-week window was used.")
            cusum_result = cumulative_sum_adaptive(
                data.copy(),
                target_col,
                mode=season_meta.get('mode', 'standard'),
                season_start_week=peak_start
            )
            data['cusum'] = cusum_result['cusum'].values
            
            data.reset_index(drop=True, inplace=True)
            data['num'] = data.index
            
            proc_data = data.copy()

            status.update(label="Analyzing...", state="running", expanded=True)
            if window_eval_seasons:
                best_window, best_score = optimize_window_size(
                    proc_data,
                    target_col,
                    hockey_date,
                    window_eval_seasons,
                    peak_start
                )
            else:
                best_window, best_score = 12, np.inf
            st.toast(f"Optimal Window Size Auto-selected: {best_window} Weeks")

            feature_col = ['slope', 'mean', 'CS_mean']
            df_analysis, data_all_analysis = make_raw(proc_data, 'analysis', best_window, target_col)
            valid_analysis_mask = df_analysis[feature_col].notna().all(axis=1)
            df_analysis = df_analysis.loc[valid_analysis_mask].reset_index(drop=True)
            data_all_analysis = data_all_analysis.loc[valid_analysis_mask].reset_index(drop=True)

            if df_analysis.empty or data_all_analysis.empty:
                st.error("No valid analysis windows were available after feature calculation.")
                st.stop()

            status.update(label="Analyzing...", state="running", expanded=True)

            result_data_t, kmeans, best_k, scaler = K_means_clustering(df_analysis)
            warning_label_t = result_data_t['label'].max()
            ED_date = find_warning_periods(result_data_t, data_all_analysis, peak_start, warning_label_t)

            boot_ensemble, scaler = train_bootstrap_ensemble(df_analysis, scaler, feature_col, B=boot_num, k_best=best_k, types='random')
            status.update(label="Analyzing...", state="running", expanded=True)

            status.update(label="Visualizing...", state="running", expanded=True)

            date_df, label_df = analyze_train_distribution(
                df_analysis,
                data_all_analysis,
                feature_col,
                boot_ensemble,
                scaler,
                peak_start,
                ED_date
            )

            season_summary_blocks = []
            for season in date_df.columns:
                _, summary = summarize_detection_progression(date_df[season])
                d_blue = format_alert_date(summary.get('blue_date'))
                d_orange = format_alert_date(summary.get('orange_date'))
                d_red = format_alert_date(summary.get('red_date'))
                season_summary_blocks.append(f"""
                <div style="margin-bottom: 18px;">
                    <div style="font-size: 18px; font-weight: 800; color: #333; margin-bottom: 10px;">
                        {int(season)}-{int(season)+1} Season
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
                </div>
                """)

            st.markdown("---")
            st.header("Analysis Report")

            analysis_period_text = (
                f"{period_meta['analysis_start'].strftime('%Y/%m/%d')} ~ "
                f"{period_meta['analysis_end'].strftime('%Y/%m/%d')}"
            )
            season_count_text = (
                f"{len(period_meta['analysis_seasons'])} seasons detected, "
                f"{len(window_eval_seasons)} complete seasons used for window optimization "
                f"({season_meta.get('mode', 'standard')} mode)"
            )

            st.markdown("""
            <style>
                .period-card-grid {
                    display: grid;
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                    gap: 20px;
                    margin: 14px 0 30px 0;
                }
                .period-card {
                    background: linear-gradient(180deg, #ffffff 0%, #f7f9fc 100%);
                    border: 1px solid #dbe4f0;
                    border-radius: 18px;
                    padding: 22px 22px 20px 22px;
                    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
                    min-height: 220px;
                    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
                }
                .period-card:hover {
                    transform: translateY(-4px);
                    box-shadow: 0 18px 36px rgba(15, 23, 42, 0.12);
                    border-color: #bfdbfe;
                }
                .period-card-title {
                    font-size: 16px;
                    font-weight: 800;
                    color: #334155;
                    margin-bottom: 14px;
                    letter-spacing: 0.25px;
                }
                .period-card-main {
                    font-size: 30px;
                    font-weight: 800;
                    color: #0f172a;
                    margin-bottom: 18px;
                    line-height: 1.3;
                }
                .period-card-label {
                    font-size: 13px;
                    font-weight: 700;
                    color: #64748b;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    margin-bottom: 6px;
                }
                .period-card-detail {
                    font-size: 16px;
                    font-weight: 600;
                    color: #1e293b;
                    line-height: 1.55;
                    margin-bottom: 16px;
                }
                .period-card-note {
                    font-size: 14px;
                    color: #475569;
                    line-height: 1.7;
                    white-space: pre-line;
                    margin-top: 0;
                }
                .report-nav {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 12px;
                    margin: 2px 0 26px 0;
                }
                .report-nav a {
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    padding: 10px 16px;
                    border-radius: 999px;
                    border: 1px solid #cbd5e1;
                    background: #ffffff;
                    color: #1e293b;
                    font-size: 14px;
                    font-weight: 700;
                    text-decoration: none;
                    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
                    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
                }
                .report-nav a:hover {
                    transform: translateY(-2px);
                    border-color: #93c5fd;
                    box-shadow: 0 10px 22px rgba(15, 23, 42, 0.10);
                }
                .report-anchor {
                    display: block;
                    position: relative;
                    top: -84px;
                    visibility: hidden;
                }
                @media (max-width: 900px) {
                    .period-card-grid {
                        grid-template-columns: 1fr;
                    }
                }
            </style>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="period-card-grid">
                <div class="period-card">
                    <div class="period-card-title">Analysis Period</div>
                    <div class="period-card-main">{analysis_period_text}</div>
                    <div class="period-card-label">Retrospective range</div>
                    <div class="period-card-detail">The full available period is used to learn a stable signal pattern.</div>
                </div>
                <div class="period-card">
                    <div class="period-card-title">Window Calibration</div>
                    <div class="period-card-main">{best_window} Weeks</div>
                    <div class="period-card-label">Selected window size</div>
                    <div class="period-card-detail">{season_count_text}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="report-nav">
                <a href="#overall-period-analysis">1. Overall</a>
                <a href="#retrospective-analysis">2. Retrospective Analysis</a>
                <a href="#season-summary">3. Season Summary</a>
            </div>
            """, unsafe_allow_html=True)

            other_dates = None
            if reference_dates is not None and len(reference_dates) > 0:
                other_dates = {reference_label or 'User Input Date': reference_dates}

            st.markdown('<span id="overall-period-analysis" class="report-anchor"></span>', unsafe_allow_html=True)
            st.subheader("1. Overall Period Analysis")
            fig_overall = overall_period_visualization_bootstrap(
                proc_data,
                target_col,
                other_dates,
                hockey_date,
                date_df,
                best_window
            )
            st.plotly_chart(fig_overall, use_container_width=True)

            st.markdown("""
            <div style="background-color: #f9f9f9; padding: 15px; border-left: 5px solid #2e86c1; margin-bottom: 20px;">
                <span style="font-size: 22px;"><strong>Overall Period Analysis</strong></span><br>
                <span style="font-size: 16px; color: #444; line-height: 1.6;">
                    <ul style="margin-top: 10px;">
                        <li>This panel shows the full analysis time series for context.</li>
                        <li>The chart keeps the visual focus on the observed signal pattern across the full period.</li>
                    </ul>
                </span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")

            st.markdown('<span id="retrospective-analysis" class="report-anchor"></span>', unsafe_allow_html=True)
            st.subheader("2. Retrospective Bootstrap Early Warning Detection")

            fig1 = early_warning_visualization_bootstrap(
                proc_data, data_all_analysis, target_col, other_dates, hockey_date, date_df, best_window
            )
            st.plotly_chart(fig1, use_container_width=True)

            st.markdown("""
            <div style="background-color: #f9f9f9; padding: 15px; border-left: 5px solid #2e86c1; margin-bottom: 20px;">
                <span style="font-size: 22px;"><strong>Retrospective Bootstrap Early Warning Detection</strong></span><br>
                <span style="font-size: 16px; color: #444; line-height: 1.6;">
                    <ul style="margin-top: 10px;">
                        <li>The model is fitted on the full retrospective period instead of a train/test split.</li>
                        <li><strong style="color: #1F77B4;">Blue / Orange / Red Dashed Lines</strong>: Mark the first dates when the cumulative bootstrap detection share becomes positive, reaches 5%, and reaches 10% within each season.</li>
                        <li><strong>Cumulative Bars</strong>: Show how many bootstrap models have already issued their first warning by each date.</li>
                    </ul>
                </span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")

            st.markdown('<span id="season-summary" class="report-anchor"></span>', unsafe_allow_html=True)
            st.subheader("3. Early Warning Timeline Summary by Season")
            if season_summary_blocks:
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
                    }}
                    .summary-card:hover {{
                        background-color: #f1f3f5 !important;
                        transform: translateY(-3px);
                    }}
                </style>
                {''.join(season_summary_blocks)}
                """, unsafe_allow_html=True)
            else:
                st.info("No seasonal warning dates were detected by the bootstrap ensemble.")

            st.markdown("""
            <div style="background-color: #f9f9f9; padding: 15px; border-left: 5px solid #2e86c1; margin-bottom: 20px;">
                <span style="font-size: 22px;"><strong>Retrospective Analysis Results</strong></span><br>
                <span style="font-size: 16px; color: #444; line-height: 1.6;">
                    <ul style="margin-top: 10px;">
                        <li>The full available period is used together for model fitting and retrospective signal detection.</li>
                        <li>The sliding-window size is selected by comparing bootstrap warning dates with hockey-stick reference dates from complete seasons only.</li>
                        <li>The final incomplete season is retained in the retrospective chart, but excluded from window-size calibration.</li>
                        <li><strong>Interactive View:</strong> Use the <b>range slider</b> at the bottom of the chart to zoom into specific periods.</li>
                    </ul>
                </span>
            </div>
            """, unsafe_allow_html=True)

            status.update(label="Analysis Complete", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Analysis Error", state="error", expanded=True)
            st.error(f"Details: {e}")
            st.exception(e)
    else:
        st.markdown("---")
        st.header("Analysis Report")
        st.info("Please review the settings in the left sidebar, then click Run Analysis to view the Analysis Report.")
