import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Import the modules used across preprocessing, modeling, and visualization.
try:
    from config import Config
    from src.preprocessing import make_raw, cumulative_sum_hybrid
    from src.clustering import (
        K_means_clustering, 
        find_warning_periods,
        train_bootstrap_ensemble, 
        analyze_train_distribution,
        analyze_distribution_with_bootstrap,
        predict_new_data_probability,
        summarize_detection_progression
    )
    from src.visualization import (
        early_warning_visualization_bootstrap, 
        overall_period_visualization_bootstrap,
        visualization_season,
        interactive_real_time_chart,
        interactive_real_time_chart_combined
    )
    from src.season_setting import (
        set_season_start_week,
        hockey_stick_regression,
        assign_analysis_periods,
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

st.set_page_config(
    page_title="Epidemic Early Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.title("Settings")
    
    st.header("1. Data Input")
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

    # Require at least four years of data before the fitting end date.
    min_data_date = raw_data[date_col].min().normalize()
    default_fit_end = min_data_date + pd.DateOffset(years=4)
    max_data_date = raw_data[date_col].max().normalize()
    if default_fit_end > max_data_date:
        default_fit_end = max_data_date

    fit_input_mode = st.radio(
        "Fitting End Input Type",
        ["Date", "Year / Week"],
        horizontal=True,
        help="Choose either a calendar date or a year/week pair."
    )

    fit_end_date_direct = st.date_input(
        "Fitting End Date",
        value=default_fit_end.to_pydatetime(),
        min_value=min_data_date.to_pydatetime(),
        max_value=max_data_date.to_pydatetime(),
        disabled=(fit_input_mode != "Date"),
        help="At least 4 years of data are required before the fitting end date."
    )

    fit_date_lookup = (
        raw_data[[date_col]]
        .assign(
            FitYear=raw_data[date_col].dt.year.astype(int),
            FitWeek=raw_data[date_col].dt.isocalendar().week.astype(int)
        )
        .drop_duplicates()
        .sort_values(date_col)
        .reset_index(drop=True)
    )

    available_fit_years = fit_date_lookup["FitYear"].drop_duplicates().tolist()
    default_fit_year = int(default_fit_end.year)
    default_fit_year_idx = (
        available_fit_years.index(default_fit_year)
        if default_fit_year in available_fit_years
        else max(0, len(available_fit_years) - 1)
    )

    fit_year = st.selectbox(
        "Fitting End Year",
        available_fit_years,
        index=default_fit_year_idx,
        disabled=(fit_input_mode != "Year / Week"),
    )

    available_fit_weeks = (
        fit_date_lookup.loc[fit_date_lookup["FitYear"] == fit_year, "FitWeek"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    default_fit_week = int(default_fit_end.isocalendar().week)
    default_fit_week_idx = (
        available_fit_weeks.index(default_fit_week)
        if default_fit_week in available_fit_weeks
        else max(0, len(available_fit_weeks) - 1)
    )

    fit_week = st.selectbox(
        "Fitting End Week",
        available_fit_weeks,
        index=default_fit_week_idx,
        disabled=(fit_input_mode != "Year / Week"),
    )

    if fit_input_mode == "Date":
        fit_end_date = pd.to_datetime(fit_end_date_direct).normalize()
    else:
        fit_candidates = fit_date_lookup.loc[
            (fit_date_lookup["FitYear"] == fit_year) &
            (fit_date_lookup["FitWeek"] == fit_week),
            date_col
        ]
        if fit_candidates.empty:
            st.error("No valid date was found for the selected year/week.")
            st.stop()
        fit_end_date = pd.to_datetime(fit_candidates.max()).normalize()
        st.caption(f"Resolved fitting end date: {fit_end_date.date()}")

    st.markdown("---")
    st.header("2. Optional Reference Dates")

    reference_file = st.file_uploader(
        "Upload Reference Date File (.xlsx)",
        type=["xlsx", "xls"],
        key="reference_date_file"
    )

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
    st.header("3. Parameters")
    
    boot_num = st.number_input("Bootstrap Iterations", 50, 2000, 200, step=50, key="boot_num_input")
    HockeyStick_type = "linear"
    
    st.markdown("---")
    run_btn = st.button("Start Analysis", type="primary")

st.title("Universal Respiratory Epidemic Early Detection System")

st.markdown("""
<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 25px; font-size: 18px; line-height: 1.6;">
    <strong style="font-size: 22px;">[System Description]</strong><br>
    This dashboard is designed for the early detection of infectious diseases.<br>
    Please upload your data in the sidebar on the left, configure the settings below, and click <strong>'Start Analysis'</strong>.<br><br>
    <em>For detailed instructions on the settings, please refer to <strong>Tab 1</strong> below.</em>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Manual & Settings Guide", "Dashboard Analysis"])
with tab1:
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
    with st.expander("0. Introduction", expanded=True):
        col1, col2 = st.columns([0.5, 2.5], gap="large")
        
        with col1:
            import os
            import base64
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            img_path = os.path.join(current_dir, "images", "intro_example.png")
            
            try:
                with open(img_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode()
                st.markdown(f'<div style="text-align: center; margin-top: 5px; margin-bottom: 40px;"><img src="data:image/png;base64,{encoded_string}" width="200" style="border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"></div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Image load error: {e}")

        with col2:
            st.markdown("""
            <div style="border-left: 5px solid #1f77b4; background-color: #f8fbff; padding: 20px 25px; border-radius: 0 8px 8px 0; margin-top: 5px; margin-bottom: 40px;">
                <div style="color: #333; line-height: 1.8;">
                    <p style="font-size: 22px; font-weight: 500; margin-bottom: 15px;">This dashboard is a tool that defines when an outbreak started using seasonal infectious disease data, and uses this to determine when the next outbreak will occur during an ongoing season.</p>
                    <p style="font-size: 22px; font-weight: 500; margin-bottom: 15px;">A 'season' is defined from the start to the end of a single outbreak wave, and the dashboard is equipped with an algorithm to detect this automatically.</p>
                    <p style="font-size: 22px; font-weight: 500; margin-bottom: 0;">Therefore, the real-time detection period can only be verified for the final, ongoing season.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    with st.expander("1. Setting Guide", expanded=False):
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
            season_df = set_season_start_week(proc_data, target_col)
            seasons = season_df['season'].to_list()
            
            peak_start, peak_len = visualization_season(proc_data, season_df, start_week=manual_start_week)
            
            fit_end_date = pd.to_datetime(fit_end_date)
            if fit_end_date < (proc_data['Date'].min() + pd.DateOffset(years=4)):
                st.error("Fitting End Date must be at least 4 years after the first available date.")
                st.stop()

            data, period_meta = assign_analysis_periods(
                proc_data,
                seasons,
                start_week=peak_start,
                fit_end_date=fit_end_date
            )
            
            status.update(label="Preprocessing...", state="running", expanded=True)
            hockey_date, hockey_df = hockey_stick_regression(data, target_col, HockeyStick_type, seasons)
            status.update(label="Preprocessing...", state="running", expanded=True)
            cusum_result = cumulative_sum_hybrid(data.copy(), target_col, season_start_week=peak_start)
            data['cusum'] = cusum_result['cusum'].values
            
            data.reset_index(drop=True, inplace=True)
            data['num'] = data.index
            
            proc_data = data.copy()

            status.update(label="Analyzing...", state="running", expanded=True)
            best_window, best_score = optimize_window_size(proc_data, target_col, hockey_date, seasons, peak_start)
            st.toast(f"Optimal Window Size Auto-selected: {best_window} Weeks")
                
            df_train, data_all_train = make_raw(proc_data, 'train', best_window, target_col)
            df_train = df_train.dropna()

            status.update(label="Analyzing...", state="running", expanded=True)

            status.update(label="Analyzing...", state="running", expanded=True)
            
            feature_col = ['slope', 'mean', 'CS_mean']
            feature_name = [r'$\beta_{\omega}$', r'$\mu_{\omega}$', r'$\widebar{S_{\omega}}$']
            
            # Baseline K-means (C0)
            result_data_t, kmeans, best_k, scaler = K_means_clustering(df_train)
            warning_label_t = result_data_t['label'].max()
            ED_date = find_warning_periods(result_data_t, data_all_train, peak_start, warning_label_t)
            
            boot_ensemble, scaler = train_bootstrap_ensemble(df_train, scaler, feature_col, B=boot_num, k_best=best_k, types='random')
            status.update(label="Analyzing...", state="running", expanded=True)

            status.update(label="Visualizing...", state="running", expanded=True)
            
            date_df, label_df = analyze_train_distribution(df_train, data_all_train, feature_col, boot_ensemble, scaler, peak_start, ED_date)
            train_season_summaries = {}
            for season in date_df.columns:
                _, summary = summarize_detection_progression(date_df[season])
                train_season_summaries[int(season)] = summary

            overall_plot_data = proc_data.copy()
            overall_plot_data['set'] = 'train'
            df_overall, data_all_overall = make_raw(overall_plot_data, 'train', best_window, target_col)
            valid_overall_mask = df_overall[feature_col].notna().all(axis=1)
            df_overall = df_overall.loc[valid_overall_mask].reset_index(drop=True)
            data_all_overall = data_all_overall.loc[valid_overall_mask].reset_index(drop=True)
            overall_date_df, overall_label_df = analyze_distribution_with_bootstrap(
                df_overall,
                data_all_overall,
                feature_col,
                boot_ensemble,
                scaler,
                peak_start
            )
            
            st.markdown("---")
            st.header("Analysis Report")

            train_display_dates = proc_data.loc[proc_data['set'] == 'train', 'Date']
            fit_display_start = train_display_dates.min() if not train_display_dates.empty else proc_data['Date'].min()

            fitting_period_text = (
                f"{fit_display_start.strftime('%Y/%m/%d')} ~ "
                f"{period_meta['fit_end_date_user'].strftime('%Y/%m/%d')}"
            )
            realtime_period_text = (
                f"{period_meta['simulation_start'].strftime('%Y/%m/%d')} ~ "
                f"{proc_data['Date'].max().strftime('%Y/%m/%d')}"
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
                    <div class="period-card-title">Fitting Period</div>
                    <div class="period-card-main">{fitting_period_text}</div>
                    <div class="period-card-label">Model fitting range</div>
                    <div class="period-card-detail">The first 3 years are reserved for feature extraction and are not used in fitting.</div>
                </div>
                <div class="period-card">
                    <div class="period-card-title">Real-time Monitoring</div>
                    <div class="period-card-main">{realtime_period_text}</div>
                    <div class="period-card-label">Simulation range</div>
                    <div class="period-card-detail">Simulation starts immediately after the selected fitting end date and proceeds season by season.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="report-nav">
                <a href="#overall-period-analysis">1. Overall</a>
                <a href="#fitting-period-analysis">2. Fitting Period</a>
                <a href="#simulation-analysis">3. Simulation</a>
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
                overall_date_df,
                best_window,
                period_meta['fit_end_date_user']
            )
            st.plotly_chart(fig_overall, use_container_width=True)

            st.markdown("""
            <div style="background-color: #f9f9f9; padding: 15px; border-left: 5px solid #2e86c1; margin-bottom: 20px;">
                <span style="font-size: 22px;"><strong>Overall Period Analysis</strong></span><br>
                <span style="font-size: 16px; color: #444; line-height: 1.6;">
                    <ul style="margin-top: 10px;">
                        <li>This panel shows the full original time series for context.</li>
                        <li>The blue shaded area marks the fitting period.</li>
                        <li>The red shaded area marks the simulation period.</li>
                    </ul>
                </span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")

            st.markdown('<span id="fitting-period-analysis" class="report-anchor"></span>', unsafe_allow_html=True)
            st.subheader("2. Bootstrap Early Warning Detection")
            
            # 새로운 시각화 함수 적용
            fig1 = early_warning_visualization_bootstrap(
                proc_data, data_all_train, target_col, other_dates, hockey_date, date_df, best_window
            )
            
            st.plotly_chart(fig1, use_container_width=True)
                
            st.markdown("""
            <div style="background-color: #f9f9f9; padding: 15px; border-left: 5px solid #2e86c1; margin-bottom: 20px;">
                <span style="font-size: 22px;"><strong>Bootstrap Early Warning Detection (Train Data)</strong></span><br>
                <span style="font-size: 16px; color: #444; line-height: 1.6;">
                    <ul style="margin-top: 10px;">
                        <li><strong style="color: #2CA02C;">Green Line (Hockey Stick)</strong>: Identifies the structural break point in the epidemic curve. This mathematical approach automatically optimizes the sliding window size by detecting the exact moment the trend shifts to exponential growth.</li>
                        <li><strong style="color: #1F77B4;">Blue / Orange / Red Dashed Lines</strong>: Mark the first dates when the cumulative bootstrap detection share becomes positive, reaches 5%, and reaches 10% within each season.</li>
                        <li><strong>Cumulative Bars</strong>: Show how many bootstrap models have already issued their first warning by each date.</li>
                    </ul>
                </span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")
            
            st.markdown('<span id="simulation-analysis" class="report-anchor"></span>', unsafe_allow_html=True)
            st.subheader("3. Real-time Surveillance")
            status.update(label="Simulation in Progress...", state="running", expanded=True)
            
            try:
                test_seasons = period_meta['test_seasons']
                if not test_seasons:
                    st.error("No valid Test data available in the selected range.")
                else:
                    status.update(label="Simulation Visualization...", state="running", expanded=True)
                    season_rt_results = []
                    season_summary_blocks = []
                    overlap_seasons = set(period_meta.get('overlap_seasons', []))

                    for season in test_seasons:
                        df_test, data_all_test = make_raw(
                            proc_data, 'test', best_window, target_col, target_season=season
                        )

                        if df_test.empty or data_all_test.empty:
                            continue

                        feature_cols_rt = ['slope', 'mean', 'CS_mean']
                        valid_mask = df_test[feature_cols_rt].notna().all(axis=1)
                        df_test = df_test.loc[valid_mask].reset_index(drop=True)
                        data_all_test = data_all_test.loc[valid_mask].reset_index(drop=True)

                        if df_test.empty or data_all_test.empty:
                            st.warning(f"{int(season)}-{int(season)+1} season was skipped because real-time features contained only missing values.")
                            continue

                        season_test_dates = proc_data.loc[
                            (proc_data['set'] == 'test') & (proc_data['Season'] == season),
                            'Date'
                        ]
                        if season_test_dates.empty:
                            continue

                        season_sim_start = season_test_dates.min()
                        season_start_display = proc_data.loc[proc_data['Season'] == season, 'Date'].min()
                        season_end_display = proc_data.loc[proc_data['Season'] == season, 'Date'].max()
                        rt_display_data = proc_data.loc[
                            (proc_data['Date'] >= season_sim_start) &
                            (proc_data['Date'] <= season_end_display)
                        ].copy()
                        fit_summary = train_season_summaries.get(int(season), {})
                        fit_red_date = fit_summary.get('red_date')
                        season_suppressed = (int(season) in overlap_seasons) and pd.notna(pd.to_datetime(fit_red_date, errors='coerce'))

                        detection_timeline = pd.DataFrame(columns=['Date', 'Cumulative_Count', 'Cumulative_Ratio', 'Level'])
                        shaded_range = None

                        if season_suppressed:
                            shaded_range = (season_sim_start, season_end_display)
                            d_blue = "Suppressed"
                            d_orange = "Suppressed"
                            d_red = "Suppressed"
                        else:
                            initial_detection_dates = date_df[season] if season in overlap_seasons and season in date_df.columns else None
                            _, _, bootstrap_dates, _ = predict_new_data_probability(
                                df_test,
                                data_all_test,
                                boot_ensemble,
                                scaler,
                                peak_start,
                                step=1,
                                initial_detection_dates=initial_detection_dates
                            )

                            final_detect_dates = bootstrap_dates.iloc[:, -1] if not bootstrap_dates.empty else pd.Series(dtype='datetime64[ns]')
                            detection_timeline, _ = summarize_detection_progression(
                                final_detect_dates,
                                monitoring_start=season_sim_start
                            )

                            _, d_blue, d_orange, d_red = interactive_real_time_chart(
                                data_all=rt_display_data,
                                detection_timeline=detection_timeline,
                                other_dates=other_dates,
                                epi=target_col,
                                shaded_range=shaded_range
                            )

                        season_rt_results.append({
                            'season': season,
                            'display_data': rt_display_data,
                            'detection_timeline': detection_timeline,
                            'shaded_range': shaded_range,
                            'season_boundary': season_start_display,
                            'simulation_start': season_sim_start,
                            'suppressed': season_suppressed,
                        })

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
                            {"<div style='font-size: 13px; color: #6b7280; margin-top: -8px;'>Simulation is gray-shaded because this season already triggered a red alert during fitting.</div>" if season_suppressed else ""}
                        </div>
                        """)

                    if season_rt_results:
                        fig_interactive_combined = interactive_real_time_chart_combined(
                            season_rt_results,
                            target_col
                        )

                        st.plotly_chart(fig_interactive_combined, use_container_width=True)
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
                            <span style="font-size: 20px; font-weight: 800; color: #333; letter-spacing: 0.5px;">Early Warning Timeline Summary by Season</span>
                        </div>
                        {''.join(season_summary_blocks)}
                        """, unsafe_allow_html=True)

                        st.divider()

            except Exception as e:
                st.error(f"Error during data slicing/prediction: {e}")
            
            status.update(label="Analysis Complete", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Analysis Error", state="error", expanded=True)
            st.error(f"Details: {e}")
            st.exception(e)

        st.markdown("""
        <div style="background-color: #f9f9f9; padding: 15px; border-left: 5px solid #2e86c1; margin-bottom: 20px;">
            <span style="font-size: 22px;"><strong>Real-Time Surveillance Results</strong></span><br>
            <span style="font-size: 16px; color: #444; line-height: 1.6;">
                <ul style="margin-top: 10px;">
                    <li>After the fitting period ends, the dashboard <strong>starts simulation immediately</strong> and tracks each season prospectively.</li>
                    <li>The secondary axis shows the <strong>cumulative number of bootstrap models</strong> that have already produced their first seasonal warning by each date.</li>
                    <li>An official seasonal alert is triggered when the cumulative share reaches <strong>10% or more</strong>, and each season can alert only once.</li>
                    <li>If a season already triggered an alert during fitting, its simulation segment is shown as a <strong>gray-shaded block</strong> and no additional alert is issued.</li>
                    <li><strong>Interactive View:</strong> Use the <b>range slider</b> at the bottom of the chart to zoom into specific periods.</li>
                    <li><strong>Summary Cards:</strong> The cards below indicate the exact dates when each risk level (Attention, Caution, Alert) was first detected by the ensemble model.</li>
                </ul>
            </span>
        </div>
        """, unsafe_allow_html=True)
