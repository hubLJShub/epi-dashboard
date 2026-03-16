import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# from mpl_toolkits.mplot3d import Axes3D

def visualization_season(data, results_df, start_week = 1, half_year_weeks = 26):
    from matplotlib.colors import ListedColormap

    # start_week 기준으로 2년동안의 범위로 설정
    season_pattern = list(range(start_week, 54)) + list(range(1, start_week))
    season_weeks = season_pattern * 2
    TOTAL_WEEKS = len(season_weeks)

    seasons = sorted(results_df['season'])
    # heatmap 행 구성: 3행 단위 (첫 번째, 두 번째, 간격)
    heatmap_rows = []
    for s in seasons:
        heatmap_rows.append(s)
        heatmap_rows.append(s)
        heatmap_rows.append(None)
    heatmap_df = pd.DataFrame(
        0,
        index=range(len(heatmap_rows)),
        columns=range(TOTAL_WEEKS)
    )

    # 주차 → 인덱스 변환 함수
    week_to_idx_map = {w: i for i, w in enumerate(season_pattern)}
    def week_to_index(week, week_to_idx_map):
        return week_to_idx_map[week]

    # peak 범위 저장
    peak_list = []
    old_epi_start = week_to_index(results_df.iloc[0,:].start_week, week_to_idx_map)
    
    for i, r in enumerate(results_df.itertuples()):
        s_row1 = i*3
        s_row2 = i*3 + 1
        
        # 피크 시점
        peak_idx = week_to_index(r.peak_week, week_to_idx_map)
        
        # 기존 유행 시즌
        old_epi_end = old_epi_start + 52  # 52주 범위

        # peak_idx가 범위 내 없으면 두 번째 패턴으로
        if not (old_epi_start <= peak_idx <= old_epi_end):
            peak_idx += 53
        peak_list.append(peak_idx+1) # 시각화는 0부터 시작해서 +1함

        # 새로운 유행 시즌
        epi_start = max(0, peak_idx - half_year_weeks-1)
        if r.peak_week==53:
            epi_start = epi_start-1
        epi_end = min(TOTAL_WEEKS - 1, peak_idx + half_year_weeks-1)

        # 첫 번째 행: 기존 유행 시즌 + peak
        heatmap_df.loc[s_row1, old_epi_start:old_epi_end] = 3 # 기존 유행 시즌 표시
        if hasattr(r, 'epi_weeks') and r.epi_weeks is not None:
            for w in r.epi_weeks:
                w_idx = week_to_index(w, week_to_idx_map)
                if not (old_epi_start <= w_idx <= old_epi_end):
                    w_idx += 53  # 두 번째 패턴으로 이동
                heatmap_df.loc[s_row1, w_idx] = 4  # 유행 기간 표시
        heatmap_df.loc[s_row1, peak_idx] = 2 # 피크 시점 표시

        # 두 번째 행: 새로운 유행 시즌 + peak
        heatmap_df.loc[s_row2, epi_start:epi_end] = 1 # 새로운 유행 시즌 표시
        heatmap_df.loc[s_row2, peak_idx] = 2 # 피크 시점 표시

        old_epi_start = epi_start # 업데이트

    cmap = ListedColormap([
        (1, 1, 1, 1.0),        # 0: white
        (0.4, 0.6, 0.9, 0.6),  # 1: steelblue
        (0.9, 0.3, 0.3, 0.8),  # 2: crimson (peak)
        (1.0, 0.7, 0.8, 0.5),  # 3: pink (epi_start~end)
        (1.0, 0.85, 0.0, 0.8)  # 4: gold (epi_weeks)
    ])


    # 시각화
    fig, ax = plt.subplots(figsize=(20, 0.5 * len(heatmap_df)), dpi=300)
    ax.grid(False)
    ax.imshow(
        heatmap_df.values,
        aspect='auto',
        cmap=cmap,
        interpolation='none',
        alpha=1
    )

    xticks = list(range(TOTAL_WEEKS))
    xtick_labels = season_pattern * 2
    ax.set_xticks(range(0, len(xticks), 2))
    ax.set_xticklabels([str(w) for w in xtick_labels[::2]], fontsize=8)

    yticks = [i*3 + 0.5 for i in range(len(seasons))]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{s}-{s+1} season" for s in seasons], fontsize=8)

    ax.set_yticks(np.arange(-0.5, len(heatmap_df), 1), minor=True)
    ax.set_xticks(np.arange(-0.5, TOTAL_WEEKS, 1), minor=True)
    ax.grid(which='minor', axis='both', linestyle='-', linewidth=0.5)

    special_y = (3 * np.arange(len(seasons))[:, None] + np.array([1.5, 2.5])).ravel()
    for y in special_y:
        ax.hlines(y, -0.5, heatmap_df.shape[1]-0.5, color='black', linewidth=1.0)
    
    ax.set_xlabel("Week")

    SEASON_WEEKS = 52

    peak_min = np.min(peak_list)
    peak_max = np.max(peak_list)
    peak_range = peak_max - peak_min
    tmp = np.ceil((SEASON_WEEKS - peak_range)/2)
    peak_start_idx = int(peak_min - tmp)

    peak_start_week = season_pattern[peak_start_idx % SEASON_WEEKS] - 1

    peak_value = [int(x - peak_start_idx + 1) for x in peak_list]
    peak_len = dict(zip(seasons, peak_value))

    ax.axvline(peak_start_idx-1, color='red', linestyle='--', alpha=0.7)
    ax.axvline(peak_start_idx+51, color='red', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return int(peak_start_week), peak_len

# K-means clustering 관련 시각화를 진행하는 함수
def K_means_visualization(result_data, input_var, var_name): 
    result_df = result_data.copy()
    result_df['label'] = result_df['label'].astype(str)
    palette = {'0': "#348ABD", '1': "#A6D854", '2': "#D62728", '3': "#D62728"}
    order = np.sort(result_df['label'].unique())
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes = axes.ravel()
    for i, col in enumerate(input_var):
        ax = axes[i]
        ax.grid(False)
        sns.boxplot(data=result_df, x='label', y=col, order=order, palette=palette, width=0.6, ax=ax, showfliers=True)
        ax.set_title(f'{var_name[i]}')
        ax.set_xlabel('Cluster')
        ax.set_ylabel(var_name[i])
        ax.set_xticklabels([f'C{c}' for c in order])
    for j in range(len(input_var), len(axes)):
        axes[j].remove()
    plt.tight_layout()
    plt.show()
# 조기 탐지 시각화
def early_warning_visualization(data, data_all, epi, result_data, ED_date, Hockey_date, other_dates, sample_window):
    ### 1. 다른 결과와 비교
    fig, ax = plt.subplots(figsize=(10, 2), dpi=400)

    x=data_all['Date']
    y=data_all[epi]
    max_y = y.max()

    # Data: 전체 데이터를 시각화
    ax.bar(data['Date'],data[epi],color='gray',alpha=0.5, label=epi, width=5.5)
    ax.axhline(y=max_y*1.3, color='black', linewidth=1)

    ax.set_xlabel('Week')
    ax.set_ylabel(epi)
    ax.grid(False)

    # Other dates
    colors = ['orange', 'red']
    markers = ['^', '*']
    line_style = ['--', '-']
    if other_dates is not None:
        for j, key in enumerate(other_dates.keys()):
            for i, date in enumerate(other_dates[key]):
                ax.scatter(date, max_y*(1.4+0.1*j), color=colors[j], marker=markers[j], s=50, label=key if i==0 else "")
                ax.vlines(date, ymin=max_y*1.3, ymax=max_y*1.6, color=colors[j], linestyle=line_style[j], linewidth=2)

    # Hockeystick
    for i, date in enumerate(Hockey_date):
        label = 'Hockey' if i == 0 else ""
        ax.scatter(date, max_y*1.2, color='green', marker='D', s=50, label=label)
        ax.vlines(x=date, ymin=0, ymax=max_y*1.3, color='green', linestyle='--', linewidth=2, alpha=0.7)

    # Baseline clustering (M0)
    for i, date in enumerate(ED_date):
        plt.scatter(date, max_y, color='blue', marker='o', s=50, label=f'Baseline clustering ($M_0$)' if i==0 else "")
        ax.vlines(date, ymin=0, ymax=max_y*1.3, color='blue', linestyle='-')

    ax.set_ylabel(epi, fontdict={'fontweight':'bold', 'fontsize':'12'})
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ticks = ax.get_yticks()
    lower = max_y * 1.3
    upper = max_y * 1.6
    labels = ['' if (lower <= t <= upper) else f'{int(t)}' for t in ticks]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(0,max_y*1.6)
    ax.set_xlim(x.min() - pd.Timedelta(weeks = sample_window-1), x.max())
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.47), ncol=5, frameon = False)
    plt.show()

    ### 2. clustering label 결과와 함께 시각화
    fig, ax = plt.subplots(figsize=(10, 3), dpi=400)

    ax.bar(data['Date'], data[epi], color='gray', alpha=0.5, label=epi, width=5.5)

    for i, date in enumerate(ED_date):
        plt.scatter(date, max_y*1.3, color='blue', marker='o', s=50, label=f'Baseline clustering ($M_0$)' if i==0 else "")
        ax.axvline(date, color='blue', linestyle='-')

    ax.set_xlabel('Week')
    ax.set_ylabel(epi)
    ax.grid(False)
    ax.set_xlim(x.min() - pd.Timedelta(weeks = sample_window-1), x.max())

    ax2 = ax.twinx()

    plot_dt =  pd.date_range(start=x.min(), end=x.max(), freq='W-SUN')
    y_series = pd.Series(data=result_data['label'].values, index=x.values)
    y_plot = y_series.reindex(plot_dt, fill_value=np.nan).astype(float)
    ax2.plot(plot_dt, y_plot.values, color='black', label='Label',  linewidth=3)
    
    ax2.grid(False)
    ax2.set_ylabel('Cluster', rotation=270, labelpad=15)
    order = np.sort(result_data['label'].astype(int).unique())
    ax2.set_yticks(order)
    ax2.set_yticklabels([f'C{c}' for c in order])

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='lower center', bbox_to_anchor=(0.5, -0.47),
            ncol=5, frameon=False, fontsize=12)

    plt.tight_layout()
    plt.show()

    return 

# def K_means_visualization(result_data, input_var, var_name): 
#     palette = {
#         0: "#348ABD", 1: "#A6D854", 2: "#D62728", 3: "#D62728",
#         '0': "#348ABD", '1': "#A6D854", '2': "#D62728", '3': "#D62728"
#     }
#     order = np.sort(result_data['label'].unique())
#     sns.set_style("whitegrid")

#     fig, axes = plt.subplots(1, 3, figsize=(9, 3))
#     axes = axes.ravel()
#     for i, col in enumerate(input_var):
#         ax = axes[i]
#         ax.grid(False)
#         # [수정됨] hue=x 변수 할당 및 legend=False 추가
#         sns.boxplot(
#             data=result_data, x='label', y=col, hue='label',
#             order=order, palette=palette,
#             width=0.6, ax=ax, showfliers=True, legend=False
#         )
#         ax.set_title(f'{var_name[i]}')
#         ax.set_xlabel('Cluster')
#         ax.set_ylabel(var_name[i])
        
#         # [수정됨] set_ticks 먼저 호출하여 경고 해결
#         ax.set_xticks(range(len(order)))
#         ax.set_xticklabels([f'C{c}' for c in order])
        
#     for j in range(len(input_var), len(axes)):
#         axes[j].remove()
#     plt.tight_layout()
#     plt.show()

# def early_warning_visualization(data_all, result_data, start_epidemic, CPD_date, covid_start, covid_end, epi, ED_date):
#     fig, ax = plt.subplots(figsize=(10, 2), dpi=400)
#     x = data_all['Date']
#     y = data_all[epi]

#     ax.bar(x, y, color='gray', alpha=0.5, label=epi, width=5.5)
#     ax.axvspan(pd.Timestamp(covid_start), pd.Timestamp(covid_end), color='gray', alpha=0.3, label='During COVID-19')
#     ax.set_xlabel('Week')
#     ax.set_ylabel(epi)
#     ax.grid(False)

#     for i in range(len(start_epidemic)):
#         plt.axvline(x=start_epidemic[i], color='r', linestyle='-', label='KDCA' if i==0 else "")
#     for i, date in enumerate(CPD_date):
#         plt.scatter(date, 80, color='orange', marker='^', s=50, label='CUSUM' if i==0 else "")
#         plt.axvline(date, color='orange', linestyle='--')
#     for i, date in enumerate(ED_date):
#         plt.scatter(date, 100, color='blue', marker='o', s=50, label='Our method' if i==0 else "")
#         ax.axvline(date, color='blue', linestyle='--')

#     plt.xlim(x.min(), x.max())
#     plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.47), ncol=5, frameon=False)
#     plt.show()

# def early_warning_visualization_bootstrap(data, data_all, epi, other_dates, Hockey_date, date_df, sample_window):
#         # 🌟 [추가된 핵심 로직] 타겟 컬럼이 'ILI'가 아니면 other_dates를 무시합니다.
#         if epi != 'ILI':
#             other_dates = None

#         ### 1. 다른 결과와 비교
#         fig1, ax = plt.subplots(figsize=(10, 2), dpi=300)

#         x=data_all['Date']
#         y=data_all[epi]
#         max_y = y.max()
#         ax.bar(data['Date'],data[epi],color='gray',alpha=0.5, label=epi, width=5.5)
#         # ax.axhline(y=max_y*1.3, color='black', linewidth=1)
#         if other_dates is not None:
#             ax.axhline(y=max_y*1.3, color='black', linewidth=1)
#         ax.set_xlabel('Week')
#         ax.set_ylabel(epi)
#         ax.grid(False)

#         # Other dates (epi가 'ILI'일 때만 그려짐)
#         colors = ['orange', 'red']
#         markers = ['^', '*']
#         line_style = ['--', '-']
#         if other_dates is not None:
#             for j, key in enumerate(other_dates.keys()):
#                 for i, date in enumerate(other_dates[key]):
#                     ax.scatter(date, max_y*(1.4+0.1*j), color=colors[j], marker=markers[j], s=50, label=key if i==0 else "")
#                     ax.vlines(date, ymin=max_y*1.3, ymax=max_y*1.6, color=colors[j], linestyle=line_style[j], linewidth=2)

#         # Hockeystick
#         for i, date in enumerate(Hockey_date):
#             label = 'Hockey' if i == 0 else ""
#             ax.scatter(date, max_y*1.1, color='green', marker='D', s=50, label=label)
#             ax.vlines(x=date, ymin=0, ymax=max_y*1.3, color='green', linestyle='--', linewidth=2, alpha=0.7)

#         # Bootstrap clustering (M1 ~ Mp)
#         rank_cols = date_df.columns
#         mode_list=[]
#         for i, col in enumerate(rank_cols):
#             iter_results = pd.to_datetime(date_df[col]).dt.normalize().dropna()
#             if not iter_results.empty:
#                 mode_date = iter_results.mode().iloc[0]
#                 mode_list.append(mode_date)
#                 low = iter_results.min()
#                 high = iter_results.max()
#                 ax.scatter(mode_date, max_y, color='blue', marker='o', s=50, label='Bootstrap' if i==0 else "")
#                 ax.vlines(mode_date, ymin=0, ymax=max_y*1.3, color='blue', linewidth=2, linestyle='-')
#                 ax.fill_between([low, high], 0, max_y*1.3, color='blue', alpha=0.3)

#         ax.set_ylabel(epi, fontdict={'fontweight':'bold', 'fontsize':'12'})
#         ax.yaxis.set_major_locator(plt.MaxNLocator(5))
#         ticks = ax.get_yticks()

#         # lower = max_y * 1.3
#         # upper = max_y * 1.6
#         # labels = ['' if (lower <= t <= upper) else f'{int(t)}' for t in ticks]
#         # ax.set_yticks(ticks)
#         # ax.set_yticklabels(labels)
#         # ax.set_ylim(0,max_y*1.6)

#         # 🌟 other_dates 유무에 따라 그래프 천장(ylim)과 y축 라벨을 다르게 설정합니다!
#         if other_dates is not None:
#             # 독감(ILI): 정답지 마커들을 위한 높은 천장 유지
#             lower = max_y * 1.3
#             upper = max_y * 1.6
#             labels = ['' if (lower <= t <= upper) else f'{int(t)}' for t in ticks]
#             ax.set_yticks(ticks)
#             ax.set_yticklabels(labels)
#             ax.set_ylim(0, max_y * 1.6) 
#         else:
#             # 수족구(HFMD): 불필요한 윗공간을 날려버려 선이 테두리에 닿게 만듦
#             labels = [f'{int(t)}' for t in ticks]
#             ax.set_yticks(ticks)
#             ax.set_yticklabels(labels)
#             ax.set_ylim(0, max_y * 1.25) # 천장을 낮춰서 시원하게 만듭니다.

#         ax.set_xlim(x.min() - pd.Timedelta(weeks = sample_window-1), x.max())

#         plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.47), ncol=5, frameon = False)

#         ### 2. 시즌 별로 비교
#         cols = date_df.columns
#         fig2, axes = plt.subplots(len(cols), 1, figsize=(10, 2 * len(cols)), dpi=300)

#         if len(cols) == 1: axes = [axes]

#         for i, col in enumerate(cols):
#             all_dates=[]

#             ax1 = axes[i]
#             data_bootstrap = pd.to_datetime(date_df[col]).dt.normalize().dropna()
#             all_dates.extend(data_bootstrap.tolist())
#             if not data_bootstrap.empty:
#                 ax1.bar(data['Date'],data[epi],color='gray',alpha=0.5, label=epi, width=5.5)
#                 ax1.scatter(mode_list[i], max_y, color='blue', marker='o', s=50, label='Bootstrap' if i==0 else "")
#                 ax1.vlines(mode_list[i], ymin=0, ymax=max_y*1.3, color='blue', linewidth=2, linestyle='-')

#                 # Other dates (epi가 'ILI'일 때만 그려짐)
#                 colors = ['orange', 'red']
#                 markers = ['^', '*']
#                 line_style = ['--', '-']
#                 if other_dates is not None:
#                     for j, key in enumerate(other_dates.keys()):
#                         ax1.scatter(other_dates[key][i], max_y*(1+0.1*j), color=colors[j], marker=markers[j], s=50, label=key if i==0 else "")
#                         ax1.vlines(other_dates[key][i], ymin=0, ymax=max_y*1.3, color=colors[j], linestyle=line_style[j], linewidth=2)
#                         all_dates.append(other_dates[key][i])
                
#                 # Hockeystick
#                 ax1.scatter(Hockey_date[i], max_y*1.3, color='green', marker='D', s=50, label='Hockey' if i == 0 else "")
#                 ax1.vlines(x=Hockey_date[i], ymin=0, ymax=max_y*1.2, color='green', linestyle='--', linewidth=2, alpha=0.7)
#                 all_dates.append(Hockey_date[i])

#                 ax1.grid(False)
#                 ax1.set_ylabel(epi)
#                 season_data = data_all[data_all['Season'] == int(col[:-2])]
#                 first_season = season_data['Season'].min()
#                 start_view = season_data['Date'].min()
#                 if int(col[:-2]) == first_season:
#                     start_view = season_data['Date'].min() - pd.Timedelta(weeks=sample_window-1)
#                 ax1.set_xlim(start_view, season_data['Date'].max())
#                 ax1.set_title(f'{int(col[:-2])}-{int(col[:-2])+1} season')
                
#                 ax2 = ax1.twinx()
#                 num_unique = len(data_bootstrap.unique())
                
#                 sns.histplot(data_bootstrap, ax=ax2, color='blue', kde=False,
#                                 stat="percent", alpha=0.5, legend=False, 
#                                 discrete=True, shrink=2.0)
#                 ax2.set_ylabel('Detection Probability')
#                 ax2.set_ylim(0, 100)
#                 ax2.grid(True, axis='y', color='gray', linestyle='--', linewidth=1, alpha=0.2)

#                 lines1, labels1 = ax1.get_legend_handles_labels()
#                 lines2, labels2 = ax2.get_legend_handles_labels()
                
#                 # 🌟 문법 오류 교정 (is -> ==)
#                 if i == 0:
#                     ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', ncol=1, frameon=False, fontsize=10)

#         plt.tight_layout()
#         return fig1, fig2


def early_warning_visualization_bootstrap(data, data_all, epi, other_dates, Hockey_date, date_df, sample_window):
        # 타겟 컬럼이 'ILI'가 아니면 other_dates를 무시합니다.
        if epi != 'ILI':
            other_dates = None
        train_seasons = [int(str(col)[:-2]) for col in date_df.columns]
        
        # 🌟 회색 막대그래프를 그리는 원본 data에서 Test 시즌(2026년 등)을 통째로 잘라냅니다!
        data = data[data['Season'].isin(train_seasons)].reset_index(drop=True)

        # 이제 x축 기준(x.max())은 완벽하게 Train의 마지막 날짜로 고정됩니다.
        x = data['Date']
        y = data[epi]
        max_y = y.max()
        # -------------------------------------------------------------------------
        train_len = len(train_seasons)

        # -------------------------------------------------------------------------
        ### 1. 동적 시각화 (Plotly) - 전체 기간 비교
        fig1 = go.Figure()

        x = data['Date']
        y = data[epi]
        max_y = y.max()

        # 배경 바 차트 (환자 수)
        fig1.add_trace(go.Bar(
            x=data['Date'], y=data[epi],
            marker_color='gray', opacity=0.5, name=epi,
            hoverinfo='x+y'
        ))

        # 검은색 가로 기준선 (other_dates가 있을 때만)
        if other_dates is not None:
            fig1.add_hline(y=max_y*1.3, line_color='black', line_width=1)

        # Other dates (KDCA, CUSUM 등)
        if other_dates is not None:
            colors = ['orange', 'red']
            markers = ['triangle-up', 'star']
            dash_styles = ['dash', 'solid']
            
            for j, key in enumerate(other_dates.keys()):
                # 🔥 핵심 수정: [:train_len]을 추가해서 Test 날짜는 그리지 않고 무시합니다!
                for i, date in enumerate(other_dates[key][:train_len]):
                    show_leg = True if i == 0 else False
                    
                    # 마커
                    fig1.add_trace(go.Scatter(
                        x=[date], y=[max_y*(1.4+0.1*j)],
                        mode='markers', marker=dict(color=colors[j], symbol=markers[j], size=10),
                        name=key, showlegend=show_leg, hoverinfo='name+x'
                    ))
                    # 세로선
                    fig1.add_shape(type='line', x0=date, x1=date, y0=max_y*1.3, y1=max_y*1.6, 
                                   line=dict(color=colors[j], dash=dash_styles[j], width=2))

        # Hockeystick
        # 🔥 핵심 수정: 여기도 [:train_len]을 추가합니다!
        for i, date in enumerate(Hockey_date[:train_len]):
            show_leg = True if i == 0 else False
            # 마커
            fig1.add_trace(go.Scatter(
                x=[date], y=[max_y*1.1],
                mode='markers', marker=dict(color='green', symbol='diamond', size=10),
                name='Hockey', showlegend=show_leg, hoverinfo='name+x'
            ))
            # 세로선
            fig1.add_shape(type='line', x0=date, x1=date, y0=0, y1=max_y*1.3, 
                           line=dict(color='green', dash='dash', width=2), opacity=0.7)

        # Bootstrap clustering (M1 ~ Mp)
        rank_cols = date_df.columns
        mode_list=[]
        for i, col in enumerate(rank_cols):
            iter_results = pd.to_datetime(date_df[col]).dt.normalize().dropna()
            if not iter_results.empty:
                mode_date = iter_results.mode().iloc[0]
                mode_list.append(mode_date)
                low = iter_results.min()
                high = iter_results.max()
                
                show_leg = True if i == 0 else False
                
                # 마커
                fig1.add_trace(go.Scatter(
                    x=[mode_date], y=[max_y],
                    mode='markers', marker=dict(color='blue', symbol='circle', size=10),
                    name='Bootstrap', showlegend=show_leg, hoverinfo='name+x'
                ))
                # 세로선
                fig1.add_shape(type='line', x0=mode_date, x1=mode_date, y0=0, y1=max_y*1.3, 
                               line=dict(color='blue', width=2))
                # 음영 (신뢰구간)

                fig1.add_vrect(x0=low, x1=high, fillcolor='blue', opacity=0.3, line_width=0)

        # Y축 천장 높이 계산
        y_max_limit = max_y * 1.6 if other_dates is not None else max_y * 1.25

        # 레이아웃(디자인 및 슬라이더) 설정
        fig1.update_layout(
            yaxis_title=epi,
            yaxis=dict(range=[0, y_max_limit], fixedrange=False),
            xaxis=dict(
                range=[x.min() - pd.Timedelta(weeks=sample_window-1), x.max()],
                rangeslider=dict(visible=True) # 🔥 하단 슬라이더 활성화!
            ),
            # 🌟 범례를 그래프 바로 위(y=1.05) 중앙(x=0.5)으로 예쁘게 올립니다!
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5), 
            margin=dict(l=20, r=20, t=60, b=20), # 범례가 들어갈 수 있게 위쪽 마진(t)을 60으로 살짝 늘려줍니다.
            hovermode="x unified",
            plot_bgcolor='white'
        )
        
        # x축, y축 테두리
        fig1.update_xaxes(showline=True, linewidth=1, linecolor='lightgray', gridcolor='whitesmoke')
        fig1.update_yaxes(showline=True, linewidth=1, linecolor='lightgray', gridcolor='whitesmoke')


        ### 2. 시즌 별 비교 (Matplotlib 유지)
        cols = date_df.columns
        fig2, axes = plt.subplots(len(cols), 1, figsize=(10, 2 * len(cols)), dpi=300)

        if len(cols) == 1: axes = [axes]

        for i, col in enumerate(cols):
            all_dates=[]

            ax1 = axes[i]
            data_bootstrap = pd.to_datetime(date_df[col]).dt.normalize().dropna()
            all_dates.extend(data_bootstrap.tolist())
            if not data_bootstrap.empty:
                ax1.bar(data['Date'],data[epi],color='gray',alpha=0.5, label=epi, width=5.5)
                ax1.scatter(mode_list[i], max_y, color='blue', marker='o', s=50, label='Bootstrap' if i==0 else "")
                ax1.vlines(mode_list[i], ymin=0, ymax=max_y*1.3, color='blue', linewidth=2, linestyle='-')

                # if other_dates is not None:
                #     for j, key in enumerate(other_dates.keys()):
                #         ax1.scatter(other_dates[key][i], max_y*(1+0.1*j), color=colors[j], marker=markers[j], s=50, label=key if i==0 else "")
                #         ax1.vlines(other_dates[key][i], ymin=0, ymax=max_y*1.3, color=colors[j], linestyle=dash_styles[j], linewidth=2)
                #         all_dates.append(other_dates[key][i])
                if other_dates is not None:
                    # 🌟 Matplotlib 전용 마커와 선 스타일을 다시 지정해줍니다!
                    mpl_markers = ['^', '*']
                    mpl_line_style = ['--', '-']
                    
                    for j, key in enumerate(other_dates.keys()):
                        ax1.scatter(other_dates[key][i], max_y*(1+0.1*j), color=colors[j], marker=mpl_markers[j], s=50, label=key if i==0 else "")
                        ax1.vlines(other_dates[key][i], ymin=0, ymax=max_y*1.3, color=colors[j], linestyle=mpl_line_style[j], linewidth=2)
                        all_dates.append(other_dates[key][i])
                
                ax1.scatter(Hockey_date[i], max_y*1.3, color='green', marker='D', s=50, label='Hockey' if i == 0 else "")
                ax1.vlines(x=Hockey_date[i], ymin=0, ymax=max_y*1.2, color='green', linestyle='--', linewidth=2, alpha=0.7)
                all_dates.append(Hockey_date[i])

                ax1.grid(False)
                ax1.set_ylabel(epi)
                # season_data = data_all[data_all['Season'] == int(col[:-2])]
                season_data = data[data['Season'] == int(col[:-2])]
                first_season = season_data['Season'].min()
                start_view = season_data['Date'].min()
                if int(col[:-2]) == first_season:
                    start_view = season_data['Date'].min() - pd.Timedelta(weeks=sample_window-1)
                ax1.set_xlim(start_view, season_data['Date'].max())
                ax1.set_title(f'{int(col[:-2])}-{int(col[:-2])+1} season')
                
                ax2 = ax1.twinx()
                
                sns.histplot(data_bootstrap, ax=ax2, color='blue', kde=False,
                                stat="percent", alpha=0.5, legend=False, 
                                discrete=True, shrink=2.0)
                ax2.set_ylabel('Detection Probability')
                ax2.set_ylim(0, 100)
                ax2.grid(True, axis='y', color='gray', linestyle='--', linewidth=1, alpha=0.2)

                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                
                if i == 0:
                    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', ncol=1, frameon=False, fontsize=10)

        plt.tight_layout()
        return fig1, fig2

def visualize_3d_incremental_detection_weekly(iteration_results):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    columns = iteration_results.columns
    
    def to_num(val):
        try:
            if pd.isnull(val): return np.nan
            return mdates.date2num(pd.to_datetime(val))
        except:
            return np.nan

    numeric_df = iteration_results.applymap(to_num)
    all_values = numeric_df.values.flatten()
    all_values = all_values[~np.isnan(all_values)]
    
    if len(all_values) == 0:
        print("No data to display.")
        return

    x_min, x_max = np.min(all_values), np.max(all_values)
    x_bins = np.arange(int(x_min) - 7, int(x_max) + 14, 7) 

    for i, col in enumerate(columns):
        data_nums = numeric_df[col].dropna().values
        if len(data_nums) == 0: continue
            
        counts, bins = np.histogram(data_nums, bins=x_bins)
        percent = (counts / len(data_nums)) * 100
        x_centers = (bins[:-1] + bins[1:]) / 2
        y_pos = np.ones_like(x_centers) * i
        mask = percent > 0
        if not np.any(mask): continue

        ax.bar3d(x_centers[mask], y_pos[mask], 0, dx=6.0, dy=0.6, dz=percent[mask], 
                 color=plt.cm.coolwarm(i / len(columns)), alpha=0.7, edgecolor='gray', linewidth=0.2)

    ax.set_xlabel('\nDetection Date (Weekly Bin)', linespacing=3)
    ax.set_ylabel('\nData Accumulated Until', linespacing=6)
    ax.set_zlabel('Bootstrap Probability (%)', linespacing=3)
    
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    ax.set_ylabel('Data Accumulated Until', labelpad=30)
    y_indices = np.arange(len(columns))
    step_size = max(1, len(columns) // 10) 
    ax.set_yticks(y_indices[::step_size])
    ax.set_yticklabels([columns[i] for i in y_indices[::step_size]], rotation=-15, va='center', ha='left', fontsize=9)

    ax.view_init(elev=30, azim=-70)
    plt.subplots_adjust(right=0.9)
    plt.show()

# def plot_bootstrap_detection_for_app(data_all_train, all_ed_sets, target_col, outbreak_season, 
#                                      start_epidemic=None, cpd_date=None, covid_start=None, covid_end=None):
#     """
#     Streamlit App을 위한 Bootstrap 조기 탐지 시각화 함수
#     - 기본: Bootstrap 탐지 구간(파란색 음영) + 최빈값(다이아몬드)
#     - ILI 특화: target_col이 'ILI'인 경우, 정답지(유행시작일, CPD, 코로나)를 오버레이
#     """
#     try:
#         # 캔버스 설정
#         fig, ax = plt.subplots(figsize=(12, 4))
        
#         # 1. 배경: 전체 환자 수 (회색 막대)
#         ax.bar(data_all_train['Date'], data_all_train[target_col], color='lightgray', label='Patients', width=5)
        
#         # ---------------------------------------------------------------------
#         # [추가됨] ILI인 경우에만 정답지(Ground Truth) 표시
#         # ---------------------------------------------------------------------
#         if target_col == 'ILI':
#             # A. COVID-19 기간 (회색 진한 음영)
#             if covid_start and covid_end:
#                 ax.axvspan(covid_start, covid_end, color='black', alpha=0.1, label='COVID-19 Period')

#             # B. Change Point Date (CPD - 초록색 점선)
#             if cpd_date:
#                 # 리스트인지 확인하고 반복문
#                 dates = cpd_date if isinstance(cpd_date, list) else [cpd_date]
#                 for idx, d in enumerate(dates):
#                     # 범례가 너무 많아지지 않게 첫 번째만 라벨 표시
#                     label = 'Change Point (CPD)' if idx == 0 else ""
#                     ax.axvline(pd.to_datetime(d), color='orange', linestyle='--', linewidth=2, label=label)

#             # C. 공식 유행 시작일 (Start Epidemic - 빨간색 점선)
#             if start_epidemic:
#                 dates = start_epidemic if isinstance(start_epidemic, list) else [start_epidemic]
#                 for idx, d in enumerate(dates):
#                     label = 'Official Epidemic Start' if idx == 0 else ""
#                     ax.axvline(pd.to_datetime(d), color='red', linestyle='-', linewidth=1.5, label=label)

#         # ---------------------------------------------------------------------
#         # 2. Bootstrap 탐지 구간 및 최빈값 표시 (공통 로직)
#         # ---------------------------------------------------------------------
#         flat_dates = [d for dates in all_ed_sets for d in dates if pd.notna(d)]
        
#         if flat_dates:
#             dates_df = pd.DataFrame({'Date': pd.to_datetime(flat_dates)})
            
#             # 시즌 계산
#             dates_df['Season'] = dates_df['Date'].apply(
#                 lambda x: x.year if x.month >= outbreak_season else x.year - 1
#             )
            
#             label_added_span = False
#             label_added_mode = False
            
#             for season in sorted(dates_df['Season'].unique()):
#                 season_data = dates_df[dates_df['Season'] == season]['Date']
#                 if season_data.empty: continue
                    
#                 start_date = season_data.min()
#                 end_date = season_data.max()
#                 if start_date == end_date:
#                     end_date = start_date + pd.Timedelta(days=6)

#                 if not label_added_span:
#                     ax.axvspan(start_date, end_date, color='blue', alpha=0.2, label='Bootstrap Detection Range')
#                     label_added_span = True
#                 else:
#                     ax.axvspan(start_date, end_date, color='blue', alpha=0.2)
                
#                 mode_date = season_data.mode()[0]
#                 match = data_all_train[data_all_train['Date'] == mode_date]
#                 y_val = match[target_col].values[0] if not match.empty else 0
                
#                 if not label_added_mode:
#                     ax.axvline(mode_date, color='blue', linestyle='-.', linewidth=1.5, label='Bootstrap Mode')
#                     ax.scatter(mode_date, y_val, color='blue', marker='D', s=50, zorder=5)
#                     label_added_mode = True
#                 else:
#                     ax.axvline(mode_date, color='blue', linestyle='-.', linewidth=1.5)
#                     ax.scatter(mode_date, y_val, color='blue', marker='D', s=50, zorder=5)

#         # 그래프 꾸미기
#         ax.set_title(f"Bootstrap-based Early Warning Detection ({target_col})", fontsize=15)
#         ax.set_xlabel("Date")
#         ax.set_ylabel("Number of Patients")
        
#         # 범례 위치 조정
#         ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4) # ncol을 4로 늘림
#         ax.grid(True, axis='y', alpha=0.3)
        
#         # X축 날짜 범위 설정
#         data_min_date = data_all_train['Date'].min()
#         data_max_date = data_all_train['Date'].max()
#         ax.set_xlim(data_min_date - pd.Timedelta(weeks=4), data_max_date + pd.Timedelta(weeks=4))
        
#         return fig

#     except Exception as e:
#         print(f"Visualization Error: {e}")
#         return None

def visualization_real_time_early_detection(data_all, date_table, prob_table, epi, other_dates, Hockey_date, bootstrap_dates, batch_size = 10):
    ### 1. 전체 기간을 예측했을 시 결과
    generated_figs = []
    fig_summary, axes = plt.subplots(2, 1, figsize=(10, 4), dpi=300)

    x = data_all['Date']
    y = data_all[epi]
    max_y = y.max()

    ax1 = axes[0]
    ax1.bar(x, y, color='gray', alpha=0.3, label=epi, width=5)
    ax1.set_ylabel(epi)
    ax1.grid(False)

    ax2 = ax1.twinx()
    ax2.plot(x, prob_table['Warning_Probability'], color='blue', alpha=0.8)
    ax2.grid(True, axis='y', color='gray', linestyle='--', linewidth=1, alpha=0.2)

    ##
    ax1 = axes[1]
    ax1.bar(x, y, color='gray', alpha=0.3, label=epi, width=5)
    ax1.set_ylabel(epi)
    ax1.grid(False)

    ax2 = ax1.twinx()
    data_bootstrap = pd.to_datetime(date_table['Detect_date']).dt.normalize().dropna()
    sns.histplot(data_bootstrap, ax=ax2, color='blue', kde=False,
                            stat="probability",
                            alpha=0.5,
                            label='Bootstrap', legend=False, discrete=True,
                            shrink=2.0)
    ax2.grid(True, axis='y', color='gray', linestyle='--', linewidth=1, alpha=0.2)

    # Other dates
    colors = ['orange', 'red']
    markers = ['^', '*']
    line_style = ['--', '-']
    if other_dates is not None:
        for j, key in enumerate(other_dates.keys()):
            for i, date in enumerate(other_dates[key]):
                ax1.scatter(date, max_y*(1+0.1*j), color=colors[j], marker=markers[j], s=50, label=key if i==0 else "")
                ax1.vlines(date, ymin=0, ymax=max_y*1.2, color=colors[j], linestyle=line_style[j], linewidth=2)

    # Hockeystick
    for i, date in enumerate(Hockey_date):
        label = 'Hockey' if i == 0 else ""
        ax1.scatter(date, max_y*1.1, color='green', marker='D', s=50, label=label)
        ax1.vlines(x=date, ymin=0, ymax=max_y*1.2, color='green', linestyle='--', linewidth=2, alpha=0.7)
    
    ax1.set_xlim(x.min(), x.max())
    plt.tight_layout()
    # plt.show()
    generated_figs.append(fig_summary)

    ### 2. 실시간으로 예측했을 때의 결과
    col_list = list(bootstrap_dates.columns)
    first_blue = None
    first_orange = None
    first_red = None

    for start in range(0, len(col_list), batch_size):

        batch_list = col_list[start:start + batch_size]
        n_rows = len(batch_list)

        fig_batch, axes = plt.subplots(n_rows, 1, figsize=(10, 2 * n_rows))

        if n_rows == 1:
            axes = [axes]

        for i, col in enumerate(batch_list):
            ax1 = axes[i]

            data = data_all.set_index('Date').loc[:col].reset_index()
            x = data['Date']
            y = data[epi]

            ax1.bar(x, y, color='gray', alpha=0.3, width=5)
            
            start_view = x.min() - pd.Timedelta(days=7)
            end_view = pd.to_datetime(col) + pd.Timedelta(days=7)
            ax1.set_xlim(start_view, end_view)
            ax1.set_ylim(0, max(y) * 1.2)
            ax1.set_title(f'End of test date: {col}')
            ax1.set_ylabel(epi)
            ax1.grid(False)

            # Other dates
            colors = ['orange', 'red']
            markers = ['^', '*']
            line_style = ['--', '-']
            if other_dates is not None:
                for j, key in enumerate(other_dates.keys()):
                    for i, date in enumerate(other_dates[key]):
                        ax1.scatter(date, max_y*(1+0.1*j), color=colors[j], marker=markers[j], s=50, label=key if i==0 else "")
                        ax1.vlines(date, ymin=0, ymax=max_y*1.2, color=colors[j], linestyle=line_style[j], linewidth=2)

            # Hockeystick
            # for i, date in enumerate(Hockey_date):
            #     ax1.scatter(date, max_y*1.1, color='green', marker='D', s=50, label='Hockey' if i == 0 else "")
            #     ax1.vlines(x=date, ymin=0, ymax=max_y*1.2, color='green', linestyle='--', linewidth=2, alpha=0.7)
            
            # Bootstrap histogram (twin axis)
            ax2 = ax1.twinx()
            bootstrap_dates[col] = bootstrap_dates[col].apply(lambda x: x[0] if isinstance(x, list) else x)
            data_bootstrap = (pd.to_datetime(bootstrap_dates[col]).dt.normalize().dropna())

            # if not data_bootstrap.empty:
            #     sns.histplot(x=data_bootstrap, stat="count", discrete=True, alpha=0.5, color='blue', shrink=2.0, ax=ax2)
            
            if not data_bootstrap.empty:
                counts = data_bootstrap.value_counts().sort_index()
                total_n = len(bootstrap_dates[col])

                first_blue = None
                first_orange = None
                first_red = None

                for date, count in counts.items():
                    ratio = count / total_n
                    if ratio >= 0.10:
                        color = 'red'
                        if first_red is None:
                            ax2.axvline(date, color='red', linestyle='--', linewidth=2, alpha=0.5)
                            first_red = date
                    elif ratio >= 0.05:
                        color = 'orange'
                        if first_orange is None:
                            ax2.axvline(date, color='orange', linestyle='--', linewidth=2, alpha=0.5)
                            first_orange = date
                    else:
                        color = 'blue'
                        if first_blue is None:
                            ax2.axvline(date, color='blue', linestyle='--', linewidth=2, alpha=0.5)
                            first_blue = date
                    ax2.bar(date, count, color=color, alpha=0.6, width=2)

            ax2.set_ylabel('Detection dates', rotation=270, labelpad=15)
            ax2.set_ylim(0, len(bootstrap_dates))

            if len(y) <= 10:
                ax1.set_xticks(x)
                ax1.set_xticklabels(x.dt.strftime('%Y-%m-%d'))

        plt.tight_layout()
        # plt.savefig(f'./ILI_result/real-time2/bootstrap_detection_{start}_{start + batch_size}.png', dpi=300)
        # plt.show()
        generated_figs.append(fig_batch)
        
    print("First blue date:", first_blue)
    print("First orange date:", first_orange)
    print("First red date:", first_red)

    return generated_figs


# def plot_incremental_test_results(incremental_prob_results, bootstrap_dates, data_all_test, target_col, 
#                                   start_epidemic=None, cpd_date=None):
#     """
#     Test 과정 시각화: 10개씩 묶어서 결과 출력
#     """
#     figs = [] # 생성된 그림들을 저장할 리스트
    
#     left_keys = list(incremental_prob_results.keys())
#     right_cols = list(bootstrap_dates.columns)
    
#     # 공통된 날짜만 추리기 (혹시 개수가 안 맞을 경우 대비)
#     # 보통은 같지만 안전하게 min 길이 사용
#     limit = min(len(left_keys), len(right_cols))
#     left_keys = left_keys[:limit]
#     right_cols = right_cols[:limit]

#     batch_size = 10 # 10개씩 묶어서 출력

#     for start in range(0, limit, batch_size):
#         try:
#             left_batch = left_keys[start:start + batch_size]
#             right_batch = right_cols[start:start + batch_size]

#             n_rows = len(left_batch)
#             if n_rows == 0: continue

#             # Figure 생성
#             fig, axes = plt.subplots(n_rows, 2, figsize=(20, 3 * n_rows)) # 높이 조금 조정
            
#             # n_rows가 1일 때 axes가 1차원 배열이 되므로 2차원으로 변환
#             if n_rows == 1:
#                 axes = np.array([axes])

#             # --- [왼쪽 컬럼] Warning Probability ---
#             for i, key in enumerate(left_batch):
#                 ax1 = axes[i, 0]

#                 # 배경 환자 수 (Bar)
#                 # key 시점까지의 데이터만 가져옴
#                 subset_mask = data_all_test['Date'] <= key
#                 data = data_all_test[subset_mask].set_index('Date').reset_index()
                
#                 x_ili = data['Date']
#                 y_ili = data[target_col]

#                 ax1.bar(x_ili, y_ili, color='gray', alpha=0.3, width=5)
#                 ax1.set_ylabel(target_col)

#                 # 확률 선 그래프 (Line)
#                 ax2 = ax1.twinx()
#                 prob_df = incremental_prob_results[key]
#                 ax2.plot(prob_df['Date'], prob_df['Warning_Probability'],
#                          alpha=0.8, color='blue', linewidth=2)
#                 ax2.set_ylabel('Warning Probability', rotation=270, labelpad=15)
#                 ax2.set_ylim(0, 1.1)

#                 ax1.set_title(f'End of test date: {pd.to_datetime(key).date()}')

#                 # 데이터가 적을 때 X축 라벨 포맷팅
#                 if len(y_ili) <= 10:
#                     ax1.set_xticks(x_ili)
#                     ax1.set_xticklabels(x_ili.dt.strftime('%Y-%m-%d'), rotation=45)

#             # --- [오른쪽 컬럼] Bootstrap Detection Histogram ---
#             for i, col in enumerate(right_batch):
#                 ax1 = axes[i, 1]

#                 # 배경 환자 수 (Bar)
#                 subset_mask = data_all_test['Date'] <= col
#                 data = data_all_test[subset_mask].set_index('Date').reset_index()
                
#                 x_ili = data['Date']
#                 y_ili = data[target_col]

#                 ax1.bar(x_ili, y_ili, color='gray', alpha=0.3, width=5)
#                 ax1.set_ylabel(target_col)

#                 # 뷰 범위 설정 (+- 7일 여유)
#                 start_view = x_ili.min() - pd.Timedelta(days=7)
#                 end_view = pd.to_datetime(col) + pd.Timedelta(days=7)
#                 ax1.set_xlim(start_view, end_view)
                
#                 # y축 범위 안전하게 설정
#                 max_val = y_ili.max() if not y_ili.empty else 10
#                 ax1.set_ylim(0, max_val * 1.2)

#                 ax2 = ax1.twinx()
                
#                 # Bootstrap 날짜 히스토그램
#                 if col in bootstrap_dates.columns:
#                     data_bootstrap = pd.to_datetime(bootstrap_dates[col]).dt.normalize().dropna()
                    
#                     if not data_bootstrap.empty:
#                         sns.histplot(
#                             x=data_bootstrap,
#                             stat="percent",
#                             discrete=True,
#                             alpha=0.5,
#                             color='blue',
#                             shrink=2.0, # 막대 얇게
#                             ax=ax2
#                         )
                
#                 ax2.set_ylabel('Detection Probability', rotation=270, labelpad=15)

#                 # KDCA 선 (빨간색)
#                 if start_epidemic:
#                     current_kdca = [d for d in start_epidemic if start_view <= pd.Timestamp(d) <= end_view]
#                     for j, k_date in enumerate(current_kdca):
#                         ax1.axvline(x=pd.Timestamp(k_date), color='red', linestyle='-', linewidth=3, alpha=0.7, 
#                                     label='KDCA' if j==0 else "")

#                 # CUSUM 점 (주황색)
#                 if cpd_date:
#                     current_cusum = [d for d in cpd_date if start_view <= pd.Timestamp(d) <= end_view]
#                     for j, c_date in enumerate(current_cusum):
#                         ax1.scatter(pd.Timestamp(c_date), max_val*0.8, color='orange', marker='^', s=100, 
#                                     label='CUSUM' if j==0 else "", zorder=5)
#                         ax1.axvline(x=pd.Timestamp(c_date), color='orange', linestyle='--', linewidth=3, alpha=0.7)

#                 ax1.set_title(f'End of test date: {pd.to_datetime(col).date()}')
                
#                 # 범례 표시
#                 lines, labels = ax1.get_legend_handles_labels()
#                 if lines:
#                     ax1.legend(loc='upper left')

#             plt.tight_layout(rect=[0, 0, 1, 0.96])
#             figs.append(fig) # 생성된 Figure 저장
            
#         except Exception as e:
#             print(f"Error creating batch plot: {e}")
#             continue

#     return figs

# def interactive_real_time_chart(data_all, prob_table, epi):
#     """
#     Plotly를 이용해 하단에 Range Slider(미니맵)가 달린 인터랙티브 차트를 생성합니다.
#     """
#     # 1. 왼쪽 Y축(환자수)과 오른쪽 Y축(위험확률)을 동시에 쓰기 위한 설정
#     fig = make_subplots(specs=[[{"secondary_y": True}]])

#     # 2. 배경: 환자 수 막대 그래프 (회색)
#     fig.add_trace(
#         go.Bar(
#             x=data_all['Date'], 
#             y=data_all[epi], 
#             name=f"{epi} Patients", 
#             marker_color='gray', 
#             opacity=0.5
#         ),
#         secondary_y=False,
#     )

#     # 3. 메인: 실시간 위험 도달 확률 선 그래프 (파란색)
#     fig.add_trace(
#         go.Scatter(
#             x=prob_table['Date'], 
#             y=prob_table['Warning_Probability'], 
#             name="Warning Probability", 
#             line=dict(color='blue', width=3)
#         ),
#         secondary_y=True,
#     )

#     # 4. ★핵심★ 하단 미니맵(Range Slider) 및 레이아웃 설정
#     fig.update_layout(
#         title_text="<b>실시간 위험 확률 모니터링 (Interactive)</b>",
#         title_x=0.5, # 제목 가운데 정렬
#         xaxis=dict(
#             rangeslider=dict(
#                 visible=True,     # 하단 슬라이더 켜기
#                 thickness=0.15,   # 슬라이더 두께
#                 bgcolor="#EAEAEA" # 슬라이더 배경색
#             ),
#             type="date"
#         ),
#         hovermode="x unified", # 마우스를 올리면 그 날짜의 모든 데이터를 한 줄에 보여줌
#         legend=dict(
#             orientation="h", # 범례 가로 배치
#             yanchor="bottom", y=-0.6, 
#             xanchor="center", x=0.5
#         ),
#         margin=dict(l=40, r=40, t=60, b=40),
#         plot_bgcolor='white'
#     )

#     # 5. Y축 디테일 설정
#     fig.update_yaxes(title_text=f"<b>{epi} 환자 수</b>", secondary_y=False, showgrid=False)
#     fig.update_yaxes(title_text="<b>위험 확률 (0~1)</b>", secondary_y=True, range=[0, 1.1], gridcolor='lightgray')

#     return fig
def interactive_real_time_chart(data_all, bootstrap_dates, other_dates, epi):
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 화면을 자르기 위해 Test 데이터의 시작과 끝 날짜를 구합니다.
    min_date = data_all['Date'].min()
    max_date = data_all['Date'].max()

    # 1. 배경: ILI 환자 수 막대 그래프 (회색) - 기본 굵기 유지
    fig.add_trace(
        go.Bar(
            x=data_all['Date'], y=data_all[epi], 
            name=f"{epi} Patients", marker_color='gray', opacity=0.3
        ),
        secondary_y=False,
    )

    # 2. 마지막 주차(최종 시뮬레이션 결과) 데이터 가져오기
    last_col = bootstrap_dates.columns[-1]
    b_dates = bootstrap_dates[last_col].apply(lambda x: x[0] if isinstance(x, list) and len(x)>0 else x)
    data_bootstrap = pd.to_datetime(b_dates).dt.normalize().dropna()

    # 3. 히스토그램 막대 및 점선 계산
    if not data_bootstrap.empty:
        counts = data_bootstrap.value_counts().sort_index()
        total_n = len(bootstrap_dates[last_col])

        x_blue, y_blue, x_orange, y_orange, x_red, y_red = [], [], [], [], [], []
        first_blue, first_orange, first_red = None, None, None

        for date, count in counts.items():
            ratio = count / total_n
            if ratio >= 0.10:
                x_red.append(date); y_red.append(count)
                if first_red is None: first_red = date
            elif ratio >= 0.05:
                x_orange.append(date); y_orange.append(count)
                if first_orange is None: first_orange = date
            else:
                x_blue.append(date); y_blue.append(count)
                if first_blue is None: first_blue = date

        # 🌟 [핵심 변경 1] 컬러 바의 굵기를 얇게 만듭니다 (3일 = 3 * 24시간 * 60분 * 60초 * 1000밀리초)
        thin_width = 3 * 24 * 60 * 60 * 1000 

        # 히스토그램 그리기 (width 속성 추가)
        if x_blue: fig.add_trace(go.Bar(x=x_blue, y=y_blue, name='Detection (Blue)', marker_color='blue', opacity=0.6, width=thin_width), secondary_y=True)
        if x_orange: fig.add_trace(go.Bar(x=x_orange, y=y_orange, name='Detection (Orange)', marker_color='orange', opacity=0.6, width=thin_width), secondary_y=True)
        if x_red: fig.add_trace(go.Bar(x=x_red, y=y_red, name='Detection (Red)', marker_color='red', opacity=0.6, width=thin_width), secondary_y=True)

        # 점선 그리기
        if first_blue: fig.add_vline(x=first_blue, line_dash="dash", line_color="blue", opacity=0.5, line_width=2, secondary_y=True)
        if first_orange: fig.add_vline(x=first_orange, line_dash="dash", line_color="orange", opacity=0.5, line_width=2, secondary_y=True)
        if first_red: fig.add_vline(x=first_red, line_dash="dash", line_color="red", opacity=0.5, line_width=2, secondary_y=True)

        b_date_str = first_blue.strftime('%Y-%m-%d') if pd.notnull(first_blue) else "Not detected"
        o_date_str = first_orange.strftime('%Y-%m-%d') if pd.notnull(first_orange) else "Not detected"
        r_date_str = first_red.strftime('%Y-%m-%d') if pd.notnull(first_red) else "Not detected"
    # 4. 하단 미니맵(Range Slider) 및 레이아웃 설정
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.15, bgcolor="#EAEAEA"), 
            type="date",
            range=[min_date, max_date] 
        ),
        hovermode="x unified",
        
        # 🌟 [핵심 변경 2] 범례를 슬라이더 밑에서 그래프 위쪽으로 이동시킵니다!
        legend=dict(
            orientation="h", 
            yanchor="bottom", y=1.02, # y값을 1보다 크게 주면 그래프 위로 올라갑니다.
            xanchor="center", x=0.5
        ),
        
        plot_bgcolor='white', margin=dict(l=40, r=40, t=80, b=40) # 위쪽 여백(t)을 조금 늘려 범례 공간 확보
    )
    
    fig.update_yaxes(title_text=f"<b>{epi}</b>", secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="<b>Detection dates</b>", secondary_y=True, gridcolor='lightgray')

    return fig, b_date_str, o_date_str, r_date_str