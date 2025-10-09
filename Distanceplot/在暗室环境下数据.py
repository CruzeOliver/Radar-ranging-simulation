import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# === 1. Data Preparation ===
distances = np.array([856, 1236, 1745, 1959, 2414, 2920])  # mm
f_theoretical = np.array([182061, 263051, 371276, 416808, 513617, 621276])
f_actual = {
    'FFT':     np.array([195234, 278906, 362578, 418359, 502031, 613593]),
    'Macleod': np.array([191689, 271911, 362837, 418322, 504064, 613468]),
    'CZT':     np.array([188261, 271933, 368621, 416035, 509003, 616847]),
    'MCZT':    np.array([184716, 266221, 368783, 416481, 511037, 617203])
}
# --- 误差数据 ---
errors_r = {
    'FFT':      np.array([61.6, 74.9, 40.9, 7.3, 54.5, 36.1]),
    'Macleod':  np.array([44.9, 42.0, 39.7, 7.1, 44.9, 36.7]),
    'CZT':      np.array([28.8, 42.1, 12.5, 3.6, 21.7, 20.8]),
    'MCZT':     np.array([12.2, 15.2, 11.7, 1.5, 12.1, 19.1])
}

errors_f = {
    'FFT':      np.array([13173, 15855, 8698, 1551, 11586, 7683]),
    'Macleod':  np.array([9628,  8860,  8439, 1514, 9553,  7808]),
    'CZT':      np.array([6200,  8882,  2655, 773,  4614,  4429]),
    'MCZT':     np.array([2655,  3170,  2493, 327,  2580,  4073])
}

# === 2. Global Style Settings ===
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

algorithms = ['FFT', 'Macleod', 'CZT', 'MCZT']

style_map = {
    'FFT':       {'color': 'red',     'marker': 'o', 'linestyle': '-', 'label': 'FFT'},
    'Macleod':   {'color': 'blue',    'marker': '^', 'linestyle': '-', 'label': 'Macleod'},
    'CZT':       {'color': 'green',   'marker': 's', 'linestyle': '-', 'label': 'CZT'},
    'MCZT':      {'color': 'black',   'marker': 'd', 'linestyle': '-', 'label': 'Macleod-CZT'}
}

def apply_common_style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(title='Algorithm', fontsize=11, title_fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=10)

# ===================================================================
# === FIGURE 1: Bar Chart (ΔR)
# ===================================================================
plt.figure(figsize=(8, 6))
ax1 = plt.gca()

bar_width = 0.2
index = np.arange(len(distances))
colors = ['#D62728', '#2CA02C', '#1F77B4', '#FF7F0E']

for i, algo in enumerate(algorithms):
    bars = plt.bar(index + i * bar_width, errors_r[algo], bar_width,
                   label=algo, color=colors[i], edgecolor='black', linewidth=0.8)
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.1f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords='offset points',
                     ha='center', va='bottom', fontsize=8)

apply_common_style(ax1,
                   xlabel='Theoretical Distance $R$ (mm)',
                   ylabel='Distance Error $\\Delta R$ (mm)',
                   title='Comparison of Distance Estimation Errors')

plt.xticks(index + bar_width * 1.5, distances)
plt.tight_layout()
plt.show()

# ===================================================================
# === FIGURE 2: plot Line Plot (Δf)
# ===================================================================
plt.figure(figsize=(8, 6))
ax2 = plt.gca()

for algo in algorithms:
    style = style_map[algo]
    plt.plot(distances, errors_f[algo],
                 linestyle=style['linestyle'],
                 marker=style['marker'],
                 color=style['color'],
                 label=style['label'],
                 linewidth=2,
                 markersize=6)

apply_common_style(ax2,
                   xlabel='Theoretical Distance $R$ (mm)',
                   ylabel='Frequency Error $\\Delta f$ (Hz)',
                   title='Trend of Frequency Estimation Errors')


plt.tight_layout()
plt.show()

# ===================================================================
# === FIGURE 3: Actual Frequency vs Theoretical Frequency
# ===================================================================

fig, ax = plt.subplots(figsize=(14, 9))

# 主图绘图
ax.plot(distances, f_theoretical, color='gray', linestyle='--', linewidth=2, label='Theoretical')
for algo in algorithms:
    style = style_map[algo]
    ax.plot(distances, f_actual[algo],
            linestyle=style['linestyle'],
            marker=style['marker'],
            color=style['color'],
            label=style['label'],
            linewidth=1.5,
            markersize=6)

ax.set_xlabel('Theoretical Distance $R$ (mm)', fontsize=14, fontweight='bold')
ax.set_ylabel('Frequency $f$ (Hz)', fontsize=14, fontweight='bold')
#ax.set_title('Comparison of Estimated Frequencies with Six Local Zooms', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.4, linestyle=':')
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, -0.15),  # 图例在 x 轴下方
    ncol=5,                       # 5 列 → 强制单行
    frameon=False,
    fontsize=10,
    columnspacing=1.8,
    handletextpad=0.8,
    handlelength=1.2
)

# ----------------------------
# 3. 插入6个局部放大区（优化布局）
# ----------------------------
inset_width = "20%"
inset_height = "22%"
#x_padding = 20
#y_padding = 20000

x_paddings = [15, 15, 15, 15,15, 15]
y_paddings_up = [25000, 25000, 10000, 10000, 10000, 10000]
y_paddings_down = [10000, 10000, 20000, 10000, 20000, 20000]

inset_boxes = [
    [0.12, 0.72, 0.20, 0.22],   # R=856
    [0.12, 0.42, 0.20, 0.22],   # R=1236
    [0.45, 0.72, 0.20, 0.22],   # R=1745
    [0.45, 0.12, 0.20, 0.22],   # R=1959
    [0.72, 0.15, 0.20, 0.22],   # R=2414
    [0.79, 0.45, 0.20, 0.22],   # R=2920
]

titles = ['$R=856$ mm', '$R=1236$ mm', '$R=1745$ mm',
          '$R=1959$ mm', '$R=2414$ mm', '$R=2920$ mm']

connect_locs = [(2, 1), (2, 1), (2, 1),
                (4, 3), (4, 3), (4, 3)]

for i in range(len(distances)):
    x0 = distances[i]
    f0 = f_theoretical[i]
    x_min, x_max = x0 - x_paddings[i], x0 + x_paddings[i]
    y_min, y_max = f0 - y_paddings_down[i], f0 + y_paddings_up[i]

    axins = ax.inset_axes(inset_boxes[i])

    # 绘制数据
    axins.plot(distances, f_theoretical, color='gray', linestyle='--', linewidth=1)
    for algo in algorithms:
        style = style_map[algo]
        axins.plot(distances, f_actual[algo],
                   linestyle=style['linestyle'],
                   marker=style['marker'],
                   color=style['color'],
                   linewidth=1.5,
                   markersize=5)

    axins.set_xlim(x_min, x_max)
    axins.set_ylim(y_min, y_max)
    axins.set_title(titles[i], fontsize=10, pad=4)
    axins.tick_params(labelsize=8)
    axins.grid(True, alpha=0.3)

    # 添加连接线
    mark_inset(ax, axins, loc1=connect_locs[i][0], loc2=connect_locs[i][1],
               fc="none", ec="gray", linestyle="--", linewidth=1.0)

# ----------------------------
# 4. 布局调整
# ----------------------------
plt.subplots_adjust(
    left=0.08,
    right=0.92,
    bottom=0.2,
    top=0.90
)

plt.show()