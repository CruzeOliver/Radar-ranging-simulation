import matplotlib.pyplot as plt
import numpy as np

# === 1. Data Preparation ===
distances = np.array([856, 1236, 1745, 1959, 2414, 2920])  # Theoretical distance R (mm)

# Distance error △R (mm)
errors_r = {
    'FFT':      [61.6, 74.9, 40.9, 7.3, 54.5, 36.1],
    'Macleod':  [44.9, 42.0, 39.7, 7.1, 44.9, 36.7],
    'CZT':      [28.8, 42.1, 12.5, 3.6, 21.7, 20.8],
    'MCZT':     [12.2, 15.2, 11.7, 1.5, 12.1, 19.1]
}

# Frequency error △f (Hz)
errors_f = {
    'FFT':      [13173, 15855, 8698, 1551, 11586, 7683],
    'Macleod':  [9628,  8860,  8439, 1514, 9553,  7808],
    'CZT':      [6200,  8882,  2655, 773,  4614,  4429],
    'MCZT':     [2655,  3170,  2493, 327,  2580,  4073]
}

# === 2. Style Settings ===
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
colors = ['#D62728', '#2CA02C', '#1F77B4', '#FF7F0E']  # FFT, Macleod, CZT, MCZT
algorithms = ['FFT', 'Macleod', 'CZT', 'MCZT']

# === 3. Figure: Grouped Bar Chart (△R) ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

bar_width = 0.2
index = np.arange(len(distances))

for i, algo in enumerate(algorithms):
    bars = ax1.bar(index + i * bar_width, errors_r[algo], bar_width,
                   label=algo, color=colors[i], edgecolor='black', linewidth=0.8)

    # Annotate values on bars
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords='offset points',
                     ha='center', va='bottom', fontsize=8)

ax1.set_xlabel('Theoretical Distance R (mm)', fontsize=12)
ax1.set_ylabel('Distance Error R (mm)', fontsize=12)
ax1.set_title('Comparison of Distance Estimation Errors', fontsize=14, fontweight='bold')
ax1.set_xticks(index + bar_width * 1.5)
ax1.set_xticklabels(distances)
ax1.legend(title='Algorithm')
ax1.grid(True, axis='y', alpha=0.3)

# === 4. Line Plot (△f) ===
for i, algo in enumerate(algorithms):
    ax2.plot(distances, errors_f[algo], marker='o', markersize=6,
             label=algo, color=colors[i], linewidth=2)

ax2.set_xlabel('Theoretical Distance R (mm)', fontsize=12)
ax2.set_ylabel('Frequency Error f (Hz)', fontsize=12)
ax2.set_title('Trend of Frequency Estimation Errors', fontsize=14, fontweight='bold')
ax2.legend(title='Algorithm')
ax2.grid(True, alpha=0.3)

# === 5. Layout & Title ===
plt.tight_layout(pad=2.0)
plt.suptitle('Performance Comparison of Algorithms under Anechoic Environment',
             fontsize=16, y=0.98, fontweight='bold')
plt.subplots_adjust(top=0.88)

# === 6. Save and Show (Recommended for Paper) ===
# Save as high-resolution PNG
#plt.savefig('algorithm_performance_comparison.png', dpi=300, bbox_inches='tight')
#plt.savefig('algorithm_performance_comparison.pdf', bbox_inches='tight')  # For LaTeX

plt.show()