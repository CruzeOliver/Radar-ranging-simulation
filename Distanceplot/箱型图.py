import matplotlib.pyplot as plt
import numpy as np

# === 1. Data Preparation: △R and △f for each algorithm across 6 distances ===
distances = [856, 1236, 1745, 1959, 2414, 2920]

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

# === 2. Prepare Data for Box Plot ===
algorithms = ['FFT', 'Macleod', 'CZT', 'MCZT']
data_r = [errors_r[algo] for algo in algorithms]
data_f = [errors_f[algo] for algo in algorithms]

colors = ['#D62728', '#2CA02C', '#1F77B4', '#FF7F0E']
boxprops = dict(facecolor='lightgray', edgecolor='black')
medianprops = dict(color='red', linewidth=2)

# === 3. Create Subplots: △R and △f Box Plots ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# -----------------------------
# (a) Box Plot for △R
# -----------------------------
bp1 = ax1.boxplot(data_r, labels=algorithms, patch_artist=True,
                  boxprops=dict(facecolor='lightgray', edgecolor='black'),
                  medianprops=medianprops, whiskerprops=dict(linewidth=1.5),
                  capprops=dict(linewidth=1.5), flierprops=dict(marker='o', alpha=0.6))

# Set face colors for each box
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax1.set_ylabel('Distance Error $\\Delta R$ (mm)', fontsize=12)
ax1.set_title('(a) Distribution of Distance Errors', fontsize=13, fontweight='bold')
ax1.grid(True, axis='y', alpha=0.3)
ax1.tick_params(axis='y', labelsize=11)

# -----------------------------
# (b) Box Plot for △f
# -----------------------------
bp2 = ax2.boxplot(data_f, labels=algorithms, patch_artist=True,
                  boxprops=dict(facecolor='lightgray', edgecolor='black'),
                  medianprops=medianprops, whiskerprops=dict(linewidth=1.5),
                  capprops=dict(linewidth=1.5), flierprops=dict(marker='o', alpha=0.6))

# Set face colors
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax2.set_ylabel('Frequency Error $\\Delta f$ (Hz)', fontsize=12)
ax2.set_title('(b) Distribution of Frequency Errors', fontsize=13, fontweight='bold')
ax2.grid(True, axis='y', alpha=0.3)
ax2.tick_params(axis='y', labelsize=11)

# === 4. Final Layout ===
plt.suptitle('Box Plot Comparison of Algorithm Performance under Anechoic Environment',
             fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.96])

# === 5. Save and Show ===
#plt.savefig('boxplot_algorithm_comparison.png', dpi=300, bbox_inches='tight')
#plt.savefig('boxplot_algorithm_comparison.pdf', bbox_inches='tight')
plt.show()