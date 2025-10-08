import matplotlib.pyplot as plt
import numpy as np

# === 1. Data Preparation ===
distances = np.array([856, 1236, 1745, 1959, 2414, 2920])  # Theoretical R (mm)

# Measured R = Theoretical R ± △R (but here we have actual deviation)
# We'll plot: measured_R = theoretical_R + error_sign * |△R|
# Since △R is absolute error, we assume it's symmetric for visualization

# Actual measured values (reconstructed from theoretical + bias implied by error)
# Note: △R = |R_measured - R_theoretical|, so R_measured could be higher or lower.
# From your data, we can back-calculate:
measured_r = {
    'FFT':      [917.6, 1310.9, 1704.1, 1966.3, 2359.5, 2883.9],
    'Macleod':  [900.9, 1278.0, 1705.3, 1966.1, 2369.1, 2883.3],
    'CZT':      [884.8, 1278.1, 1732.5, 1955.4, 2392.3, 2899.2],
    'MCZT':     [868.2, 1251.2, 1733.3, 1957.5, 2401.9, 2900.9]
}

# Errors (△R) — used as error bars
errors_r = {
    'FFT':      [61.6, 74.9, 40.9, 7.3, 54.5, 36.1],
    'Macleod':  [44.9, 42.0, 39.7, 7.1, 44.9, 36.7],
    'CZT':      [28.8, 42.1, 12.5, 3.6, 21.7, 20.8],
    'MCZT':     [12.2, 15.2, 11.7, 1.5, 12.1, 19.1]
}

# === 2. Style Settings ===
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728']  # CZT, MCZT, FFT, Macleod
markers = ['o', 's', '^', 'D']
algorithms = ['FFT', 'Macleod', 'CZT', 'MCZT']

# === 3. Create Error Bar Plot ===
plt.figure(figsize=(12, 7))

for i, algo in enumerate(algorithms):
    plt.errorbar(distances, measured_r[algo], yerr=errors_r[algo],
                 fmt=markers[i], capsize=5, capthick=1.5,
                 label=algo, color=colors[i], markersize=8, linewidth=2,
                 elinewidth=1.2)

# Add theoretical reference line
plt.plot(distances, distances, 'k--', linewidth=2, label='Theoretical Value', alpha=0.7)

# === 4. Customize Plot ===
plt.xlabel('Theoretical Distance $R$ (mm)', fontsize=12)
plt.ylabel('Measured Distance $R_{\\mathrm{meas}}$ (mm)', fontsize=12)
plt.title('Measured Distance vs. Theoretical Distance with Error Bars',
          fontsize=14, fontweight='bold')

plt.legend(title='Algorithm', fontsize=11)
plt.grid(True, alpha=0.3)

# Ensure equal scaling so diagonal is 45°
plt.axis('equal')
# Optional: set axis limits for better visual alignment
x_min, x_max = 800, 3000
y_min, y_max = 800, 3000
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Add diagonal guide line (y = x)
plt.plot([x_min, x_max], [y_min, y_max], 'k--', linewidth=1, alpha=0.5)

# === 5. Layout & Save ===
plt.tight_layout()
#plt.savefig('distance_errorbar_plot.png', dpi=300, bbox_inches='tight')
#plt.savefig('distance_errorbar_plot.pdf', bbox_inches='tight')  # For LaTeX

plt.show()