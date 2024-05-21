import matplotlib.pyplot as plt
import numpy as np

# Data
Boxdata1 = [62.8, np.nan, np.nan]
Boxdata2 = [63, np.nan, np.nan]
Boxdata3 = [64.1, 60.3, 65.0]
Boxdata4 = [71.6, np.nan, np.nan]
Mapdata1 = [50.0, np.nan, np.nan]
Mapdata2 = [51.5, 52.5, 58.2]
Mapdata3 = [53.7, np.nan, np.nan]
Mapdata4 = [53.2, np.nan, np.nan]

# X-axis labels
x_labels = ['start', 'tuned', 'full']

# X-axis values (indices for the labels)
x = range(len(x_labels))

# Plotting the Boxdata
plt.figure(figsize=(12,8))
plt.plot(x, Boxdata1, label='YoloV5 (Box)', marker='o', linestyle='-', linewidth=2)
plt.plot(x, Boxdata2, label='YoloV8 (Box)', marker='o', linestyle='-', linewidth=2)
plt.plot(x, Boxdata3, label='YoloV9 (Box)', marker='o', linestyle='-', linewidth=2)
plt.plot(x, Boxdata4, label='GELAN (Box)', marker='o', linestyle='-', linewidth=2)

# Plotting the Mapdata
plt.plot(x, Mapdata1, label='YoloV5 (Map)', marker='x', linestyle='--', linewidth=2)
plt.plot(x, Mapdata2, label='YoloV8 (Map)', marker='x', linestyle='--', linewidth=2)
plt.plot(x, Mapdata3, label='YoloV9 (Map)', marker='x', linestyle='--', linewidth=2)
plt.plot(x, Mapdata4, label='GELAN (Map)', marker='x', linestyle='--', linewidth=2)

# Adding titles and labels
plt.title('Performance Comparison of Different Models', fontsize=14)
plt.xlabel('Stages', fontsize=12)
plt.ylabel('Values', fontsize=12)

# Adding x-axis labels
plt.xticks(x, x_labels, fontsize=10)

# Adding legend
plt.legend(loc='upper left', fontsize=10)

# Adding grid for better readability
plt.grid(True, linestyle='--', alpha=0.6)

# Save the figure
plt.savefig('performance_comparison_plot.png', dpi=300, bbox_inches='tight')

# Display the graph
plt.show()
