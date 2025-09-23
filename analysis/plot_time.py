# Dual-axis line chart with solid right/top spines and dark-blue time line
import matplotlib.pyplot as plt

iterations = list(range(0, 11))
times = [40.8, 49.6, 56.5, 60.6, 59.7, 60.5, 60.0, 62.0, 63.9, 63.5, 63.2]
activations = [0.0, 6.0, 6.1, 6.0, 6.0, 6.0, 6.0, 6.0, 6.1, 6.1, 6.1]

baseline = times[0]
pct_increase = [((t - baseline) / baseline) * 100.0 for t in times]

plt.figure(figsize=(7.5, 4.5))
ax = plt.gca()
ax2 = ax.twinx()

# Left y-axis
line1, = ax.plot(iterations, activations, marker='o', linewidth=2, linestyle='--', label='Activations per task')

# Right y-axis (deep blue)
deep_blue = '#1f4e79'
line2, = ax2.plot(iterations, pct_increase, marker='s', linewidth=2,
                  color=deep_blue, label='Time increase vs. baseline (%)')

ax.set_xlabel('Iterations per activation', fontsize=14, fontweight='bold')
ax.set_ylabel('Activations per task', fontsize=14, fontweight='bold')
ax2.set_ylabel('Time increase vs. baseline (%)', fontsize=14, fontweight='bold')
ax.set_xticks(iterations)

ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)

# Solid right/top spines
for a in (ax, ax2):
    a.spines['top'].set_visible(True)
    a.spines['top'].set_linewidth(1.2)
ax2.spines['right'].set_visible(True)
ax2.spines['right'].set_linewidth(1.2)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# plt.title('Activations (left) and time increase vs. baseline (right)', fontsize=14, fontweight='bold')
ax.legend([line1, line2], [line1.get_label(), line2.get_label()], loc='lower right')

plt.tight_layout()
plt.savefig('iters_activations_timepct_v2.png', dpi=200)