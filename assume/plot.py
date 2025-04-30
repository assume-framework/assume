import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True, gridspec_kw={'width_ratios': [1, 1]})

# Cement Mills (Left Plot)
ax1.plot([0, 5, 5, 20, 20, 24], [0, 0, 41, 41, 0, 0], color='orangered', linewidth=1.5)
ax1.axhline(y=34, linestyle='--', color='black', linewidth=2)

ax1.add_patch(patches.Rectangle((5, 34), 15, 7, facecolor='skyblue', edgecolor='none'))
ax1.text(7, 36.5, 'Load curtailment potential', fontsize=10, color='white')

ax1.add_patch(patches.Rectangle((20, 0), 4, 41, facecolor='skyblue', edgecolor='black', linestyle='--', linewidth=1))
ax1.text(20.5, 15, 'Load shifting potential', fontsize=10, color='white', rotation=90)

ax1.set_xlim(0, 24)
ax1.set_ylim(0, 45)
ax1.set_xlabel('Hours')
ax1.set_ylabel('Load (MW)')
ax1.set_title('Cement Mills', loc='left')

# Raw Mills (Right Plot)
ax2.plot([0, 4, 4, 20, 20, 25], [0, 0, 26, 26, 0, 0], color='orangered', linewidth=1.5)
ax2.axhline(y=22, linestyle='--', color='black', linewidth=2)

ax2.add_patch(patches.Rectangle((4, 22), 16, 4, facecolor='skyblue', edgecolor='none'))
ax2.text(6, 23.5, 'Load curtailment potential', fontsize=10, color='white')

ax2.add_patch(patches.Rectangle((20, 0), 5, 26, facecolor='skyblue', edgecolor='black', linestyle='--', linewidth=1))
ax2.text(20.5, 10, 'Load shifting potential', fontsize=10, color='white', rotation=90)

ax2.set_xlim(0, 25)
ax2.set_xlabel('Hours')
ax2.set_title('Raw Mills', loc='left')

# Custom Legend
line_max = mlines.Line2D([], [], color='orangered', linewidth=2, label='Load curve at maximum operating speed')
line_norm = mlines.Line2D([], [], color='black', linewidth=2, linestyle='--', label='Load curve at normal operating speed')
fig.legend(handles=[line_norm, line_max], loc='lower center', ncol=2, fontsize=10, frameon=False)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()
