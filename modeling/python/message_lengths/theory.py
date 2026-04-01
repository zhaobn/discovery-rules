# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

# %%
# Parameters
epsilon = 0.5     # approximate length for "not sure"
n       = 2       # approximate length for a primitive message
d_vals  = [1, 3, 6]   # three conditions
md_range = np.linspace(3, 10, 100)

def compute(md, d, n, epsilon):
    coverage = np.minimum(md/d, 0.9)
    expressed = np.minimum(md, d)
    diversity = 1 / (d - md + np.exp(md - d))
    return coverage * diversity * expressed * n + (1 - coverage) * epsilon



# def compute(md, d, n, epsilon):
#     coverage = 1 - np.exp(-md / d)
#     expressed = np.minimum(md, d)
#     diversity = md / d * np.exp(-md / d)
#     return coverage * diversity * expressed * n + (1 - coverage) * epsilon


colors  = ['#e05c4b', '#4b9fe0', '#5ec97a']
labels  = [f'd = {d}' for d in d_vals]

fig, ax = plt.subplots(figsize=(9, 6))
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.12, top=0.92)

for d, color in zip(d_vals, colors):
    y = compute(md_range, d, n, epsilon)
    ax.plot(md_range, y, color=color, linewidth=2.2,
            label=f'd = {d}', marker='o', markersize=3.5)

ax.set_xlabel('Effective capacity', fontsize=13)
ax.set_ylabel('Message length', fontsize=13)
ax.set_title('Simulated message length vs effective capacity',
    # '$\\mathcal{L} = \\frac{m_d}{d}\\cdot m_d \\cdot n + '
    # '\\left(1 - \\frac{m_d}{d}\\right)\\cdot\\epsilon'
    # '\\quad\\left[\\frac{m_d}{d} \\to 0.9 \\ \\mathrm{if}\\ m_d > d\\right]$',
    fontsize=13, pad=10)
ax.legend(loc='upper left', framealpha=0.3)

#ax.set_facecolor('#111827')
#fig.patch.set_facecolor('#0f172a')
#ax.tick_params(colors='lightgrey')
#ax.xaxis.label.set_color('lightgrey')
#ax.yaxis.label.set_color('lightgrey')
ax.title.set_color('black')
ax.grid(True, color='#1e293b', linewidth=0.1)

plt.show()
# %%
