# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# %%
# Read the CSV file
df = pd.read_csv('results/full_dumb_history.csv')
df = pd.read_csv('results/full_default_2025-11-30_14-48-07.csv')

# Group by condition and step, then calculate mean and std across runs and seeds
grouped = df.groupby(['condition', 'step'])['score'].agg(['mean', 'std', 'sum', 'count']).reset_index()

# Calculate 95% confidence interval (1.96 * standard error)
grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
grouped['ci'] = 1.96 * grouped['se']

# Create the plot
plt.figure(figsize=(12, 7))

# Get unique conditions
conditions = grouped['condition'].unique()
colors = sns.color_palette("husl", len(conditions))

# Plot each condition
for i, condition in enumerate(conditions):
    data = grouped[grouped['condition'] == condition]
    
    # Add 1 to step to show as 1-40 instead of 0-39
    x = data['step'] + 1
    y = data['mean']
    ci = data['ci']
    se = data['se']
    
    # Cap values to minimum of 0.1 for log scale (can't display 0 on log scale)
    y = np.maximum(y, 0.1)
    y_lower = np.maximum(y - ci, 0.1)
    y_upper = np.maximum(y + ci, 0.1)
    
    # Plot mean line
    plt.plot(x, y, label=condition, color=colors[i], linewidth=2)
    
    # Plot confidence band
    plt.fill_between(x, y_lower, y_upper, alpha=0.2, color=colors[i])

# Set log scale for y-axis
plt.yscale('log')

# Labels and title
plt.xlabel('Step', fontsize=12)
plt.ylabel('Score (log scale)', fontsize=12)
plt.title('Performance Over Time by Condition', fontsize=14, fontweight='bold')
plt.legend(title='Condition', fontsize=10)
plt.grid(True, alpha=0.3, which='both', linestyle='--')

# Set x-axis limits
plt.xlim(1, 40)

plt.tight_layout()
plt.savefig('performance_over_time.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved as 'performance_over_time.png'")