# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# %%
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10
})

max_step = 40

files = {
    "Default":    "results/default_2026-02-18_17-03-34.csv",
    "Hypothesis": "results/hypo_history.csv",
    "Hybrid":     "results/hybridcombined_history.csv",
}

def prepare_df(path):
    df = pd.read_csv(path)
    df = df.sort_values(['condition', 'run', 'seed', 'step'])
    df['score'] = df.groupby(['condition', 'run', 'seed'])['score'].cummax()

    filled = []
    for (cond, run, seed), group in df.groupby(['condition', 'run', 'seed']):
        score_series = group.set_index('step')['score'].reindex(range(1, max_step + 1))
        score_series = score_series.ffill().fillna(0)
        chunk = pd.DataFrame({
            'condition': cond,
            'run':       run,
            'seed':      seed,
            'step':      range(1, max_step + 1),
            'score':     score_series.values,
        })
        filled.append(chunk)

    df_filled = pd.concat(filled, ignore_index=True)

    grouped = df_filled.groupby(['condition', 'step'])['score'].agg(['mean', 'std', 'count']).reset_index()
    grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
    grouped['ci'] = 1.96 * grouped['se']
    return grouped

# Load all
results = {name: prepare_df(path) for name, path in files.items()}

# Print conditions per file for debugging
for name, grouped in results.items():
    print(f"{name}: {sorted(grouped['condition'].unique())}")

# Consistent colours across all panels
all_conditions = sorted(set(
    cond
    for grouped in results.values()
    for cond in grouped['condition'].unique()
))
palette   = sns.color_palette("colorblind", len(all_conditions))
color_map = dict(zip(all_conditions, palette))

# Global y-limits
all_means = pd.concat([r['mean'] for r in results.values()])
y_min = max(all_means[all_means > 0].min() * 0.8, 0.1)
y_max = all_means.max() * 1.3

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

for ax, (title, grouped) in zip(axes, results.items()):
    for condition in all_conditions:
        data = grouped[grouped['condition'] == condition]
        if data.empty:
            print(f"  WARNING: '{condition}' missing from {title}")
            continue

        x       = data['step']
        y       = np.maximum(data['mean'], 0.1)
        y_lower = np.maximum(y - data['ci'], 0.1)
        y_upper = np.maximum(y + data['ci'], 0.1)

        ax.plot(x, y,
                label=condition,
                color=color_map[condition],
                linewidth=2.5)
        ax.fill_between(x, y_lower, y_upper,
                        alpha=0.2,
                        color=color_map[condition])

    ax.set_yscale('log')
    ax.set_ylim(y_min, y_max)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Step")
    ax.grid(True, axis='y', linestyle='--', alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)

axes[0].set_ylabel("Score (log scale, mean ± 95% CI)")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           title="Condition",
           loc='center right',
           bbox_to_anchor=(1.0, 0.5),
           frameon=False)

plt.tight_layout(rect=[0, 0, 0.88, 1])
plt.savefig("performance_comparison.pdf", dpi=300, bbox_inches="tight")
plt.show()





# %%
# Read the CSV file
#df = pd.read_csv('results/default_2026-02-18_17-03-34.csv')
df = pd.read_csv('results/hypo_history.csv')
#df = pd.read_csv('results/hybridcombined_history.csv')

max_step = 40
all_steps = pd.DataFrame({'step': range(1, max_step + 1)})

expanded_runs = []
for (cond, run, seed), group in df.groupby(['condition', 'run', 'seed']):
    
    group = group.sort_values('step')
    
    # Track highest score so far
    group['score'] = group['score'].cummax()
    
    # Merge with full step range
    merged = all_steps.merge(group, on='step', how='left')
    
    # Fill identifying columns
    merged['condition'] = cond
    merged['run'] = run
    merged['seed'] = seed
    
    # Forward-fill score, then fill initial NaN with 0
    merged['score'] = merged['score'].ffill().fillna(0)
    
    expanded_runs.append(merged)

df = pd.concat(expanded_runs, ignore_index=True)


# %%
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
# %%
