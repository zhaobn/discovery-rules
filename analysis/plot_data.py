# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
subject_data =


# %%
subject_data = pd.read_csv("../data/subject_data.csv")
subject_data['assignment'] = pd.Categorical(subject_data['assignment'], categories=['easy', 'medium', 'hard'], ordered=True)

plt.figure(figsize=(12, 6))
sns.scatterplot(data=subject_data, x='assignment', y='total_points', hue='assignment', style='assignment', s=100)
plt.yscale('log')
plt.title('Total Points per Condition (Log Scale)')
plt.xlabel('Assignment')
plt.ylabel('Total Points (Log Scale)')
plt.legend(title='Assignment')
plt.show()

# %%
