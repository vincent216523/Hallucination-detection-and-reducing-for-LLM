import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import scipy as sp
from scipy.stats import ttest_ind, mannwhitneyu, entropy


inference_results = list(Path("./data/").rglob("*.pickle"))

df_list = list()
for results_file in inference_results:
    with open(results_file, "rb") as infile:
        df_list.append(pd.DataFrame(pickle.loads(infile.read())))
results = pd.concat(df_list, ignore_index=True)

# Plot histogram
first_logits = np.stack([sp.special.softmax(i[j]) for i, j in zip(results['logits'], results['start_pos'])])
results['first_logits'] = first_logits.tolist()
results['first_logits_entropy'] = results['first_logits'].apply(lambda x: entropy(x))
# Plot the distribution of entropy for each flag
plt.figure(figsize=(10, 6))
sns.histplot(data=results, x='first_logits_entropy', hue='correct', kde=True, bins=20, palette="Set1")

# Add labels and title
plt.title('Distribution of Softmax Entropy for True/False', fontsize=16)
plt.xlabel('Entropy', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(alpha=0.3)

# Show the plot
plt.show()


# Statistical tests
data_true = results[results['correct'] == True]['first_logits_entropy']
data_false = results[results['correct'] == False]['first_logits_entropy']

t_stat, p_value = ttest_ind(data_true, data_false, equal_var=False)
print(f"t-test: p-value: {p_value:.4g}, stat:{t_stat:.4g}. Signifcant: {p_value<0.05}")
stat, p_value = mannwhitneyu(data_true, data_false, alternative='two-sided')
print(f"MWU-test (2-sided): p-value: {p_value:.4g}. Signifcant: {p_value<0.05}")
stat, p_value = mannwhitneyu(data_true, data_false, alternative='less')
print(f"MWU-test (true < false): p-value: {p_value:.4g}. Signifcant: {p_value<0.05}")
stat, p_value = mannwhitneyu(data_true, data_false, alternative='greater')
print(f"MWU-test (false < true): p-value: {p_value:.4g}. Signifcant: {p_value<0.05}")
