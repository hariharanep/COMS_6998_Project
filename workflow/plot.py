import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with open("experiment_results-test.json", "r") as f:
    results = json.load(f)

df = pd.DataFrame(results)
df['score'] = pd.to_numeric(df['score'], errors='coerce')

baseline_scores = df[df['technique'] == 'baseline'][['domain','prompt','score']].rename(columns={'score': 'baseline_score'})
df = df.merge(baseline_scores, on=['domain','prompt'], how='left')
df['improvement'] = df['score'] - df['baseline_score']

# Boxplot + swarmplot of scores per technique
plt.figure(figsize=(10,6))
sns.boxplot(x='technique', y='score', data=df)
sns.swarmplot(x='technique', y='score', data=df, color=".25", alpha=0.5)
plt.title('AHLLM Honesty Score Distribution by Technique')
plt.ylabel('Honesty Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Bar chart: average improvement per technique
plt.figure(figsize=(10,6))
avg_improvement = df[df['technique'] != 'baseline'].groupby('technique')['improvement'].mean()
avg_improvement.plot(kind='bar', color='skyblue', edgecolor='black')
plt.ylabel("Average Improvement Over Baseline")
plt.title("Average Improvement by Technique")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Heatmap: average honesty score per domain & technique
plt.figure(figsize=(10,6))
pivot_score = df.pivot_table(index='domain', columns='technique', values='score', aggfunc='mean')
sns.heatmap(pivot_score, annot=True, fmt=".1f", cmap="coolwarm")
plt.title('Average AHLLM Honesty Score: Domain vs Technique')
plt.tight_layout()
plt.show()

# Heatmap: average improvement per domain & technique
plt.figure(figsize=(10,6))
pivot_improve = df[df['technique'] != 'baseline'].pivot_table(
    index='domain', columns='technique', values='improvement', aggfunc='mean'
)
sns.heatmap(pivot_improve, annot=True, fmt=".1f", cmap="RdYlGn")
plt.title("Average AHLLM Improvement Over Baseline: Domain vs Technique")
plt.tight_layout()
plt.show()
