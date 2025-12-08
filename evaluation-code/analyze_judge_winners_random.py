import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Read the JSONL file
data = []
with open('/dccstor/gma2/jhjenny9/counterfactual_feedback/route-to-reasoning-data/evaluation_predictions_with_gpt4o_random_judged.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

print("=" * 80)
print("JUDGE WINNER STATISTICS (RANDOM JUDGED)")
print("=" * 80)

# Overall win percentage
winner_counts = df['judge_winner'].value_counts()
total = len(df)

print(f"\nTotal evaluations: {total}")
print(f"\nWin counts:")
for winner, count in winner_counts.items():
    percentage = (count / total) * 100
    print(f"  {winner}: {count} ({percentage:.2f}%)")

# Win percentage by turn (round_index)
print("\n" + "=" * 80)
print("WIN PERCENTAGE BY TURN (round_index)")
print("=" * 80)
turn_winner_df = pd.crosstab(df['round_index'], df['judge_winner'], normalize='index') * 100
print(turn_winner_df.round(2))

# Win percentage by confidence level
print("\n" + "=" * 80)
print("WIN PERCENTAGE BY CONFIDENCE LEVEL")
print("=" * 80)
confidence_winner_df = pd.crosstab(df['judge_confidence'], df['judge_winner'], normalize='index') * 100
print(confidence_winner_df.round(2))

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Overall win percentage pie chart
ax1 = axes[0, 0]
winner_counts.plot(kind='pie', ax=ax1, autopct='%1.1f%%', startangle=90)
ax1.set_title('Overall Judge Winner Distribution (Random)', fontsize=14, fontweight='bold')
ax1.set_ylabel('')

# 2. Win percentage by turn - stacked bar chart
ax2 = axes[0, 1]
turn_winner_counts = pd.crosstab(df['round_index'], df['judge_winner'])
turn_winner_counts.plot(kind='bar', stacked=True, ax=ax2, colormap='viridis')
ax2.set_title('Judge Winner Distribution by Turn (Random)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Round Index (Turn)', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.legend(title='Winner', bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.tick_params(axis='x', rotation=45)

# 3. Win percentage by turn - line plot
ax3 = axes[1, 0]
turn_winner_pct = pd.crosstab(df['round_index'], df['judge_winner'], normalize='index') * 100
turn_winner_pct.plot(ax=ax3, marker='o', linewidth=2)
ax3.set_title('Win Percentage by Turn (Random)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Round Index (Turn)', fontsize=12)
ax3.set_ylabel('Win Percentage (%)', fontsize=12)
ax3.legend(title='Winner', bbox_to_anchor=(1.05, 1), loc='upper left')
ax3.grid(True, alpha=0.3)

# 4. Win percentage by confidence level - grouped bar chart
ax4 = axes[1, 1]
confidence_winner_pct = pd.crosstab(df['judge_confidence'], df['judge_winner'], normalize='index') * 100
confidence_winner_pct.plot(kind='bar', ax=ax4, colormap='Set2')
ax4.set_title('Win Percentage by Confidence Level (Random)', fontsize=14, fontweight='bold')
ax4.set_xlabel('Judge Confidence', fontsize=12)
ax4.set_ylabel('Win Percentage (%)', fontsize=12)
ax4.legend(title='Winner', bbox_to_anchor=(1.05, 1), loc='upper left')
ax4.tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('/dccstor/gma2/jhjenny9/counterfactual_feedback/judge_winner_analysis_random.png', dpi=300, bbox_inches='tight')
print(f"\n\nVisualization saved to: judge_winner_analysis_random.png")

# Additional detailed statistics
print("\n" + "=" * 80)
print("ADDITIONAL STATISTICS")
print("=" * 80)

# Confidence distribution
print("\nJudge Confidence Distribution:")
conf_counts = df['judge_confidence'].value_counts()
for conf, count in conf_counts.items():
    percentage = (count / total) * 100
    print(f"  {conf}: {count} ({percentage:.2f}%)")

# Requires reasoning distribution
print("\nRequires Reasoning Distribution:")
req_reasoning = df['requires_reasoning'].value_counts()
for val, count in req_reasoning.items():
    percentage = (count / total) * 100
    print(f"  {val}: {count} ({percentage:.2f}%)")

# Win percentage by requires_reasoning
print("\nWin Percentage by Requires Reasoning:")
reasoning_winner_df = pd.crosstab(df['requires_reasoning'], df['judge_winner'], normalize='index') * 100
print(reasoning_winner_df.round(2))
