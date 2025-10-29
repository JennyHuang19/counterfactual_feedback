"""
Exploratory Data Analysis for loaded_conversations.json
Analyzes conversation data from the naturally occurring feedback dataset
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load the data
print("Loading data...")
with open("/data/lingo/jhuang9/counterfactual_feedback/data/filtered_conversations_ur235.json", "r") as f:
    data = json.load(f)

print(f"Total conversations loaded: {len(data)}")
print("=" * 80)

# Convert to DataFrame for easier analysis
df = pd.DataFrame(data)

# Basic Dataset Overview
print("\n1. DATASET OVERVIEW")
print("=" * 80)
print(f"Number of conversations: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"\nData types:")
print(df.dtypes)

# Basic statistics
print("\n2. BASIC STATISTICS")
print("=" * 80)
print("\nDataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# Model distribution
print("\n3. MODEL DISTRIBUTION")
print("=" * 80)
model_counts = df['model'].value_counts()
print(model_counts)
print(f"\nTotal unique models: {df['model'].nunique()}")

# Language distribution
print("\n4. LANGUAGE DISTRIBUTION")
print("=" * 80)
language_counts = df['language'].value_counts()
print(language_counts)
print(f"\nTotal unique languages: {df['language'].nunique()}")

# Category distribution
print("\n5. CATEGORY DISTRIBUTION")
print("=" * 80)
category_counts = df['category'].value_counts()
print(category_counts)
print(f"\nTotal unique categories: {df['category'].nunique()}")

# Label distribution
print("\n6. LABEL DISTRIBUTION")
print("=" * 80)
label_counts = df['label'].value_counts()
print(label_counts)
print(f"\nLabel proportions:")
print(df['label'].value_counts(normalize=True))

# Feedback turn analysis
print("\n7. FEEDBACK TURN ANALYSIS")
print("=" * 80)
print(f"Min feedback turn: {df['feedback_turn'].min()}")
print(f"Max feedback turn: {df['feedback_turn'].max()}")
print(f"Mean feedback turn: {df['feedback_turn'].mean():.2f}")
print(f"Median feedback turn: {df['feedback_turn'].median()}")
print("\nFeedback turn distribution:")
print(df['feedback_turn'].value_counts().sort_index())

# Conversation context length analysis
print("\n8. CONVERSATION CONTEXT LENGTH ANALYSIS")
print("=" * 80)
df['context_length'] = df['conversation_context'].apply(len)
print(f"Min context length: {df['context_length'].min()}")
print(f"Max context length: {df['context_length'].max()}")
print(f"Mean context length: {df['context_length'].mean():.2f}")
print(f"Median context length: {df['context_length'].median()}")

# Text length analysis (total characters in conversation)
print("\n9. TEXT LENGTH ANALYSIS")
print("=" * 80)
df['total_chars'] = df['conversation_context'].apply(
    lambda x: sum(len(turn['content']) for turn in x)
)
print(f"Min total characters: {df['total_chars'].min()}")
print(f"Max total characters: {df['total_chars'].max()}")
print(f"Mean total characters: {df['total_chars'].mean():.2f}")
print(f"Median total characters: {df['total_chars'].median()}")

# Role distribution in conversations
print("\n10. ROLE DISTRIBUTION IN CONVERSATIONS")
print("=" * 80)
all_roles = []
for conv in df['conversation_context']:
    for turn in conv:
        all_roles.append(turn['role'])
role_counts = Counter(all_roles)
print("Role counts across all conversations:")
for role, count in role_counts.items():
    print(f"  {role}: {count}")

# Cross-tabulations
print("\n11. CROSS-TABULATIONS")
print("=" * 80)

print("\nLabel by Category:")
print(pd.crosstab(df['category'], df['label'], margins=True))

print("\n\nLabel by Model:")
print(pd.crosstab(df['model'], df['label'], margins=True))

print("\n\nLabel by Language:")
print(pd.crosstab(df['language'], df['label'], margins=True))

# Unique conversation IDs
print("\n12. CONVERSATION ID UNIQUENESS")
print("=" * 80)
print(f"Total conversations: {len(df)}")
print(f"Unique conversation IDs: {df['conversation_id'].nunique()}")
print(f"Duplicate conversation IDs: {len(df) - df['conversation_id'].nunique()}")

# Sample conversations
print("\n13. SAMPLE CONVERSATIONS")
print("=" * 80)
print("\nFirst conversation structure:")
sample = data[0]
print(f"Conversation ID: {sample['conversation_id']}")
print(f"Model: {sample['model']}")
print(f"Language: {sample['language']}")
print(f"Category: {sample['category']}")
print(f"Label: {sample['label']}")
print(f"Feedback turn: {sample['feedback_turn']}")
print(f"Number of turns in context: {len(sample['conversation_context'])}")
print("\nConversation turns:")
for i, turn in enumerate(sample['conversation_context']):
    content_preview = turn['content'][:100] + "..." if len(turn['content']) > 100 else turn['content']
    print(f"  Turn {i+1} ({turn['role']}): {content_preview}")

# Generate visualizations
print("\n14. GENERATING VISUALIZATIONS")
print("=" * 80)

# Create a figure with multiple subplots
fig = plt.figure(figsize=(16, 12))

# 1. Model distribution
plt.subplot(3, 3, 1)
model_counts.plot(kind='barh')
plt.title('Model Distribution')
plt.xlabel('Count')
plt.tight_layout()

# 2. Language distribution
plt.subplot(3, 3, 2)
language_counts.plot(kind='bar')
plt.title('Language Distribution')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# 3. Category distribution
plt.subplot(3, 3, 3)
category_counts.plot(kind='bar', color='green')
plt.title('Category Distribution')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# 4. Label distribution
plt.subplot(3, 3, 4)
label_counts.plot(kind='bar', color='orange')
plt.title('Label Distribution')
plt.ylabel('Count')
plt.xlabel('Label')
plt.tight_layout()

# 5. Feedback turn distribution
plt.subplot(3, 3, 5)
df['feedback_turn'].hist(bins=20, edgecolor='black')
plt.title('Feedback Turn Distribution')
plt.xlabel('Feedback Turn')
plt.ylabel('Frequency')
plt.tight_layout()

# 6. Context length distribution
plt.subplot(3, 3, 6)
df['context_length'].hist(bins=20, edgecolor='black', color='purple')
plt.title('Context Length Distribution')
plt.xlabel('Number of Turns')
plt.ylabel('Frequency')
plt.tight_layout()

# 7. Total characters distribution
plt.subplot(3, 3, 7)
df['total_chars'].hist(bins=30, edgecolor='black', color='red')
plt.title('Total Characters Distribution')
plt.xlabel('Total Characters')
plt.ylabel('Frequency')
plt.tight_layout()

# 8. Label by Category heatmap
plt.subplot(3, 3, 8)
label_by_category = pd.crosstab(df['category'], df['label'])
sns.heatmap(label_by_category, annot=True, fmt='d', cmap='YlOrRd')
plt.title('Label by Category Heatmap')
plt.tight_layout()

# 9. Label by Model
plt.subplot(3, 3, 9)
label_by_model = pd.crosstab(df['model'], df['label'])
sns.heatmap(label_by_model, annot=True, fmt='d', cmap='Blues')
plt.title('Label by Model Heatmap')
plt.tight_layout()

plt.savefig('jenny-edits/eda_visualizations.png', dpi=300, bbox_inches='tight')
print("Visualizations saved to: jenny-edits/eda_visualizations.png")

# Additional detailed statistics
print("\n15. SUMMARY STATISTICS FOR NUMERIC COLUMNS")
print("=" * 80)
print(df[['feedback_turn', 'context_length', 'total_chars', 'label']].describe())


print("\n" + "=" * 80)
print("EDA COMPLETE!")
print("=" * 80)
