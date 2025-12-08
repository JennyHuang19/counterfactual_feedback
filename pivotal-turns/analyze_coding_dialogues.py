#!/usr/bin/env python3
"""
Analyze coding_dialogues.csv to identify non-coding content
"""

import pandas as pd
import json
import re
from collections import Counter
from tqdm import tqdm

# Load the data
print("Loading dataset...")
df = pd.read_csv('pivotal-turns/coding_dialogues.csv')
print(f"Total dialogues: {len(df):,}")
print(f"Columns: {df.columns.tolist()}\n")

# Parse first user message from each conversation
def get_first_message(conv_json):
    try:
        conv = json.loads(conv_json)
        return conv[0]['content'] if conv else ""
    except:
        return ""

print("Parsing first messages...")
df['first_message'] = df['conversation'].apply(get_first_message)

# Define coding indicators
CODING_KEYWORDS = [
    'code', 'function', 'python', 'javascript', 'java', 'program', 'script',
    'algorithm', 'debug', 'error', 'api', 'database', 'sql', 'html', 'css',
    'class', 'method', 'variable', 'compile', 'syntax', 'import', 'library',
    'framework', 'django', 'react', 'node', 'git', 'repository', 'bug',
    'implementation', 'c++', 'ruby', 'php', 'typescript', 'rust', 'go',
    'regex', 'json', 'xml', 'array', 'list', 'dictionary', 'loop'
]

NON_CODING_KEYWORDS = [
    'story', 'poem', 'poetry', 'novel', 'fantasy', 'recipe', 'cook',
    'imagine', 'creative writing', 'essay', 'character', 'protagonist',
    'romance', 'adventure', 'medieval', 'dragon', 'magic', 'wizard',
    'once upon', 'narrative', 'fiction', 'songwriter', 'lyrics', 'song',
    'plot', 'chapter', 'villain', 'hero', 'journey'
]

# Score each dialogue
def score_dialogue(text):
    if not isinstance(text, str):
        return 0, 0

    text_lower = text.lower()

    coding_score = sum(1 for keyword in CODING_KEYWORDS if keyword in text_lower)
    non_coding_score = sum(1 for keyword in NON_CODING_KEYWORDS if keyword in text_lower)

    return coding_score, non_coding_score

print("Scoring dialogues...")
tqdm.pandas()
df[['coding_score', 'non_coding_score']] = df['first_message'].progress_apply(
    lambda x: pd.Series(score_dialogue(x))
)

# Categorize dialogues
def categorize(row):
    if row['coding_score'] > 0 and row['non_coding_score'] == 0:
        return 'likely_coding'
    elif row['non_coding_score'] > 0 and row['coding_score'] == 0:
        return 'likely_non_coding'
    elif row['coding_score'] > row['non_coding_score']:
        return 'probably_coding'
    elif row['non_coding_score'] > row['coding_score']:
        return 'probably_non_coding'
    else:
        return 'unclear'

df['category'] = df.apply(categorize, axis=1)

# Print statistics
print("\n" + "="*60)
print("CATEGORIZATION RESULTS")
print("="*60)
print(df['category'].value_counts())
print(f"\nPercentages:")
print(df['category'].value_counts(normalize=True) * 100)

# Show examples of non-coding dialogues
print("\n" + "="*60)
print("SAMPLE NON-CODING DIALOGUES (First 10)")
print("="*60)

non_coding = df[df['category'].str.contains('non_coding')]
for i, (idx, row) in enumerate(non_coding.head(10).iterrows()):
    print(f"\n--- Example {i+1} (Row {idx}) ---")
    print(f"Category: {row['category']}")
    print(f"Coding score: {row['coding_score']}, Non-coding score: {row['non_coding_score']}")
    print(f"First message: {row['first_message'][:300]}...")

# Show examples of unclear dialogues
print("\n" + "="*60)
print("SAMPLE UNCLEAR DIALOGUES (First 5)")
print("="*60)

unclear = df[df['category'] == 'unclear']
for i, (idx, row) in enumerate(unclear.head(5).iterrows()):
    print(f"\n--- Example {i+1} (Row {idx}) ---")
    print(f"First message: {row['first_message'][:300]}...")

# Save results
output_file = 'pivotal-turns/coding_dialogues_analysis.csv'
df[['conversation_id', 'category', 'coding_score', 'non_coding_score', 'first_message']].to_csv(
    output_file, index=False
)
print(f"\n\nFull analysis saved to: {output_file}")

# Save non-coding examples
non_coding_file = 'pivotal-turns/non_coding_dialogues.csv'
df[df['category'].str.contains('non_coding')].to_csv(non_coding_file, index=False)
print(f"Non-coding dialogues saved to: {non_coding_file}")
print(f"Total non-coding dialogues: {len(df[df['category'].str.contains('non_coding')]):,}")

# Additional analysis: check language field
print("\n" + "="*60)
print("LANGUAGE DISTRIBUTION")
print("="*60)
if 'language' in df.columns:
    print(df['language'].value_counts().head(10))

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
