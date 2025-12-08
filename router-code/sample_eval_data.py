import pandas as pd
import json
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Read the CSV file
df = pd.read_csv('/dccstor/gma2/jhjenny9/counterfactual_feedback/pivotal-turns-data/expanded_routing_decisions.csv')

print(f"Total rows in CSV: {len(df)}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nrequires_reasoning distribution:")
print(df['requires_reasoning'].value_counts())

# Stratified sampling: 125 True, 125 False
n_per_class = 125

# Sample 125 from each class
df_true = df[df['requires_reasoning'] == True].sample(n=n_per_class, random_state=42)
df_false = df[df['requires_reasoning'] == False].sample(n=n_per_class, random_state=42)

# Combine and shuffle
df_sample = pd.concat([df_true, df_false]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nSampled {len(df_sample)} rows")
print(f"True: {(df_sample['requires_reasoning'] == True).sum()}")
print(f"False: {(df_sample['requires_reasoning'] == False).sum()}")

# Convert to JSONL
output_path = '/dccstor/gma2/jhjenny9/counterfactual_feedback/eval_sample_12_8_2025.jsonl'

with open(output_path, 'w') as f:
    for idx, row in df_sample.iterrows():
        # Convert row to dict and then to JSON
        json_line = row.to_dict()
        f.write(json.dumps(json_line) + '\n')

print(f"\nSaved to: {output_path}")

# Show first few examples
print("\nFirst 3 examples:")
for idx, row in df_sample.head(3).iterrows():
    print(f"\n{idx}. requires_reasoning: {row['requires_reasoning']}")
    print(f"   conversation_id: {row['conversation_id']}")
    print(f"   round_index: {row['round_index']}")
