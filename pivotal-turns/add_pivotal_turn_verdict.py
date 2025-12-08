import pandas as pd
import json

# Load the datasets
nf2500 = pd.read_csv('pivotal-turns-data/nf2500_11_30_2025.csv')
pivotal_turns = pd.read_csv('pivotal-turns-data/pivotal_turns_1000.csv')

# Join on conversation_id
merged = nf2500.merge(pivotal_turns[['conversation_id', 'routing_decisions']],
                      on='conversation_id',
                      how='left')

# Create expanded dataset with one row per turn
expanded_rows = []

for idx, row in merged.iterrows():
    conversation_id = row['conversation_id']
    conversation = json.loads(row['conversation'])

    # Parse routing decisions if available
    routing_decisions_map = {}
    if pd.notna(row['routing_decisions']):
        routing_decisions = json.loads(row['routing_decisions'])
        for decision in routing_decisions:
            turn_idx = decision['turn_index']
            requires_reasoning = decision['requires_reasoning']
            routing_decisions_map[turn_idx] = requires_reasoning

    # Create a row for each turn
    for turn_idx in range(len(conversation)):
        # Get conversation up to and including this turn
        conversation_so_far = conversation[:turn_idx + 1]

        # Determine routing decision for this turn
        routing_decision = "yes" if routing_decisions_map.get(turn_idx, False) else "no"

        # Create new row
        new_row = {
            'conversation_id': conversation_id,
            'turn_index': turn_idx,
            'conversation': json.dumps(conversation_so_far),
            'routing_decision': routing_decision,
            'model': row['model'],
            'timestamp': row['timestamp'],
            'turn': row['turn'],
            'language': row['language'],
            'openai_moderation': row['openai_moderation'],
            'detoxify_moderation': row['detoxify_moderation'],
            'toxic': row['toxic'],
            'redacted': row['redacted'],
            'label': row['label']
        }

        expanded_rows.append(new_row)

# Create the final dataframe
expanded_df = pd.DataFrame(expanded_rows)

# Save to CSV
output_path = 'pivotal-turns-data/combined_dataset.csv'
expanded_df.to_csv(output_path, index=False)

print(f"Combined dataset saved to {output_path}")
print(f"Total rows: {len(expanded_df)}")
print(f"\nSample of routing decisions:")
print(expanded_df['routing_decision'].value_counts())
print(f"\nFirst few rows:")
print(expanded_df.head())
