"""
Expand routing decisions from pivotal turns dataset into individual rows.

This script transforms the dataset so that each routing decision becomes its own row,
with the conversation history up to that point included as plain text.

Input: pivotal-turns-data/nf_pt_12_05_2025.csv
Output: pivotal-turns-data/expanded_routing_decisions.csv
"""

import pandas as pd
import json
import os
from typing import List, Dict, Any


def format_conversation_history(messages: List[Dict[str, Any]], round_index: int) -> str:
    """
    Convert messages to plain text format up to the user message for given round.

    Args:
        messages: List of message dicts with 'role' and 'content' fields
        round_index: The round index (0-indexed)

    Returns:
        Plain text string with conversation up to and including user message at round_index
    """
    # For round_index n, include messages [0, ..., 2n]
    # This captures all messages up to and including the user message that starts round n
    end_index = 2 * round_index + 1  # +1 because we want to include the user message

    relevant_messages = messages[:end_index]

    # Format as plain text
    formatted = []
    for msg in relevant_messages:
        role = msg.get('role', 'unknown').capitalize()
        content = msg.get('content', '')
        formatted.append(f"{role}: {content}")

    return "\n\n".join(formatted)


def expand_routing_decisions(input_file: str, output_file: str, test_mode: bool = False) -> pd.DataFrame:
    """
    Expand routing decisions from the input CSV into individual rows.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        test_mode: If True, only process first 5 rows for testing

    Returns:
        Expanded DataFrame
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)

    if test_mode:
        print("Running in TEST MODE - processing only first 5 rows")
        df = df.head(5)

    print(f"Loaded {len(df)} conversations")

    # Collect all expanded rows
    expanded_rows = []
    errors = []

    for idx, row in df.iterrows():
        conversation_id = row['conversation_id']

        try:
            # Parse JSON strings
            routing_decisions = json.loads(row['routing_decisions'])
            conversation = json.loads(row['conversation'])

            # Expand each routing decision
            for decision in routing_decisions:
                round_index = decision['round_index']
                requires_reasoning = decision['requires_reasoning']
                justification = decision['justification']

                # Validate round_index
                expected_message_count = 2 * round_index + 1
                if len(conversation) < expected_message_count:
                    print(f"Warning: conversation {conversation_id} has only {len(conversation)} messages, "
                          f"but round_index {round_index} requires at least {expected_message_count} messages. Skipping.")
                    errors.append({
                        'conversation_id': conversation_id,
                        'error': f'Invalid round_index {round_index} for conversation length {len(conversation)}'
                    })
                    continue

                # Build conversation history
                conversation_history = format_conversation_history(conversation, round_index)

                # Add to results
                expanded_rows.append({
                    'conversation_id': conversation_id,
                    'round_index': round_index,
                    'requires_reasoning': requires_reasoning,
                    'justification': justification,
                    'conversation_history': conversation_history
                })

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON for conversation {conversation_id}: {e}")
            errors.append({
                'conversation_id': conversation_id,
                'error': f'JSON decode error: {str(e)}'
            })
        except Exception as e:
            print(f"Error processing conversation {conversation_id}: {e}")
            errors.append({
                'conversation_id': conversation_id,
                'error': str(e)
            })

    # Create new dataframe
    expanded_df = pd.DataFrame(expanded_rows)

    # Print statistics
    print(f"\n=== Processing Complete ===")
    print(f"Original conversations: {len(df)}")
    print(f"Expanded rows: {len(expanded_df)}")
    print(f"Errors: {len(errors)}")

    if len(expanded_df) > 0:
        print(f"\nRouting decisions per conversation:")
        decisions_per_conv = expanded_df.groupby('conversation_id').size()
        print(f"  Mean: {decisions_per_conv.mean():.2f}")
        print(f"  Median: {decisions_per_conv.median():.0f}")
        print(f"  Max: {decisions_per_conv.max():.0f}")

        print(f"\nRequires reasoning breakdown:")
        print(f"  True: {(expanded_df['requires_reasoning'] == True).sum()}")
        print(f"  False: {(expanded_df['requires_reasoning'] == False).sum()}")

    # Save to CSV
    if not test_mode:
        expanded_df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
    else:
        print(f"\nTest mode - results NOT saved")

    # Save errors if any
    if errors:
        error_file = output_file.replace('.csv', '_errors.csv')
        pd.DataFrame(errors).to_csv(error_file, index=False)
        print(f"Errors saved to {error_file}")

    return expanded_df


def main():
    """Main execution function."""
    # Set up paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, 'pivotal-turns-data', 'nf_pt_12_05_2025.csv')
    output_file = os.path.join(base_dir, 'pivotal-turns-data', 'expanded_routing_decisions.csv')

    # First run in test mode
    print("=" * 60)
    print("RUNNING IN TEST MODE")
    print("=" * 60)
    test_df = expand_routing_decisions(input_file, output_file, test_mode=True)

    if len(test_df) > 0:
        print("\n=== Sample output ===")
        print(f"First row conversation_history preview (first 200 chars):")
        print(test_df['conversation_history'].iloc[0][:200] + "...")
        print(f"\nColumns: {test_df.columns.tolist()}")
        print(f"\nFirst row:")
        for col in test_df.columns:
            if col != 'conversation_history':  # Skip long text
                print(f"  {col}: {test_df[col].iloc[0]}")

    # Ask user to confirm before running on full dataset
    print("\n" + "=" * 60)
    response = input("Test looks good? Run on full dataset? (y/n): ")

    if response.lower() == 'y':
        print("\n" + "=" * 60)
        print("RUNNING ON FULL DATASET")
        print("=" * 60)
        full_df = expand_routing_decisions(input_file, output_file, test_mode=False)
        print("\nDone!")
    else:
        print("\nSkipping full dataset processing.")


if __name__ == "__main__":
    main()
