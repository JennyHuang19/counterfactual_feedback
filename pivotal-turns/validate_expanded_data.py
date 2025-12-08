"""
Generate validation samples from expanded routing decisions dataset.

This script creates a text report with random samples from the expanded dataset,
cross-referenced with the original data, to enable manual validation of data quality.

Output: pivotal-turns-data/validation_samples.txt
"""

import pandas as pd
import json
import random
import os


def get_assistant_response_for_round(original_conversation: list, round_index: int) -> str:
    """
    Get the assistant response that follows the user message at round_index.

    Args:
        original_conversation: List of message dicts from original data
        round_index: The round index (0-indexed)

    Returns:
        Assistant response content, or "[No assistant response]" if not available
    """
    # The assistant response for round_index is at index 2*round_index + 1
    assistant_msg_index = 2 * round_index + 1

    if assistant_msg_index < len(original_conversation):
        return original_conversation[assistant_msg_index].get('content', '[No content]')
    return "[No assistant response available]"


def count_messages_in_history(conversation_history: str) -> tuple:
    """Count user and assistant messages in the formatted history."""
    user_count = conversation_history.count('User:')
    assistant_count = conversation_history.count('Assistant:')
    return user_count, assistant_count


def validate_sample(sample: pd.Series, original_df: pd.DataFrame) -> dict:
    """
    Cross-validate a sample with original data.

    Returns validation info including checks and assistant response.
    """
    # Find original conversation
    orig_row = original_df[original_df['conversation_id'] == sample['conversation_id']].iloc[0]
    orig_conversation = json.loads(orig_row['conversation'])

    # Get assistant response
    assistant_response = get_assistant_response_for_round(orig_conversation, sample['round_index'])

    # Count messages
    user_count, assistant_count = count_messages_in_history(sample['conversation_history'])
    total_messages = user_count + assistant_count
    expected_messages = 2 * sample['round_index'] + 1

    # Check if ends with user message
    history_lines = sample['conversation_history'].strip().split('\n\n')
    if history_lines:
        last_line = history_lines[-1].strip()
        ends_with_user = last_line.startswith('User:')
    else:
        ends_with_user = False

    return {
        'assistant_response': assistant_response,
        'message_count': total_messages,
        'expected_count': expected_messages,
        'count_matches': total_messages == expected_messages,
        'ends_with_user': ends_with_user
    }


def generate_validation_report(n_samples: int = 25):
    """Main function to generate validation report."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load data
    print("Loading datasets...")
    expanded_df = pd.read_csv(os.path.join(base_dir, 'pivotal-turns-data', 'expanded_routing_decisions.csv'))
    original_df = pd.read_csv(os.path.join(base_dir, 'pivotal-turns-data', 'nf_pt_12_05_2025.csv'))

    print(f"Expanded dataset: {len(expanded_df)} rows")
    print(f"Original dataset: {len(original_df)} conversations")

    # Random sample
    random.seed(42)  # For reproducibility
    samples = expanded_df.sample(n=min(n_samples, len(expanded_df)))

    print(f"\nGenerating {len(samples)} validation samples...")

    # Generate report
    output_path = os.path.join(base_dir, 'pivotal-turns-data', 'validation_samples.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("EXPANDED ROUTING DECISIONS - VALIDATION SAMPLES\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated {len(samples)} random samples for manual review\n")
        f.write(f"Random seed: 42 (for reproducibility)\n")
        f.write("=" * 80 + "\n\n")

        for idx, (_, sample) in enumerate(samples.iterrows(), 1):
            try:
                validation = validate_sample(sample, original_df)

                f.write("=" * 80 + "\n")
                f.write(f"SAMPLE {idx} of {len(samples)}\n")
                f.write("=" * 80 + "\n")
                f.write(f"Conversation ID: {sample['conversation_id']}\n")
                f.write(f"Round Index: {sample['round_index']}\n")
                f.write(f"Requires Reasoning: {sample['requires_reasoning']}\n\n")

                f.write("JUSTIFICATION:\n")
                f.write(f"{sample['justification']}\n\n")

                f.write("CONVERSATION HISTORY:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{sample['conversation_history']}\n")
                f.write("-" * 80 + "\n\n")

                f.write("[VERIFICATION] ASSISTANT RESPONSE (from original data):\n")
                f.write("-" * 80 + "\n")
                # Truncate if too long
                response_text = validation['assistant_response']
                if len(response_text) > 1000:
                    f.write(f"{response_text[:1000]}...\n[Response truncated - showing first 1000 characters]\n")
                else:
                    f.write(f"{response_text}\n")
                f.write("-" * 80 + "\n\n")

                f.write("[VALIDATION CHECKS]\n")
                check_mark = "✓" if validation['count_matches'] else "✗"
                f.write(f"{check_mark} Message count: {validation['message_count']} (expected: {validation['expected_count']})\n")

                check_mark = "✓" if validation['ends_with_user'] else "✗"
                f.write(f"{check_mark} Ends with: User message\n")

                f.write("\n\n")

            except Exception as e:
                f.write(f"\n[ERROR processing sample {idx}: {str(e)}]\n\n")
                print(f"Error processing sample {idx}: {e}")

    print(f"\nValidation report saved to: {output_path}")
    print("\nYou can now manually review the samples to verify:")
    print("1. Justifications align with conversation content")
    print("2. Conversation history is correctly sliced")
    print("3. No data corruption or misalignment")


if __name__ == "__main__":
    generate_validation_report(n_samples=25)
