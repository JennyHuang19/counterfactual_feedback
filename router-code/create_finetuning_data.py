"""
Convert expanded routing decisions dataset to OpenAI fine-tuning format (JSONL).

This script transforms the routing decisions dataset into JSONL format suitable
for fine-tuning an OpenAI model to predict whether conversations require reasoning.

Input: pivotal-turns-data/expanded_routing_decisions.csv
Output: train.jsonl (80%), val.jsonl (20%)
"""

import pandas as pd
import json
import os
import random


# System prompt for the routing task
SYSTEM_PROMPT = """You are a routing system that analyzes conversations to determine whether they require advanced reasoning capabilities. Analyze the conversation presented up to the current turn and respond with 'true' if the next turn in the conversation requires reasoning (e.g., complex debugging, deep analysis), or 'false' if it can be handled by a standard non-reasoning model."""


def create_training_example(conversation_history: str, requires_reasoning: bool) -> dict:
    """
    Convert a single row into OpenAI fine-tuning format.

    Args:
        conversation_history: Plain text conversation history
        requires_reasoning: Boolean indicating if reasoning is required

    Returns:
        Dictionary in OpenAI fine-tuning format with messages array
    """
    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": conversation_history
            },
            {
                "role": "assistant",
                "content": "true" if requires_reasoning else "false"
            }
        ]
    }


def stratified_train_test_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Perform stratified train/test split on dataframe.

    Args:
        df: Dataframe to split
        test_size: Fraction of data to use for test set
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, test_df)
    """
    random.seed(random_state)

    # Separate by class
    true_df = df[df['requires_reasoning'] == True].copy()
    false_df = df[df['requires_reasoning'] == False].copy()

    # Shuffle
    true_indices = list(true_df.index)
    false_indices = list(false_df.index)
    random.shuffle(true_indices)
    random.shuffle(false_indices)

    # Split each class
    true_split = int(len(true_indices) * (1 - test_size))
    false_split = int(len(false_indices) * (1 - test_size))

    train_indices = true_indices[:true_split] + false_indices[:false_split]
    test_indices = true_indices[true_split:] + false_indices[false_split:]

    train_df = df.loc[train_indices].copy()
    test_df = df.loc[test_indices].copy()

    return train_df, test_df


def write_jsonl(df: pd.DataFrame, output_path: str):
    """
    Write dataframe to JSONL format.

    Args:
        df: Dataframe with conversation_history and requires_reasoning columns
        output_path: Path to output JSONL file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            example = create_training_example(
                row['conversation_history'],
                row['requires_reasoning']
            )
            # Write as single-line JSON
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print(f"Wrote {len(df)} examples to {output_path}")


def validate_jsonl(file_path: str) -> bool:
    """
    Validate JSONL format and print statistics.

    Args:
        file_path: Path to JSONL file to validate

    Returns:
        True if validation passed, False otherwise
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"\nValidating {file_path}:")
    print(f"  Total examples: {len(lines)}")

    true_count = 0
    false_count = 0
    errors = []

    for i, line in enumerate(lines, 1):
        try:
            obj = json.loads(line)
            # Verify structure
            assert "messages" in obj, "Missing 'messages' field"
            assert len(obj["messages"]) == 3, f"Expected 3 messages, got {len(obj['messages'])}"
            assert obj["messages"][0]["role"] == "system", "First message must be system"
            assert obj["messages"][1]["role"] == "user", "Second message must be user"
            assert obj["messages"][2]["role"] == "assistant", "Third message must be assistant"
            assert obj["messages"][2]["content"] in ["true", "false"], \
                f"Assistant content must be 'true' or 'false', got '{obj['messages'][2]['content']}'"

            if obj["messages"][2]["content"] == "true":
                true_count += 1
            else:
                false_count += 1

        except Exception as e:
            errors.append(f"Line {i}: {str(e)}")

    if errors:
        print(f"  ERRORS FOUND:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"    {error}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more errors")
        return False

    print(f"  'true' examples: {true_count}")
    print(f"  'false' examples: {false_count}")
    print(f"  Class balance: {true_count / len(lines) * 100:.1f}% true, {false_count / len(lines) * 100:.1f}% false")
    print(f"  Validation: PASSED ✓")
    return True


def main():
    """Main function to convert dataset to JSONL format."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, 'pivotal-turns-data', 'expanded_routing_decisions.csv')
    train_file = os.path.join(base_dir, 'pivotal-turns-data', 'train.jsonl')
    val_file = os.path.join(base_dir, 'pivotal-turns-data', 'val.jsonl')

    print("=" * 80)
    print("CONVERTING ROUTING DECISIONS TO OPENAI FINE-TUNING FORMAT")
    print("=" * 80)

    # Step 1: Load and prepare data
    print("\nStep 1: Loading data...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} examples from {input_file}")

    # Extract relevant columns
    data = df[['conversation_history', 'requires_reasoning']].copy()

    # Print dataset statistics
    print(f"\nDataset statistics:")
    print(f"  Total examples: {len(data)}")
    print(f"  'true' examples: {(data['requires_reasoning'] == True).sum()}")
    print(f"  'false' examples: {(data['requires_reasoning'] == False).sum()}")

    # Step 2: Create stratified train/val split
    print("\nStep 2: Creating stratified train/validation split (80/20)...")
    train_df, val_df = stratified_train_test_split(
        data,
        test_size=0.2,  # 20% validation
        random_state=42
    )

    print(f"\nSplit results:")
    print(f"  Training samples: {len(train_df)}")
    print(f"    'true': {(train_df['requires_reasoning'] == True).sum()}")
    print(f"    'false': {(train_df['requires_reasoning'] == False).sum()}")
    print(f"  Validation samples: {len(val_df)}")
    print(f"    'true': {(val_df['requires_reasoning'] == True).sum()}")
    print(f"    'false': {(val_df['requires_reasoning'] == False).sum()}")

    # Step 3 & 4: Generate JSONL files
    print("\nStep 3: Generating JSONL files...")
    write_jsonl(train_df, train_file)
    write_jsonl(val_df, val_file)

    # Step 5: Validate generated files
    print("\nStep 4: Validating generated JSONL files...")
    train_valid = validate_jsonl(train_file)
    val_valid = validate_jsonl(val_file)

    print("\n" + "=" * 80)
    if train_valid and val_valid:
        print("SUCCESS! JSONL files generated and validated.")
        print("\nNext steps:")
        print("1. Upload files to OpenAI:")
        print(f"   openai files create -f {train_file} -p fine-tune")
        print(f"   openai files create -f {val_file} -p fine-tune")
        print("\n2. Create fine-tuning job:")
        print("   openai fine-tuning jobs create \\")
        print("     --training-file file-abc123 \\")
        print("     --validation-file file-xyz456 \\")
        print("     --model gpt-4o-mini-2024-07-18")
    else:
        print("ERROR: Validation failed. Please check the output above.")
    print("=" * 80)


if __name__ == "__main__":
    main()
