"""
Load and filter WildChat dataset for coding-related conversations with >4 turns.
"""

from datasets import load_dataset
import pandas as pd
import json
import os
import numpy as np


def convert_to_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to JSON-serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj


def load_wildchat_coding_conversations(min_turns=3):
    """
    Load WildChat dataset and filter for coding-related conversations with >min_turns.

    Args:
        min_turns: Minimum number of turns required (default: 3)

    Returns:
        Filtered dataset containing only coding conversations with >min_turns turns
    """
    print("Loading WildChat dataset...")
    dataset = load_dataset("allenai/WildChat")

    print(f"Dataset loaded. Available splits: {list(dataset.keys())}")

    # Work with the train split (or modify as needed)
    data = dataset['train']

    print(f"Total conversations: {len(data)}")

    # Filter for coding-related conversations
    print("Filtering for coding-related conversations...")
    coding_keywords = [
        'code', 'programming', 'python', 'javascript', 'java',
        'function', 'debug', 'syntax', '{','(', '::', '=', '```', '<',
        'algorithm', 'api', 'database', 'git', 'repository', 'compile',
        'html', 'css', 'typescript', 'c++', 'pytorch', 'tensorflow',
        'sql', 'bash', 'variable', 'array',
        'array', 'string', 'int', 'float', 'import'
    ]
    # filter for non-story/writing/fiction related conversations
    story_keywords = [
        'story', 'writing', 'fiction', 'book', 'novel', 'poem', 'poetry', 'narrative',
        'character', 'plot', 'chapter', 'author', 'literature', 'fantasy',
        'horror', 'romance', 'drama', 'mystery', 'thriller', 'sci-fi', 
        'imagery', 'dialogue', 'theme', 'prose', 'verse', 'stanza'
    ]

    def is_coding_conversation(example):
        """Check if conversation is related to coding."""
        conversation = example.get('conversation', [])
        if not isinstance(conversation, list):
            return False


        for message in conversation:
            content = message.get('content', '').lower()
            # Exclude if any story-related keywords are found
            if any(keyword in content for keyword in story_keywords):
                return False
            # Check content of all messages for coding keywords
            if any(keyword in content for keyword in coding_keywords):
                return True
        return False

    coding_data = data.filter(is_coding_conversation)
    print(f"Coding conversations: {len(coding_data)}")

    # Filter for English conversations
    print("Filtering for English conversations...")

    def is_english_conversation(example):
        """Check if all messages in conversation are in English."""
        conversation = example.get('conversation', [])
        if not isinstance(conversation, list) or len(conversation) == 0:
            return False
        return all(msg.get('language') == 'English' for msg in conversation)

    coding_english = coding_data.filter(is_english_conversation)
    print(f"English coding conversations: {len(coding_english)}")

    # Filter for conversations with more than min_turns turns
    print(f"Filtering for conversations with >{min_turns} turns...")

    def has_enough_turns(example):
        """Check if the 'turn' field in the dataset is greater than min_turns."""
        return example.get('turn', 0) > min_turns

    filtered_data = coding_english.filter(has_enough_turns)
    print(f"Final filtered conversations: {len(filtered_data)}")

    return filtered_data


def main():
    """Main function to load and explore the filtered dataset."""
    filtered_dataset = load_wildchat_coding_conversations(min_turns=4)

    print("\n=== Dataset Summary ===")
    print(f"Total filtered conversations: {len(filtered_dataset)}")

    if len(filtered_dataset) > 0:
        print("\n=== Sample Conversation ===")
        print(filtered_dataset[0])

        print("\n=== Dataset Features ===")
        print(filtered_dataset.features)

        # Save to CSV
        print("\n=== Saving to CSV ===")
        output_path = os.path.join(os.path.dirname(__file__), "coding_dialogues.csv")

        # Convert to pandas DataFrame
        df = filtered_dataset.to_pandas()

        # If conversation is a list of dicts, convert to JSON string for CSV storage
        if 'conversation' in df.columns:
            df['conversation'] = df['conversation'].apply(lambda x: json.dumps(convert_to_serializable(x)))

        df.to_csv(output_path, index=False)
        print(f"Dataset saved to: {output_path}")
        print(f"Total rows saved: {len(df)}")

    return filtered_dataset


if __name__ == "__main__":
    dataset = main()
