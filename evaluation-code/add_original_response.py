#!/usr/bin/env python3
"""
Script to add original_response column to evaluation_predictions_with_gpt4o.jsonl
by extracting assistant responses from nf_pt_12_05_2025.csv at specified round_index.
"""

import json
import csv
from typing import Dict, Any, Optional

# File paths
JSONL_FILE = "route-to-reasoning-data/evaluation_predictions_with_gpt4o.jsonl"
CSV_FILE = "pivotal-turns-data/nf_pt_12_05_2025.csv"

def load_csv_conversations(csv_path: str) -> Dict[str, Any]:
    """Load CSV and build lookup dictionary by conversation_id."""
    print(f"Loading CSV file: {csv_path}")
    conversations = {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            conv_id = row['conversation_id']
            # Only store the first occurrence of each conversation_id
            if conv_id not in conversations:
                conversations[conv_id] = row

    print(f"Loaded {len(conversations)} unique conversations from CSV")
    return conversations

def extract_assistant_response(conversation_json: str, round_index: int) -> Optional[str]:
    """
    Extract the assistant response at the given round_index from conversation JSON.

    Args:
        conversation_json: JSON string containing array of messages
        round_index: 0-indexed turn number

    Returns:
        Assistant's content at that turn, or None if error
    """
    try:
        messages = json.loads(conversation_json)

        # Calculate message index: assistant at round N is at index 2N+1
        message_index = 2 * round_index + 1

        # Check if index is valid
        if message_index >= len(messages):
            print(f"  ERROR: Index {message_index} out of bounds (conversation has {len(messages)} messages)")
            return None

        # Get message and verify it's from assistant
        message = messages[message_index]
        if message['role'] != 'assistant':
            print(f"  ERROR: Expected assistant at index {message_index}, got {message['role']}")
            return None

        return message['content']

    except json.JSONDecodeError as e:
        print(f"  ERROR: Failed to parse conversation JSON: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"  ERROR: Failed to extract message: {e}")
        return None

def process_jsonl(jsonl_path: str, conversations: Dict[str, Any]) -> list:
    """Process JSONL file and add original_response to each entry."""
    print(f"\nProcessing JSONL file: {jsonl_path}")

    entries = []
    stats = {
        'total': 0,
        'success': 0,
        'missing_id': 0,
        'extraction_error': 0
    }

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            stats['total'] += 1
            entry = json.loads(line.strip())

            conv_id = entry['conversation_id']
            round_index = entry['round_index']

            # Look up conversation in CSV
            if conv_id not in conversations:
                print(f"Line {line_num}: Missing conversation_id {conv_id}")
                entry['original_response'] = None
                stats['missing_id'] += 1
            else:
                # Extract original response
                csv_row = conversations[conv_id]
                original_response = extract_assistant_response(
                    csv_row['conversation'],
                    round_index
                )

                if original_response is None:
                    print(f"Line {line_num}: Failed to extract response for {conv_id}, round {round_index}")
                    stats['extraction_error'] += 1
                else:
                    stats['success'] += 1

                entry['original_response'] = original_response

            entries.append(entry)

    print(f"\nProcessing complete:")
    print(f"  Total entries: {stats['total']}")
    print(f"  Successful: {stats['success']}")
    print(f"  Missing conversation_id: {stats['missing_id']}")
    print(f"  Extraction errors: {stats['extraction_error']}")

    return entries

def write_jsonl(jsonl_path: str, entries: list):
    """Write entries back to JSONL file."""
    print(f"\nWriting {len(entries)} entries to {jsonl_path}")

    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print("Write complete!")

def main():
    print("=" * 70)
    print("Adding original_response column to JSONL dataset")
    print("=" * 70)

    # Load CSV conversations
    conversations = load_csv_conversations(CSV_FILE)

    # Process JSONL entries
    entries = process_jsonl(JSONL_FILE, conversations)

    # Write back to file
    write_jsonl(JSONL_FILE, entries)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

if __name__ == "__main__":
    main()
