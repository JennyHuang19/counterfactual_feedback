#!/usr/bin/env python3
"""
Extract pivotal turns from analysis results into a simplified format.

This script creates a CSV or simplified JSON format that's easier to work with
for further analysis, visualization, or model training.
"""

import json
import csv
import argparse
from typing import List, Dict, Any

def extract_pivotal_turns_to_list(conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract pivotal turn information into a simplified list.

    Args:
        conversations: List of conversations with pivotal_turn_analysis

    Returns:
        List of simplified pivotal turn records
    """
    extracted = []

    for conv in conversations:
        analysis = conv.get('pivotal_turn_analysis', {})

        if analysis.get('status') == 'success':
            # Count assistant turns
            conv_context = conv.get('conversation_context', [])
            total_assistant_turns = sum(1 for turn in conv_context if turn.get('role') == 'assistant')
            total_turns = len(conv_context)

            # Get the conversation up to the pivotal turn
            pivotal_turn_idx = None
            assistant_turn_count = 0
            target_turn = analysis.get('pivotal_turn_number')

            for i, turn in enumerate(conv_context):
                if turn.get('role') == 'assistant':
                    assistant_turn_count += 1
                    if assistant_turn_count == target_turn:
                        pivotal_turn_idx = i
                        break

            # Build context before pivotal turn
            context_before = []
            if pivotal_turn_idx is not None:
                context_before = conv_context[:pivotal_turn_idx]

            record = {
                'conversation_id': conv.get('conversation_id', ''),
                'model': conv.get('model', ''),
                'category': conv.get('category', ''),
                'pivotal_turn_number': analysis.get('pivotal_turn_number'),
                'pivotal_turn_content': analysis.get('pivotal_turn_content', ''),
                'reason': analysis.get('reason', ''),
                'total_assistant_turns': total_assistant_turns,
                'total_turns': total_turns,
                'context_before_pivotal': context_before,
                'full_conversation_context': conv_context
            }

            extracted.append(record)

    return extracted

def save_to_csv(records: List[Dict[str, Any]], output_path: str):
    """
    Save extracted records to CSV format.

    Args:
        records: List of pivotal turn records
        output_path: Path to save CSV file
    """
    if not records:
        print("No records to save")
        return

    # Define CSV columns (excluding nested structures)
    fieldnames = [
        'conversation_id',
        'model',
        'category',
        'pivotal_turn_number',
        'total_assistant_turns',
        'total_turns',
        'pivotal_turn_content',
        'reason'
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            # Create a flattened version for CSV
            csv_record = {k: record.get(k, '') for k in fieldnames}
            writer.writerow(csv_record)

    print(f"Saved {len(records)} records to CSV: {output_path}")

def save_to_json(records: List[Dict[str, Any]], output_path: str):
    """
    Save extracted records to JSON format.

    Args:
        records: List of pivotal turn records
        output_path: Path to save JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(records)} records to JSON: {output_path}")

def print_summary(records: List[Dict[str, Any]]):
    """
    Print a summary of the extracted records.

    Args:
        records: List of pivotal turn records
    """
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)

    print(f"\nTotal records extracted: {len(records)}")

    # Model distribution
    models = {}
    for record in records:
        model = record.get('model', 'unknown')
        models[model] = models.get(model, 0) + 1

    print("\nModel distribution:")
    for model, count in sorted(models.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model}: {count}")

    # Turn distribution
    turns = {}
    for record in records:
        turn = record.get('pivotal_turn_number')
        turns[turn] = turns.get(turn, 0) + 1

    print("\nPivotal turn distribution:")
    for turn, count in sorted(turns.items()):
        percentage = (count / len(records) * 100) if records else 0
        print(f"  Turn {turn}: {count} ({percentage:.1f}%)")

    # Average conversation length
    avg_turns = sum(r.get('total_turns', 0) for r in records) / len(records) if records else 0
    avg_assistant_turns = sum(r.get('total_assistant_turns', 0) for r in records) / len(records) if records else 0

    print(f"\nAverage conversation length:")
    print(f"  Total turns: {avg_turns:.1f}")
    print(f"  Assistant turns: {avg_assistant_turns:.1f}")

    print("\n" + "="*60)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Extract pivotal turns into simplified format')
    parser.add_argument('--input', '-i',
                       required=True,
                       help='Input JSON file with pivotal turn analysis results')
    parser.add_argument('--output-json', '-oj',
                       help='Output simplified JSON file')
    parser.add_argument('--output-csv', '-oc',
                       help='Output CSV file')
    parser.add_argument('--sample', '-s', type=int,
                       help='Extract only first N records (for testing)')

    args = parser.parse_args()

    # Validate that at least one output format is specified
    if not args.output_json and not args.output_csv:
        print("Error: Please specify at least one output format (--output-json or --output-csv)")
        return 1

    # Load results
    print(f"Loading results from {args.input}...")
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        print(f"Loaded {len(conversations)} conversations")
    except FileNotFoundError:
        print(f"Error: Could not find input file {args.input}")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file - {e}")
        return 1

    # Extract pivotal turns
    print("Extracting pivotal turn information...")
    records = extract_pivotal_turns_to_list(conversations)

    # Apply sampling if requested
    if args.sample:
        records = records[:args.sample]
        print(f"Sampled first {len(records)} records")

    # Print summary
    print_summary(records)

    # Save to requested formats
    if args.output_json:
        save_to_json(records, args.output_json)

    if args.output_csv:
        save_to_csv(records, args.output_csv)

    print("\nExtraction complete!")

    return 0

if __name__ == "__main__":
    exit(main())
