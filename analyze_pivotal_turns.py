#!/usr/bin/env python3
"""
Script to analyze the results of pivotal turn identification.

Provides summary statistics and insights about which turns are most commonly identified as pivotal.
"""

import json
import argparse
from collections import Counter
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import os

def analyze_pivotal_turns(conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze pivotal turn results and generate statistics.

    Args:
        conversations: List of conversations with pivotal_turn_analysis

    Returns:
        Dictionary containing analysis results
    """
    # Initialize counters and data structures
    turn_distribution = Counter()
    reasons_by_turn = {}

    for conv in conversations:
        analysis = conv.get('pivotal_turn_analysis', {})
        turn_num = analysis.get('pivotal_turn_number')

        if turn_num:
            turn_distribution[turn_num] += 1

            # Collect reasons for each turn number
            if turn_num not in reasons_by_turn:
                reasons_by_turn[turn_num] = []
            reasons_by_turn[turn_num].append(analysis.get('reason', ''))

    # Calculate statistics
    total = len(conversations)

    # Find most common turn numbers
    most_common_turns = turn_distribution.most_common(5)

    return {
        'total_conversations': total,
        'turn_distribution': dict(turn_distribution),
        'most_common_turns': most_common_turns,
        'reasons_by_turn': reasons_by_turn
    }

def print_analysis(analysis: Dict[str, Any]):
    """
    Print the analysis results in a readable format.
    Assumes all analyses were successful.

    Args:
        analysis: Analysis results dictionary
    """
    print("\n" + "="*60)
    print("PIVOTAL TURN ANALYSIS SUMMARY")
    print("="*60)

    print(f"\nTotal conversations: {analysis['total_conversations']}")

    print("\n" + "-"*60)
    print("TURN DISTRIBUTION")
    print("-"*60)

    if analysis['most_common_turns']:
        print("\nMost commonly identified pivotal turns:")
        total = analysis['total_conversations']
        for turn_num, count in analysis['most_common_turns']:
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  Turn {turn_num}: {count} occurrences ({percentage:.1f}%)")

        # Print histogram
        print("\nTurn distribution (histogram):")
        max_count = max(count for _, count in analysis['most_common_turns'])
        for turn_num, count in sorted(analysis['most_common_turns'], key=lambda x: x[0]):
            bar_length = int(40 * count / max_count)
            bar = "█" * bar_length
            print(f"  Turn {turn_num:2d}: {bar} {count}")

    print("\n" + "-"*60)
    print("EXAMPLE REASONS BY TURN")
    print("-"*60)

    # Show a few example reasons for the most common turns
    for turn_num, count in analysis['most_common_turns'][:3]:
        print(f"\nTurn {turn_num} examples:")
        reasons = analysis['reasons_by_turn'].get(turn_num, [])
        # Show up to 3 unique reasons
        unique_reasons = list(set(reasons))[:3]
        for i, reason in enumerate(unique_reasons, 1):
            print(f"  {i}. {reason}")

    print("\n" + "="*60)

def plot_turn_distribution(analysis: Dict[str, Any], conversations: List[Dict[str, Any]], output_path: str = 'results/turn_distribution.png'):
    """
    Create a comprehensive plot showing both pivotal turn distribution and total conversation turn distribution.

    Args:
        analysis: Analysis results dictionary
        conversations: List of conversations for computing total turn distribution
        output_path: Path to save the plot image
    """
    turn_distribution = analysis['turn_distribution']

    if not turn_distribution:
        print("No turn distribution data to plot.")
        return

    # Calculate total assistant turns distribution
    assistant_turns_distribution = Counter()
    for conv in conversations:
        conv_context = conv.get('conversation_context', [])
        assistant_turns = sum(1 for turn in conv_context if turn.get('role') == 'assistant')
        if assistant_turns > 0:
            assistant_turns_distribution[assistant_turns] += 1

    # Prepare pivotal turn data
    pivotal_turns = sorted(turn_distribution.keys())
    pivotal_counts = [turn_distribution[turn] for turn in pivotal_turns]
    total = analysis['total_conversations']
    pivotal_percentages = [(count / total * 100) if total > 0 else 0 for count in pivotal_counts]

    # Prepare total turns data
    total_turns = sorted(assistant_turns_distribution.keys())
    total_turn_counts = [assistant_turns_distribution[turn] for turn in total_turns]
    total_turn_percentages = [(count / total * 100) if total > 0 else 0 for count in total_turn_counts]

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Pivotal Turn Distribution
    bars1 = ax1.bar(pivotal_turns, pivotal_counts, color='steelblue', alpha=0.8, edgecolor='black')

    # Add value labels on top of bars
    for turn, count, pct in zip(pivotal_turns, pivotal_counts, pivotal_percentages):
        ax1.text(turn, count + max(pivotal_counts) * 0.01, f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_xlabel('Pivotal Turn Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Conversations', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Pivotal Turns', fontsize=14, fontweight='bold')
    ax1.set_xticks(pivotal_turns)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Plot 2: Total Assistant Turns Distribution
    bars2 = ax2.bar(total_turns, total_turn_counts, color='coral', alpha=0.8, edgecolor='black')

    # Add value labels on top of bars
    for turn, count, pct in zip(total_turns, total_turn_counts, total_turn_percentages):
        ax2.text(turn, count + max(total_turn_counts) * 0.01, f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_xlabel('Total Assistant Turns in Conversation', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Conversations', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of Total Conversation Length', fontsize=14, fontweight='bold')
    ax2.set_xticks(total_turns)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add summary text to the figure
    fig.suptitle(f'Pivotal Turn Analysis - Total: {total} Conversations',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Close the plot to free memory
    plt.close()

def find_interesting_cases(conversations: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Find interesting edge cases in the results.
    Args:
        conversations: List of conversations with pivotal_turn_analysis

    Returns:
        Dictionary with categorized interesting cases
    """
    interesting = {
        'early_turns': [],  # Pivotal turn is 1
        'late_turns': [],   # Pivotal turn is >= 3
        'multi_turn_conversations': []  # Conversations with many turns
    }

    for conv in conversations:
        analysis = conv.get('pivotal_turn_analysis', {})
        turn_num = analysis.get('pivotal_turn_number')
        conv_context = conv.get('conversation_context', [])
        assistant_turns = sum(1 for turn in conv_context if turn.get('role') == 'assistant')

        if turn_num: # this handles none types.
            if turn_num == 1:
                interesting['early_turns'].append({
                    'conversation_id': conv.get('conversation_id'),
                    'turn': turn_num,
                    'reason': analysis.get('reason'),
                    'total_assistant_turns': assistant_turns
                })
            elif turn_num >= 3:
                interesting['late_turns'].append({
                    'conversation_id': conv.get('conversation_id'),
                    'turn': turn_num,
                    'reason': analysis.get('reason'),
                    'total_assistant_turns': assistant_turns
                })

            if assistant_turns >= 6:
                interesting['multi_turn_conversations'].append({
                    'conversation_id': conv.get('conversation_id'),
                    'turn': turn_num,
                    'reason': analysis.get('reason'),
                    'total_assistant_turns': assistant_turns
                })

    return interesting

def print_interesting_cases(interesting: Dict[str, List[Dict[str, Any]]]):
    """
    Print interesting edge cases.

    Args:
        interesting: Dictionary of interesting cases
    """
    print("\n" + "="*60)
    print("INTERESTING CASES")
    print("="*60)

    # Early failures
    if interesting['early_turns']:
        print(f"\nEarly failures (Turn 1): {len(interesting['early_turns'])} cases")
        print("These conversations failed from the very first assistant response.")
        for case in interesting['early_turns'][:3]:
            print(f"  - {case['conversation_id'][:16]}...")
            print(f"    Reason: {case['reason']}")

    # Late failures
    if interesting['late_turns']:
        print(f"\nLate failures (Turn 3+): {len(interesting['late_turns'])} cases")
        print("These conversations deteriorated after initially going well.")
        for case in interesting['late_turns'][:3]:
            print(f"  - {case['conversation_id'][:16]}... (Turn {case['turn']}/{case['total_assistant_turns']})")
            print(f"    Reason: {case['reason']}")

    # Multi-turn conversations
    if interesting['multi_turn_conversations']:
        print(f"\nMulti-turn conversations (4+ assistant turns): {len(interesting['multi_turn_conversations'])} cases")
        for case in interesting['multi_turn_conversations'][:3]:
            print(f"  - {case['conversation_id'][:16]}... (Pivotal: Turn {case['turn']}/{case['total_assistant_turns']})")
            print(f"    Reason: {case['reason']}")

    print("\n" + "="*60)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Analyze pivotal turn identification results')
    parser.add_argument('--input', '-i',
                       default='results/UR2_pivotal_turns.json',
                       help='Input JSON file with pivotal turn analysis results')
    parser.add_argument('--output', '-o',
                       default='results/analysis_UR2_pivotal_turns.json',
                       help='Optional: Save analysis summary to JSON file')
    parser.add_argument('--plot', '-p',
                       default='results/turn_distribution.png',
                       help='Path to save the turn distribution plot (default: results/turn_distribution.png)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip generating the plot')
    args = parser.parse_args()

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

    # Analyze results
    analysis = analyze_pivotal_turns(conversations)
    print_analysis(analysis)

    # Generate plot
    if not args.no_plot:
        try:
            plot_turn_distribution(analysis, conversations, args.plot)
        except Exception as e:
            print(f"Warning: Could not generate plot: {e}")
            print("Continuing with analysis...")

    # Find interesting cases
    interesting = find_interesting_cases(conversations)
    print_interesting_cases(interesting)

    # Save analysis if output specified
    if args.output:
        output_data = {
            'analysis': analysis,
            'interesting_cases': interesting
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nAnalysis saved to: {args.output}")

    return 0

if __name__ == "__main__":
    exit(main())
