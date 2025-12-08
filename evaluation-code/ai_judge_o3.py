#!/usr/bin/env python3
"""
AI Judge using o3 to compare original_response vs reasoning_response.
Evaluates which response leads to better conversation quality.
"""

import json
import os
import random
import time
from typing import Dict, Any, Optional
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

JUDGE_SYSTEM_PROMPT = """You are an expert AI judge evaluating conversation quality. Your task is to compare two different assistant responses to the same user query and determine which response would lead to better conversation quality and outcomes.

**PRIMARY FOCUS: Correctness of Reasoning**

When evaluating, prioritize these criteria in order:

1. **Factual Accuracy**:
   - Is the response/code provided factually correct?
   - Are there any errors or inaccuracies in the response/code?

2. **Reasoning Quality**:
   - For technical/coding issues: Does it correctly identify the root cause and propose a valid solution?
   - Does it avoid making incorrect claims, faulty assumptions, or logical fallacies?

3. **Problem-Solving Effectiveness**:
   - Does it effectively address and resolve the user's issue?
   - Would following this response lead to a successful outcome?

4. **Long-term Conversation Quality**:
   - Would this response lead to a productive continuation of the conversation?
   - Does it set the user on the right path?

5. **Helpfulness and Clarity**:
   - Is it clear, well-structured, and helpful to the user?

You will be shown:
- The conversation history leading up to this turn
- Two possible responses (labeled Response A and Response B)

Output your judgment in the following JSON format:
{
    "winner": "A" or "B",
    "confidence": "high", "medium", or "low",
}

Be objective and focus on which response would genuinely lead to better outcomes for the user. Prioritize correctness of solution over stylistic preferences."""

def create_judge_prompt(conversation_history: str, response_a: str, response_b: str) -> str:
    """Create the prompt for the AI judge."""
    return f"""# Conversation History

{conversation_history}

---

# Response A

{response_a}

---

# Response B

{response_b}

---

**Task**: Compare Response A and Response B. Which response would lead to better conversation quality and outcomes for the user? Provide your answer in JSON format."""

def _make_api_call(prompt: str) -> Dict[str, Any]:
    """Helper function to make the actual API call."""
    response = client.chat.completions.create(
        model="o3",
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def call_o3_judge(conversation_history: str, response_a: str, response_b: str,
                  max_retries: int = 3, timeout_seconds: int = 300) -> Optional[Dict[str, Any]]:
    """Call o3 API to judge which response is better.

    Args:
        conversation_history: The conversation context
        response_a: First response to compare
        response_b: Second response to compare
        max_retries: Maximum number of retry attempts
        timeout_seconds: Timeout in seconds (default: 300 = 5 minutes)
    """

    prompt = create_judge_prompt(conversation_history, response_a, response_b)

    for attempt in range(max_retries):
        try:
            # Use ThreadPoolExecutor to enable timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_make_api_call, prompt)
                result = future.result(timeout=timeout_seconds)
                return result

        except FuturesTimeoutError:
            print(f"  Timeout after {timeout_seconds} seconds on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                print(f"  Retrying...")
            else:
                print("  Max retries reached, skipping this entry")
                return None

        except Exception as e:
            print(f"  Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"  Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("  Max retries reached, skipping this entry")
                return None

    return None

def process_entry(entry: Dict[str, Any], entry_idx: int) -> Dict[str, Any]:
    """Process a single entry and add judge results."""

    print(f"\nProcessing entry {entry_idx + 1}...")
    print(f"  conversation_id: {entry['conversation_id']}")
    print(f"  round_index: {entry['round_index']}")

    # Get responses
    original_response = entry.get('original_response', '')
    reasoning_response = entry.get('reasoning_response', '')

    if not original_response or not reasoning_response:
        print("  Skipping: Missing response(s)")
        entry['judge_winner'] = None
        entry['judge_explanation'] = "Missing response data"
        entry['judge_confidence'] = None
        return entry

    # Randomize order to avoid position bias
    if random.random() < 0.5:
        response_a = original_response
        response_b = reasoning_response
        a_label = "original"
        b_label = "reasoning"
    else:
        response_a = reasoning_response
        response_b = original_response
        a_label = "reasoning"
        b_label = "original"

    # Store which response is which (for analysis)
    entry['judge_response_a'] = a_label
    entry['judge_response_b'] = b_label

    # Call judge
    judgment = call_o3_judge(
        entry['conversation_history'],
        response_a,
        response_b
    )

    if judgment:
        # Map A/B to original/reasoning
        winner_position = judgment['winner']
        winner = a_label if winner_position == 'A' else b_label

        entry['judge_winner'] = winner
        entry['judge_confidence'] = judgment.get('confidence', 'unknown')
        entry['judge_explanation'] = judgment.get('explanation', '')

        print(f"  Winner: {winner} (confidence: {entry['judge_confidence']})")
    else:
        entry['judge_winner'] = None
        entry['judge_confidence'] = None
        entry['judge_explanation'] = "Failed to get judgment"
        print("  Failed to get judgment")

    return entry

def process_file(input_file: str, output_file: str):
    """Process a JSONL file and add judge results."""

    print("\n" + "="*70)
    print(f"Processing: {input_file}")
    print("="*70)

    # Read all entries
    entries = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            entries.append(json.loads(line.strip()))

    print(f"Loaded {len(entries)} entries")

    # Process each entry
    processed_entries = []
    stats = {
        'total': len(entries),
        'success': 0,
        'failed': 0,
        'original_wins': 0,
        'reasoning_wins': 0
    }

    for idx, entry in enumerate(entries):
        processed_entry = process_entry(entry, idx)
        processed_entries.append(processed_entry)

        # Update stats
        if processed_entry['judge_winner']:
            stats['success'] += 1
            if processed_entry['judge_winner'] == 'original':
                stats['original_wins'] += 1
            elif processed_entry['judge_winner'] == 'reasoning':
                stats['reasoning_wins'] += 1
        else:
            stats['failed'] += 1

        # Rate limiting - small delay between requests
        time.sleep(0.5)

    # Write results
    print(f"\n\nWriting results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # Print summary
    print("\n" + "="*70)
    print("Summary Statistics")
    print("="*70)
    print(f"Total entries: {stats['total']}")
    print(f"Successful judgments: {stats['success']}")
    print(f"Failed judgments: {stats['failed']}")
    print(f"Original wins: {stats['original_wins']} ({stats['original_wins']/stats['success']*100:.1f}%)" if stats['success'] > 0 else "Original wins: 0")
    print(f"Reasoning wins: {stats['reasoning_wins']} ({stats['reasoning_wins']/stats['success']*100:.1f}%)" if stats['success'] > 0 else "Reasoning wins: 0")
    print("="*70)

def main():
    """Main function to process both files."""

    files_to_process = [
        ("route-to-reasoning-data/evaluation_predictions_with_gpt4o.jsonl",
         "route-to-reasoning-data/evaluation_predictions_with_gpt4o_judged.jsonl"),
        ("route-to-reasoning-data/evaluation_predictions_with_gpt4o_random.jsonl",
         "route-to-reasoning-data/evaluation_predictions_with_gpt4o_random_judged.jsonl")
    ]

    for input_file, output_file in files_to_process:
        try:
            process_file(input_file, output_file)
        except Exception as e:
            print(f"\n\nERROR processing {input_file}: {e}")
            continue

    print("\n\n" + "="*70)
    print("All files processed!")
    print("="*70)

if __name__ == "__main__":
    main()
