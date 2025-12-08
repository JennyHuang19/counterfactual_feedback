"""
Generate GPT-4o responses for evaluation dataset.

This script reads evaluation_predictions.jsonl, generates GPT-4o responses for ALL
conversations in the dataset, and outputs a new dataset with all original fields
plus a new 'reasoning_response' field containing the GPT-4o generated response.

Usage:
    python generate_gpt4o_responses.py

Requirements:
    - OPENAI_API_KEY environment variable must be set
    - evaluation_predictions.jsonl must exist in to-route-to-reasoning/

Outputs:
    - to-route-to-reasoning/evaluation_predictions_with_gpt4o.jsonl
"""

import json
import os
import re
import time
from openai import OpenAI
from tqdm import tqdm


def parse_conversation_history(history: str) -> list:
    """
    Parse conversation history string into OpenAI API message format.

    Args:
        history: Conversation string with "User:" and "Assistant:" prefixes

    Returns:
        List of message dicts: [{"role": "user", "content": "..."}, ...]
    """
    messages = []

    # Split on "User:" or "Assistant:" while keeping the delimiter
    parts = re.split(r'(User:|Assistant:)', history)

    # Remove empty strings and strip whitespace
    parts = [p.strip() for p in parts if p.strip()]

    # Process pairs of (role_marker, content)
    i = 0
    while i < len(parts) - 1:
        role_marker = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""

        if role_marker == "User:":
            # Skip ahead to find the actual content (next part that's not a role marker)
            if content in ["User:", "Assistant:"]:
                i += 1
                continue
            messages.append({"role": "user", "content": content.strip()})
        elif role_marker == "Assistant:":
            # Skip ahead to find the actual content
            if content in ["User:", "Assistant:"]:
                i += 1
                continue
            messages.append({"role": "assistant", "content": content.strip()})

        i += 2

    return messages


def get_gpt4o_response(messages: list, client: OpenAI, max_retries: int = 3) -> tuple:
    """
    Get GPT-4o response for a conversation.

    Args:
        messages: List of message dicts for OpenAI API
        client: OpenAI client instance
        max_retries: Maximum number of retry attempts

    Returns:
        Tuple of (response_text, input_tokens, output_tokens, error_message)
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=1.0,
                max_tokens=2048
            )

            content = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            return content, input_tokens, output_tokens, None

        except Exception as e:
            error_msg = str(e)

            # Retry on rate limit errors with exponential backoff
            if "rate_limit" in error_msg.lower() and attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"\nRate limit hit, waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue

            # Return error message if max retries reached or other error
            print(f"\nError generating response: {error_msg}")
            return f"[ERROR: {error_msg}]", 0, 0, error_msg

    return "[ERROR: Max retries exceeded]", 0, 0, "Max retries exceeded"


def main():
    """Main processing function."""
    print("=" * 80)
    print("GENERATING GPT-4O RESPONSES")
    print("=" * 80)

    # Set up paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, 'route-to-reasoning-data', 'evaluation_predictions_random.jsonl')
    output_file = os.path.join(base_dir, 'route-to-reasoning-data', 'evaluation_predictions_with_gpt4o_random.jsonl')

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY=your-api-key")
        return

    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Load input data
    print(f"\nInput file: {input_file}")
    print(f"Output file: {output_file}")

    # Count total records
    with open(input_file, 'r', encoding='utf-8') as f:
        total_records = sum(1 for _ in f)
    print(f"\nTotal records to process: {total_records}")

    # Process records
    print("\nGenerating GPT-4o responses...\n")
    print("Note: Generating responses for ALL records\n")

    total_input_tokens = 0
    total_output_tokens = 0
    error_count = 0
    generated_count = 0

    # Open output file for writing
    with open(output_file, 'w', encoding='utf-8') as out_f:
        with open(input_file, 'r', encoding='utf-8') as in_f:
            for idx, line in enumerate(tqdm(in_f, total=total_records, desc="Processing"), 1):
                # Parse input record
                record = json.loads(line)

                # Extract conversation history
                conversation_history = record['conversation_history']

                # Parse to messages
                messages = parse_conversation_history(conversation_history)

                # Get GPT-4o response
                response, input_tokens, output_tokens, error = get_gpt4o_response(messages, client)

                # Track tokens and errors
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                if error:
                    error_count += 1

                # Add reasoning_response field
                record['reasoning_response'] = response
                generated_count += 1

                # Print cost estimate every 10 generated responses
                if generated_count % 10 == 0:
                    input_cost = (total_input_tokens / 1_000_000) * 2.50
                    output_cost = (total_output_tokens / 1_000_000) * 10.00
                    total_cost = input_cost + output_cost
                    print(f"\n[After {generated_count} responses] Cost so far: ${total_cost:.2f} (Input: {total_input_tokens:,} tokens, Output: {total_output_tokens:,} tokens)")

                # Write to output file immediately (incremental save)
                json.dump(record, out_f, ensure_ascii=False)
                out_f.write('\n')

    # Print summary
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nTotal records processed: {total_records}")
    print(f"Generated responses: {generated_count}")
    print(f"Successful: {generated_count - error_count}")
    print(f"Errors: {error_count}")

    print(f"\nToken usage:")
    print(f"  Input tokens:  {total_input_tokens:,}")
    print(f"  Output tokens: {total_output_tokens:,}")
    print(f"  Total tokens:  {total_input_tokens + total_output_tokens:,}")

    # Estimate cost (GPT-4o pricing as of 2024)
    # Input: $2.50 per 1M tokens, Output: $10.00 per 1M tokens
    input_cost = (total_input_tokens / 1_000_000) * 2.50
    output_cost = (total_output_tokens / 1_000_000) * 10.00
    total_cost = input_cost + output_cost

    print(f"\nEstimated cost:")
    print(f"  Input:  ${input_cost:.2f}")
    print(f"  Output: ${output_cost:.2f}")
    print(f"  Total:  ${total_cost:.2f}")

    print(f"\nOutput saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
