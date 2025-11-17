#!/usr/bin/env python3
"""
Script to identify the most pivotal turn in conversations using a judge model.

This script:
1. Loads conversations from UR_original_context (or any specified file)
2. For each conversation_context, uses a judge model to identify the single turn
   that, if revised, would most improve the overall trajectory of the dialogue
3. Saves the results with the identified pivotal turn and reasoning
"""

import json
import os
import time
import argparse
from typing import List, Dict, Any, Optional, Tuple
from together import Together
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class PivotalTurnIdentifier:
    """Identifies the most pivotal turn in conversations using a judge model."""

    def __init__(self, api_key: Optional[str] = None, judge_model: str = "meta-llama/Meta-Llama-3.3-70B-Instruct-Turbo"):
        """
        Initialize the identifier with Together API.

        Args:
            api_key: Together API key (if None, will look for TOGETHER_API_KEY env var)
            judge_model: The model to use as judge
        """
        if api_key is None:
            api_key = os.getenv('TOGETHER_API_KEY')

        if not api_key:
            raise ValueError("Together API key is required. Set TOGETHER_API_KEY environment variable or pass api_key parameter.")

        self.client = Together(api_key=api_key)
        self.judge_model = judge_model
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'errors': 0,
            'parse_errors': 0
        }

    def create_judge_prompt(self, conversation_context: List[Dict[str, str]]) -> str:
        """
        Create a prompt for the judge model to identify the pivotal turn.

        Args:
            conversation_context: The conversation history

        Returns:
            The prompt string for the judge model
        """
        # Build conversation history with turn numbers for assistant turns only
        conversation_text = ""
        assistant_turn_count = 0

        for turn in conversation_context:
            role = turn['role'].upper()
            content = turn['content']

            if role == "ASSISTANT":
                assistant_turn_count += 1
                conversation_text += f"ASSISTANT (Turn {assistant_turn_count}): {content}\n\n"
            else:
                conversation_text += f"{role}: {content}\n\n"

        system_prompt = """You are an expert dialogue analyst. Your task is to identify the one turn in a multi-turn conversation whose revision would most improve the overall trajectory of the dialogue.

Follow these principles:

Goal: Select the single most pivotal assistant turn that caused the conversation to deteriorate or derailed alignment with the user's intent.

Counterfactual Criterion: Choose the turn such that, if it were re-written appropriately, it would give the highest chance of the conversation proceeding positively thereafter.

Global Context Awareness:
- Infer the user's overarching intent across the full dialogue, not just the local turn.
- Consider whether the assistant misunderstood constraints, ignored instructions, contradicted prior turns, or mishandled feedback.
- Pay attention to cumulative signals of dissatisfaction.

Look for Structural Errors:
- Misinterpretations of instructions
- Repeated failures that accumulate into a breakdown
- Turns where expectations diverged from delivery
- Moments where a misunderstanding first appears

Output:
Return the turn number of the problematic assistant turn (counting the first assistant turn as turn 1 (1-indexed)).
Provide a one-sentence justification explaining why this turn is the most impactful point of failure.
Do not rewrite the turn—only identify it.

Your output format must be exactly:
turn: <turn_number>
reason: <one-sentence explanation>"""

        full_prompt = f"""{system_prompt}

Here is the conversation to analyze:

{conversation_text}

Please identify the most pivotal assistant turn."""

        return full_prompt

    def call_judge_model(self, conversation_context: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """
        Call the judge model to identify the pivotal turn.

        Args:
            conversation_context: The conversation history

        Returns:
            Dictionary with 'turn' (int) and 'reason' (str), or None if failed
        """
        try:
            prompt = self.create_judge_prompt(conversation_context)

            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}], # 'prompt' is from the user.
                max_tokens=300,
                temperature=0.1,  # Low temperature for consistent analysis
                top_p=0.9,
            )

            response_text = response.choices[0].message.content.strip()
            return self.parse_judge_response(response_text)

        except Exception as e:
            print(f"Error calling judge model: {e}")
            self.stats['errors'] += 1
            return None

    def parse_judge_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse the judge model's response to extract turn number and reason.

        Args:
            response_text: The raw response from the judge model

        Returns:
            Dictionary with 'turn' and 'reason', or None if parsing failed
        """
        try:
            lines = response_text.strip().split('\n')
            turn_num = None
            reason = None

            for line in lines:
                line = line.strip()
                if line.lower().startswith('turn:'):
                    # Extract turn number
                    turn_str = line.split(':', 1)[1].strip()
                    # Try to extract just the number
                    try:
                        turn_num = int(''.join(filter(str.isdigit, turn_str)))
                    except ValueError:
                        print(f"Warning: Could not parse turn number from: {turn_str}")
                        continue
                elif line.lower().startswith('reason:'):
                    reason = line.split(':', 1)[1].strip()

            if turn_num is not None and reason:
                return {
                    'turn': turn_num,
                    'reason': reason,
                    'raw_response': response_text
                }
            else:
                print(f"Warning: Could not parse response. Turn: {turn_num}, Reason: {reason}")
                print(f"Raw response: {response_text}")
                self.stats['parse_errors'] += 1
                return None

        except Exception as e:
            print(f"Error parsing judge response: {e}")
            print(f"Response text: {response_text}")
            self.stats['parse_errors'] += 1
            return None

    def get_turn_content(self, conversation_context: List[Dict[str, str]], turn_number: int) -> Optional[str]:
        """
        Get the content of a specific assistant turn.

        Args:
            conversation_context: The conversation history
            turn_number: The turn number (1-indexed)

        Returns:
            The content of the specified turn, or None if not found
        """
        assistant_turn_count = 0
        for turn in conversation_context:
            if turn['role'] == 'assistant': # count only assistant turns in the indexing.
                assistant_turn_count += 1 # first turn is assigned the number 1.
                if assistant_turn_count == turn_number:
                    return turn['content']
        return None

    def process_conversation(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single conversation to identify its pivotal turn.

        Args:
            conversation: The conversation dictionary

        Returns:
            Processed conversation with pivotal turn information
        """
        conversation_context = conversation.get('conversation_context', [])

        if not conversation_context:
            print(f"Warning: Empty conversation context for {conversation.get('conversation_id', 'unknown')}")
            return {
                **conversation,
                'pivotal_turn_analysis': {
                    'status': 'error',
                    'error': 'Empty conversation context'
                }
            }

        # Call judge model
        judge_result = self.call_judge_model(conversation_context)

        if judge_result:
            # Get the actual turn content
            turn_content = self.get_turn_content(conversation_context, judge_result['turn'])

            analysis = {
                'status': 'success',
                'pivotal_turn_number': judge_result['turn'],
                # 'pivotal_turn_content': turn_content,
                'reason': judge_result['reason'],
                # 'raw_judge_response': judge_result['raw_response'],
                'analysis_timestamp': datetime.now().isoformat()
            }
            self.stats['successful'] += 1
        else:
            analysis = {
                'status': 'error',
                'error': 'Failed to get or parse judge response'
            }
            self.stats['errors'] += 1

        return {
            **conversation,
            'pivotal_turn_analysis': analysis
        }

    def process_conversations(self, conversations: List[Dict[str, Any]],
                            max_conversations: Optional[int] = None,
                            delay_between_calls: float = 0.5) -> List[Dict[str, Any]]:
        """
        Process all conversations to identify pivotal turns.

        Args:
            conversations: List of conversation dictionaries
            max_conversations: Maximum number to process (for testing)
            delay_between_calls: Delay between API calls to avoid rate limits

        Returns:
            List of processed conversations with pivotal turn analysis
        """
        processed_conversations = []
        total = min(len(conversations), max_conversations) if max_conversations else len(conversations)

        print(f"Processing {total} conversations...")

        for i, conversation in enumerate(conversations[:max_conversations] if max_conversations else conversations):
            if i % 10 == 0:
                print(f"Progress: {i}/{total} conversations processed")

            processed_conv = self.process_conversation(conversation)
            processed_conversations.append(processed_conv)

            self.stats['total_processed'] += 1

            # Add delay between API calls for rate limiting
            if delay_between_calls > 0 and i < total - 1:
                time.sleep(delay_between_calls)

        print(f"Progress: {total}/{total} conversations processed (complete)")
        return processed_conversations

    def print_stats(self):
        """Print processing statistics."""
        print(f"\n=== Processing Statistics ===")
        print(f"Total processed: {self.stats['total_processed']}")
        print(f"Successful: {self.stats['successful']}")
        print(f"Errors: {self.stats['errors']}")
        print(f"Parse errors: {self.stats['parse_errors']}")

        if self.stats['total_processed'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total_processed']) * 100
            print(f"Success rate: {success_rate:.1f}%")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Identify pivotal turns in conversations using a judge model')
    parser.add_argument('--input', '-i',
                       default='data/UR2_original_context.json',
                       help='Input JSON file with conversations (default: data/UR2_original_context.json)')
    parser.add_argument('--output', '-o',
                       default='results/UR2_pivotal_turns.json',
                       help='Output JSON file for results (default: input_name_pivotal_turns.json)')
    parser.add_argument('--max-conversations', '-n', type=int,
                       help='Maximum number of conversations to process (for testing)')
    parser.add_argument('--delay', '-d', type=float, default=0.5,
                       help='Delay between API calls in seconds (default: 0.5)')
    parser.add_argument('--judge-model', '-j',
                       default='meta-llama/Llama-3.3-70B-Instruct-Turbo',
                       help='Judge model to use (default: meta-llama/Llama-3.3-70B-Instruct-Turbo)')

    args = parser.parse_args()

    # Check if Together API key is available
    if not os.getenv('TOGETHER_API_KEY'):
        print("Error: TOGETHER_API_KEY environment variable not set.")
        print("Please set it with: export TOGETHER_API_KEY='your-api-key-here'")
        return 1

    # Load conversations
    print(f"Loading conversations from {args.input}...")
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

    # Filter for conversations with feedback_turn >= 3
    print(f"Filtering for conversations with feedback_turn >= 3...")
    filtered_conversations = [
        conv for conv in conversations
        if conv.get('feedback_turn', 0) >= 3
    ]
    conversations = filtered_conversations

    # Generate output filename if not provided
    if not args.output:
        input_basename = os.path.basename(args.input).replace('.json', '')
        args.output = f"data/{input_basename}_pivotal_turns.json"

    print(f"Results will be saved to: {args.output}")

    # Initialize identifier model.
    try:
        identifier = PivotalTurnIdentifier(judge_model=args.judge_model)
        print(f"Using judge model: {args.judge_model}")
    except ValueError as e:
        print(f"Error initializing identifier: {e}")
        return 1

    # Process conversations
    processed_conversations = identifier.process_conversations(
        conversations,
        max_conversations=args.max_conversations,
        delay_between_calls=args.delay
    )

    # Save results
    print(f"\nSaving results to {args.output}...")
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(processed_conversations, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(processed_conversations)} processed conversations")
    except Exception as e:
        print(f"Error saving results: {e}")
        return 1

    # Print statistics
    identifier.print_stats()

    print(f"\nProcessing complete!")
    print(f"Results saved to: {args.output}")

    return 0

if __name__ == "__main__":
    exit(main())
