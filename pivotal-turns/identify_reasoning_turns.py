import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# Insert your OpenAI API key here
client = OpenAI()

ROUTING_JUDGE_SYSTEM_PROMPT = """
You are an Expert Dialogue Analyst and AI Routing Specialist.

Your Task:
You will be provided with a multi-turn conversation log between a USER and a STANDARD_AI_ASSISTANT (a fast, non-reasoning model).
Your goal is to identify specific turns where the STANDARD_AI_ASSISTANT failed because the task required "System 2" capabilities (Deep Reasoning, Precise Code Analysis, or Complex Planning) which the standard model lacks.

These labeled turns will be used to train a Router that redirects difficult turns in a dialogue to a Reasoning Model (e.g., o1, gpt-4o, DeepSeek-R1).

Definitions:
1. REASONING_REQUIRED (Label: YES) - The Standard AI fails due to:
   - **Algorithmic Errors**: The assistant makes an error in algorithmic logic (e.g., sorting, searching, recursion) or data structure manipulation that requires step-by-step reasoning.
   - **Persistent Issues**: If the user reports the same or similar mistake/issue/error 3 or more times across the conversation, ALL turns where this issue persists should be flagged as requiring reasoning.
   - **Hallucination Pattern** (error_category: "hallucination"): The assistant claims with confidence that something "should work" or "looks correct", then the user reports the exact same error, and the assistant tries only a trivial variation. This pattern indicates the assistant failed to deeply analyze the problem.
   - **Deep Debugging**: The user reports an error, and the Standard AI suggests generic fixes (e.g., "check your imports") or hallucinates that the code is correct because it "looks" standard, failing to spot logic gaps or subtle typos (e.g., `init` vs `__init__`).
   - **Low-Level/Memory Errors**: Pointer logic, `ctypes` structures, memory leaks, or Rust lifetime constraints.
   - **Tensor/Data Shape Errors**: Mismatched dimensions in machine learning models (numpy/torch) that require tracing data flow.
   - **Other Complex Reasoning**: Any other scenario where the assistant's failure stems from a lack of deep reasoning, multi-step logic, or complicated analysis.

2. STANDARD_CAPABLE (Label: NO):
   - **User Misunderstanding**: The assistant correctly understood and responded, but the user misunderstood the assistant's request or instructions. These should NOT be flagged as requiring reasoning.
   - **Environment/Setup Issues**: The assistant provided correct code/instructions, but the error is due to the user's environment (e.g., missing dependencies, wrong Python version, OS-specific issues, network problems, file permission errors). These should NOT be flagged as requiring reasoning.
   - **Knowledge Retrieval**: Simple factual questions.
   - **Boilerplate Generation**: Writing standard code templates that don't require integrating conflict constraints.
   - **Tone/Safety**: Failures due to refusals or style issues.

Instructions for each multi-turn conversation:
1. Carefully scan the entire multi-turn dialogue for the negative feedback provided by the user (e.g., "Error", "Traceback", "Still not working", "AttributeError").
2. Next, for each round (one user message + one assistant response = one round, 0-indexed), determine whether that turn's assistant response could be the source for downstream issues/errors.
3. If the issue could have been addressed by more reasoning at that round, label that round as `requires_reasoning: true`.
4. Track if the same negative-feedback/issue/error appears multiple times. If the user flags the same or similar mistake 3+ times, that round MUST be labeled as requires_reasoning.

IMPORTANT: analyze every round in the conversation and its placement in context of the entire conversation arc. return a <bool> for each round indicating whether routing to a reasoning model at that round is necessary, along with a brief explanation.

Output Format (JSON):
{
  "routing_decisions": [
    {
      "round_index": <int, index of the round (0-indexed). Each round = one user message + one assistant response.>,
      "requires_reasoning": <bool>,
      "justification": "Specific explanation (keep it to 1 sentence). Key phrase: 'Assistant failed to understand/do [X] and instead did [Y]'."
    }
  ]
}
"""


def analyze_conversation_turns(conversation_json_str):
    """
    Analyzes a conversation to identify which assistant turns required reasoning capabilities.

    Args:
        conversation_json_str: JSON string of a list of conversation turns,
        where each turn looks like:
        {
            "content": "...",
            "role": "user" or "assistant",
            ...
        }

    Returns:
        dict: Parsed JSON response with analysis and routing_decisions
    """
    conversation = json.loads(conversation_json_str)

    # Build messages for the model
    messages = [{"role": "system", "content": ROUTING_JUDGE_SYSTEM_PROMPT}]

    for turn in conversation:
        # Only need role + content for judging
        messages.append({
            "role": turn["role"],
            "content": turn["content"]
        })

    # Model call
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=messages,
        reasoning_effort="minimal",
        verbosity="low",
        response_format={"type": "json_object"}
    )

    # Parse the JSON response
    result = json.loads(response.choices[0].message.content.strip())
    return result


if __name__ == "__main__":
    # Load the negative feedback dataset
    input_file = "/dccstor/gma2/jhjenny9/counterfactual_feedback/pivotal-turns-data/nf4000_12_05_2025.csv"
    output_file = "/dccstor/gma2/jhjenny9/counterfactual_feedback/pivotal-turns-data/pt316_12_05_2025_2.csv"

    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} conversations from {input_file}")

    # Process each conversation
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing conversations"):
        conversation_id = row['conversation_id']
        conversation_json = row['conversation']

        try:
            analysis_result = analyze_conversation_turns(conversation_json)

            # Store the full result
            results.append({
                'conversation_id': conversation_id,
                'analysis': analysis_result.get('analysis', ''),
                'routing_decisions': json.dumps(analysis_result.get('routing_decisions', [])),
                'num_reasoning_turns': sum(
                    1 for decision in analysis_result.get('routing_decisions', [])
                    if decision.get('requires_reasoning', False)
                ),
                'status': 'SUCCESS'
            })

        except Exception as e:
            print(f"\nError processing conversation {conversation_id}: {e}")
            results.append({
                'conversation_id': conversation_id,
                'routing_decisions': '[]',
                'num_reasoning_turns': 0,
                'status': f'ERROR: {str(e)}'
            })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)

    print(f"\nResults saved to {output_file}")
    print(f"\nSummary:")
    print(f"Total conversations analyzed: {len(results_df)}")
    print(f"Successful analyses: {(results_df['status'] == 'SUCCESS').sum()}")
    print(f"Errors: {(results_df['status'] != 'SUCCESS').sum()}")
    print(f"\nConversations with reasoning-required turns: {(results_df['num_reasoning_turns'] > 0).sum()}")
    print(f"Total reasoning-required turns identified: {results_df['num_reasoning_turns'].sum()}")

    # Show distribution of reasoning turns
    print(f"\nDistribution of reasoning-required turns per conversation:")
    print(results_df['num_reasoning_turns'].value_counts().sort_index())
