import json
from openai import OpenAI

# Insert your OpenAI API key here
client = OpenAI()

JUDGE_SYSTEM_PROMPT = """
You are a conversation-quality judge. Your task is to determine whether a multi-turn conversation
between a user and an AI assistant contains negative feedback from the user.

Negative feedback includes ANY sign that the user is dissatisfied with the assistant’s output:

1. IMPLICIT negative feedback:
   - Re-asking the same question in different wording. 
   - Implying the earlier answer wasn’t good.
   - Requesting revisions, rewrites, or improvements.
   - Correcting the assistant’s content.
   - Adding constraints that the assistant failed to follow earlier.
   - Indicating that the answer didn’t meet the goal or missed important aspects.

2. EXPLICIT negative feedback:
   - "Contains Error."
   - "Shows Error."
   - "Invalid."
   - "still showing the following error."
   - “You misunderstood.”
   - “That’s not what I asked.”
   - “Redo this; the previous answer wasn’t good.”

3. REQUESTS FOR DEEPER OR MORE CAREFUL THINKING (treated as negative feedback):
   Any instruction suggesting the prior answer was too shallow, incomplete, or insufficient.
   Examples:
     - “Reason more carefully.”
     - “Provide a more elaborate answer.”
     - “Can you explain in more detail?”
     - “Be more rigorous / more precise.”

If ANY turn plausibly indicates dissatisfaction—explicitly, implicitly, or by requesting a deeper
rewrite of the assistant’s previous response—you must label the conversation as:

    "negative feedback"

If there is NO sign of dissatisfaction, label the conversation as:

    "no negative feedback"

Output ONLY one of these two strings.
- "negative feedback"
- "no negative feedback"
MAKE SURE YOU DO NOT OUTPUT ANYTHING ELSE.
"""


def classify_negative_feedback(conversation_json_str):
    """
    conversation_json_str: JSON string of a list of conversation turns,
    where each turn looks like:
    {
        "content": "...",
        "role": "user" or "assistant",
        ...
    }
    """

    conversation = json.loads(conversation_json_str)

    # Build messages for the model
    messages = [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}]

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
        verbosity="low"
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm

    # Load the first 500 rows from coding_dialogues.csv
    df = pd.read_csv("/dccstor/gma2/jhjenny9/counterfactual_feedback/pivotal-turns-data/coding_dialogues.csv", skiprows=range(1, 4000))

    print(f"Loaded {len(df)} conversations from coding_dialogues.csv")

    # Process each conversation
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Classifying conversations"):
        conversation_json = row['conversation']

        try:
            label = classify_negative_feedback(conversation_json)
            results.append({
                'conversation_id': row['conversation_id'],
                'label': label
            })
            print("conversation_id:", row['conversation_id'], "label:", label)
        except Exception as e:
            print(f"\nError processing conversation {row['conversation_id']}: {e}")
            results.append({
                'conversation_id': row['conversation_id'],
                'label': 'ERROR'
            })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("pivotal-turns/negative_feedback_classifications_4000-11363.csv", index=False)

    print(f"\nSummary:")
    print(results_df['label'].value_counts())
