import json
from typing import Any
# unit test for load_judge_data. filters for UR2/3 and conversation IDs present in all three datasets.

path_original_context: str = 'data/filtered_conversations_ur235_mode3.json'
path_feedback_context: str = 'data/feedback.json'
path_alternative_gen_context: str = 'data/alternative.json'
limit: int = 2000  # or set to an integer value to limit the number of conversations

print(f"Loading feedback dataset from {path_feedback_context}...")
with open(path_feedback_context, 'r', encoding='utf-8') as f:
    feedback_context_list = json.load(f)
print(f"Loaded {len(feedback_context_list)} conversations from feedback dataset")

print(f"Loading original context dataset from {path_original_context}...")
with open(path_original_context, 'r', encoding='utf-8') as f:
    original_context_list = json.load(f)
print(f"Loaded {len(original_context_list)} conversations from original context dataset")

if path_alternative_gen_context:
    print(f"Loading alternative generation context dataset from {path_alternative_gen_context}...")
    with open(path_alternative_gen_context, 'r', encoding='utf-8') as f:
        alternative_gen_context_list = json.load(f)
    print(f"Loaded {len(alternative_gen_context_list)} conversations from alternative generation context dataset")


# Filter to only the conversation IDs that are present in all three datasets and appear in category UR2 or UR3 in original_context_list.
valid_original_context_list = []
valid_feedback_context_list = []
valid_alternative_gen_context_list = []
for orig_conv in original_context_list:
    conv_id = orig_conv.get('conversation_id', None)
    # Skip repeat conversation IDs in original context list, because the list has appearances of the same conversation stopped at different points.
    if sum(1 for c in original_context_list if c.get('conversation_id', None) == conv_id) > 1:
        continue
    
    category = orig_conv.get('category', None)
    if conv_id is None or category not in ['UR2']:
        continue
    
    # Find matching feedback context
    feedback_conv = next((conv for conv in feedback_context_list if conv.get('conversation_id', None) == conv_id), None)
    if feedback_conv is None:
        continue
    
    # Find matching alternative generation context if provided
    if path_alternative_gen_context:
        alternative_gen_conv = next((conv for conv in alternative_gen_context_list if conv.get('conversation_id', None) == conv_id), None)
        if alternative_gen_conv is None:
            continue

    valid_original_context_list.append(orig_conv)
    valid_feedback_context_list.append(feedback_conv)   
    if path_alternative_gen_context:
        valid_alternative_gen_context_list.append(alternative_gen_conv)

print(f"Found {len(valid_original_context_list)} valid conversations with feedback and alternative generations to compare.")

if limit:
    valid_original_context_list = valid_original_context_list[:limit]
    valid_feedback_context_list = valid_feedback_context_list[:limit]
    if path_alternative_gen_context:
        valid_alternative_gen_context_list = valid_alternative_gen_context_list[:limit]
    print(f"Limited to first {limit} conversations for testing.")

# save each list to json files for verification.
with open('data/valid_original_context.json', 'w', encoding='utf-8') as f:
    json.dump(valid_original_context_list, f, indent=2)
# with open('data/valid_feedback_context.json', 'w', encoding='utf-8') as f:
#     json.dump(valid_feedback_context_list, f, indent=2)
# if path_alternative_gen_context:
#     with open('data/valid_alternative_gen_context.json', 'w', encoding='utf-8') as f:
#         json.dump(valid_alternative_gen_context_list, f, indent=2)

