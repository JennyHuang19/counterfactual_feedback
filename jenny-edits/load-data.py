
import os
import datasets
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

TOKEN = os.getenv("HF_TOKEN")
LOADING_MODE = 3

extraction_df = datasets.load_dataset("shachardon/naturally_occurring_feedback")["train"]
lmsys_dataset = datasets.load_dataset("lmsys/lmsys-chat-1m", token=TOKEN)["train"]
lmsys_dataset = lmsys_dataset.to_pandas()


# use the conversation_id and feedback_turn from the extraction_df to get the conversations
# from the lmsys dataset
# Filter only for UR2, UR3, or UR5 categories and english language.
conversations = []
for i in range(len(extraction_df)):
    if len(conversations) >= 5000:  #
        break
    row = extraction_df[i]
    category = row["category"]


    # Skip if not UR2, UR3, or UR5
    if category not in ["UR2", "UR3", "UR5"]:
        continue

    conversation_id = row["conversation_id"]
    feedback_turn = row["feedback_turn"]
    org_row = lmsys_dataset[lmsys_dataset["conversation_id"] == conversation_id].iloc[0]

    conversation = org_row["conversation"]

    language = org_row["language"]
    if language != "English":
        continue

    conversation_context = []
    if LOADING_MODE == 1:
      for i in range(feedback_turn * 2 - 2, feedback_turn * 2):
          conversation_context.append(conversation[i])
    elif LOADING_MODE == 2: # The user prompt, assistant response, AND the user's feedback message.
      for i in range(feedback_turn * 2 - 2, feedback_turn * 2 + 1):
          conversation_context.append(conversation[i])
    elif LOADING_MODE == 3: # The entire conversation up to and including the feedback turn.
      for i in range(0, feedback_turn * 2 + 1):
          conversation_context.append(conversation[i])
    else: # All turns from start to finish, not just the context around the feedback.
      conversation_context = conversation

    conversations.append({"conversation_id": conversation_id, "model": org_row["model"],
                          "conversation_context": conversation_context, "feedback_turn": feedback_turn,
                          "language": org_row["language"], "category": row["category"], "label": row["label"]})
    # print(f"Conversation context {i+1} is {conversation_context[:2]}...")
    print(f"Processed {i+1}/{len(extraction_df)}", end="\r")
print(conversations[:5])
#  save conversations to a json file
import json
with open(f"data/filtered_conversations_ur235_mode3.json", "w") as f:
    json.dump(conversations, f, indent=4)

