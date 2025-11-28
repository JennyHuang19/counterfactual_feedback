import re
import time
from vllm import LLLM, SamplingParams

# -------------------------
# 0. CONFIG
# -------------------------

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"   # <-- change to your model
questions = [
    "If Sarah has 12 apples and gives 3 to John, then buys 5 more, how many apples does she have?",
    "A rectangle has sides 3 and 7. What is its area?",
    "Tom has twice as many marbles as Lisa. Lisa has 8. How many do they have together?",
]
answers = ["14", "21", "24"]

# -------------------------
# 1. PROMPT TEMPLATES
# -------------------------

def make_light_prompt(q):
    return f"""You are a helpful assistant.

Question: {q}

Give only the final answer, as briefly as possible.
Answer:"""

def make_deep_prompt(q):
    return f"""You are a helpful assistant.

Question: {q}

Think step by step and show your reasoning.
At the end, write: "Final answer: <answer>".
Reasoning and answer:"""


# -------------------------
# 2. SAMPLING CONFIGS
# -------------------------

light_params = SamplingParams(
    temperature=0.0,
    max_tokens=32,
)

deep_params = SamplingParams(
    temperature=0.0,
    max_tokens=256,
)

# -------------------------
# 3. LOAD MODEL
# -------------------------

print("Loading model...")
llm = LLM(model=MODEL_NAME)

# -------------------------
# 4. RUN LIGHT REASONING
# -------------------------

light_prompts = [make_light_prompt(q) for q in questions]

print("\nRunning LIGHT reasoning...")
start = time.perf_counter()
light_outputs = []
light_times = []

for prompt in light_prompts:
    t0 = time.perf_counter()
    out = llm.generate([prompt], light_params)[0]
    t1 = time.perf_counter()
    light_outputs.append(out)
    light_times.append(t1 - t0)

total_light_time = time.perf_counter() - start


# -------------------------
# 5. RUN DEEP REASONING
# -------------------------

deep_prompts = [make_deep_prompt(q) for q in questions]

print("\nRunning DEEP reasoning...")
start = time.perf_counter()
deep_outputs = []
deep_times = []

for prompt in deep_prompts:
    t0 = time.perf_counter()
    out = llm.generate([prompt], deep_params)[0]
    t1 = time.perf_counter()
    deep_outputs.append(out)
    deep_times.append(t1 - t0)

total_deep_time = time.perf_counter() - start


# -------------------------
# 6. ANSWER EXTRACTION
# -------------------------

def extract_light_answer(text):
    return text.strip().split("\n")[0].strip()

def extract_deep_answer(text):
    m = re.search(r"Final answer:\s*(.+)", text)
    return m.group(1).strip() if m else text.strip().split("\n")[-1].strip()

light_preds = [extract_light_answer(out.outputs[0].text) 
               for out in light_outputs]
deep_preds  = [extract_deep_answer(out.outputs[0].text) 
               for out in deep_outputs]


# -------------------------
# 7. ACCURACY + TOKEN USAGE
# -------------------------

def accuracy(preds, gold):
    return sum(p == g for p, g in zip(preds, gold)) / len(gold)

acc_light = accuracy(light_preds, answers)
acc_deep  = accuracy(deep_preds, answers)

light_toks = sum(o.outputs[0].token_count for o in light_outputs) / len(light_outputs)
deep_toks  = sum(o.outputs[0].token_count for o in deep_outputs) / len(deep_outputs)


# -------------------------
# 8. REPORT
# -------------------------

print("\n====================== RESULTS ======================")

print("\nACCURACY")
print(f"  Light reasoning accuracy: {acc_light:.3f}")
print(f"  Deep reasoning accuracy:  {acc_deep:.3f}")

print("\nTOKEN USAGE (proxy for compute)")
print(f"  Light avg output tokens: {light_toks:.1f}")
print(f"  Deep avg output tokens:  {deep_toks:.1f}")

print("\nLATENCY PER QUERY")
print(f"  Light avg latency: {sum(light_times)/len(light_times):.3f} s")
print(f"  Deep avg latency:  {sum(deep_times)/len(deep_times):.3f} s")

print("\nTOTAL WALL-CLOCK TIME")
print(f"  Light total time: {total_light_time:.3f} s")
print(f"  Deep total time:  {total_deep_time:.3f} s")

print("\nPREDICTIONS")
print("  Light:", light_preds)
print("  Deep: ", deep_preds)

print("\nDetailed per-query latency:")
for i, (lt, dt) in enumerate(zip(light_times, deep_times)):
    print(f"  Q{i}: light={lt:.3f}s | deep={dt:.3f}s")
