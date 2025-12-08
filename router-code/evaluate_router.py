"""
Evaluate fine-tuned router model on eval_sample.jsonl.

This script evaluates a fine-tuned OpenAI model's performance at predicting
whether conversations require advanced reasoning capabilities.

Usage:
    python evaluate_router.py

Requirements:
    - OPENAI_API_KEY environment variable must be set
    - eval_sample.jsonl must exist in router-data/

Outputs:
    - router-data/evaluation_results.json: Metrics (accuracy, precision, recall, F1, confusion matrix)
    - router-data/evaluation_predictions.jsonl: All input data plus model predictions
"""

import json
import os
import pandas as pd
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm


# Fine-tuned model ID
MODEL_ID = "ft:gpt-4o-mini-2024-07-18:lingo:router-train-reasoning-model-2:CkY5kt9r"

# System prompt for routing
SYSTEM_PROMPT = "You are a routing system that analyzes conversations to determine whether they require advanced reasoning capabilities. Analyze the conversation presented up to the current turn and respond with 'true' if the next turn in the conversation requires reasoning (e.g., complex debugging, deep analysis), or 'false' if it can be handled by a standard non-reasoning model."


def load_eval_data(file_path: str) -> list:
    """
    Load evaluation data from JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of dictionaries, each containing a conversation example
    """
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def get_prediction(client: OpenAI, messages: list, model_id: str) -> str:
    """
    Get prediction from fine-tuned model.

    Args:
        client: OpenAI client instance
        messages: List of message dicts (system + user prompts)
        model_id: Fine-tuned model ID

    Returns:
        Predicted label ("true" or "false")
    """
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0,  # Deterministic predictions
            max_tokens=10,  # Only need "true" or "false"
        )
        prediction = response.choices[0].message.content.strip().lower()

        # Normalize prediction to "true" or "false"
        if prediction not in ["true", "false"]:
            print(f"Warning: Unexpected prediction '{prediction}', defaulting to 'false'")
            return "false"

        return prediction

    except Exception as e:
        print(f"Error getting prediction: {e}")
        return "false"  # Default to false on error


def evaluate_model(eval_file: str, model_id: str) -> tuple:
    """
    Evaluate model on eval dataset.

    Args:
        eval_file: Path to evaluation JSONL file
        model_id: Fine-tuned model ID

    Returns:
        Tuple of (predictions_list, metrics_dict)
    """
    print("=" * 80)
    print("EVALUATING ROUTER MODEL")
    print("=" * 80)
    print(f"\nModel: {model_id}")
    print(f"Eval file: {eval_file}")

    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Load evaluation data
    print("\nLoading evaluation data...")
    examples = load_eval_data(eval_file)
    print(f"Loaded {len(examples)} examples")

    # Run predictions
    print("\nRunning predictions...")
    predictions_with_data = []
    ground_truths = []
    predictions_only = []

    for i, example in enumerate(tqdm(examples, desc="Evaluating")):
        # Create messages for the model
        input_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["conversation_history"]}
        ]

        # Extract ground truth
        ground_truth = str(example["requires_reasoning"]).lower()  # Convert boolean to "true"/"false"

        # Get prediction
        prediction = get_prediction(client, input_messages, model_id)

        # Store prediction with original data
        result = example.copy()
        result["prediction"] = prediction
        result["correct"] = prediction == ground_truth
        predictions_with_data.append(result)

        ground_truths.append(ground_truth)
        predictions_only.append(prediction)

    # Calculate metrics
    print("\nCalculating metrics...")
    y_true = [1 if gt == "true" else 0 for gt in ground_truths]
    y_pred = [1 if pred == "true" else 0 for pred in predictions_only]

    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)

    # Per-class metrics (for "true" class, positive class = 1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=1
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model_id": model_id,
        "total_samples": len(examples),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": {
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn)
        },
        "class_distribution": {
            "true_samples": int(sum(y_true)),
            "false_samples": int(len(y_true) - sum(y_true))
        },
        "correct_predictions": int(sum([p["correct"] for p in predictions_with_data])),
        "incorrect_predictions": int(len(predictions_with_data) - sum([p["correct"] for p in predictions_with_data]))
    }

    return predictions_with_data, metrics


def print_metrics(metrics: dict):
    """Print metrics in a readable format."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nTotal samples: {metrics['total_samples']}")
    print(f"\nClass distribution:")
    print(f"  'true' samples: {metrics['class_distribution']['true_samples']}")
    print(f"  'false' samples: {metrics['class_distribution']['false_samples']}")

    print(f"\nOverall Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['correct_predictions']}/{metrics['total_samples']})")

    print(f"\nMetrics for 'true' class (requires reasoning):")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")

    cm = metrics['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted False  Predicted True")
    print(f"  Actual False   {cm['true_negatives']:15d}  {cm['false_positives']:14d}")
    print(f"  Actual True    {cm['false_negatives']:15d}  {cm['true_positives']:14d}")
    print("=" * 80)


def main():
    """Main evaluation function."""
    # Set up paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eval_file = os.path.join(base_dir, 'router-data', 'eval_sample_12_8_2025.jsonl')
    results_file = os.path.join(base_dir, 'router-data', 'evaluation_results_12_8_2025.json')
    predictions_file = os.path.join(base_dir, 'router-data', 'evaluation_predictions_12_8_2025.jsonl')

    # Check if eval file exists
    if not os.path.exists(eval_file):
        print(f"Error: Evaluation file not found at {eval_file}")
        return

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY=your-api-key")
        return

    # Run evaluation
    predictions_with_data, metrics = evaluate_model(eval_file, MODEL_ID)

    # Print results
    print_metrics(metrics)

    # Save results
    print(f"\nSaving results...")

    # Save metrics JSON
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  Metrics saved to: {results_file}")

    # Save predictions JSONL (all original data + predictions)
    with open(predictions_file, 'w', encoding='utf-8') as f:
        for prediction in predictions_with_data:
            json.dump(prediction, f, ensure_ascii=False)
            f.write('\n')
    print(f"  Predictions saved to: {predictions_file}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
