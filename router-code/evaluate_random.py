"""
Evaluate random baseline on eval_sample.jsonl.

This script generates random predictions (50/50 chance of true/false) as a baseline
to compare against the fine-tuned router model's performance.

Usage:
    python evaluate_random.py

Requirements:
    - eval_sample.jsonl must exist in router-data/

Outputs:
    - router-data/evaluation_results_random.json: Metrics (accuracy, precision, recall, F1, confusion matrix)
    - router-data/evaluation_predictions_random.jsonl: All input data plus random predictions
"""

import json
import os
import random
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm


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


def get_random_prediction() -> str:
    """
    Generate a random prediction.

    Returns:
        Random prediction: "true" or "false" with 50/50 probability
    """
    return random.choice(["true", "false"])


def evaluate_random(eval_file: str, seed: int = 42) -> tuple:
    """
    Evaluate random baseline on eval dataset.

    Args:
        eval_file: Path to evaluation JSONL file
        seed: Random seed for reproducibility

    Returns:
        Tuple of (predictions_list, metrics_dict)
    """
    print("=" * 80)
    print("EVALUATING RANDOM BASELINE")
    print("=" * 80)
    print(f"\nEval file: {eval_file}")
    print(f"Random seed: {seed}")

    # Set random seed for reproducibility
    random.seed(seed)

    # Load evaluation data
    print("\nLoading evaluation data...")
    examples = load_eval_data(eval_file)
    print(f"Loaded {len(examples)} examples")

    # Run predictions
    print("\nGenerating random predictions...")
    predictions_with_data = []
    ground_truths = []
    predictions_only = []

    for example in tqdm(examples, desc="Generating"):
        # Extract ground truth
        ground_truth = str(example["requires_reasoning"]).lower()  # Convert boolean to "true"/"false"

        # Get random prediction
        prediction = get_random_prediction()

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
        "model": "random_baseline",
        "random_seed": seed,
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
    print(f"\nModel: {metrics['model']}")
    print(f"Random seed: {metrics['random_seed']}")
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
    eval_file = os.path.join(base_dir, 'router-data', 'eval_sample.jsonl')
    results_file = os.path.join(base_dir, 'router-data', 'evaluation_results_random.json')
    predictions_file = os.path.join(base_dir, 'router-data', 'evaluation_predictions_random.jsonl')

    # Check if eval file exists
    if not os.path.exists(eval_file):
        print(f"Error: Evaluation file not found at {eval_file}")
        return

    # Run evaluation
    predictions_with_data, metrics = evaluate_random(eval_file, seed=42)

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
