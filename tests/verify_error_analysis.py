
import csv
from pathlib import Path
from collections import Counter, defaultdict


ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    ROOT
    / "outputs/test_error_analysis/test_predictions.csv"
)


def main():

    if not CSV_PATH.exists():
        raise FileNotFoundError(CSV_PATH)

    with CSV_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        records = list(csv.DictReader(file))

    if not records:
        raise ValueError("Prediction CSV is empty.")

    total = len(records)

    correct = [
        row for row in records
        if row["classification_correct"].lower() == "true"
    ]

    errors = [
        row for row in records
        if row["classification_correct"].lower() == "false"
    ]

    print("\n========== DATASET VERIFICATION ==========")

    print("Total records:", total)
    print("Correct predictions:", len(correct))
    print("Incorrect predictions:", len(errors))

    print(
        "Accuracy:",
        round(len(correct) / total, 4)
    )

    error_groups = Counter(
        (
            row["actual_class"],
            row["predicted_class"]
        )
        for row in errors
    )

    print("\nMisclassification breakdown:")

    for (actual, predicted), count in sorted(error_groups.items()):

        print(
            f"{actual} -> {predicted}: {count}"
        )

    class_stats = defaultdict(list)

    for row in errors:

        class_stats[row["actual_class"]].append(
            float(row["confidence"])
        )

    print("\nAverage confidence of incorrect predictions:")

    for class_name, values in sorted(class_stats.items()):

        average = sum(values) / len(values)

        print(
            f"{class_name}: {average:.4f}"
        )

    normal_errors = [
        row for row in errors
        if row["actual_class"] == "Normal"
    ]

    normal_fp = sum(
        row["normal_false_positive"].lower() == "true"
        for row in normal_errors
    )

    print("\nNormal-image segmentation audit:")
    print("Misclassified normal images:", len(normal_errors))
    print("With segmentation false positives:", normal_fp)

    print("\n========== VERIFICATION COMPLETED ==========")


if __name__ == "__main__":
    main()
