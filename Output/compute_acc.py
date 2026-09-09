import argparse
import json


def evaluate(path):
    # Resumed runs may duplicate failures; prefer successful records for each sample.
    records = {}
    config = None
    with open(path, encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if config is None:
                config = row['config']
            elif row['config'] != config:
                raise ValueError('Results contain different experiment configurations')
            key = (row['id'], row['text_hash'])
            if key not in records or row['status'] == 'ok':
                records[key] = row
    if not records:
        raise ValueError('Results file is empty')
    rows = list(records.values())
    successful = [row for row in rows if row['status'] == 'ok']
    correct = sum(row['prediction'] == row['label'] for row in successful)
    # Count failed cases as missing predictions in the denominator to avoid inflated scores.
    labels = sorted({row['label'] for row in rows} | {row['prediction'] for row in successful})
    f1_values = []
    for label in labels:
        tp = sum(row['label'] == label and row.get('prediction') == label for row in rows)
        fp = sum(row['label'] != label and row.get('prediction') == label for row in rows)
        fn = sum(row['label'] == label and row.get('prediction') != label for row in rows)
        f1_values.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    result = {
        'total': len(rows), 'successful': len(successful), 'failed': len(rows) - len(successful),
        'accuracy': correct / len(rows), 'macro_f1': sum(f1_values) / len(f1_values),
        'initial_accuracy': sum(row['initial_prediction'] == row['label'] for row in successful) / len(rows),
        'prediction_correction': sum(row['initial_prediction'] != row['label'] and row['prediction'] == row['label'] for row in successful),
        'prediction_degradation': sum(row['initial_prediction'] == row['label'] and row['prediction'] != row['label'] for row in successful),
    }
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute accuracy, macro-F1, and prediction correction/degradation counts')
    parser.add_argument('path')
    args = parser.parse_args()
    print(json.dumps(evaluate(args.path), ensure_ascii=False, indent=2))
