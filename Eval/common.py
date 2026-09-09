import argparse
import json
from pathlib import Path
import sys

# Support running scripts from the repository root or the Eval directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_utils import load_task


def arguments(name):
    parser = argparse.ArgumentParser(description=f'{name.upper()} text classification baseline')
    parser.add_argument('--task', choices=['trial', 'article', 'crime'], default='article')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--limit', type=int, default=0, help='Maximum samples per split; 0 uses all samples')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default=f'Output/{name}')
    parser.add_argument('--model', help='Pretrained model name or local path')
    if name == 'cnn':
        parser.add_argument('--finetune-bert', action='store_true', help='Fine-tune BERT jointly; BERT is frozen by default')
    args = parser.parse_args()
    if args.model is None:
        args.model = default_model(name, args.task)
    if min(args.epochs, args.batch_size, args.max_length) < 1 or args.lr <= 0 or args.limit < 0:
        parser.error('epochs, batch-size, max-length, and lr must be positive; limit must be nonnegative')
    if name == 'cnn' and args.max_length < 5:
        parser.error('CNN requires max-length of at least 5')
    return args


def default_model(name, task):
    if name == 'cnn':
        return 'bert-base-uncased' if task == 'trial' else 'bert-base-chinese'
    return {'legal_bert': 'nlpaueb/legal-bert-base-uncased',
            'lawformer': 'thunlp/Lawformer'}[name]


def prepare_data(args):
    data, labels = load_task(args.task)
    if args.limit:
        data = {split: rows[:args.limit] for split, rows in data.items()}
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory is not empty; choose a new directory with --output')
    output.mkdir(parents=True, exist_ok=True)
    mapping = {label: i for i, label in enumerate(labels)}
    texts = {split: [r['text'] for r in rows] for split, rows in data.items()}
    targets = {split: [mapping[r['label']] for r in rows] for split, rows in data.items()}
    save_json(output / 'config.json', vars(args))
    save_json(output / 'labels.json', labels)
    return data, texts, targets, labels, output


def save_json(path, value):
    with Path(path).open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def metrics(truth, scores):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    scores = np.asarray(scores)
    prediction = scores.argmax(axis=1)
    result = {'accuracy': float(accuracy_score(truth, prediction)),
              'macro_f1': float(f1_score(truth, prediction, average='macro', zero_division=0))}
    for k in (5, 10):
        top = np.argsort(scores, axis=1)[:, -min(k, scores.shape[1]):]
        result[f'top_{k}_accuracy'] = float(np.mean([label in row for label, row in zip(truth, top)]))
    return result


def save_results(output, rows, labels, scores):
    mapping = {label: i for i, label in enumerate(labels)}
    result = metrics([mapping[row['label']] for row in rows], scores)
    save_json(output / 'metrics.json', result)
    with (output / 'predictions.jsonl').open('w', encoding='utf-8') as f:
        for row, score in zip(rows, scores):
            prediction = labels[max(range(len(labels)), key=lambda i: score[i])]
            f.write(json.dumps({'id': row['id'], 'label': row['label'], 'prediction': prediction}, ensure_ascii=False) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
