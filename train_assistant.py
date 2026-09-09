import argparse
import json
from pathlib import Path

from data_utils import load_task


def load_examples(path, split, task):
    data, _ = load_task(task)
    allowed = {row['text']: row['label'] for row in data[split]}
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            config = row['config']
            if config['task'] != task or config['split'] != split or config['method'] != 'debate-single':
                raise ValueError(f"{path} must contain debate-single results for {task}/{split}")
            if row['text'] not in allowed or row['label'] != allowed[row['text']]:
                raise ValueError("Sample split or label does not match the dataset")
            if row['status'] == 'ok':
                records[row['id']] = row
    examples = []
    for row in records.values():
        for debate_round in row['rounds']:
            for comment in debate_round['comments']:
                # Use 1 when the expert position matches the ground truth, and 0 otherwise.
                examples.append((row['text'], f"Position: {comment['label']}\nOpinion: {comment['opinion']}",
                                 int(comment['label'] == row['label'])))
    if not examples:
        raise ValueError(f"{path} contains no usable debate comments")
    return examples


def main():
    parser = argparse.ArgumentParser(description="Train a reliability classifier from unassisted debate records")
    parser.add_argument('--task', choices=['trial', 'article', 'crime'], default='trial')
    parser.add_argument('--train', required=True)
    parser.add_argument('--valid', required=True)
    parser.add_argument('--model', default='bert-base-multilingual-cased')
    parser.add_argument('--output', default='assistant_model')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error('epochs, batch-size, and lr must be positive')
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory is not empty; choose a new model directory')
    train = load_examples(args.train, 'train', args.task)
    valid = load_examples(args.valid, 'valid', args.task)
    if len({x[2] for x in train}) < 2:
        raise ValueError('Training comments must include correct and incorrect positions; generate more samples')
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
    from sklearn.metrics import accuracy_score, f1_score

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, id2label={0: 'incorrect', 1: 'reliable'},
        label2id={'incorrect': 0, 'reliable': 1},
    ).to(device)

    def collate(batch):
        cases, opinions, labels = zip(*batch)
        inputs = tokenizer(list(cases), list(opinions), padding=True, truncation=True,
                           max_length=512, return_tensors='pt')
        inputs['labels'] = torch.tensor(labels)
        return inputs

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid, batch_size=args.batch_size, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_f1 = -1
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        truth, prediction = [], []
        with torch.no_grad():
            for batch in valid_loader:
                truth.extend(batch['labels'].tolist())
                batch = {key: value.to(device) for key, value in batch.items()}
                prediction.extend(model(**batch).logits.argmax(dim=-1).cpu().tolist())
        f1 = f1_score(truth, prediction, labels=[0, 1], average='macro', zero_division=0)
        print(f"epoch={epoch + 1} loss={total_loss / len(train_loader):.4f} "
              f"valid_acc={accuracy_score(truth, prediction):.4f} valid_macro_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(output)
            tokenizer.save_pretrained(output)
    with (output / 'training_config.json').open('w', encoding='utf-8') as f:
        json.dump(dict(vars(args), train_examples=len(train), valid_examples=len(valid),
                       best_valid_macro_f1=best_f1), f, ensure_ascii=False, indent=2)
    print(f'Assistant model saved to {output}')


if __name__ == '__main__':
    main()
