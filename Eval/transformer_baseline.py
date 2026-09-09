from common import arguments, prepare_data, metrics, save_results


def run(name):
    args = arguments(name)
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, texts, targets, labels, output = prepare_data(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_class = AutoModelForSequenceClassification
    if name == 'cnn':
        from cnn_model import BertCNN
        model_class = BertCNN
    model = model_class.from_pretrained(
        args.model, num_labels=len(labels), id2label=dict(enumerate(labels)),
        label2id={label: i for i, label in enumerate(labels)},
    ).to(device)
    if name == 'cnn':
        model.config.finetune_bert = args.finetune_bert
        model.bert.requires_grad_(args.finetune_bert)
    # Longformer position indices start after padding_idx.
    position_limit = model.config.max_position_embeddings
    if model.config.model_type == 'longformer':
        position_limit -= model.config.pad_token_id + 1
    if args.max_length > position_limit:
        raise ValueError(f'max-length exceeds the model limit {position_limit}')
    is_longformer = model.config.model_type == 'longformer'

    def collate(batch):
        cases, truth = zip(*batch)
        inputs = tokenizer(list(cases), padding='max_length' if name == 'cnn' else True, truncation=True,
                           max_length=args.max_length, return_tensors='pt')
        if is_longformer:
            # Use global attention for the classification token and local attention elsewhere.
            inputs['global_attention_mask'] = torch.zeros_like(inputs['input_ids'])
            inputs['global_attention_mask'][:, 0] = 1
        if is_longformer:
            inputs.pop('token_type_ids', None)
        inputs['labels'] = torch.tensor(truth)
        return {key: value.to(device) for key, value in inputs.items()}

    loaders = {split: DataLoader(list(zip(texts[split], targets[split])), batch_size=args.batch_size,
                                 shuffle=split == 'train', collate_fn=collate) for split in data}

    def predict(loader):
        model.eval()
        scores = []
        with torch.no_grad():
            for batch in loader:
                scores.extend(model(**batch).logits.cpu().tolist())
        return scores

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    best_f1 = -1
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0
        for batch in loaders['train']:
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        result = metrics(targets['valid'], predict(loaders['valid']))
        print(f"epoch={epoch + 1} loss={loss_sum / len(loaders['train']):.4f} valid={result}")
        if result['macro_f1'] > best_f1:
            best_f1 = result['macro_f1']
            model.save_pretrained(output / 'model')
            tokenizer.save_pretrained(output / 'model')
    # Evaluate the best validation checkpoint once on the test set.
    del model
    model = model_class.from_pretrained(output / 'model').to(device)
    save_results(output, data['test'], labels, predict(loaders['test']))
