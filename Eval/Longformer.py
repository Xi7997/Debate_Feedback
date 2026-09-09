import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import LongformerTokenizer, LongformerForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW
from torch.nn import DataParallel


train_file_path = '../Dataset/Legal Document Prediction/article_train.json'
valid_file_path = '../Dataset/Legal Document Prediction/article_valid.json'
test_file_path = '../Dataset/Legal Document Prediction/article_test.json'


def load_and_sample_data(file_path, fraction=0.1):
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f'Loading {file_path}'):
            data = json.loads(line)
            texts.append(data['fact'])  # 使用'fact'字段
            labels.append(data['relevant_articles'])  # 使用'accusation'字段

    total_samples = int(len(texts) * fraction)
    sampled_texts = texts[:total_samples]
    sampled_labels = labels[:total_samples]

    return sampled_texts, sampled_labels



X_train, y_train = load_and_sample_data(train_file_path)
X_valid, y_valid = load_and_sample_data(valid_file_path)
X_test, y_test = load_and_sample_data(test_file_path)



def create_label_mapping(labels):
    unique_labels = list(set(labels))
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    return label_mapping



label_mapping = create_label_mapping(y_train + y_valid + y_test)

y_train_encoded = [label_mapping[label] for label in y_train]
y_valid_encoded = [label_mapping[label] for label in y_valid]
y_test_encoded = [label_mapping[label] for label in y_test]

print("Data preprocessing successful!")

tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')



def encode_texts(texts, max_length=512):
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')


print("Begin to encoding")

train_encodings = encode_texts(X_train)
valid_encodings = encode_texts(X_valid)
test_encodings = encode_texts(X_test)
print("Encoding success!")

train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'],
                              torch.tensor(y_train_encoded))
valid_dataset = TensorDataset(valid_encodings['input_ids'], valid_encodings['attention_mask'],
                              torch.tensor(y_valid_encoded))
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'],
                             torch.tensor(y_test_encoded))
print("Begin to start dataloader.")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096',
                                                            num_labels=len(label_mapping))
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = DataParallel(model)
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)



def train_model(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids, attention_mask, labels = [item.to(device) for item in batch]

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss.mean()
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(dataloader)



def evaluate_model(model, dataloader):
    model.eval()
    preds, true_labels = [], []
    for batch in dataloader:
        input_ids, attention_mask, labels = [item.to(device) for item in batch]
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
    return preds, true_labels



epochs = 5
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_loss = train_model(model, train_loader, optimizer)
    print(f"Epoch {epoch + 1} Loss: {train_loss:.4f}")


    y_valid_pred, y_valid_true = evaluate_model(model, valid_loader)
    valid_accuracy = accuracy_score(y_valid_true, y_valid_pred)
    valid_f1 = f1_score(y_valid_true, y_valid_pred, average='weighted')
    print(f"Validation Accuracy: {valid_accuracy:.2f}, Validation F1 Score: {valid_f1:.2f}")


y_pred, y_true = evaluate_model(model, test_loader)



def top_k_accuracy(y_true, y_pred_prob, k=5):
    top_k_preds = np.argsort(y_pred_prob, axis=1)[:, -k:]
    return np.mean([y_true[i] in top_k_preds[i] for i in range(len(y_true))])


test_logits = []
model.eval()
for batch in test_loader:
    input_ids, attention_mask, labels = [item.to(device) for item in batch]
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits.cpu()
    test_logits.append(logits)


test_logits = torch.cat(test_logits, dim=0).numpy()


top_5_accuracy = top_k_accuracy(y_test_encoded, np.array(test_logits), k=5)
top_10_accuracy = top_k_accuracy(y_test_encoded, np.array(test_logits), k=10)


accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')

print(f'Test Accuracy: {accuracy:.2f}')
print(f'Test F1 Score: {f1:.2f}')
print(f'Top-5 Accuracy: {top_5_accuracy:.2f}')
print(f'Top-10 Accuracy: {top_10_accuracy:.2f}')


output_file = 'longformer_multiclass_predictions.txt'
with open(output_file, 'w') as f:
    for truth, pred in zip(y_true, y_pred):
        f.write(f"{truth}   {pred}\n")  # 每行写入 "Ground_truth   Predict"

print(f'Predictions saved to {output_file}')


if isinstance(model, DataParallel):
    model.module.save_pretrained('longformer_multiclass_classification_model_with_gpus')
else:
    model.save_pretrained('longformer_multiclass_classification_model')
