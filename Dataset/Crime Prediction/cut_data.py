# -*- coding: UTF-8 -*-
train_path = "accusation_train.json"
test_path = "accusation_test.json"
valid_path = "accusation_valid.json"
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# only save 1/100 and save as "present_data.json"

def load_and_sample_data(file_path, fraction=0.01):
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f'Loading {file_path}'):
                data = json.loads(line)
                texts.append(data['fact'])
                labels.append(data['accusation'])
    total_samples = int(len(texts) * fraction)
    sampled_texts = texts[:total_samples]
    sampled_labels = labels[:total_samples]
    return sampled_texts, sampled_labels

X_train, y_train = load_and_sample_data(train_path)
X_valid, y_valid = load_and_sample_data(valid_path)
X_test, y_test = load_and_sample_data(test_path)

# save as "present_train_data.json"

with open("present_train_data.json", "w", encoding='utf-8') as f:
    for text, label in zip(X_train, y_train):
        f.write(json.dumps({"fact": text, "accusation": label},ensure_ascii=False) + "\n")

# save as "present_valid_data.json"

with open("present_valid_data.json", "w", encoding='utf-8') as f:
    for text, label in zip(X_valid, y_valid):
        f.write(json.dumps({"fact": text, "accusation": label},ensure_ascii=False) + "\n")

# save as "present_test_data.json"

with open("present_test_data.json", "w", encoding='utf-8') as f:
    for text, label in zip(X_test, y_test):
        f.write(json.dumps({"fact": text, "accusation": label},ensure_ascii=False) + "\n")

print("Data preprocessing successful!")
print("Data saved successfully!")
print("length of train data: ", len(X_train))
print("length of valid data: ", len(X_valid))
print("length of test data: ", len(X_test))
print("Data saved successfully!")




