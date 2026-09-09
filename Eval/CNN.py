import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


train_file_path = '../Dataset/Legal Document Prediction/article_train.json'
valid_file_path = '../Dataset/Legal Document Prediction/article_valid.json'
test_file_path = '../Dataset/Legal Document Prediction/article_test.json'



def load_and_sample_data(file_path, fraction=0.25):
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f'Loading {file_path}'):
            data = json.loads(line)
            texts.append(data['fact'])
            #             labels.append(data['accusation'])  # 使用'accusation'字段
            labels.append(data['relevant_articles'])


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


max_vocab_size = 10000
max_sequence_length = 500

tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_valid_seq = tokenizer.texts_to_sequences(X_valid)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length, padding='post', truncating='post')
X_valid_pad = pad_sequences(X_valid_seq, maxlen=max_sequence_length, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length, padding='post', truncating='post')


strategy = tf.distribute.MirroredStrategy()

with strategy.scope():

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=max_vocab_size, output_dim=128, input_length=max_sequence_length),
        tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(label_mapping), activation='softmax')
    ])


    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


history = model.fit(X_train_pad, np.array(y_train_encoded),
                    epochs=10,
                    batch_size=32,
                    validation_data=(X_valid_pad, np.array(y_valid_encoded)))


y_pred_prob = model.predict(X_test_pad)
y_pred = np.argmax(y_pred_prob, axis=1)



def top_k_accuracy(y_true, y_pred_prob, k=5):
    top_k_preds = np.argsort(y_pred_prob, axis=1)[:, -k:]
    return np.mean([y_true[i] in top_k_preds[i] for i in range(len(y_true))])


top_5_accuracy = top_k_accuracy(y_test_encoded, y_pred_prob, k=5)
top_10_accuracy = top_k_accuracy(y_test_encoded, y_pred_prob, k=10)


accuracy = accuracy_score(y_test_encoded, y_pred)
f1 = f1_score(y_test_encoded, y_pred, average='weighted')

print(f'Test Accuracy: {accuracy:.2f}')
print(f'Test F1 Score: {f1:.2f}')
print(f'Top-5 Accuracy: {top_5_accuracy:.2f}')
print(f'Top-10 Accuracy: {top_10_accuracy:.2f}')


output_file = 'multiclass_predictions.txt'
with open(output_file, 'w') as f:
    for truth, pred in zip(y_test_encoded, y_pred):
        f.write(f"{truth}   {pred}\n")  # 每行写入 "Ground_truth   Predict"

print(f'Predictions saved to {output_file}')


model.save('cnn_multiclass_classification_model.h5')

