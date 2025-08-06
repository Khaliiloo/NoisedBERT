# Deveoped by Khalil Abdulgawad
# For my research paper that will be published in Springer Nature

import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
import torch
import torch_optimizer as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, \
    ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ArabicToxicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return input_ids, attention_mask, label


def collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])
    attention_mask = torch.stack([item[1] for item in batch])
    labels = torch.tensor([item[2] for item in batch])
    return input_ids, attention_mask, labels


def preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data['label'] = data['Majority_Label'].apply(lambda x: 1 if x == 'Offensive' else 0)
    return data


def oversample_data(data):
    X = data['Comment']
    y = data['label']
    ros = RandomOverSampler(random_state=42)
    x_resampled, y_resampled = ros.fit_resample(X.values.reshape(-1, 1), y)
    balanced_data = pd.DataFrame({
        'Comment': x_resampled.flatten(),
        'label': y_resampled
    })
    return balanced_data


def introduce_label_noise(data, noise_level=0.1):
    data['label'] = pd.to_numeric(data['label'], errors='coerce')
    num_to_flip = int(len(data) * noise_level)
    # flip_indices = data.index[0:num_to_flip]
    # data.loc[flip_indices, 'label'] = 1 - data.loc[flip_indices, 'label']
    # return data
    flip_indices = np.random.choice(
        data.index,
        size=num_to_flip,
        replace=False
    )
    data.loc[flip_indices, 'label'] = 1 - data.loc[flip_indices, 'label']
    return data


def plot_confusion_matrix(labels, predictions):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Offensive', 'Offensive'],
                yticklabels=['Non-Offensive', 'Offensive'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


def plot_two_confusion_matrices(labels1, predictions1, labels2, predictions2, noise_level=0.1):
    cm1 = confusion_matrix(labels1, predictions1)
    cm2 = confusion_matrix(labels2, predictions2)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot the first confusion matrix
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Offensive', 'Offensive'],
                yticklabels=['Non-Offensive', 'Offensive'], ax=axes[0], annot_kws={"size": 16})
    axes[0].set_xlabel('Predicted Labels', fontsize=16)
    axes[0].set_ylabel('True Labels', fontsize=16)
    axes[0].set_title('Noise ' + str(int(100*noise_level)) + '% on Unbalanced Data', fontsize=20)
    axes[0].tick_params(axis='both', labelsize=14)

    # Plot the second confusion matrix
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Offensive', 'Offensive'],
                yticklabels=['Non-Offensive', 'Offensive'], ax=axes[1], annot_kws={"size": 16})
    axes[1].set_xlabel('Predicted Labels', fontsize=16)
    axes[1].set_ylabel('True Labels', fontsize=16)
    axes[1].set_title('Noise ' + str(int(100*noise_level)) + '% on Balanced Data', fontsize=20)
    axes[1].tick_params(axis='both', labelsize=14)

    plt.tight_layout()
    plt.show()


def train_and_evaluate(data, model, tokenizer, device, epochs=10, batch_size=30, noise_level=0.1):
    train_data, test_data = train_test_split(data, test_size=0.4, random_state=42, shuffle=True)
    # print('train_data', train_data.loc[train_data['label']])
    # Introduce label noise
    train_data = introduce_label_noise(train_data, noise_level)
    train_texts = train_data['Comment'].tolist()
    train_labels = train_data['label'].tolist()
    test_texts = test_data['Comment'].tolist()
    test_labels = test_data['label'].tolist()

    train_dataset = ArabicToxicDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # optimizer = AdamW(model.parameters(), lr=2e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    model.train()
    for epoch in range(epochs):
        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    test_dataset = ArabicToxicDataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

    return all_labels, all_predictions



def main():
    data = preprocess_data('ardata.csv')
    unbalanced_data = data.copy()
    # Oversample the data
    balanced_data = oversample_data(data)
    # Load the BERT model and tokenizer
    model = BertForSequenceClassification.from_pretrained('asafaya/bert-base-arabic', num_labels=2)
    tokenizer = BertTokenizer.from_pretrained('asafaya/bert-base-arabic')

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    # Train and evaluate with label noise
    print("Training and evaluating with label noise...")
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for i in noise_levels:
        print(f"Training and evaluating with noise level: {i * 0.1}")
        all_labels_unbalanced, all_predictions_unbalanced = train_and_evaluate(unbalanced_data, model, tokenizer, device,
                                                                               noise_level=i * 0.1)

        # Train and evaluate on oversampled data with label noise
        print("Training and evaluating on balanced data with label noise...")
        all_labels_balanced, all_predictions_balanced = train_and_evaluate(balanced_data, model, tokenizer, device,
                                                                           noise_level=i * 0.1)
        # plot_confusion_matrix(all_labels_balanced, all_predictions_balanced)
        plot_two_confusion_matrices(all_labels_unbalanced, all_predictions_unbalanced, all_labels_balanced,
                                    all_predictions_balanced, noise_level=i * 0.1)


if __name__ == "__main__":
    main()
