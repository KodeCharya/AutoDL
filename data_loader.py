import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transformers import AutoTokenizer
import json

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ImageDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.dataset = ImageFolder(data_path, transform=transform)
        self.classes = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def get_tabular_loaders(data_path, test_size, stratify, batch_size, seed, task):
    """Creates DataLoaders for tabular data."""
    df = pd.read_csv(data_path)

    # Assume last column is target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Preprocessing
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    le = None
    if task == 'classification':
        le = LabelEncoder()
        y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y if stratify and task == 'classification' else None, random_state=seed
    )

    train_dataset = TabularDataset(X_train, y_train)
    test_dataset = TabularDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, (scaler, le)

def get_image_loaders(data_path, batch_size, test_size, seed):
    """Creates DataLoaders for image data."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageDataset(data_path, transform=transform)
    
    # Create train/test split
    train_idx, test_idx = train_test_split(
        list(range(len(dataset))),
        test_size=test_size,
        random_state=seed,
        stratify=dataset.dataset.targets if hasattr(dataset.dataset, 'targets') else None
    )

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, test_loader, dataset.classes

def get_text_loaders(data_path, batch_size, test_size, seed, tokenizer_name, max_length):
    """Creates DataLoaders for text data."""
    if data_path.endswith('.jsonl'):
        with open(data_path, 'r') as f:
            lines = f.readlines()
        data = [json.loads(line) for line in lines]
        texts = [d['text'] for d in data]
        labels = [d['label'] for d in data]
    elif data_path.endswith('.txt'):
        with open(data_path, 'r') as f:
            lines = f.readlines()
        # Assume label is first word
        labels = [line.split()[0] for line in lines]
        texts = [' '.join(line.split()[1:]) for line in lines]
    else: # directory of text files
        texts, labels = [], []
        for label in os.listdir(data_path):
            label_path = os.path.join(data_path, label)
            if os.path.isdir(label_path):
                for fname in os.listdir(label_path):
                    with open(os.path.join(label_path, fname), 'r') as f:
                        texts.append(f.read())
                        labels.append(label)

    le = LabelEncoder()
    labels = le.fit_transform(labels)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, le

def detect_data_type(data_path):
    """Detects the data type from the data_path."""
    if os.path.isfile(data_path) and data_path.endswith('.csv'):
        return 'tabular'
    elif os.path.isdir(data_path):
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    return 'image'
                if file.endswith('.txt'):
                    return 'text'
    elif os.path.isfile(data_path) and data_path.endswith('.jsonl'):
        return 'text'
    elif os.path.isfile(data_path) and data_path.endswith('.txt'):
        return 'text'

    raise ValueError(f"Unsupported data type or path: {data_path}")