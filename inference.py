import torch
import joblib
import pandas as pd
from torchvision import transforms
from transformers import AutoTokenizer
import argparse
import os

from models import TabularMLP, get_image_model, get_text_model
from data_loader import TabularDataset, ImageDataset, TextDataset, detect_data_type

def predict(data_path, model_path):
    """Makes predictions on new data."""
    data_type = detect_data_type(data_path)

    if data_type == 'tabular':
        model_state = torch.load(f"{model_path}/model.pth")
        scaler = joblib.load(f"{model_path}/scaler.joblib")
        le = joblib.load(f"{model_path}/label_encoder.joblib")

        df = pd.read_csv(data_path)
        X = scaler.transform(df.values)
        model = TabularMLP(X.shape[1], len(le.classes_), [128, 64], 0.1)
        model.load_state_dict(model_state)
        model.eval()

        with torch.no_grad():
            outputs = model(torch.tensor(X, dtype=torch.float32))
            preds = torch.argmax(outputs, dim=1)
            return le.inverse_transform(preds.numpy())

    elif data_type == 'image':
        model_state = torch.load(f"{model_path}/model.pth")
        # we need to know the number of classes from the saved model
        # for now, we assume it is 2
        model = get_image_model('resnet50', 2)
        model.load_state_dict(model_state)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        from PIL import Image
        image = Image.open(data_path)
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image)
            preds = torch.argmax(outputs, dim=1)
            return preds.numpy()

    elif data_type == 'text':
        model_state = torch.load(f"{model_path}/model.pth")
        le = joblib.load(f"{model_path}/label_encoder.joblib")
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        model = get_text_model('distilbert-base-uncased', len(le.classes_))
        model.load_state_dict(model_state)
        model.eval()

        with open(data_path, 'r') as f:
            text = f.read()

        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        with torch.no_grad():
            outputs = model(encoding['input_ids'], attention_mask=encoding['attention_mask'])
            preds = torch.argmax(outputs.logits, dim=1)
            return le.inverse_transform(preds.numpy())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data to predict on')
    parser.add_argument('--model_path', type=str, default='final_model', help='Path to the saved model')
    args = parser.parse_args()

    predictions = predict(args.data_path, args.model_path)
    print(predictions)