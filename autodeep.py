
import argparse
import yaml
import pandas as pd
import os
from utils import set_seed, get_logger
from data_loader import detect_data_type, get_tabular_loaders, get_image_loaders, get_text_loaders
from models import get_tabular_model, get_image_model, get_text_model
from trainer import Trainer
import torch
import joblib

def main():
    parser = argparse.ArgumentParser(description='AutoDL-deep: Automatic Deep Learning')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to the config file')
    parser.add_argument('--task', type=str, help='Task type (classification or regression)')
    parser.add_argument('--device', type=str, help='Device to use (cpu or cuda)')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--model_name', type=str, help='Name of the model to use')
    parser.add_argument('--model_path', type=str, default='final_model', help='Path to save the model')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with CLI args
    if args.task: config['task'] = args.task
    if args.device: config['device'] = args.device
    if args.epochs: config['training']['epochs'] = args.epochs
    if args.batch_size: config['training']['batch_size'] = args.batch_size
    if args.lr: config['training']['learning_rate'] = args.lr
    if args.model_name: config['model']['name'] = args.model_name
    config['data_path'] = args.data_path
    config['model_path'] = args.model_path

    # Setup
    set_seed(config['seed'])
    logger = get_logger(config['logging']['log_file'], config['logging']['level'])
    device = config['device'] if config['device'] != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(config['model_path']):
        os.makedirs(config['model_path'])

    # Data Loading
    data_type = detect_data_type(config['data_path'])
    logger.info(f"Detected data type: {data_type}")

    if data_type == 'tabular':
        df = pd.read_csv(config['data_path'])
        if len(df.iloc[:, -1].unique()) < 10:
            task = 'classification'
            output_dim = len(df.iloc[:, -1].unique())
        else:
            task = 'regression'
            output_dim = 1

        train_loader, test_loader, (scaler, le) = get_tabular_loaders(
            config['data_path'],
            config['test_size'],
            config['stratify'],
            config['training']['batch_size'],
            config['seed'],
            task
        )
        input_dim = scaler.n_features_in_
        model = get_tabular_model(input_dim, output_dim, config['model']['tabular']['mlp']['layers'], config['model']['tabular']['mlp']['dropout'])
        joblib.dump(scaler, f"{config['model_path']}/scaler.joblib")
        if le:
            joblib.dump(le, f"{config['model_path']}/label_encoder.joblib")

    elif data_type == 'image':
        train_loader, test_loader, classes = get_image_loaders(
            config['data_path'],
            config['training']['batch_size'],
            config['test_size'],
            config['seed']
        )
        num_classes = len(classes)
        model = get_image_model(config['model']['image']['name'], num_classes, config['model']['image']['pretrained'], config['model']['image']['freeze_backbone'])
        task = 'classification'
    elif data_type == 'text':
        train_loader, test_loader, le = get_text_loaders(
            config['data_path'],
            config['training']['batch_size'],
            config['test_size'],
            config['seed'],
            config['model']['text']['name'],
            config['model']['text']['max_length']
        )
        num_classes = len(le.classes_)
        model = get_text_model(config['model']['text']['name'], num_classes)
        task = 'text'
        joblib.dump(le, f"{config['model_path']}/label_encoder.joblib")

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])

    trainer = Trainer(
        model,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        device,
        config['training']['epochs'],
        config['training']['mixed_precision'],
        task,
        config['model_path']
    )

    trainer.train()

if __name__ == '__main__':
    main()
