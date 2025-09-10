
import torch
from tqdm import tqdm
from utils import get_classification_metrics, get_regression_metrics, save_model
import os

class Trainer:
    def __init__(self, model, train_loader, test_loader, optimizer, scheduler, device, epochs, mixed_precision, task, model_path):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.epochs = epochs
        self.mixed_precision = mixed_precision
        self.task = task
        self.scaler = torch.amp.GradScaler(enabled=mixed_precision)
        self.model_path = model_path

    def train(self):
        """Trains the model."""
        self.model.to(self.device)
        for epoch in range(self.epochs):
            self.model.train()
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                if self.task == 'text':
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    with torch.amp.autocast(device_type=self.device, enabled=self.mixed_precision):
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss
                else:
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    with torch.amp.autocast(device_type=self.device, enabled=self.mixed_precision):
                        outputs = self.model(inputs)
                        if self.task == 'classification':
                            loss = torch.nn.functional.cross_entropy(outputs, labels.long())
                        else: # regression
                            loss = torch.nn.functional.mse_loss(outputs, labels.unsqueeze(1))

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            self.scheduler.step()
            self.evaluate(epoch)
        
        self.save_model()

    def evaluate(self, epoch):
        """Evaluates the model."""
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in self.test_loader:
                if self.task == 'text':
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = torch.argmax(outputs.logits, dim=1)
                else:
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    if self.task == 'classification':
                        preds = torch.argmax(outputs, dim=1)
                    else: # regression
                        preds = outputs.squeeze()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        if self.task == 'classification':
            metrics = get_classification_metrics(all_labels, all_preds)
            print(f"Epoch {epoch+1} - Validation metrics: {metrics}")
        else:
            metrics = get_regression_metrics(all_labels, all_preds)
            print(f"Epoch {epoch+1} - Validation metrics: {metrics}")

    def save_model(self):
        """Saves the model."""
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        save_model(self.model, self.model_path)
