
import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModelForSequenceClassification

class TabularMLP(nn.Module):
    def __init__(self, input_dim, output_dim, layers, dropout):
        super().__init__()
        
        layer_list = []
        for i, layer_size in enumerate(layers):
            if i == 0:
                layer_list.append(nn.Linear(input_dim, layer_size))
            else:
                layer_list.append(nn.Linear(layers[i-1], layer_size))
            layer_list.append(nn.ReLU())
            layer_list.append(nn.Dropout(dropout))
            
        self.layers = nn.Sequential(*layer_list)
        self.output = nn.Linear(layers[-1], output_dim)

    def forward(self, x):
        x = self.layers(x)
        x = self.output(x)
        return x

def get_image_model(model_name, num_classes, pretrained=True, freeze_backbone=True):
    """Creates a pretrained image model."""
    if model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    if model_name == "resnet50":
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "efficientnet_b0":
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    return model

def get_text_model(model_name, num_classes):
    """Creates a pretrained text model."""
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
    return model

def get_tabular_model(input_dim, output_dim, layers, dropout):
    """Creates a tabular model."""
    return TabularMLP(input_dim, output_dim, layers, dropout)
