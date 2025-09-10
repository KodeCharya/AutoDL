
import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

def generate_tabular_data(path, task, n_samples=1000, n_features=10):
    """Generates tabular data for classification or regression."""
    if task == 'classification':
        X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=5, n_redundant=0, n_classes=2, random_state=42)
    else:
        X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=5, noise=0.1, random_state=42)
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = y
    df.to_csv(path, index=False)

def generate_image_data(path, n_classes=2, n_images_per_class=10):
    """Generates a dummy image dataset."""
    from PIL import Image
    for i in range(n_classes):
        class_path = os.path.join(path, f'class_{i}')
        os.makedirs(class_path, exist_ok=True)
        for j in range(n_images_per_class):
            img = Image.new('RGB', (100, 100), color = (i*50, j*10, 0))
            img.save(os.path.join(class_path, f'image_{j}.png'))

def generate_text_data(path, n_samples=100, n_classes=2):
    """Generates a dummy text dataset."""
    with open(path, 'w') as f:
        for i in range(n_samples):
            label = i % n_classes
            f.write(f'label_{label} This is a sample text for class {label}.\n')

if __name__ == '__main__':
    os.makedirs('demo_data', exist_ok=True)
    generate_tabular_data('demo_data/tabular_classification.csv', 'classification')
    generate_tabular_data('demo_data/tabular_regression.csv', 'regression')
    generate_image_data('demo_data/image_data')
    generate_text_data('demo_data/text_data.txt')
    print("Demo data generated in 'demo_data' directory.")
