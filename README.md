
# AutoDL-deep

A Python AutoDL framework that, given only a dataset path, automatically detects the data type (tabular CSV, image folder, text files / JSON), builds a deep-learning-first training pipeline using modern best practices, trains, evaluates, and saves a ready-to-serve model and preprocessing pipeline.

## Features

- **Automatic Data Type Detection:** Automatically detects tabular, image, and text data.
- **Deep Learning First:** Uses modern deep learning models for all data types.
- **Best Practices:** Implements best practices like mixed precision training, learning rate scheduling, and early stopping.
- **Extensible:** Easily extensible with new models and data loaders.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kodechrya/AutoDL.git
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Generate Demo Data (Optional):**
   ```bash
   python generate_demo_data.py
   ```

2. **Run Training:**
   ```bash
   python autodeep.py --data_path /path/to/your/data
   ```

   For example, to train on the generated demo data:
   ```bash
   python autodeep.py --data_path demo_data/tabular_classification.csv
   ```

## CLI Arguments

- `--data_path`: Path to the data (required).
- `--config`: Path to the config file (default: `configs/default.yaml`).
- `--task`: Task type (e.g., `classification`, `regression`). Default: `auto`.
- `--device`: Device to use (e.g., `cpu`, `cuda`). Default: `auto`.
- `--epochs`: Number of epochs.
- `--batch_size`: Batch size.
- `--lr`: Learning rate.
- `--model_name`: Name of the model to use.

## Model Export

The trained model and preprocessors are saved in the `final_model/` directory. An `inference.py` script is provided for making predictions on new data.

## How to use `inference.py`

```python
from inference import predict

predictions = predict('path/to/new_data.csv', 'final_model')
print(predictions)
```
