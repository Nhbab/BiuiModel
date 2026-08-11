import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def get_datasets(cfg):
    # If a CSV or data path is provided, you can extend this to load real data.
    # For demo purposes we generate a synthetic classification dataset.
    n_samples = cfg.get('n_samples', 2000)
    input_dim = cfg.get('input_dim', 20)
    n_classes = cfg.get('num_classes', 2)
    X, y = make_classification(n_samples=n_samples, n_features=input_dim, n_informative=10, n_redundant=2, n_classes=n_classes, random_state=42)
    X = X.astype('float32')
    y = y.astype('int64')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=cfg.get('val_split', 0.2), random_state=42)
    # convert to numpy arrays suitable for torch tensors in training script
    return X_train, y_train, X_val, y_val