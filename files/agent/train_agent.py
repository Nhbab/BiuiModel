import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from agent.data import get_datasets
from agent.model import SimpleMLP
from agent.utils import save_checkpoint, maybe_upload


def train(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train, y_train, X_val, y_val = get_datasets(cfg)
    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(list(zip(X_val, y_val)), batch_size=cfg['batch_size'], shuffle=False)

    model = SimpleMLP(cfg['input_dim'], cfg['hidden_dim'], cfg['num_classes']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device).long()
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
        train_loss = total_loss / total
        train_acc = correct / total

        # validation
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device).float()
                yb = yb.to(device).long()
                logits = model(xb)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)
        val_acc = correct / total

        print(f"Epoch {epoch}/{cfg['epochs']} - train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} val_acc: {val_acc:.4f}")

    os.makedirs(cfg['output_dir'], exist_ok=True)
    checkpoint_path = os.path.join(cfg['output_dir'], cfg.get('checkpoint_name', 'model.pt'))
    save_checkpoint(model, optimizer, cfg, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    # optionally upload to moithub (pluggable)
    maybe_upload(checkpoint_path, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='train/config.yaml')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--output-dir', type=str)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    if args.epochs:
        cfg['epochs'] = args.epochs
    if args.batch_size:
        cfg['batch_size'] = args.batch_size
    if args.output_dir:
        cfg['output_dir'] = args.output_dir

    train(cfg)