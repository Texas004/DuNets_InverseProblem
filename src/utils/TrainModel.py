''' "TrainModel.py" is a module that contains functions for training machine learning models.
It includes functions for loading data, preprocessing data, training models, and evaluating model performance.
The trained models will be saved in the 'models' folder, and the training logs will be saved in the 'logs' folder.
'''

import os
import torch


def _forward_step(model, inputs, targets, device):
    """inputs / targets 都是张量；模型内部自带 sigma0 默认初始化。"""
    outputs = model(inputs.to(device))
    return outputs, targets.to(device)


def _batch_size(targets):
    return targets.size(0) if targets.dim() > 1 else 1


def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs, device, save_path, grad_accum_steps: int = 1):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    model.to(device)
    best_val_loss = float("inf")
    early_stopping_counter = 0
    grad_accum_steps = max(1, int(grad_accum_steps))

    for epoch in range(num_epochs):
        model.train()
        running_loss, n_train = 0.0, 0
        optimizer.zero_grad()
        accum = 0

        for inputs, targets in train_loader:
            outputs, targets = _forward_step(model, inputs, targets, device)
            raw_loss = criterion(outputs, targets)
            (raw_loss / grad_accum_steps).backward()

            bs = _batch_size(targets)
            running_loss += raw_loss.item() * bs
            n_train += bs
            accum += 1

            if accum >= grad_accum_steps:
                optimizer.step()
                optimizer.zero_grad()
                accum = 0

        if accum > 0:                      # 尾批残留
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss = running_loss / max(n_train, 1)

        # ---- validation ----
        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs, targets = _forward_step(model, inputs, targets, device)
                loss = criterion(outputs, targets)
                bs = _batch_size(targets)
                val_loss += loss.item() * bs
                n_val += bs
        val_loss /= max(n_val, 1)

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train: {epoch_loss:.6f}, Val: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model (val={best_val_loss:.6f})")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= 5:
                print("Early stopping triggered.")
                break