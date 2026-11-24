''' "TrainModel.py" is a module that contains functions for training machine learning models.
It includes functions for loading data, preprocessing data, training models, and evaluating model performance.
The trained models will be saved in the 'models' folder, and the training logs will be saved in the 'logs' folder.
'''

import os
import torch
import time


def _forward_step(model, inputs, targets, device):
    outputs = model(inputs.to(device))
    return outputs, targets.to(device)


def _batch_size(targets):
    return targets.size(0) if targets.dim() > 1 else 1

def _relative_error(outputs, targets):

    if outputs.dim() == 1:
        out2 = outputs.reshape(1, -1)
        gt2 = targets.reshape(1, -1)
    else:
        out2 = outputs.reshape(outputs.shape[0], -1)
        gt2 = targets.reshape(targets.shape[0], -1)
    diff_norm = torch.norm(out2 - gt2, p=2, dim=1)
    gt_norm = torch.norm(gt2, p=2, dim=1) + 1e-12
    return diff_norm / gt_norm


def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs, device, save_path, grad_accum_steps: int = 1):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    model.to(device)
    best_rel_loss = float("inf")
    early_stopping_counter = 0
    grad_accum_steps = max(1, int(grad_accum_steps))

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss, n_train = 0.0, 0
        optimizer.zero_grad()
        accum = 0
        rel_loss = 0.0

        print(f"Epoch {epoch+1}/{num_epochs} - Training...")

        for inputs, targets in train_loader:
            outputs, targets = _forward_step(model, inputs, targets, device)
            raw_loss = _relative_error(outputs, targets).mean()
            (raw_loss / grad_accum_steps).backward()

            bs = _batch_size(targets)
            running_loss += raw_loss.item() * bs
            rel_loss += raw_loss.item() * bs
            n_train += bs
            accum += 1

            if accum >= grad_accum_steps:
                optimizer.step()
                optimizer.zero_grad()
                accum = 0

        if accum > 0:                      
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss = running_loss / max(n_train, 1)
        epoch_rel = rel_loss / max(n_train, 1) 

        # ---- validation ----
        model.eval()
        rel_loss, n_val = 0.0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs, targets = _forward_step(model, inputs, targets, device)
                bs = _batch_size(targets)
                rel_loss += _relative_error(outputs, targets).mean().item() * bs
                n_val += bs
        rel_loss /= max(n_val, 1)
        

        print(f"Train: {epoch_loss:.6f}, average relative error: {epoch_rel:.6f},"
              f"Time: {time.time() - start_time:.2f}s\n\n")

        if rel_loss < best_rel_loss:
            best_rel_loss = rel_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model (rel={best_rel_loss:.6f})")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= 5:
                print("Early stopping triggered.")
                break
