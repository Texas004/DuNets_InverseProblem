import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from config.config import DATA_DIR, WEIGHTS_DIR
from model.lcg_net import LCGNet
from utils.TrainModel import train_model


def _ensure_data_file(filepath, num_samples, dimension):
    if os.path.exists(filepath):
        print(f"Data file {filepath} found.")
        return True

    agree = input(
        f"Data file {filepath} not found, do you want to generate it now? (y/n): "
    )
    if agree.lower() != "y":
        print("Please run `python src/maincg.py` and answer y to generate the data.")
        return False

    import data.inteq_data as cg_data

    matrix = np.random.rand(dimension, dimension)
    measurements, targets = cg_data.generate_data(matrix, num_samples, dimension)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez_compressed(
        filepath,
        A=matrix.astype(np.float32),
        phi=targets.astype(np.float32),
        f=measurements.astype(np.float32),
    )
    print(f"Saved CG data to {filepath}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Train the LCG model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-9)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--dimension", type=int, default=20)
    parser.add_argument("--Lambda", type=float, default=1e-5)
    parser.add_argument("--iter", type=int, default=30)
    args = parser.parse_args()
    print(args)

    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")
    if args.num_samples < 2:
        raise ValueError("--num-samples must be at least 2")
    if args.dimension < 1:
        raise ValueError("--dimension must be positive")

    filepath = os.path.join(DATA_DIR, "CGdata.npz")
    if not _ensure_data_file(filepath, args.num_samples, args.dimension):
        return

    with np.load(filepath) as npzfile:
        matrix = npzfile["A"].astype(np.float32)
        measurements = npzfile["f"].astype(np.float32)
        targets = npzfile["phi"].astype(np.float32)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"A must be a square matrix; got shape {matrix.shape}")
    if measurements.ndim != 2 or targets.shape != measurements.shape:
        raise ValueError(
            "f and phi must be 2D arrays with matching shapes; "
            f"got f={measurements.shape}, phi={targets.shape}"
        )
    if measurements.shape[0] != matrix.shape[0]:
        raise ValueError(
            f"A dimension {matrix.shape[0]} does not match data dimension "
            f"{measurements.shape[0]}"
        )

    dimension, sample_count = targets.shape
    if sample_count < 2:
        raise ValueError(f"CG data must contain at least 2 samples; got {sample_count}")
    print(f"Data: Samples={sample_count}, Dimension={dimension}")

    dataset = TensorDataset(
        torch.from_numpy(measurements.T.copy()),
        torch.from_numpy(targets.T.copy()),
    )
    val_size = min(
        max(1, int(round(args.val_ratio * len(dataset)))),
        len(dataset) - 1,
    )
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(2026),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )

    criterion = nn.MSELoss()
    torch.manual_seed(2026)
    num_epochs = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrix_tensor = torch.from_numpy(matrix).to(device=device)
    A_reg = matrix_tensor.T @ matrix_tensor + args.Lambda * torch.eye(args.dimension, device = device)
    A_T = matrix_tensor.T
    model = LCGNet(A_reg=A_reg, A_T=A_T, step=dimension)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    save_path = os.path.join(WEIGHTS_DIR, "lcg_best.pt")
    print(f"Training on {device}; model weights will be saved to {save_path}")

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
