import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config.config as cfg
from utils.TrainModel import train_model
import model.lpd_net as Model_Module


# ---------- Dataset ----------
class EITDataset(torch.utils.data.Dataset):
    """y -> conductivity_gt。init conductivity by LPDNet.forward(sigma0=None) beginning ones."""
    def __init__(self, y_all, sigma_gt_all):
        self.y_all       = y_all              # (S, m)
        self.sigma_gt_all = sigma_gt_all      # (S, n)

    def __len__(self):
        return self.y_all.shape[0]

    def __getitem__(self, i):
        return torch.from_numpy(self.y_all[i]), torch.from_numpy(self.sigma_gt_all[i])


def _collate_single(batch):
    "batch_size must be 1; keep 1D."
    return batch[0]


def main():
    # ---------- 参数 ----------
    p = argparse.ArgumentParser(description="Train LPD model for EIT")
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--batch-size", type=int,   default=1,
                   help="LPDNet only supports batch_size=1")
    p.add_argument("--lr",         type=float, default=0.1)
    p.add_argument("--val-ratio",  type=float, default=0.2)
    p.add_argument("--device",     type=str,   default="cpu",
                   help="LPDNet is CPU-only, please do not use cuda")
    p.add_argument("--grad-accum-steps", type=int, default=1,
               help="gradient accumulation steps, equivalent batch = batch_size * grad_accum_steps")
    args = p.parse_args()

    if args.batch_size != 1:
        raise ValueError("LPDNet only supports batch_size=1")
    if args.device != "cpu":
        print(f"[Warning] LPDNet depends on SciPy sparse solver, only supports CPU;"
              f"ignoring --device={args.device}, using cpu.")
        args.device = "cpu"

    # ---------- check Data file ----------
    if not os.path.exists(cfg.DATA_FILE):
        agree = input(f"Data file {cfg.DATA_FILE} not found, do you want to generate it now?(y/n): ")
        if agree.lower() == 'y':
            import data.eit_data as eit_data
            gen = eit_data.EIT_Data()                
            gen.save_datafile(cfg.DATA_FILE)          
        else:
            print("Please run `python src/data/eit_data.py` to generate the data first.")
            return
    else:
        print(f"Data file {cfg.DATA_FILE} found.")

    # ---------- get npz ----------
    with np.load(cfg.DATA_FILE) as npzfile:
        V            = npzfile["voltage"]                    # (S, P, nb)
        S, P, nb     = V.shape
        y_all        = V.reshape(S, P * nb).astype(np.float32)
        sigma_gt_all = npzfile["conductivity"].astype(np.float32)   # (S, n)
        current      = npzfile["current"].astype(np.float32)        # (P, n)
        grid_density = int(npzfile["grid_density"])
        radius       = float(npzfile["radius"])
        basis        = int(npzfile["basis"])
    print(f"Data: Samples={S}, Patterns={P}, Elements={nb}, Conductivity={sigma_gt_all.shape[1]}")

    # ---------- reconstruct grid object (LPDNet needs) ----------
    print("Reconstructing grid object...")
    from core.squareGrid import Area_grid as AG
    grid = AG(grid_density=grid_density, radius=radius)
    print("Grid reconstructed")

    # ---------- Dataset / DataLoader ----------
    dataset = EITDataset(y_all, sigma_gt_all)
    val_size = max(1, int(round(args.val_ratio * len(dataset))))
    train_size = len(dataset) - val_size
    torch.manual_seed(2026)
    if train_size < 1:
        raise RuntimeError(f"samples too few({len(dataset)}, can't split into train/val with ratio")

    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(2026),
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=1, shuffle=True,
        collate_fn=_collate_single, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False,
        collate_fn=_collate_single, num_workers=0,
    )

    # ---------- Model ----------
    model = Model_Module.LPDNet(
        grid=grid,
        patterns=current,
        basis=basis,
        dtype=torch.float64,
    )

    # ---------- optimizer / loss ----------
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # ---------- save path ----------
    os.makedirs(cfg.WEIGHTS_DIR, exist_ok=True)
    save_path = os.path.join(cfg.WEIGHTS_DIR, "lpd_best.pt")
    print(f"Model weights will be saved to {save_path}")

    # ---------- train ----------
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=args.device,
        save_path=save_path,
        grad_accum_steps=args.grad_accum_steps,   
    )


if __name__ == "__main__":
    main()