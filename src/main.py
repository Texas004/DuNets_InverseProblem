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
    """y -> conductivity_gt。初始电导率由 LPDNet.forward(sigma0=None) 自动 ones。"""
    def __init__(self, y_all, sigma_gt_all):
        self.y_all       = y_all              # (S, m)
        self.sigma_gt_all = sigma_gt_all      # (S, n)

    def __len__(self):
        return self.y_all.shape[0]

    def __getitem__(self, i):
        return torch.from_numpy(self.y_all[i]), torch.from_numpy(self.sigma_gt_all[i])


def _collate_single(batch):
    """batch_size 必须为 1；保持 1D。"""
    return batch[0]





def main():
    # ---------- 参数 ----------
    p = argparse.ArgumentParser(description="Train LPD model for EIT")
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--batch-size", type=int,   default=1,
                   help="LPDNet 仅支持 batch_size=1")
    p.add_argument("--lr",         type=float, default=1e-7)
    p.add_argument("--val-ratio",  type=float, default=0.2)
    p.add_argument("--device",     type=str,   default="cpu",
                   help="LPDNet 是 CPU-only，请勿用 cuda")
    p.add_argument("--grad-accum-steps", type=int, default=1,
               help="梯度累积步数，等效 batch = batch_size * grad_accum_steps")
    args = p.parse_args()

    if args.batch_size != 1:
        raise ValueError("LPDNet 目前只支持 batch_size=1")
    if args.device != "cpu":
        print(f"[警告] LPDNet 依赖 SciPy 稀疏求解器，仅支持 CPU；"
              f"忽略 --device={args.device}，使用 cpu。")
        args.device = "cpu"

    # ---------- 数据文件检查 / 生成 ----------
    if not os.path.exists(cfg.DATA_FILE):
        agree = input(f"数据文件 {cfg.DATA_FILE} 不存在，是否现在生成？(y/n): ")
        if agree.lower() == 'y':
            import data.eit_data as eit_data
            gen = eit_data.EIT_Data()                 # 用默认超参
            gen.save_datafile(cfg.DATA_FILE)          # 显式指定输出路径
        else:
            print("请先运行 `python src/data/eit_data.py` 生成数据。")
            return
    else:
        print(f"数据文件 {cfg.DATA_FILE} 已找到。")

    # ---------- 读取 npz ----------
    with np.load(cfg.DATA_FILE) as npzfile:
        V            = npzfile["voltage"]                    # (S, P, nb)
        S, P, nb     = V.shape
        y_all        = V.reshape(S, P * nb).astype(np.float32)
        sigma_gt_all = npzfile["conductivity"].astype(np.float32)   # (S, n)
        current      = npzfile["current"].astype(np.float32)        # (P, n)
        grid_density = int(npzfile["grid_density"])
        radius       = float(npzfile["radius"])
        basis        = int(npzfile["basis"])
    print(f"数据: S={S}, P={P}, nb={nb}, n={sigma_gt_all.shape[1]}")

    # ---------- 重建 grid 对象（LPDNet 需要） ----------
    from core.squareGrid import Area_grid as AG
    grid = AG(grid_density=grid_density, radius=radius)

    # ---------- Dataset / DataLoader ----------
    dataset = EITDataset(y_all, sigma_gt_all)
    val_size = max(1, int(round(args.val_ratio * len(dataset))))
    train_size = len(dataset) - val_size
    if train_size < 1:
        raise RuntimeError(f"样本太少（{len(dataset)}），无法切分训练/验证。")

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

    # ---------- 模型 ----------
    model = Model_Module.LPDNet(
        grid=grid,
        patterns=current,
        basis=basis,
        dtype=torch.float64,        # 与 LPDNet 默认保持一致；CPU 上 OK
    )

    # ---------- 优化器 / 损失 ----------
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # ---------- 保存路径 ----------
    os.makedirs(cfg.WEIGHTS_DIR, exist_ok=True)
    save_path = os.path.join(cfg.WEIGHTS_DIR, "lpd_best.pt")

    # ---------- 训练 ----------
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=args.device,
        save_path=save_path,
        grad_accum_steps=args.grad_accum_steps,    # ← 新增
    )


if __name__ == "__main__":
    main()