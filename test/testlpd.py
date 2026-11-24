"""Compare the trained LPD network against its untrained initialization.

For every sample in ``src/data/EITdata.npz`` the script reconstructs the
conductivity with two models:

* ``trained``   - ``LPDNet`` loaded from ``src/data/weights/lpd_best.pt``;
* ``untrained`` - a freshly initialized ``LPDNet``.

The reported metric is the relative reconstruction error

    rel_err = ||sigma_pred - sigma_gt||_2 / ||sigma_gt||_2

In addition to a console table, the script writes a CSV with the per-sample
errors, a bar chart comparing both models and a grid with ground truth,
trained reconstruction, untrained reconstruction and absolute error maps.

Examples
--------
    python test/testlpd.py
    python test/testlpd.py --max-samples 3 --num-plots 3
    python test/testlpd.py --split val
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import torch

from core.squareGrid import Area_grid
from model.lpd_net import LPDNet


DEFAULT_DATA = SRC_DIR / "data" / "EITdata.npz"
DEFAULT_WEIGHTS = SRC_DIR / "data" / "weights" / "lpd_best.pt"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "lpd_test"


# ----------------------------------------------------------------------
# Loading helpers
# ----------------------------------------------------------------------
def load_npz(path: Path) -> dict:
    with np.load(path) as npz:
        return {key: npz[key] for key in npz.files}


def build_grid(data: dict) -> Area_grid:
    grid = Area_grid(
        grid_density=int(data["grid_density"]),
        radius=float(data["radius"]),
    )
    checks = {
        "coords": np.allclose(grid.coords, data["coords"], atol=1e-6),
        "edges": np.array_equal(grid.edges, data["edges"]),
        "boundary": np.array_equal(grid.boundary, data["boundary"]),
        "mask": np.array_equal(grid.mask, data["mask"]),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Grid metadata stored in the npz does not match Area_grid: "
            f"{checks}"
        )
    return grid


def build_model(
    data: dict,
    grid: Area_grid,
    outer: int,
    inner: int,
    tv: float,
) -> LPDNet:
    return LPDNet(
        grid=grid,
        patterns=data["current"].astype(np.float64),
        basis=int(data["basis"]),
        tv=tv,
        outer=outer,
        inner=inner,
        dtype=torch.float64,
    )


def load_weights(model: LPDNet, path: Path) -> LPDNet:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load weights from {path}. Check that the checkpoint "
            "matches the current LPDNet architecture (especially --inner, "
            "which must equal the length of tau_acc).\n"
            f"{exc}"
        ) from exc
    return model


def split_indices(n: int, split: str, val_ratio: float, seed: int) -> np.ndarray:
    if split == "all":
        return np.arange(n)

    train_size = n - max(1, int(round(val_ratio * n)))
    val_size = n - train_size
    if train_size < 1:
        raise ValueError(f"Not enough samples ({n}) for a train/val split.")
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(
        range(n), [train_size, val_size], generator=generator
    )
    subset = train_subset if split == "train" else val_subset
    return np.asarray(sorted(subset.indices), dtype=int)


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def evaluate(
    model: LPDNet,
    y_all: np.ndarray,
    sigma_gt_all: np.ndarray,
    indices: np.ndarray,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions = np.empty(
        (len(indices), sigma_gt_all.shape[1]), dtype=np.float64
    )
    errors = np.empty(len(indices), dtype=np.float64)

    with torch.no_grad():
        for row, index in enumerate(indices):
            y = torch.from_numpy(np.asarray(y_all[index], dtype=np.float64))
            pred = model(y).detach().cpu().numpy().reshape(-1)
            predictions[row] = pred
            denominator = np.linalg.norm(sigma_gt_all[index]) + 1e-12
            errors[row] = (
                np.linalg.norm(pred - sigma_gt_all[index]) / denominator
            )
            print(
                f"  [{label}] sample {int(index):2d}: "
                f"rel_err={errors[row]:.6f}",
                flush=True,
            )
    return predictions, errors


# ----------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------
def print_table(
    indices: np.ndarray,
    rel_trained: np.ndarray,
    rel_untrained: np.ndarray,
) -> None:
    header = (
        f"{'sample':>7} | {'trained':>10} | "
        f"{'untrained':>10} | {'improvement':>11}"
    )
    print()
    print(header)
    print("-" * len(header))
    for index, tr, un in zip(indices, rel_trained, rel_untrained):
        print(
            f"{int(index):>7d} | {tr:>10.6f} | {un:>10.6f} | {un - tr:>11.6f}"
        )
    print("-" * len(header))
    print(
        f"{'mean':>7} | {rel_trained.mean():>10.6f} | "
        f"{rel_untrained.mean():>10.6f} | "
        f"{rel_untrained.mean() - rel_trained.mean():>11.6f}"
    )
    print()


def write_csv(
    path: Path,
    indices: np.ndarray,
    rel_trained: np.ndarray,
    rel_untrained: np.ndarray,
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sample", "trained_rel_error", "untrained_rel_error", "improvement"]
        )
        for index, tr, un in zip(indices, rel_trained, rel_untrained):
            writer.writerow(
                [int(index), f"{tr:.8f}", f"{un:.8f}", f"{un - tr:.8f}"]
            )


def plot_summary(
    path: Path,
    indices: np.ndarray,
    rel_trained: np.ndarray,
    rel_untrained: np.ndarray,
) -> None:
    width = 0.38
    positions = np.arange(len(indices), dtype=float)
    figure, axis = plt.subplots(
        figsize=(max(7.0, 0.6 * len(indices) + 4.0), 4.0)
    )
    axis.bar(
        positions - width / 2,
        rel_trained,
        width,
        label="Trained LPD",
        color="#1f77b4",
    )
    axis.bar(
        positions + width / 2,
        rel_untrained,
        width,
        label="Untrained LPD",
        color="#ff7f0e",
    )
    axis.axhline(
        rel_trained.mean(),
        color="#1f77b4",
        linestyle="--",
        linewidth=1.0,
        label=f"Trained mean = {rel_trained.mean():.4f}",
    )
    axis.axhline(
        rel_untrained.mean(),
        color="#ff7f0e",
        linestyle=":",
        linewidth=1.0,
        label=f"Untrained mean = {rel_untrained.mean():.4f}",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels([str(int(index)) for index in indices])
    axis.set_xlabel("Sample index")
    axis.set_ylabel("Relative reconstruction error")
    axis.set_title("LPD reconstruction error: trained vs. untrained")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def nodes_to_image(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = np.full(mask.shape, np.nan, dtype=np.float64)
    image[mask] = np.asarray(values).reshape(-1)
    return image


def plot_samples(
    path: Path,
    indices: np.ndarray,
    sigma_gt_all: np.ndarray,
    pred_trained: np.ndarray,
    pred_untrained: np.ndarray,
    mask: np.ndarray,
) -> None:
    rows = len(indices)
    columns = 5
    extent = (-1.0, 1.0, -1.0, 1.0)

    gt_images = [nodes_to_image(sigma_gt_all[index], mask) for index in indices]
    trained_images = [nodes_to_image(values, mask) for values in pred_trained]
    untrained_images = [
        nodes_to_image(values, mask) for values in pred_untrained
    ]
    trained_errors = [
        np.abs(trained - gt)
        for trained, gt in zip(trained_images, gt_images)
    ]
    untrained_errors = [
        np.abs(untrained - gt)
        for untrained, gt in zip(untrained_images, gt_images)
    ]

    vmin = float(np.nanmin([np.nanmin(image) for image in gt_images]))
    vmax = float(np.nanmax([np.nanmax(image) for image in gt_images]))
    error_max = max(
        1e-12,
        float(
            np.nanmax(
                [
                    np.nanmax(image)
                    for image in trained_errors + untrained_errors
                ]
            )
        ),
    )

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(20.0, 3.2 * rows + 0.8),
        squeeze=False,
        constrained_layout=True,
    )

    for row, index in enumerate(indices):
        gt = gt_images[row]
        trained = trained_images[row]
        untrained = untrained_images[row]
        trained_error = trained_errors[row]
        untrained_error = untrained_errors[row]

        panels = (
            (gt, "viridis", vmin, vmax),
            (trained, "viridis", vmin, vmax),
            (untrained, "viridis", vmin, vmax),
            (trained_error, "magma", 0.0, error_max),
            (untrained_error, "magma", 0.0, error_max),
        )
        for column, (image, cmap, lower, upper) in enumerate(panels):
            axis = axes[row, column]
            axis.imshow(
                image,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=lower,
                vmax=upper,
            )
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                titles = (
                    "Ground truth",
                    "Trained LPD",
                    "Untrained LPD",
                    "|Trained - GT|",
                    "|Untrained - GT|",
                )
                axis.set_title(titles[column])
        axes[row, 0].set_ylabel(f"Sample {int(index)}")

    conductivity_mappable = plt.cm.ScalarMappable(
        norm=Normalize(vmin=vmin, vmax=vmax), cmap="viridis"
    )
    error_mappable = plt.cm.ScalarMappable(
        norm=Normalize(vmin=0.0, vmax=error_max), cmap="magma"
    )
    figure.colorbar(
        conductivity_mappable,
        ax=axes[:, :3].ravel().tolist(),
        shrink=0.55,
        label="Conductivity",
    )
    figure.colorbar(
        error_mappable,
        ax=axes[:, 3:].ravel().tolist(),
        shrink=0.55,
        label="Absolute error",
    )
    figure.savefig(path, dpi=170)
    plt.close(figure)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare trained and untrained LPD networks on EIT data."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--split",
        choices=("all", "train", "val"),
        default="all",
        help="Which samples to evaluate (default: the whole dataset).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only the first N selected samples.",
    )
    parser.add_argument(
        "--num-plots",
        type=int,
        default=5,
        help="Number of samples in the reconstruction grid.",
    )
    parser.add_argument("--outer", type=int, default=16)
    parser.add_argument("--inner", type=int, default=250)
    parser.add_argument("--tv", type=float, default=8.5e-7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data}")
    if not args.weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = load_npz(args.data)
    voltage = data["voltage"].astype(np.float64)  # (S, P, nb)
    sample_count = voltage.shape[0]
    y_all = voltage.reshape(sample_count, -1)
    sigma_gt_all = data["conductivity"].astype(np.float64)  # (S, n)
    mask = data["mask"].astype(bool)

    print(
        f"Dataset: {sample_count} samples, {y_all.shape[1]} measurements, "
        f"{sigma_gt_all.shape[1]} conductivity nodes"
    )
    grid = build_grid(data)

    indices = split_indices(
        sample_count, args.split, args.val_ratio, args.seed
    )
    if args.max_samples is not None:
        indices = indices[: max(0, int(args.max_samples))]
    if len(indices) == 0:
        raise ValueError("No samples selected for evaluation.")
    print(f"Evaluating samples: {indices.tolist()}")

    untrained = build_model(data, grid, args.outer, args.inner, args.tv)
    trained = build_model(data, grid, args.outer, args.inner, args.tv)
    load_weights(trained, args.weights)

    print(
        "Step-size parameters:"
        f"\n  trained   gamma={trained.gamma.item():.6f}, "
        f"tau_coeffs={trained.tau_coeffs.detach().numpy()}, "
        f"sigma_coeffs={trained.sigma_coeffs.detach().numpy()}"
        f"\n            tau_acc mean={trained.tau_acc.mean().item():.4f}, "
        f"std={trained.tau_acc.std().item():.4f}, "
        f"range=[{trained.tau_acc.min().item():.4f}, "
        f"{trained.tau_acc.max().item():.4f}]"
        f"\n  untrained gamma={untrained.gamma.item():.6f}, "
        f"tau_coeffs={untrained.tau_coeffs.detach().numpy()}, "
        f"sigma_coeffs={untrained.sigma_coeffs.detach().numpy()}"
        f"\n            tau_acc mean={untrained.tau_acc.mean().item():.4f}, "
        f"std={untrained.tau_acc.std().item():.4f}, "
        f"range=[{untrained.tau_acc.min().item():.4f}, "
        f"{untrained.tau_acc.max().item():.4f}]"
    )

    print("\nEvaluating trained network ...")
    pred_trained, rel_trained = evaluate(
        trained, y_all, sigma_gt_all, indices, "trained"
    )
    print("\nEvaluating untrained network ...")
    pred_untrained, rel_untrained = evaluate(
        untrained, y_all, sigma_gt_all, indices, "untrained"
    )

    print_table(indices, rel_trained, rel_untrained)

    write_csv(
        args.out_dir / "lpd_compare_metrics.csv",
        indices,
        rel_trained,
        rel_untrained,
    )
    plot_summary(
        args.out_dir / "lpd_compare_summary.png",
        indices,
        rel_trained,
        rel_untrained,
    )

    num_plots = min(max(0, int(args.num_plots)), len(indices))
    if num_plots > 0:
        plot_samples(
            args.out_dir / "lpd_compare_samples.png",
            indices[:num_plots],
            sigma_gt_all,
            pred_trained[:num_plots],
            pred_untrained[:num_plots],
            mask,
        )

    np.savez_compressed(
        args.out_dir / "lpd_compare_predictions.npz",
        indices=indices,
        sigma_gt=sigma_gt_all[indices],
        pred_trained=pred_trained,
        pred_untrained=pred_untrained,
        rel_trained=rel_trained,
        rel_untrained=rel_untrained,
    )

    print(
        f"Saved metrics, plots and predictions to {args.out_dir}\n"
        f"  trained   mean rel_err = {rel_trained.mean():.6f}\n"
        f"  untrained mean rel_err = {rel_untrained.mean():.6f}"
    )


if __name__ == "__main__":
    main()
