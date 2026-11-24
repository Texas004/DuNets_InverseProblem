import argparse
import numpy as np
from config.config import DATA_DIR 
import os



def generate_data(
    A,
    num_samples: int = 200,
    N: int = 20,
    noise: float = 5e-3,
):
    if noise < 0:
        raise ValueError("noise must be non-negative")

    coef = np.random.rand(num_samples, 4)
    y = np.linspace(0, 1, N)
    y = np.array([np.ones_like(y), y, y**2, y**3])
    phi = coef @ y
    phi = phi.T
    f = A @ phi
    f = f + np.random.normal(0.0, noise, size=f.shape)
    return f, phi

def main():

    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=20)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--noise", type=float, default=5e-3)

    args = p.parse_args()
    num_samples = args.num_samples
    N = args.N
    #A = np.random.rand(N, N)
    #A = A.T @ A
    x = np.linspace(0.5/N, 1-0.5/N, N)
    A = 1.0/N * np.exp(np.outer(x, x))
    #A = np.eye(N)

    f, phi = generate_data(A, num_samples, N, noise=args.noise)

    filepath = os.path.join(DATA_DIR, "CGdata.npz")

    np.savez_compressed(
        filepath,

        A = A.astype(np.float32),

        phi = phi.astype(np.float32),

        f = f.astype(np.float32)
    )
    print(f"Saved EIT data to {filepath}")


if __name__ == "__main__":

    main()
