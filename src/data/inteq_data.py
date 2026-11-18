import argparse
import numpy as np
from config.config import DATA_DIR 
import os



def generate_data(A, num_samples:int = 200, N:int = 20):
    coef = np.random.rand(num_samples, 4)
    y = np.linspace(0, 1, N)
    y = np.array([np.ones_like(y), y, y**2, y**3])
    phi = coef @ y
    phi = phi.T
    f = A @ phi
    return f, phi

def main():

    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=20)
    p.add_argument("--num_samples", type=int, default=200)

    args = p.parse_args()
    num_samples = args.num_samples
    N = args.N
    A = np.random.rand(N, N)

    f, phi = generate_data(A, num_samples, N)

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


    