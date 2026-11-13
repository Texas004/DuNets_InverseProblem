'''
The file 'eit_data.py' contains functions and classes for generating and processing EIT data.
Including EIT conductivity data, EIT current data, and observated voltage data.

The EIT data will be used for training and testing the EIT reconstruction model, and it will be saved in the 'data' folder.
'''

import argparse
import numpy as np
from core.EITForwardOperator import forward_operator
from core.squareGrid import Area_grid as AG
from config.config import DATA_FILE
import os

class EIT_Conductivity_Data(AG):

    def __init__(self, basis:int = 15, grid_density:int = 31, radius:float = 0.94, sample_size:int = 10):

        super().__init__(grid_density = grid_density, radius = radius)

        self.basis = basis
        self.sample_size = sample_size

    def gaussian_basis(self, width=None):
        lo,hi=-0.82,0.82
        centers=np.linspace(lo,hi,self.basis)
        C=np.array([(x,y) for y in centers for x in centers])
        if width is None: width=(centers[1]-centers[0])*0.85
        B=np.exp(-((self.coords[:,None,0]-C[None,:,0])**2+
               (self.coords[:,None,1]-C[None,:,1])**2)/(2*width**2))
        # Normalize columns to make coefficients interpretable and conditioning better.
        B/=np.maximum(np.max(B,axis=0,keepdims=True),1e-12)
        return B,C

    def ground_truth(self, seed:int = 2026):
        rng = np.random.default_rng(seed)

        x,y=self.coords[:,0],self.coords[:,1]
        s=np.ones((self.sample_size, len(self.coords)))
        # Smooth-edged inclusions make the inverse problem less sensitive to pixel aliasing.

         # base parameter: [cx, cy, rx, ry, amp]
        base1 = np.array([-0.34,  0.25, 0.22, 0.18,  0.75])
        base2 = np.array([ 0.30, -0.24, 0.20, 0.22, -0.40])

        # perturbation scales: [cx, cy, rx, ry, amp]
        scale1 = np.array([0.02, 0.02, 0.01, 0.01, 0.03])
        scale2 = np.array([0.02, 0.02, 0.01, 0.01, 0.03])

        p1 = base1 + rng.normal(size=(self.sample_size, 5)) * scale1
        p2 = base2 + rng.normal(size=(self.sample_size, 5)) * scale2

        p1[:, 2:4] = np.maximum(p1[:, 2:4], 0.05)
        p2[:, 2:4] = np.maximum(p2[:, 2:4], 0.05)

        def blob(cx,cy,rx,ry,amp):
           r=((x-cx)/rx)**2+((y-cy)/ry)**2
           return amp/(1+np.exp(np.clip((r-1)*12,-50,50)))

        for i in range(self.sample_size):
            cx, cy, rx, ry, amp = p1[i]
            s[i] += blob(cx, cy, rx, ry, amp)

            cx, cy, rx, ry, amp = p2[i]
            s[i] += blob(cx, cy, rx, ry, amp)

        return np.clip(s,0.45,1.9)


class Neumann_Data:


    def __init__(self, basis:int = 15, electrodes:int = 52, patterns:int = 32, coords:np.ndarray = None, boundary:np.ndarray = None):
        
        self.basis = basis
        self.electrodes = electrodes
        self.patterns = patterns
        self.coords = coords
        self.boundary = boundary
        self.motivated_current = None


    def make_motivated_current(self):

        # Distribute electrodes over boundary nodes, then use smooth current profiles.
        bxy=self.coords[self.boundary]
        theta=np.arctan2(bxy[:,1],bxy[:,0])
        centers=np.linspace(-np.pi, np.pi, self.electrodes, endpoint=False)
        # Each pattern is a bipolar pair of Gaussian electrode current profiles.
        patterns=[]
        for k in range(self.patterns):
            c=centers[k % self.electrodes]
            d=2*np.pi*(max(1,self.electrodes//4))/self.electrodes
            def circdist(a,b): return np.angle(np.exp(1j*(a-b)))
            pos=np.exp(-0.5*(circdist(theta,c)/0.20)**2)
            neg=np.exp(-0.5*(circdist(theta,c+d)/0.20)**2)
            p=pos/(pos.sum()+1e-15)-neg/(neg.sum()+1e-15)
            # Scale current pattern to unit L2 norm for stable numerical scale.
            p=p/(np.linalg.norm(p)+1e-15)
            f=np.zeros(len(self.coords)); f[self.boundary]=p
            patterns.append(f)
        return np.asarray(patterns)


class EIT_Data:


    def __init__(self, grid_density:int = 31,radius:float = 0.94, basis:int = 15, electrodes:int = 52, patterns:int = 32, sample_size:int = 10, seed:int = 2026):
        self.grid_density = grid_density
        self.radius = radius
        self.basis = basis
        self.electrodes = electrodes
        self.patterns = patterns
        self.sample_size = sample_size
        self.seed = seed

        
        # Generate the EIT conductivity data
        self.conductivity_data_generator = EIT_Conductivity_Data(basis=self.basis, grid_density=self.grid_density, radius=self.radius, sample_size=self.sample_size)
        self.conductivity_data = self.conductivity_data_generator.ground_truth(self.seed)

        # Generate the grid and boundary nodes
        self.grid = self.conductivity_data_generator

        # Generate the Neumann data (motivated current patterns)
        self.neumann_data_generator = Neumann_Data(basis=self.basis, electrodes=self.electrodes, patterns=self.patterns, coords=self.grid.coords, boundary=self.grid.boundary)
        self.motivated_current = self.neumann_data_generator.make_motivated_current()

        # Compute the observed voltage data using the forward operator
        self.observed_voltage_data = forward_operator(self.grid, self.motivated_current, self.conductivity_data)

    def save_datafile(self, filepath:str):


        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        S = self.sample_size
        P = self.patterns
        nb = len(self.grid.boundary)

        #  voltage changed from (S, P*nb) to (S, P, nb) 
        V = self.observed_voltage_data.reshape(S, P, nb)

        # compute mean and std for normalization
        V_mean = V.mean(axis=(0, 1), keepdims=True)   # (1,1,nb)
        V_std = V.std(axis=(0, 1), keepdims=True) + 1e-8

        np.savez_compressed(
            filepath,
            #  data
            conductivity=self.conductivity_data.astype(np.float32),   # (S, n)
            voltage=V.astype(np.float32),                             # (S, P, nb)
            current=self.motivated_current.astype(np.float32),        # (P, n)

            # normalization parameters
            voltage_mean=V_mean.astype(np.float32),                   # (1,1,nb)
            voltage_std=V_std.astype(np.float32),                     # (1,1,nb)

            # grid metadata
            coords=self.grid.coords.astype(np.float32),               # (n, 2)
            edges=self.grid.edges.astype(np.int32),                   # (E, 2)
            boundary=self.grid.boundary.astype(np.int32),             # (nb,)
            mask=self.grid.mask.astype(bool),                         # (d, d)
            X=self.grid.X.astype(np.float32),
            Y=self.grid.Y.astype(np.float32),
            index=self.grid.index.astype(np.int32),

            #
            grid_density=np.int32(self.grid_density),
            radius=np.float32(self.radius),
            basis=np.int32(self.basis),
            electrodes=np.int32(self.electrodes),
            patterns=np.int32(self.patterns),
            sample_size=np.int32(self.sample_size),
            seed=np.int32(self.seed),
        )
        print(f"Saved EIT data to {filepath}")
        return filepath

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--grid",type=int,default=31)
    p.add_argument("--basis",type=int,default=15)
    p.add_argument("--electrodes",type=int,default=52)
    p.add_argument("--patterns",type=int,default=32)
    p.add_argument("--samples",type=int,default=10)
    p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--out",type=str,default=DATA_FILE)
    args = p.parse_args()
    data = EIT_Data(
        grid_density=args.grid,
        basis=args.basis,
        electrodes=args.electrodes,
        patterns=args.patterns,
        sample_size=args.samples,
        seed=args.seed,
    )
    data.save_datafile(args.out)

if __name__ == "__main__":
    main()