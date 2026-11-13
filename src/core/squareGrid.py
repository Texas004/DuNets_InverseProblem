import numpy as np


class Area_grid:

    '''
    The Area_grid class is used to generate a grid of points within a circular area.
    In this class, the grid density and radius of the circular area can be specified.
    The get_grid method generates the grid points, edges, and boundary nodes within the circular area.

    Parameters
    ----------
    grid_density: The density of the grid points within the circular area. Default is 31.

    radius: The radius of the circular area. Default is 0.94.

    Attributes
    -------
    X: The x_cordinates of the grid points.

    Y: The y_cordinates of the grid points.

    mask: A boolean array indicating the points within the circular area.

    coords: An array of the (x, y) coordinates of the grid points.

    edges: An array of edges connecting the grid points.

    boundary: An array of indices for the boundary nodes.

    index: An array mapping grid coordinates to indices.

    '''


    def __init__(self, grid_density:int = 31,radius:float = 0.94):
        
        self.grid_density = grid_density
        self.radius = radius

        ax = np.linspace(-1.0, 1.0, self.grid_density)
        self.X, self.Y = np.meshgrid(ax, ax, indexing="xy")
        self.mask = self.X*self.X + self.Y*self.Y <= self.radius*self.radius
        self.coords = np.column_stack([self.X[self.mask], self.Y[self.mask]])
        self.index = -np.ones(self.mask.shape, dtype=int)
        self.index[self.mask] = np.arange(self.mask.sum())
        self.edges=[]
        # nearest-neighbour graph edges; conductance includes conductivity average and dx scale

        for i,j in zip(*np.nonzero(self.mask)):
            a=self.index[i,j]
            for di,dj in ((1,0),(0,1)):
                ii,jj=i+di,j+dj
                if ii<self.grid_density and jj<self.grid_density and self.mask[ii,jj]:
                    self.edges.append((a,self.index[ii,jj]))

        self.edges=np.asarray(self.edges,dtype=int)
        # boundary nodes are nodes adjacent to the outside of the disk mask
        self.boundary=[]
        for i,j in zip(*np.where(self.mask)):
            if any((i+di<0 or i+di>=self.grid_density or j+dj<0 or j+dj>=self.grid_density or not self.mask[i+di,j+dj])
                   for di,dj in ((1,0),(-1,0),(0,1),(0,-1))):
                self.boundary.append(self.index[i,j])
        self.boundary=np.asarray(self.boundary,dtype=int)
        theta=np.arctan2(self.coords[self.boundary,1],self.coords[self.boundary,0])
        self.boundary=self.boundary[np.argsort(theta)]
