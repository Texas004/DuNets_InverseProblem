import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve



def forward_operator(grid,
                     motivated_current: np.ndarray,
                     conductivity: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    grid : Grid class which contains the mesh information
    motivated_current : (P, n) float
    conductivity : (n,) or (S, n) float

    Returns
    -------
    V : (P*nb,) if conductivity is 1-D
        (S, P*nb) if conductivity is 2-D
    """
    patterns = motivated_current.shape[0]
    n = len(grid.coords)
    edge_i = grid.edges[:, 0]
    edge_j = grid.edges[:, 1]
    dx = 2.0 / (grid.X.shape[0] - 1)

    # ---------- conductivity: accept (n,) or (S, n) ----------
    conductivity = np.asarray(conductivity, dtype=float)
    single_sample = (conductivity.ndim == 1)
    if single_sample:
        conductivity = conductivity[None, :]          # (1, n)
    S = conductivity.shape[0]
    if conductivity.shape[1] != n:
        raise ValueError(
            f"conductivity has {conductivity.shape[1]} nodes, expected {n}"
        )

    # ---------- measurement matrix M (independent of conductivity) ----------
    nb = len(grid.boundary)
    rows, cols, vals = [], [], []
    for k in range(nb):
        rows += [k, k]
        cols += [k, (k + nb // 2) % nb]
        vals += [1.0, -1.0]
    M = coo_matrix((vals, (rows, cols)), shape=(nb, nb)).tocsr()
    n_meas = M.shape[0]

    # ---------- gauge (independent of conductivity) ----------
    interior_nodes = np.setdiff1d(np.arange(n), grid.boundary)
    gauge = interior_nodes[0]

    # ---------- rhs (independent of conductivity) ----------
    rhs = motivated_current.astype(float, copy=True)
    rhs[:, gauge] = 0.0

    # ---------- per-sample forward solve ----------
    all_V = np.empty((S, patterns, n_meas), dtype=float)

    for s in range(S):
        sigma = conductivity[s]

        # edge conductance: harmonic mean is stable across conductivity jumps
        si = sigma[edge_i]
        sj = sigma[edge_j]
        g = 2.0 * si * sj / (si + sj + 1e-12) / (dx ** 2)

        i, j = edge_i, edge_j
        rr = np.concatenate([i, j, i, j])
        cc = np.concatenate([i, j, j, i])
        vv = np.concatenate([g, g, -g, -g])

        # gauge: remove all entries touching `gauge`, then set L[gauge,gauge]=1
        keep = (rr != gauge) & (cc != gauge)
        rr2 = np.concatenate([rr[keep], [gauge]])
        cc2 = np.concatenate([cc[keep], [gauge]])
        vv2 = np.concatenate([vv[keep], [1.0]])
        L = coo_matrix((vv2, (rr2, cc2)), shape=(n, n)).tocsr()

        # solve all current patterns at once
        U = spsolve(L, rhs.T).T                       # (P, n)
        U_bd = U[:, grid.boundary]                    # (P, nb)
        all_V[s] = U_bd @ M.T                         # (P, nb)

    # ---------- output shape ----------
    if single_sample:
        return all_V[0].ravel()                       # (P*nb,) 
    return all_V.reshape(S, -1)                       # (S, P*nb)