"""Per-node Vm contrasts across unique tetrahedral mesh edges, in mV."""
import numpy as np


def build_neighbour_graph(tetrahedra, n_nodes):
    """One unique edge per neighbouring pair; shared tetrahedra do not add weight."""
    tetrahedra = np.asarray(tetrahedra)
    if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4 or not len(tetrahedra):
        raise ValueError('Neighbour fields require tetrahedral connectivity (cells, 4)')
    if not np.issubdtype(tetrahedra.dtype, np.integer):
        raise ValueError('Tetrahedral node indices must be integers')
    if np.any(tetrahedra < 0) or np.any(tetrahedra >= n_nodes):
        raise ValueError('Tetrahedral node index is outside the reference mesh')
    if np.any(np.diff(np.sort(tetrahedra, axis=1), axis=1) == 0):
        raise ValueError('A tetrahedron contains repeated node indices')

    # Every tetrahedron has six undirected edges.
    local_pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    edges = tetrahedra[:, local_pairs].reshape(-1, 2).copy()
    edges.sort(axis=1)
    keys = np.unique(edges[:, 0].astype(np.int64) * n_nodes + edges[:, 1])
    source = keys // n_nodes
    destination = keys % n_nodes
    degree = np.bincount(source, minlength=n_nodes) + np.bincount(destination, minlength=n_nodes)
    return source, destination, degree


def neighbour_fields(vm_frame, graph):
    """Return mean |Vi-Vj|, Vi-mean(Vj), and sum_edges (Vi-Vj)^2 for one frame.

    There is no division by physical edge length. These are voltage contrasts,
    not Cartesian gradients. Isolated nodes have NaN contrasts and degree zero.
    Processing one frame avoids allocating an edges-by-times array on the HPC.
    """
    source, destination, degree = graph
    values = np.asarray(vm_frame, dtype=np.float64)
    if values.shape != degree.shape or not np.isfinite(values).all():
        raise ValueError('Vm frame must contain one finite value per mesh node')

    differences = values[source] - values[destination]
    absolute_sum = np.bincount(source, weights=np.abs(differences), minlength=len(values))
    absolute_sum += np.bincount(destination, weights=np.abs(differences), minlength=len(values))
    signed_sum = np.bincount(source, weights=differences, minlength=len(values))
    signed_sum += np.bincount(destination, weights=-differences, minlength=len(values))

    unsigned = np.full(len(values), np.nan, dtype=np.float64)
    signed = np.full(len(values), np.nan, dtype=np.float64)
    np.divide(absolute_sum, degree, out=unsigned, where=degree > 0)
    np.divide(signed_sum, degree, out=signed, where=degree > 0)
    # A NumPy reduction avoids launching a BLAS thread team inside every writer.
    energy = float(np.sum(differences * differences, dtype=np.float64))
    return unsigned.astype(np.float32), signed.astype(np.float32), energy
