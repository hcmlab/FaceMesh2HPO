import numpy as np
from scipy.linalg import svd


def procrustes_align(source_mesh: np.ndarray, ref_mesh: np.ndarray):
    """
    Align source (N,3) to ref (N,3) via rigid + uniform scale.
    Returns transformed source.
    """
    # Center both
    src_mu = source_mesh.mean(axis=0)
    ref_mu = ref_mesh.mean(axis=0)
    src_centered = source_mesh - src_mu
    ref_centered = ref_mesh - ref_mu

    # Scale to unit variance
    src_scale = np.sqrt(np.sum(src_centered ** 2) / len(source_mesh))
    ref_scale = np.sqrt(np.sum(ref_centered ** 2) / len(ref_mesh))
    src_unit = src_centered / src_scale
    ref_unit = ref_centered / ref_scale

    # SVD for rotation
    H = src_unit.T @ ref_unit
    U, _, Vt = svd(H)
    R = Vt.T @ U.T

    # Reflection check (det(R) = -1 → flip)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Transform
    aligned = (ref_scale / src_scale) * (R @ src_unit.T).T + ref_mu
    return aligned
