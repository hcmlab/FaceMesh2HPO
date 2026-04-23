import numpy as np


def compute_mask(old_mask, new_mask):
    if old_mask is None:
        return new_mask
    idx = 0
    full_mask = []
    for i, mask in enumerate(old_mask):
        if mask:
            full_mask.append(new_mask[idx])
            idx += 1
        else:
            full_mask.append(False)
    point_mask = np.asarray(full_mask)
    return point_mask
