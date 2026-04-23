import os
import pathlib

import numpy as np
import torch
import tqdm
from captum.attr import IntegratedGradients
from lightning import LightningModule
from loguru import logger
from torch.utils.data import DataLoader


def investigate_feature_importance(full_output_dir, lightning_module: LightningModule, val_dataset, parallel: bool, reset: bool = False):
    """Compute and cache global feature importance for a validation set.

    The method uses Captum's `IntegratedGradients` on the trained Lightning module
    to attribute importance to each input feature. It averages absolute attributions
    across the batch and channel dimensions to produce a single importance score per
    point/feature. Results are cached to `global_feature_importance.npy` inside the
    provided `full_output_dir` so subsequent calls can load the saved array unless
    `reset=True` forces recomputation.

    Parameters:
      full_output_dir: Directory where the NumPy cache file is read/written.
      lightning_module: The trained `FaceMeshLightningModule`. If None, the method
        expects that a previously computed cache exists and will be loaded.
      val_dataset: Dataset used to compute attributions. If `lightning_module` is
        None (loading from cache), this can be None as well.
      reset: If True, recompute attributions even if a cache exists.

    Returns:
      np.ndarray: The global feature importance array. For point-cloud data_dict it has
      shape (C, P) or (P,) depending on the model's output shape; in either case,
      the last dimension corresponds to point-wise importance used for plotting.
    """
    global_feature_importance_file = os.path.join(full_output_dir, 'global_feature_importance.npy')
    if not reset and os.path.exists(global_feature_importance_file):
        global_feature_importance = np.load(global_feature_importance_file)
        logger.debug(
            f'Loaded global feature importance from {global_feature_importance_file} (shape: {global_feature_importance.shape}))')
    else:
        logger.debug(f'Start investigating feature importance in {full_output_dir}...')
        lightning_module.cuda()
        lightning_module.eval()
        attributions = []
        fold = pathlib.Path(full_output_dir).parts[-1]
        batch_size = 50
        test_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4 if parallel else 0)
        for data_dict in tqdm.tqdm(test_loader, desc=f'IntegratedGradients ({fold})'):
            for key, value in data_dict.items():
                data_dict[key] = value.to(lightning_module.device)
                value.requires_grad = True
            inputs = data_dict['x']
            kwargs = {key: value for key, value in data_dict.items() if key not in ['x', 'y']}

            def forward_fn(inputs: torch.Tensor) -> torch.Tensor:
                return lightning_module(inputs, **kwargs)[0]

            feature_attribution_method = IntegratedGradients(forward_fn)
            attr = feature_attribution_method.attribute(inputs, internal_batch_size=inputs.size(0))
            attributions.append(attr.detach().cpu())
        attributions = torch.cat(attributions)

        # Per-feature global importance across dataset
        global_feature_importance = attributions.abs().mean(dim=0).cpu().detach().numpy()
        np.save(global_feature_importance_file, global_feature_importance)

        logger.debug('Finished investigating feature importance.')
    return global_feature_importance
