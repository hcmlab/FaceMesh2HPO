import math
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class RobustNorm(nn.Module):
    """
    Normalization layer that falls back to GroupNorm when batch size is 1.
    In small batch training (e.g., batch size 1), BatchNorm fails. This layer
    switches between BatchNorm and GroupNorm to handle such cases.
    """

    def __init__(self, num_features, num_groups=8):
        """
        Initialize the RobustNorm layer.
        :param num_features: Number of features in the input tensor.
        :param num_groups: Number of groups for GroupNorm. If 1, BatchNorm1d is used for batch sizes > 1.
        """
        super().__init__()
        ng = min(num_groups, num_features)
        while num_features % ng != 0:
            ng -= 1
        self.norm = nn.GroupNorm(ng, num_features)

    def forward(self, x):
        # Check if spatial dimensions are too small for GroupNorm
        if x.dim() >= 3:
            spatial_size = x.shape[2:].numel()  # Product of all spatial dimensions
            if spatial_size <= 1:
                # Skip normalization or use alternative
                return x
        return self.norm(x)


class TransformationNet(nn.Module):
    """
    T-Net (Transformation Network) for PointNet.
    Learns an affine transformation matrix for input points or features to achieve invariance.
    """

    def __init__(self, input_dim, kernel, output_dim, base_potence):
        """
        Initialize the TransformationNet.
        :param input_dim: Dimension of the input (e.g., 3 for XYZ).
        :param kernel: Kernel size for the first convolution.
        :param output_dim: Dimension of the learned transformation matrix (output_dim x output_dim).
        :param base_potence: Base power of 2 for defining layer widths.
        """
        super(TransformationNet, self).__init__()
        self.output_dim = output_dim
        n_64 = int(math.pow(2, base_potence))
        n_128 = int(math.pow(2, base_potence + 1))
        n_256 = int(math.pow(2, base_potence + 2))
        n_512 = int(math.pow(2, base_potence + 3))
        n_1024 = int(math.pow(2, base_potence + 4))
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_dim, n_64, (kernel, 1)),
            RobustNorm(n_64),
            nn.ReLU(),
            nn.Conv2d(n_64, n_128, (1, 1)),
            RobustNorm(n_128),
            nn.ReLU(),
            nn.Conv2d(n_128, n_1024, (1, 1)),
            RobustNorm(n_1024),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_1024, n_512),
            RobustNorm(n_512, 1),
            nn.ReLU(),
            nn.Linear(n_512, n_256),
            RobustNorm(n_256, 1),
            nn.ReLU(),
            nn.Linear(n_256, self.output_dim * self.output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of TransformationNet.
        :param x: Input tensor of shape [batch, input_dim, points, 1].
        :return: Transformation matrix of shape [batch, output_dim, output_dim].
        """
        batch_size = x.shape[0]
        num_points = x.shape[2]

        # Adjust kernel size if input is too small
        first_conv = self.feature_extractor[0]
        if num_points < first_conv.kernel_size[0]:
            # Use a smaller kernel or padding
            # For simplicity, we can pad the input spatially
            padding_needed = first_conv.kernel_size[0] - num_points
            x = F.pad(x, (0, 0, 0, padding_needed))

        x = self.feature_extractor(x)
        x = torch.max(x, dim=2, keepdim=True)[0]
        x = torch.flatten(x, start_dim=1)
        x = x.reshape(batch_size, -1)
        x = self.classifier(x)

        identity_matrix = torch.eye(self.output_dim, device=x.device)
        x = x.view(-1, self.output_dim, self.output_dim) + identity_matrix
        return x


class BasePointNet(nn.Module):
    """
    Base PointNet architecture that extracts global features from point clouds.
    Includes input transformation (T-Net) and feature transformation (T-Net).
    """

    def __init__(self, point_dimension, base_potence):
        """
        Initialize the BasePointNet.
        :param point_dimension: Dimension of input points (e.g., 3).
        :param base_potence: Base power of 2 for defining layer widths.
        """
        super(BasePointNet, self).__init__()
        n_64 = int(math.pow(2, base_potence))
        n_128 = int(math.pow(2, base_potence + 1))
        self.n_256 = int(math.pow(2, base_potence + 2))

        self.input_transform = TransformationNet(input_dim=point_dimension, kernel=point_dimension,
                                                 output_dim=point_dimension, base_potence=base_potence)
        self.feature_transform = TransformationNet(input_dim=n_64, kernel=1, output_dim=n_64, base_potence=base_potence)

        self.conv_1 = nn.Conv2d(point_dimension, n_64, (point_dimension, 1))
        self.conv_2 = nn.Conv2d(n_64, n_64, (1, 1))
        self.conv_3 = nn.Conv2d(n_64, n_64, (1, 1))
        self.conv_4 = nn.Conv2d(n_64, n_128, (1, 1))
        self.conv_5 = nn.Conv2d(n_128, self.n_256, (1, 1))

        self.bn_1 = RobustNorm(n_64, 8)
        self.bn_2 = RobustNorm(n_64, 8)
        self.bn_3 = RobustNorm(n_64, 8)
        self.bn_4 = RobustNorm(n_128, 8)
        self.bn_5 = RobustNorm(self.n_256, 8)

    def forward(self, x):
        """
        Forward pass of BasePointNet.
        :param x: Input point cloud of shape [batch, point_dimension, points].
        :return: Global feature vector of shape [batch, n_256].
        """
        input_transform = self.input_transform(x.unsqueeze(-1))  # T-Net tensor [batch, 3, 3]
        x = x.transpose(2, 1)  # [batch, 3, 478] -> [batch, 478, 3]
        x = torch.bmm(x, input_transform)  # Batch matrix-matrix product
        x = x.transpose(2, 1)  # [batch, 478, 3] -> [batch, 3, 478]
        x = x.unsqueeze(-1)

        # Adjust for small number of points in conv_1 (kernel size 3)
        num_points = x.shape[2]
        if num_points < self.conv_1.kernel_size[0]:
            padding_needed = self.conv_1.kernel_size[0] - num_points
            x = F.pad(x, (0, 0, 0, padding_needed))

        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))

        feature_transform = self.feature_transform(x)  # T-Net tensor [batch, 64, 64]
        x = x.squeeze(dim=-1)
        x = x.transpose(2, 1)
        x = torch.bmm(x, feature_transform)  # local point features [batch, 200, 64]
        x = x.transpose(2, 1)
        x = x.unsqueeze(-1)
        x = F.relu(self.bn_3(self.conv_3(x)))
        x = F.relu(self.bn_4(self.conv_4(x)))
        x = F.relu(self.bn_5(self.conv_5(x)))
        x = torch.max(x, dim=2, keepdim=True)[0]
        x = torch.flatten(x, start_dim=1)

        return x


class ClassificationPointNet(nn.Module):
    """
    Classification model using PointNet architecture.
    Supports additional metadata (e.g., age, gender) concatenated to global features.
    """

    def __init__(self, num_classes, dropout=0.3, point_dimension=3, base_potence=6, meta_data: List[str] = []):
        """
        Initialize the ClassificationPointNet.
        :param num_classes: Number of output classes.
        :param dropout: Dropout rate for the classifier head.
        :param point_dimension: Dimension of input points.
        :param base_potence: Base power of 2 for defining layer widths.
        :param meta_data: List of metadata keys to include in classification.
        """
        super(ClassificationPointNet, self).__init__()
        self.meta_data = meta_data
        n_64 = int(math.pow(2, base_potence))
        n_128 = int(math.pow(2, base_potence + 1))
        n_256 = int(math.pow(2, base_potence + 2))

        self.base_pointnet = BasePointNet(point_dimension=point_dimension, base_potence=base_potence)
        self.classifier_head = nn.Sequential(
            nn.Linear(n_256 + len(meta_data), n_128),
            RobustNorm(n_128, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_128, n_64),
            RobustNorm(n_64, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_64, num_classes),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of ClassificationPointNet.
        :param x: Input point cloud of shape [batch, point_dimension, points].
        :param kwargs: Additional metadata values corresponding to self.meta_data.
        :return: Tuple of (predictions, global_features).
                 predictions: Logits of shape [batch, num_classes].
                 global_features: Global feature vector of shape [batch, n_256].
        """
        features = self.base_pointnet(x)
        features_with_meta = features
        for meta_key in self.meta_data:
            if meta_key in kwargs:
                meta_value = kwargs[meta_key]
                meta_value = meta_value.unsqueeze(-1)
                features_with_meta = torch.concat((features_with_meta, meta_value), dim=1)
        pred = self.classifier_head(features_with_meta)
        return pred, features
