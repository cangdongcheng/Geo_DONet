"""Geometry-conditioned AT and temporal-PCA prediction."""
import torch
from torch import nn


def build_mlp(input_dim, width, depth):
    """Build `depth` Linear/Tanh layers, each with `width` hidden units."""
    layers = []
    current = input_dim
    for _ in range(depth):
        layers.append(nn.Sequential(nn.Linear(current, width), nn.Tanh()))
        current = width
    return nn.Sequential(*layers)


class FeatureDeepONet(nn.Module):
    """Predict D nodewise features from geometry theta and spatial coordinate x.

    Shared branch/trunk embeddings are combined multiplicatively.  Each output
    has its own learned reduction of that interaction:

        y[b,n,d] = sum_w branch[b,w] * trunk[n,w] * head[d,w] + bias[d]

    Output 0 is activation time; outputs 1..D-1 are aligned temporal-PCA
    coefficients.  Keeping the legacy ``_branch_g``/``_trunk`` names allows an
    old AT-only Geo_DeepONet checkpoint to initialise these two encoders.
    """

    def __init__(self, geo_dim=60, coord_dim=4, width=200, depth=4, output_dim=6):
        super().__init__()
        self.geo_dim = geo_dim
        self.coord_dim = coord_dim
        self.width = width
        self.depth = depth
        self.output_dim = output_dim

        # 1. Geometry branch: one global theta vector per heart.
        self._branch_g = build_mlp(geo_dim, width, depth)

        # 2. Spatial trunk: the four Cobiveco coordinates at each canonical node.
        self._trunk = build_mlp(coord_dim, width, depth)

        # 3. Learned output heads: AT followed by K temporal-PCA coefficients.
        self.head = nn.Linear(width, output_dim)

    def forward(self, theta, coords):
        """theta: (B, G), coords: (M, 4) -> features: (B, M, 1+K)."""
        branch = self._branch_g(theta)       # (B, W)
        trunk = self._trunk(coords)          # (N, W)
        output = torch.einsum("bw,nw,dw->bnd", branch, trunk, self.head.weight)
        return output + self.head.bias

    def config(self):
        return {
            'geo_dim': self.geo_dim,
            'coord_dim': self.coord_dim,
            'width': self.width,
            'depth': self.depth,
            'output_dim': self.output_dim,
        }
