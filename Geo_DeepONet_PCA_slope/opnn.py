"""Multi-output DeepONet for AT, slope, and aligned PCA controls."""
import torch
import torch.nn as nn


def build_mlp(input_dim, width, depth):
    layers = []
    current = input_dim
    for _ in range(depth):
        layers.append(nn.Sequential(nn.Linear(current, width), nn.Tanh()))
        current = width
    return nn.Sequential(*layers)


class FeatureDeepONet(nn.Module):
    """Geometry branch + spatial trunk with one learned head per feature.

    y[b,n,d] = sum_w branch[b,w] * trunk[n,w] * head[d,w] + bias[d]
    """
    def __init__(self, geo_dim=60, coord_dim=4, width=200, depth=4,
                 output_dim=10):
        super().__init__()
        self.geo_dim = geo_dim
        self.coord_dim = coord_dim
        self.width = width
        self.depth = depth
        self.output_dim = output_dim
        self.branch = build_mlp(geo_dim, width, depth)
        self.trunk = build_mlp(coord_dim, width, depth)
        self.head = nn.Linear(width, output_dim)

    def forward(self, theta, coords):
        branch = self.branch(theta)
        trunk = self.trunk(coords)
        output = torch.einsum("bw,nw,dw->bnd", branch, trunk,
                              self.head.weight)
        return output + self.head.bias

    def config(self):
        return dict(geo_dim=self.geo_dim, coord_dim=self.coord_dim,
                    width=self.width, depth=self.depth,
                    output_dim=self.output_dim)
