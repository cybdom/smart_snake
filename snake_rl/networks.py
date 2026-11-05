"""Custom neural-network architecture used by the PPO agent.

SB3 default policies are NOT used: the observation is processed by a hand-designed
two-branch feature extractor, followed by custom (separate) actor and critic MLP heads.

           grid (5 x 14 x 14)                     features (14)
                  |                                     |
   Conv3x3(32)+ReLU -> Conv3x3(64, s2)+ReLU      Linear(64)+ReLU
   Conv3x3(64)+ReLU -> Flatten                   Linear(64)+ReLU
   Linear(128)+LayerNorm+ReLU                           |        features (skip)
                  \\_______________ concat ______________/_______________/
                                (128 + 64 + 14 = 206)
                    /                         \\
          actor: 128-64 (ReLU)          critic: 128-64 (ReLU)
          -> 3 action logits            -> state value V(s)
"""

from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class SnakeFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, cnn_out: int = 128, mlp_out: int = 64):
        grid_space = observation_space["grid"]
        feat_space = observation_space["features"]
        n_feat = feat_space.shape[0]
        super().__init__(observation_space, features_dim=cnn_out + mlp_out + n_feat)
        c = grid_space.shape[0]

        # Spatial branch: first layer at full resolution (local patterns: head next to body/wall),
        # then a strided conv halves the map (14 -> 7) and a last conv aggregates wider context.
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, *grid_space.shape)).shape[1]
        # LayerNorm keeps the (initially noisy) spatial embedding on the same scale as the
        # vector branch, so it does not drown out the informative hand-crafted features.
        self.cnn_head = nn.Sequential(nn.Linear(n_flat, cnn_out), nn.LayerNorm(cnn_out), nn.ReLU())

        # Vector branch: local danger / direction information.
        self.mlp = nn.Sequential(
            nn.Linear(n_feat, mlp_out),
            nn.ReLU(),
            nn.Linear(mlp_out, mlp_out),
            nn.ReLU(),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        g = self.cnn_head(self.cnn(obs["grid"]))
        f = self.mlp(obs["features"])
        # raw features are also passed through (skip connection) -> fast early learning
        return torch.cat([g, f, obs["features"]], dim=1)


def policy_kwargs() -> dict:
    return dict(
        features_extractor_class=SnakeFeatureExtractor,
        features_extractor_kwargs=dict(cnn_out=128, mlp_out=64),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
        activation_fn=nn.ReLU,
        share_features_extractor=True,
    )
