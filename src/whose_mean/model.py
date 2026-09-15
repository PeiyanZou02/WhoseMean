"""The 512 px pix2pix generator, discriminator, and image losses."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Generator512(nn.Module):
    """Six-scale U-Net. A 512 px image reaches an 8 x 8 bottleneck."""

    def __init__(self) -> None:
        super().__init__()

        def down(input_channels: int, output_channels: int, normalize: bool = True) -> nn.Sequential:
            layers: list[nn.Module] = [nn.Conv2d(input_channels, output_channels, 4, 2, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(output_channels))
            return nn.Sequential(*layers)

        def up(input_channels: int, output_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, 4, 2, 1),
                nn.InstanceNorm2d(output_channels),
            )

        self.e1 = down(3, 32, False)
        self.e2 = down(32, 64)
        self.e3 = down(64, 128)
        self.e4 = down(128, 256)
        self.e5 = down(256, 512)
        self.e6 = down(512, 512)
        self.d1 = up(512, 512)
        self.d2 = up(1024, 256)
        self.d3 = up(512, 128)
        self.d4 = up(256, 64)
        self.d5 = up(128, 32)
        self.out = nn.ConvTranspose2d(64, 3, 4, 2, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        a = F.leaky_relu(self.e1(image), 0.2)
        b = F.leaky_relu(self.e2(a), 0.2)
        c = F.leaky_relu(self.e3(b), 0.2)
        d = F.leaky_relu(self.e4(c), 0.2)
        e = F.leaky_relu(self.e5(d), 0.2)
        z = F.leaky_relu(self.e6(e), 0.2)
        u = F.relu(self.d1(z))
        v = F.relu(self.d2(torch.cat([u, e], 1)))
        w = F.relu(self.d3(torch.cat([v, d], 1)))
        q = F.relu(self.d4(torch.cat([w, c], 1)))
        r = F.relu(self.d5(torch.cat([q, b], 1)))
        return torch.tanh(self.out(torch.cat([r, a], 1)))


class Discriminator512(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 3, 1, 1),
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([source, target], 1))


def edge_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    horizontal = F.l1_loss(
        output[:, :, :, 1:] - output[:, :, :, :-1],
        target[:, :, :, 1:] - target[:, :, :, :-1],
    )
    vertical = F.l1_loss(
        output[:, :, 1:, :] - output[:, :, :-1, :],
        target[:, :, 1:, :] - target[:, :, :-1, :],
    )
    return horizontal + vertical
