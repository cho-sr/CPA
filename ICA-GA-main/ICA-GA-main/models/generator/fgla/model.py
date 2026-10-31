# --------------------------------------------------
# Fast Generation-Based Gradient Leakage Attacks:
# An Approach to Generate Training Data Directly From the Gradient
# https://github.com/pigeon-dove/FGLA/blob/master/gen_attack/model.py
# --------------------------------------------------

import torch.nn as nn

from ...base_model import BaseModel


class FglaGenerator(BaseModel):
    class ResConv(nn.Module):
        def __init__(self, channel_size):
            super().__init__()
            self.act = nn.LeakyReLU()
            self.conv = nn.Sequential(
                nn.Conv2d(channel_size, channel_size, 3, padding=1, bias=False),
                nn.BatchNorm2d(channel_size),
                self.act,
                nn.Conv2d(channel_size, channel_size, 3, padding=1, bias=False),
                nn.BatchNorm2d(channel_size),
            )

        def forward(self, x):
            out = self.conv(x) + x
            return self.act(out)


class FglaGenerator18(FglaGenerator):
    def __init__(self, *args, **kwargs):
        super(FglaGenerator18, self).__init__(*args, **kwargs)

        self.conv = nn.Sequential(
            # 14 * 14
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            self.ResConv(256),
            self.ResConv(256),
            self.ResConv(256),
            # 28 * 28
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            self.ResConv(128),
            self.ResConv(128),
            self.ResConv(128),
            # 56 * 56
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            self.ResConv(64),
            self.ResConv(64),
            self.ResConv(64),
            # 112 * 112
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            self.ResConv(32),
            self.ResConv(32),
            self.ResConv(32),
            # 224 * 224
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            self.ResConv(16),
            self.ResConv(16),
            self.ResConv(16),
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = x.view(-1, 512, 7, 7)
        out = self.conv(out)
        return out


class FglaGenerator50(FglaGenerator):
    def __init__(self, *args, **kwargs):
        super(FglaGenerator50, self).__init__(*args, **kwargs)

        self.conv = nn.Sequential(
            # 14 * 14
            nn.ConvTranspose2d(2048, 1024, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            self.ResConv(1024),
            self.ResConv(1024),
            self.ResConv(1024),
            # 28 * 28
            nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            self.ResConv(512),
            self.ResConv(512),
            self.ResConv(512),
            # 56 * 56
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            self.ResConv(256),
            self.ResConv(256),
            self.ResConv(256),
            # 112 * 112
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            self.ResConv(128),
            self.ResConv(128),
            self.ResConv(128),
            # 224 * 224
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            self.ResConv(64),
            self.ResConv(64),
            self.ResConv(64),
            nn.Conv2d(64, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = x.view(-1, 2048, 7, 7)
        out = self.conv(out)
        return out
