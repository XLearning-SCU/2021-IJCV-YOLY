#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch

class VAE(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.encoder = Encoder(size)
        self.decoder = Decoder(size)

    def getLatent(self, means, var):
        log_var = var
        epsilon = torch.randn(means.size()).cuda()

        sigma = torch.exp(0.5 * log_var)
        z = means + sigma * epsilon

        self.means = means
        self.var = var
        return z

    def sample(self):
        z = self.getLatent(self.means, self.var)
        return self.decoder(z)

    def forward(self, data):
        means, var = self.encoder(data)
        z = self.getLatent(means, var)
        return self.decoder(z)

    def getLoss(self):
        log_var = self.var
        lossKL = 0.5 * torch.sum(log_var.exp() + self.means * self.means - 1 - log_var)
        loss = lossKL
        return loss


class Encoder(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 5, 1, 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 5, 1, 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 5, 1, 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 5, 1, 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2)
        )
        self.fc1 = torch.nn.Linear(int(128 * (size[1] // 16) * (size[2] // 16)), 100)
        self.fc2 = torch.nn.Linear(int(128 * (size[1] // 16) * (size[2] // 16)), 100)

    def forward(self, data):
        data = self.conv1(data)
        data = self.conv2(data)
        data = self.conv3(data)
        data = self.conv4(data)
        data = data.view(data.size(0), -1)
        means = self.fc1(data)
        var = self.fc2(data)
        return means, var


class Decoder(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear0 = torch.nn.Linear(100, int(128 * (size[1] // 16) * (size[2] // 16)))
        self.size = size

        self.de = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(128, 64, 5, 1, 2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(64, 32, 5, 1, 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(True),
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(32, 16, 5, 1, 2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(True),
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(16, 3, 5, 1, 2),
            torch.nn.Sigmoid()
        )

    def forward(self, data):
        data = self.linear0(data)
        data = data.view(1, -1, self.size[1] // 16, self.size[2] // 16)

        data = self.de(data)

        return data
