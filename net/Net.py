#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

class Net(torch.nn.Module):
    def __init__(self, out_channel):
        super().__init__()
        #        torch.nn.Conv1d()
        #        self
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, out_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.LeakyReLU(inplace=True)
            #                torch.nn.ReLU(inplace = True)
            #                torch.nn.MaxPool2d(2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            #                torch.nn.BatchNorm2d(32),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.LeakyReLU(inplace=True)
            #                torch.nn.ReLU(inplace = True)
            #                torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            #                torch.nn.BatchNorm2d(64),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.LeakyReLU(inplace=True)
            #                torch.nn.ReLU(inplace = True)
            #                torch.nn.MaxPool2d(2)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            #                torch.nn.BatchNorm2d(128),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.LeakyReLU(inplace=True)
            # torch.nn.ReLU(inplace = True)
            # torch.nn.MaxPool2d(2)
        )
        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            torch.nn.Sigmoid()
        )

    def forward(self, data):
        data = self.conv1(data)
        data = self.conv2(data)
        data = self.conv3(data)
        data = self.conv4(data)
        data = self.final(data)
        return data

# class My_Net(torch.nn.Module):
#    def __init__(self, out_channel):
#        super().__init__()
#        #        torch.nn.Conv1d()
##        self
#        self.conv1 = torch.nn.Sequential(
#                torch.nn.Conv2d(3, 3, 5, 1, 2),
#                torch.nn.BatchNorm2d(3),
#                torch.nn.LeakyReLU(inplace = True)
##                torch.nn.ReLU(inplace = True)
##                torch.nn.MaxPool2d(2)
#                )
#        self.conv2 = torch.nn.Sequential(
#                torch.nn.Conv2d(3, 3, 5, 1, 2),
##                torch.nn.BatchNorm2d(32),
#                torch.nn.BatchNorm2d(3),
#                torch.nn.LeakyReLU(inplace = True)
##                torch.nn.ReLU(inplace = True)
##                torch.nn.MaxPool2d(2)
#                )
#        self.conv3 = torch.nn.Sequential(
#                torch.nn.Conv2d(3, 3, 5, 1, 2),
##                torch.nn.BatchNorm2d(64),
#                torch.nn.BatchNorm2d(3),
#                torch.nn.LeakyReLU(inplace = True)
##                torch.nn.ReLU(inplace = True)
##                torch.nn.MaxPool2d(2)
#                )
#        self.conv4 = torch.nn.Sequential(
#                torch.nn.Conv2d(3, 3, 5, 1, 2),
##                torch.nn.BatchNorm2d(128),
#                torch.nn.BatchNorm2d(3),
#                torch.nn.LeakyReLU(inplace = True)
##                torch.nn.ReLU(inplace = True)
##                torch.nn.MaxPool2d(2)
#                )
#        self.final = torch.nn.Sequential(
#                torch.nn.Conv2d(3, out_channel, 5, 1, 2),
#                torch.nn.Sigmoid()
#                )
#
#    def forward(self, data):
#        data = self.conv1(data)
#        data = self.conv2(data)
#        data = self.conv3(data)
#        data = self.conv4(data)
#        data = self.final(data)
#        return data
