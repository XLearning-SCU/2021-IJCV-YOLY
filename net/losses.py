import torch
from torch import nn
import numpy as np
from torch.nn import functional


class StdLoss(nn.Module):
    def __init__(self):
        """
        Loss on the variance of the image.
        Works in the grayscale.
        If the image is smooth, gets zero
        """
        super(StdLoss, self).__init__()
        blur = (1 / 25) * np.ones((5, 5))
        blur = blur.reshape(1, 1, blur.shape[0], blur.shape[1])
        self.mse = nn.MSELoss()
        self.blur = nn.Parameter(data=torch.cuda.FloatTensor(blur), requires_grad=False)
        image = np.zeros((5, 5))
        image[2, 2] = 1
        image = image.reshape(1, 1, image.shape[0], image.shape[1])
        self.image = nn.Parameter(data=torch.cuda.FloatTensor(image), requires_grad=False)
        self.gray_scale = GrayscaleLayer()

    def forward(self, x):
        x = self.gray_scale(x)
        return self.mse(functional.conv2d(x, self.image), functional.conv2d(x, self.blur))

class GrayscaleLayer(nn.Module):
    def __init__(self):
        super(GrayscaleLayer, self).__init__()

    def forward(self, x):
        return torch.mean(x, 1, keepdim=True)
