#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 10:36:59 2020

@author: lester
"""
from collections import namedtuple
from cv2.ximgproc import guidedFilter
from net.losses import StdLoss
from utils.imresize import imresize, np_imresize
from utils.image_io import *
from skimage.color import rgb2hsv
import torch
import torch.nn as nn
from net.vae import VAE
import numpy as np
from net.Net import Net
from options import options

def get_dark_channel(image, w=15):
    """
    Get the dark channel prior in the (RGB) image data.
    Parameters
    -----------
    image:  an M * N * 3 numpy array containing data ([0, L-1]) in the image where
        M is the height, N is the width, 3 represents R/G/B channels.
    w:  window size
    Return
    -----------
    An M * N array for the dark channel prior ([0, L-1]).
    """
    M, N, _ = image.shape
    padded = np.pad(image, ((w // 2, w // 2), (w // 2, w // 2), (0, 0)), 'edge')
    darkch = np.zeros((M, N))
    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i:i + w, j:j + w, :])  # CVPR09, eq.5
    return darkch


def get_atmosphere(image, p=0.0001, w=15):
    """Get the atmosphere light in the (RGB) image data.
    Parameters
    -----------
    image:      the 3 * M * N RGB image data ([0, L-1]) as numpy array
    w:      window for dark channel
    p:      percentage of pixels for estimating the atmosphere light
    Return
    -----------
    A 3-element array containing atmosphere light ([0, L-1]) for each channel
    """
    image = image.transpose(1, 2, 0)
    # reference CVPR09, 4.4
    darkch = get_dark_channel(image, w)
    M, N = darkch.shape
    flatI = image.reshape(M * N, 3)
    flatdark = darkch.ravel()
    searchidx = (-flatdark).argsort()[:int(M * N * p)]  # find top M * N * p indexes
    # return the highest intensity for each channel
    return np.max(flatI.take(searchidx, axis=0), axis=0)


DehazeResult_RW = namedtuple("DehazeResult", ['learned', 't', 'a'])


class Dehaze(object):

    def __init__(self, image_name, image, opt):
        self.image_name = image_name
        self.image = image
        self.num_iter = opt.num_iter
        self.ambient_net = None
        self.image_net = None
        self.mask_net = None
        self.ambient_val = None
        self.mse_loss = None
        self.learning_rate = opt.learning_rate
        self.parameters = None
        self.current_result = None
        self.output_path = "output/" + opt.datasets + '/' + opt.name + '/'

        self.data_type = torch.cuda.FloatTensor
        self.clip = opt.clip
        self.blur_loss = None
        self.best_result = None
        self.best_result_ssim = None
        self.image_net_inputs = None
        self.mask_net_inputs = None
        self.image_out = None
        self.mask_out = None
        self.ambient_out = None
        self.total_loss = None
        self._init_all()

    def _init_images(self):
        self.original_image = self.image.copy()
        factor = 1
        image = self.image
        image_size = 1000

        while image.shape[1] >= image_size or image.shape[2] >= image_size:
            new_shape_x, new_shape_y = self.image.shape[1] / factor, self.image.shape[2] / factor
            new_shape_x -= (new_shape_x % 32)
            new_shape_y -= (new_shape_y % 32)
            image = np_imresize(self.image, output_shape=(new_shape_x, new_shape_y))
            factor += 1

        self.image = image
        self.image_torch = np_to_torch(self.image).type(torch.cuda.FloatTensor)

    def _init_nets(self):
        image_net = Net(out_channel=3)

        self.image_net = image_net.type(self.data_type)

        mask_net = Net(out_channel=1)

        self.mask_net = mask_net.type(self.data_type)

    def _init_ambient(self):
        ambient_net = VAE(self.image.shape)

        self.ambient_net = ambient_net.type(torch.cuda.FloatTensor)
        atmosphere = get_atmosphere(self.image)

        self.ambient_val = nn.Parameter(data=torch.cuda.FloatTensor(atmosphere.reshape((1, 3, 1, 1))),
                                        requires_grad=False)

    def _init_parameters(self):
        parameters = [p for p in self.image_net.parameters()] + \
                     [p for p in self.mask_net.parameters()] + \
                     [p for p in self.ambient_net.parameters()]

        self.parameters = parameters

    def _init_loss(self):
        self.mse_loss = torch.nn.MSELoss().type(self.data_type)
        self.blur_loss = StdLoss().type(self.data_type)

    def _init_inputs(self):
        self.image_net_inputs = np_to_torch(self.image).cuda().type(self.data_type)
        self.mask_net_inputs = np_to_torch(self.image).cuda().type(self.data_type)
        self.ambient_net_input = np_to_torch(self.image).cuda().type(self.data_type)

    def _init_all(self):
        self._init_images()
        self._init_nets()
        self._init_ambient()
        self._init_inputs()
        self._init_parameters()
        self._init_loss()

    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.num_iter):
            optimizer.zero_grad()
            self._optimization_closure(j)
            self._obtain_current_result(j)
            self._plot_closure(j)
            optimizer.step()
            if j % 200 == 0:
                self.finalize(steps=j)

    def _optimization_closure(self, step):
        """
        :param step: the number of the iteration
        :return:
        """

        self.image_out = self.image_net(self.image_net_inputs)
        self.ambient_out = self.ambient_net(self.ambient_net_input)
        self.mask_out = self.mask_net(self.mask_net_inputs)

        self.mseloss = self.mse_loss(self.mask_out * self.image_out + (1 - self.mask_out) * self.ambient_out,
                                     self.image_torch)

        hsv = np_to_torch(rgb2hsv(torch_to_np(self.image_out).transpose(1, 2, 0)))
        cap_prior = hsv[:, :, :, 2] - hsv[:, :, :, 1]
        self.cap_loss = self.mse_loss(cap_prior, torch.zeros_like(cap_prior))
        vae_loss = self.ambient_net.getLoss()

        self.total_loss = self.mseloss
        self.total_loss += vae_loss
        self.total_loss += 1.0 * self.cap_loss
        self.total_loss += 0.001 * self.blur_loss(self.ambient_out)
        if step < 1000:
            self.total_loss += self.mse_loss(self.ambient_out, self.ambient_val * torch.ones_like(self.ambient_out))
        self.total_loss.backward(retain_graph=True)

    def _obtain_current_result(self, step):
        if step % 5 == 0:
            image_out_np = np.clip(torch_to_np(self.image_out), 0, 1)
            mask_out_np = np.clip(torch_to_np(self.mask_out), 0, 1)
            ambient_out_np = np.clip(torch_to_np(self.ambient_out), 0, 1)
            mask_out_np = self.t_matting(mask_out_np)

            self.current_result = DehazeResult_RW(learned=image_out_np, t=mask_out_np, a=ambient_out_np)

    def _plot_closure(self, step):
        print('Iteration %05d    Loss %f %f %0.4f%% \n' % (
            step, self.total_loss.item(),
            self.cap_loss,
            self.cap_loss / self.total_loss.item()), '\r', end='')

    def finalize(self, steps=800):

        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
            os.mkdir(self.output_path + 'Normal/')
            os.mkdir(self.output_path + 'Matting/')


        final_a = np_imresize(self.current_result.a, output_shape=self.image.shape[1:])
        final_t = np_imresize(self.current_result.t, output_shape=self.image.shape[1:])

        post = np.clip((self.image - ((1 - final_t) * final_a)) / final_t, 0, 1)
        save_image(self.image_name, post, self.output_path + 'Normal/' + str(steps) + '/')

        final_t = self.t_matting(final_t)
        post = np.clip((self.image - ((1 - final_t) * final_a)) / final_t, 0, 1)
        save_image(self.image_name, post, self.output_path + 'Matting/' + str(steps) + '/')

    def t_matting(self, mask_out_np):
        refine_t = guidedFilter(self.image.transpose(1, 2, 0).astype(np.float32),
                                mask_out_np[0].astype(np.float32), 50, 1e-4)
        if self.clip:
            return np.array([np.clip(refine_t, 0.1, 1)])
        else:
            return np.array([np.clip(refine_t, 0, 1)])



def dehazing(opt):
    torch.cuda.set_device(opt.cuda)

    hazy_add = 'data/' + opt.datasets + '/*.jpg'

    for item in sorted(glob.glob(hazy_add)):
        print(item)
        name = item.split('.')[0].split('/')[2]
        print(name)
        
        hazy_img = prepare_image(item)
        
        dh = Dehaze(name, hazy_img, opt)
        dh.optimize()
        dh.finalize()
    
if __name__ == "__main__":
    dehazing(options)

