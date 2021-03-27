#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 21:29:27 2020

@author: lester
"""

def write_log(file_name, title, psnr, ssim):
    fp = open(file_name, "a+")
    fp.write(title+ ':\n')
    fp.write('PSNR:%0.6f\n'%psnr)
    fp.write('SSIM:%0.6f\n'%ssim)
    fp.close()
