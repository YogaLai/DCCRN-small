#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import matplotlib.pyplot as plt

def show_params(nnet):
    print("=" * 40, "Model Parameters", "=" * 40)
    num_params = 0
    for module_name, m in nnet.named_modules():
        if module_name == '':
            for name, params in m.named_parameters():
                print(name, params.size())
                i = 1
                for j in params.size():
                    i = i * j
                num_params += i
    print('[*] Parameter Size: {}'.format(num_params))
    print("=" * 98)


def show_model(nnet):
    print("=" * 40, "Model Structures", "=" * 40)
    for module_name, m in nnet.named_modules():
        if module_name == '':
            print(m)
    print("=" * 98)


def visualize_mask(mask):
    mask_mag = mask[0][0]
    plt.imshow(mask_mag)

def spec2complex(x, fft_len):
    """
    convert input to torch complexed tensor
    """
    real = x[:,:fft_len//2+1]
    imag = x[:,fft_len//2+1:]

    return torch.complex(real, imag)