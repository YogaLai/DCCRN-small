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

def get_crm(y, s, masking_mode='E'):
    """
    y: noisy spectrogram
    s: clean spectrogram
    """
    mask_real = (y.real * s.real + y.imag * s.imag) / (y.real**2 + y.imag**2)
    mask_imag = (y.real * s.imag - y.imag * s.real) / (y.real**2 + y.imag**2)

    if masking_mode == 'E':
        mask_mags = (mask_real**2+mask_imag**2)**0.5
        real_phase = mask_real/(mask_mags+1e-8)
        imag_phase = mask_imag/(mask_mags+1e-8)
        mask_phase = torch.atan2(imag_phase, real_phase) 
        mask_mags = torch.tanh(mask_mags)
        return mask_mags, mask_phase

    else:
        return mask_real + 1j * mask_imag

def visualize_mask(mask):
    mask_mag = mask[0][0]
    plt.imshow(mask_mag)