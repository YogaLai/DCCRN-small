#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display
import numpy as np

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

# def visualize_spec(noisy_path, clean_path, estimate_path):
#     noisy, sample_rate = sf.read(noisy_path)
#     clean, sample_rate = sf.read(clean_path)
#     estimate, sample_rate = sf.read(estimate_path)
#     # spec = stft(wav)
#     plt.subplot(1,3,1)
#     plt.specgram(noisy, 512, sample_rate, cmap='jet')
#     plt.subplot(1,3,2)
#     plt.specgram(clean, 512, sample_rate, cmap='jet')
#     plt.subplot(1,3,3)
#     plt.specgram(estimate, 512, sample_rate, cmap='jet')
#     plt.show()

def save_spec_img(path, filename):
    wav, sample_rate = librosa.load(path)
    spec = librosa.stft(wav)
    spec = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
    plt.figure()
    librosa.display.specshow(spec)
    plt.colorbar()
    plt.savefig(f'visualization/{filename}.png')
  
def visualize_spec():
    save_spec_img('D:/yoga/noisy.wav', 'noisy')
    save_spec_img('D:/yoga/clean.wav', 'clean')
    save_spec_img('D:/yoga/estimate.wav', 'estimate')

if __name__ == '__main__':
    visualize_spec()
