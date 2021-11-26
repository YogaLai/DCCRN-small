import torch
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
    parser.add_argument('--filename_path',            type=str,   help='path to the filenames text file', required=True)
    parser.add_argument('--batch_size',                type=int,   help='batch size', default=12)
    parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=20)
    parser.add_argument('--lr',             type=float,   help='learning rate', default=1e-4)
    parser.add_argument('--exp_name',             type=str,   help='experiment name', default='')
    parser.add_argument('--loadmodel', help='load model')
    parser.add_argument('--seed', type=int, default=100, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()

    
