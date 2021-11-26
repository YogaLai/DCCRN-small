import torch
import argparse
from DCCRN import DCCRN
import soundfile as sf
from dataloader import DNSDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path',            type=str,   help='path to the filenames text file', required=True)
    parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
    parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=20)
    parser.add_argument('--lr',             type=float,   help='learning rate', default=1e-4)
    parser.add_argument('--exp_name',             type=str,   help='experiment name', default='')
    parser.add_argument('--loadmodel', help='load model')
    parser.add_argument('--seed', type=int, default=100, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()

    train_loader = torch.utils.data.DataLoader(DNSDataset(args.json_path), batch_size=args.batch_size, shuffle=True)
    model = DCCRN(rnn_units=256,masking_mode='E',use_clstm=True,kernel_num=[32, 64, 128, 256, 256, 256])

    for mix, clean, noise in train_loader:
        output = model(mix)
        sf.write('input.wav', mix[0].detach().numpy(),16000)
        sf.write('output.wav', output[1].detach().numpy(),16000)
        print(output)


    
