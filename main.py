import torch
import argparse
from DCCRN import DCCRN
import soundfile as sf
from dataloader import DNSDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path',            type=str,   help='path to the filenames text file', required=True)
    parser.add_argument('--batch_size',                type=int,   help='batch size', default=2)
    parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=20)
    parser.add_argument('--lr',             type=float,   help='learning rate', default=1e-4)
    parser.add_argument('--exp_name',             type=str,   help='experiment name', default='')
    parser.add_argument('--loadmodel', help='load model')
    parser.add_argument('--seed', type=int, default=100, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    writer = SummaryWriter('logs/' + args.exp_name)

    train_loader = torch.utils.data.DataLoader(DNSDataset(args.json_path), batch_size=args.batch_size, shuffle=True)
    model = DCCRN(rnn_units=256, masking_mode='E',use_clstm=False).cuda()
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999))
    start_epoch = 0
    iter = 0 

    for epoch in range(start_epoch, args.num_epochs):
        with tqdm(total=len(train_loader.dataset)) as pbar:
            for mix, clean, noise in train_loader:
                optimizer.zero_grad()
                mix, clean = mix.cuda(), clean.cuda()
                output = model(mix.cuda())  # [B, fft//2, 4803]
                real_loss, imag_loss = model.mask_mse_loss(output, mix, clean)
                loss = real_loss + imag_loss
                loss.backward()
                optimizer.step()

                writer.add_scalar('Train_iter/real_loss', real_loss.data, iter)
                writer.add_scalar('Train_iter/imag_loss', imag_loss.data, iter)
                iter += 1

                pbar.set_description(
                f"real_loss: {real_loss.item():.5f}; imag_loss: {imag_loss.item():.5f}"
                )
                pbar.update(mix.size(0))
                # sf.write('input.wav', mix[0].detach().numpy(),16000)
                # sf.write('output.wav', output[1][0].detach().numpy(),16000)


    
