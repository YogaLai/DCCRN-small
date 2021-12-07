import torch
import argparse
from DCCRN_TCN import DCCRN
import soundfile as sf
from dataloader import DNSDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path',            type=str,   help='path to the filenames text file', required=True)
    parser.add_argument('--batch_size',                type=int,   help='batch size', default=2)
    parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=20)
    parser.add_argument('--lr',             type=float,   help='learning rate', default=1e-4)
    parser.add_argument('--exp_name',             type=str,   help='experiment name', default='')
    parser.add_argument('--loadmodel', help='load model')
    parser.add_argument('--cal_batch_size', type=int, default=4, help='batch_size is the size of loading batch. cal_batch_size is the number of caluation')
    parser.add_argument('--loadmodel', type=str, help="checkpoint path")
    parser.add_argument('--seed', type=int, default=100, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    writer = SummaryWriter('logs/' + args.exp_name)

    if not os.path.isdir('savemodel/' + args.exp_name):
        os.makedirs('savemodel/' + args.exp_name)

    train_loader = torch.utils.data.DataLoader(DNSDataset(args.json_path), batch_size=args.batch_size, shuffle=True)
    model = DCCRN(rnn_units=256, masking_mode='E',use_clstm=False, out_mask=False).cuda()
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999))
    start_epoch = 0
    iter = 0 
    # frame_dur = int(37.5 / 1000 * 16000)

    

    for epoch in range(start_epoch, args.num_epochs):
        total_loss = 0
        with tqdm(total=len(train_loader.dataset)) as pbar:
            for mix, clean in train_loader:
                mix = mix.view(mix.size(0)*args.cal_batch_size, mix.size(1)//args.cal_batch_size)
                clean = clean.view(clean.size(0)*args.cal_batch_size, clean.size(1)//args.cal_batch_size)
                # for index in range(0, mix.size(0)-args.cal_batch_size+1, args.cal_batch_size):
                #     optimizer.zero_grad()
                #     input, target = mix[index:index+args.cal_batch_size, :].cuda(), clean[index:index+args.cal_batch_size, :].cuda()
                #     outputs = model(input)  # [B, fft//2, 4803]
                #     loss = model.loss(outputs[1], target, loss_mode='SI-SNR')
                #     loss.backward()
                #     optimizer.step()
                optimizer.zero_grad()
                mix, clean = mix.cuda(), clean.cuda()
                outputs = model(mix)  # [B, fft//2, 4803]
                loss = model.loss(outputs[1], clean, loss_mode='SI-SNR')
                loss.backward()
                optimizer.step()
                writer.add_scalar('Train_iter/loss', loss.data, iter)
                iter += 1

                total_loss += float(loss)

                pbar.set_description(
                f"real_loss: {loss.item():.5f}"
                )
                pbar.update(mix.size(0)//args.cal_batch_size)
                    

        writer.add_scalar('Train_epoch/total_loss', total_loss/len(train_loader.dataset)/4, iter)
        

        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': total_loss,
            },  f'savemodel/{args.exp_name}/checkpoint_{epoch}.tar')


    
