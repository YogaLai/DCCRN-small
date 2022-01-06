import torch
import argparse
from DCCRN_TCN import DCTCAD
from DCCRN import DCCRN
import soundfile as sf
from dataloader import DNSDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from validate import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path',            type=str,   help='path to the filenames json file', required=True)
    parser.add_argument('--val_json_path',            type=str,   help='path to the validation filenames json file', required=True)
    parser.add_argument('--val_reverb_json_path',            type=str,   help='path to the reverb validation filenames json file', required=True)
    parser.add_argument('--batch_size',                type=int,   help='batch size', default=2)
    parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=20)
    parser.add_argument('--lr',             type=float,   help='learning rate', default=1e-3)
    parser.add_argument('--exp_name',             type=str,   help='experiment name', default='')
    parser.add_argument('--cal_batch_size', type=int, default=8, help='batch_size is the size of loading batch. cal_batch_size is the number of caluation')
    parser.add_argument('--loadmodel', type=str, help="checkpoint path")
    parser.add_argument('--model_name', type=str, default="tcn")
    parser.add_argument('--seed', type=int, default=100, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    writer = SummaryWriter('logs/' + args.exp_name)

    if not os.path.isdir('savemodel/' + args.exp_name):
        os.makedirs('savemodel/' + args.exp_name)

    train_loader = torch.utils.data.DataLoader(DNSDataset(args.json_path), batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(DNSDataset(args.val_json_path), batch_size=1, shuffle=False)
    # val_reverb_loader = torch.utils.data.DataLoader(DNSDataset(args.val_reverb_json_path), batch_size=1, shuffle=False)
    if args.model_name == 'tcn':
        model = DCTCAD(rnn_units=256, masking_mode='E', kernel_num=[16,32,64,128,128,256,256], use_clstm=False, out_mask=False).cuda()
    else:
        model = DCCRN(rnn_units=256, masking_mode='E', out_mask=False).cuda()

    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    start_epoch = 0
    iter = 0 
    best_pesq = 0
    # frame_dur = int(37.5 / 1000 * 16000)

    if args.loadmodel:
        ckt = torch.load(args.loadmodel)
        model.load_state_dict(ckt['state_dict'])
        start_epoch = ckt['epoch'] + 1
        iter = int(start_epoch * len(train_loader.dataset) / args.batch_size) + 1
        optimizer.load_state_dict(ckt['optimizer'])
        scheduler = ckt['scheduler']
        best_pesq = ckt['best_pesq']
        print('load model successfully')    

    for epoch in range(start_epoch, args.num_epochs):
        total_loss = 0
        with tqdm(total=len(train_loader.dataset)) as pbar:
            model.train()
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
                # snr_loss, rmse_loss = model.loss(outputs, clean, loss_mode='SI-SNR+RMSE')
                # loss = (snr_loss + 2*rmse_loss)/3
                snr_loss, mel_loss = model.loss(outputs, clean, loss_mode='SI-SNR+LMS')
                loss = (snr_loss + 2*mel_loss)/3
                loss.backward()
                optimizer.step()

                writer.add_scalar('Train_iter/snr_loss', snr_loss.data, iter)
                writer.add_scalar('Train_iter/rmse_loss', mel_loss.data, iter)
                iter += 1

                total_loss += float(loss)

                pbar.set_description(
                f"loss: {loss.item():.5f}"
                )
                pbar.update(mix.size(0)//args.cal_batch_size)

        total_val_loss = validate_pesq(model, val_loader)
        # total_val_reverb_loss = validate_pesq(model, val_reverb_loader)
        total_val_loss /= len(val_loader.dataset)
        # total_val_reverb_loss /= len(val_reverb_loader.dataset)

        scheduler.step(total_val_loss)
        writer.add_scalar('Train_epoch/total_loss', total_loss/len(train_loader.dataset)/4, epoch)
        writer.add_scalar('Val_epoch/pesq', total_val_loss, epoch)
        # writer.add_scalar('Val_epoch/reverb_pesq', total_val_reverb_loss, epoch)
        
        if total_val_loss > best_pesq:
            best_pesq = total_val_loss
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_pesq': best_pesq,
                # 'reverb_pesq': total_val_reverb_loss,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler,
            },  f'savemodel/{args.exp_name}/checkpoint_best.tar')

        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': total_loss,
            'scheduler': scheduler,
            'optimizer': optimizer.state_dict(),
            'best_pesq': best_pesq,
            'pesq': total_val_loss,
            # 'reverb_pesq': total_val_reverb_loss
        },  f'savemodel/{args.exp_name}/checkpoint_{epoch}.tar')
        
            

    
