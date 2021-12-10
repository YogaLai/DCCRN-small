from tqdm import tqdm
import torch
import pandas as pd
from asteroid.metrics import get_metrics


def validate_sisnr(model, val_loader):
    with tqdm(total=len(val_loader.dataset)) as pbar:
        model.eval()
        total_val_loss = 0
        for mix, clean in val_loader:
            with torch.no_grad():
                mix, clean = mix.cuda(), clean.cuda()
                outputs = model(mix)  # [B, fft//2, 4803]
                val_loss = model.loss(outputs[1], clean, loss_mode='SI-SNR')
                total_val_loss += float(val_loss)

                pbar.set_description(
                f"val_loss: {val_loss.item():.5f}"
                )
                pbar.update(mix.size(0))
    
    return total_val_loss

def validate_pesq(model, val_loader):
    with tqdm(total=len(val_loader.dataset)) as pbar:
        model.eval()
        total_val_loss = 0
        for mix, clean in val_loader:
            with torch.no_grad():
                # mix, clean = mix.cuda(), clean.cuda()
                outputs = model(mix.cuda()) 
                utt_metrics = get_metrics(
                    mix=mix.data.numpy(),
                    clean=clean.data.numpy(),
                    estimate=outputs[1].cpu().data.numpy(),
                    sample_rate=16000,
                    metrics_list=["pesq"],
                )

                total_val_loss += utt_metrics["pesq"]

                pbar.set_description(
                f"pesq: {utt_metrics['pesq']:.5f}"
                )
                pbar.update(mix.size(0))

    return total_val_loss