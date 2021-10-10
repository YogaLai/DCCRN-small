import torchaudio
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt

def load_wav(path, sr=16000):
   wav, _ = librosa.load(path, sr=sr)
   return torch.tensor(wav)

class STFT():
    def __init__(self, n_fft, hop_length, win_length, input_type='real_imag'):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.input_type = input_type
        self.stft = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    def wav2spec(self, wav):
        # x = self.stft(signal)   # [F, T]
        complex_spec = torch.stft(wav, self.n_fft, self.hop_length, self.win_length, window=torch.hann_window(self.win_length, device=wav.device),
                              return_complex=True) # [num_freqs, num_frames]
        if self.input_type == 'real_imag':
            real, imag = complex_spec.real, complex_spec.imag
            return real, imag, complex_spec

        elif self.input_type == 'mag_phase':
            mag, phase = torch.abs(complex_spec), torch.angle(complex_spec)
            return mag, phase

        else:
            raise NotImplementedError("No such input type")

def plot_distribution(x):
    plt.hist(x)
    plt.show()

if __name__ == '__main__':
    real_imag_stft = STFT(512, 6, 25)
    mag_phase_stft = STFT(512, 6, 25, input_type='mag_phase')
    wav = load_wav('D:/yoga/110_1/audio_singal_processing/DNS-Challenge/datasets/clean/emotional_speech/crema_d/1001_DFA_ANG_XX.wav')
    # real, imag, complex_spec = real_imag_stft.wav2spec(wav)
    # log_pow_imag = torch.log(torch.pow(imag, 2)+1e-10)
    # plot_distribution(log_pow_imag)
    mag, phase = mag_phase_stft.wav2spec(wav)
    log_pow_mag = torch.log(torch.pow(mag, 2)+1e-10)    # after taking mag, do the log power operation. The distribution will be more like Gaussain
    plot_distribution(mag)

