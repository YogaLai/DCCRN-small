import torchaudio
import torch
# import librosa
import matplotlib.pyplot as plt
from torch.utils import data
import json
import soundfile as sf

def load_wav(path, sr=16000):
   wav, _ = librosa.load(path, sr=sr)
   return torch.tensor(wav)

class STFT():
    def __init__(self, n_fft, hop_length, win_length, output_type='real_imag'):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.output_type = output_type
        self.stft = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    def wav2spec(self, wav):
        # x = self.stft(signal)   # [F, T]
        complex_spec = torch.stft(wav, self.n_fft, self.hop_length, self.win_length, window=torch.hann_window(self.win_length, device=wav.device),
                              return_complex=True) # [num_freqs, num_frames]
        if self.output_type == 'real_imag':
            real, imag = complex_spec.real, complex_spec.imag
            return real, imag

        elif self.output_type == 'mag_phase':
            mag, phase = torch.abs(complex_spec), torch.angle(complex_spec)
            return mag, phase

        elif self.output_type == 'complex':
            return complex_spec

        else:
            raise NotImplementedError("No such input type")

def plot_distribution(x):
    plt.hist(x)
    plt.show()

class DNSDataset(data.Dataset):
    """Deep Noise Suppression (DNS) Challenge's dataset.
    Args
        json_dir (str): path to the JSON directory (from the recipe).
    References
        "The INTERSPEECH 2020 Deep Noise Suppression Challenge: Datasets,
        Subjective Testing Framework, and Challenge Results", Reddy et al. 2020.
    """

    dataset_name = "DNS"

    def __init__(self, json_path):

        super(DNSDataset, self).__init__()
        with open(json_path, "r") as f:
            filename_json = json.load(f)

        self.mix = filename_json['mix']
        self.clean = filename_json['clean']
        self.noise = filename_json['noise']

    def __len__(self):
        return len(self.mix)

    def __getitem__(self, idx):
        """Gets a mixture/sources pair.
        Returns:
            mixture, vstack([source_arrays])
        """
        # Load mixture
        x = torch.from_numpy(sf.read(self.mix[idx], dtype="float32")[0])
        # Load clean
        speech = torch.from_numpy(sf.read(self.clean[idx], dtype="float32")[0])
        # Load noise
        noise = torch.from_numpy(sf.read(self.noise[idx], dtype="float32")[0])
        return x, speech, noise

if __name__ == '__main__':
    # torch stft test 
    # real_imag_stft = STFT(512, 6, 25)
    # mag_phase_stft = STFT(512, 6, 25, input_type='mag_phase')
    # wav = load_wav('D:/yoga/110_1/audio_singal_processing/DNS-Challenge/datasets/clean/emotional_speech/crema_d/1001_DFA_ANG_XX.wav')
    # # real, imag, complex_spec = real_imag_stft.wav2spec(wav)
    # # log_pow_imag = torch.log(torch.pow(imag, 2)+1e-10)
    # # plot_distribution(log_pow_imag)
    # mag, phase = mag_phase_stft.wav2spec(wav)
    # log_pow_mag = torch.log(torch.pow(mag, 2)+1e-10)    # after taking mag, do the log power operation. The distribution will be more like Gaussain
    # plot_distribution(mag)
    from torch.utils.data import DataLoader
    train_set = DNSDataset('sort_filename.json')
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=4,
        num_workers=4,
        drop_last=True,
    )    
    for x, clean, noise in train_loader:
        print(x)

