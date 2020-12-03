import librosa
from NMF import NMF_MuR, divide_magphase, merge_magphase
import os
import numpy as np
import soundfile as sf
# clean speech path
speech_dir = "data/train/clean"
noise_dir = "data/train/noise"

stft_clean = []
stft_noise = []
count = 0
num_FFT = 512
hop_size = 128
window = 'hamming'
for speech in os.listdir(speech_dir):
    s_path = os.path.join(speech_dir, speech)
    y, sr = librosa.load(s_path, sr=16000)
    S = librosa.stft(y, n_fft=num_FFT, hop_length=hop_size, window=window)
    print('processed {}'.format(count))
    count += 1
    stft_clean.append(S)
stft_clean = np.concatenate(stft_clean, 1)[:, :80000]
magnitude_clean_train, phase_clean_train = divide_magphase(stft_clean, power=1)
W_clean_train, H_clean_train = NMF_MuR(magnitude_clean_train, 40, 300, 10, const_W=False, init_W=0)
count = 0
for noise in os.listdir(noise_dir):
    n_path = os.path.join(noise_dir, noise)
    y, sr = librosa.load(n_path, sr=16000)
    S = librosa.stft(y, n_fft=num_FFT, hop_length=hop_size, window=window)
    print('processed {}'.format(count))
    count += 1
    stft_noise.append(S)

stft_noise = np.concatenate(stft_noise, 1)
magnitude_noise_train, phase_noise_train = divide_magphase(stft_noise, power=1)
W_noise, H_noise = NMF_MuR(magnitude_noise_train, 40, 200, 10, const_W=False, init_W=0)

#test
test_dir = "data/test/mix"
out_dir = "NMF_enhanced"
os.makedirs(out_dir, exist_ok=True)
for test_ in os.listdir(test_dir):
    test_path = os.path.join(test_dir, test_)
    y, sr = librosa.load(test_path, sr=16000)
    stft_noisy_test = librosa.stft(y, n_fft=num_FFT, hop_length=hop_size, window=window)
    magnitude_noisy_test, phase_noisy_test = divide_magphase(stft_noisy_test, power=1)

    W_noisy = np.concatenate([W_clean_train, W_noise], axis=1)
    _, H_reconstructed_noisy = NMF_MuR(magnitude_noisy_test, 2 * 40, 200, 10, const_W=True,
                                   init_W=W_noisy)

    H_reconstructed_clean = H_reconstructed_noisy[:40, :]
    H_reconstructed_noise = H_reconstructed_noisy[40:, :]

    magnitude_reconstructed_clean = np.matmul(W_clean_train, H_reconstructed_clean)
    magnitude_reconstructed_noise = np.matmul(W_noise, H_reconstructed_noise)

    wiener_gain = np.power(magnitude_reconstructed_clean, 2) / (
            np.power(magnitude_reconstructed_clean, 2) + np.power(magnitude_reconstructed_noise, 2))
    magnitude_estimated_clean = wiener_gain * magnitude_noisy_test

    # Reconstruct
    stft_reconstructed_clean = merge_magphase(magnitude_estimated_clean, phase_noisy_test)
    signal_reconstructed_clean = librosa.istft(stft_reconstructed_clean, hop_length=hop_size, window=window)
    sf.write(os.path.join(out_dir, test_), signal_reconstructed_clean, 16000)


