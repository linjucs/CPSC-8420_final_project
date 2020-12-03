
import librosa
import torch
import numpy as np
from scipy import interpolate
from pesq import pesq
import argparse
import time
import os
import soundfile
import matplotlib.pyplot as plt
import pickle
from pystoi import stoi


def reconstruction(magnitude, phase, n_fft=512):
    window = torch.hann_window(n_fft)
    real = magnitude * torch.cos(phase)
    imag = magnitude * torch.sin(phase)

    stft = torch.cat([real.unsqueeze(-1), imag.unsqueeze(-1)], 2)
    waveform = torch.istft(stft, n_fft, n_fft // 2, window=window)

    return waveform


def cal_lsd(enhanced_waveform, target_waveform):
    if len(enhanced_waveform) == len(target_waveform):
        pass
    else:
        minlenth = min(len(enhanced_waveform), len(target_waveform))
        enhanced_waveform = enhanced_waveform[: minlenth]
        target_waveform = target_waveform[: minlenth]
    original_spectrogram = librosa.core.stft(enhanced_waveform, n_fft=2048)
    target_spectrogram = librosa.core.stft(target_waveform, n_fft=2048)

    original_log = np.log10(np.abs(original_spectrogram) ** 2 + 1e-8)
    target_log = np.log10(np.abs(target_spectrogram) ** 2 + 1e-8)
    original_target_squared = (original_log - target_log) ** 2
    original_target_squared_HF = (original_log[512:, :] - target_log[512:, :]) ** 2
    target_lsd = np.mean(np.sqrt(np.mean(original_target_squared, axis=0)))
    target_lsd_HF = np.mean(np.sqrt(np.mean(original_target_squared_HF, axis=0)))

    return target_lsd, target_lsd_HF


def cal_snr(predicted_waveform, target_waveform):
    if len(predicted_waveform) == len(target_waveform):
        pass
    else:
        minlenth = min(len(predicted_waveform), len(target_waveform))
        predicted_waveform = predicted_waveform[: minlenth]
        target_waveform = target_waveform[: minlenth]
    # dB
    signal = np.sum(target_waveform ** 2)
    noise = np.sum((predicted_waveform - target_waveform) ** 2)
    snr = 10 * np.log10(signal / noise)

    return snr

def cal_stoi(predicted_waveform, target_waveform, fs=16000):
    if len(predicted_waveform) == len(target_waveform):
        pass
    else:
        minlenth = min(len(predicted_waveform), len(target_waveform))
        predicted_waveform = predicted_waveform[: minlenth]
        target_waveform = target_waveform[: minlenth]

    d = stoi(target_waveform, predicted_waveform, fs, extended=False)
    return d

def cal_pesq(predicted_waveform, target_waveform, sr=16000):
    if len(predicted_waveform) == len(target_waveform):
        pass
    else:
        minlenth = min(len(predicted_waveform), len(target_waveform))
        predicted_waveform = predicted_waveform[: minlenth]
        target_waveform = target_waveform[: minlenth]
    p = pesq(sr, target_waveform, predicted_waveform, 'nb')

    return p

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ae_attn based speech bandwidth extension')
    parser.add_argument('--batch_size', default=1, type=int, help='train batch size')
    parser.add_argument('--snr', default="full", type=str, help='specific snr condition')
    parser.add_argument('--log_dir', default="logs", type=str, help='summary data for tensorboard')
    parser.add_argument('--test_dir', default="NMF_res", type=str, help='test set')
    opts = parser.parse_args()
    batch_size = opts.batch_size
    test_dir = opts.test_dir
    tblog_fdr = opts.log_dir  # summary data for tensorboard
    # time info is used to distinguish dfferent training sessions
    run_time = time.strftime('%Y%m%d_%H%M', time.gmtime())  # 20180625_1742

    tblog_path = os.path.join(os.getcwd(), tblog_fdr, run_time)  # summary data for tensorboard
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_devices = [0]

    # data transform
    ori_sr = 16000
    count = 0
    crn_pesq_total = []
    crn_lsd_total = []
    crn_stoi_total = []
    ori_pesq_total = []
    ori_lsd_total = []
    ori_stoi_total = []

    for file_ in os.listdir(os.path.join(test_dir, "mix")):
        cur_snr = file_.split('_')[-1][:-4]
        filepath = os.path.join(test_dir, "mix", file_)
        noisy_signal, _ = librosa.load(filepath, sr=ori_sr)
        clean_name = os.path.basename(filepath).split('_')[:-2]
        clean_name = "_".join(clean_name) + '.wav'
        #clean_path = os.path.join(test_dir, "clean", clean_name)
        clean_path = os.path.join(test_dir, "clean", os.path.basename(filepath))
        enh_path = os.path.join(test_dir, "enh", os.path.basename(filepath))
        clean_signal, _ = librosa.load(clean_path, sr=ori_sr)
        enh_signal, _ = librosa.load(enh_path, sr=ori_sr)

        ori_pesq = cal_pesq(noisy_signal, clean_signal)
        ori_stoi = cal_stoi(noisy_signal, clean_signal)
        ori_lsd, _ = cal_lsd(noisy_signal, clean_signal)
        ori_lsd_total.append(ori_lsd)
        ori_pesq_total.append(ori_pesq)
        ori_stoi_total.append(ori_stoi)

        crn_pesq = cal_pesq(enh_signal, clean_signal)
        crn_stoi = cal_stoi(enh_signal, clean_signal)
        crn_lsd, crn_lsd_hf = cal_lsd(enh_signal, clean_signal)
        crn_lsd_total.append(crn_lsd)
        crn_pesq_total.append(crn_pesq)
        crn_stoi_total.append(crn_stoi)

        print(
            f'processed {count} filename {file_}, ori PESQ {round(ori_pesq, 3)}, crn pesq {round(crn_pesq, 3)}, ori STOI {round(ori_stoi, 3)} crn stoi {round(crn_stoi, 3)}, ori LSD {round(ori_lsd, 3)}, crn LSD {round(crn_lsd, 3)}')
        count += 1



    crn_pesq_mean = np.mean(np.array(crn_pesq_total))
    crn_lsd_mean = np.mean(np.array(crn_lsd_total))
    crn_stoi_mean = np.mean(np.array(crn_stoi_total))

    ori_pesq_mean = np.mean(np.array(ori_pesq_total))
    ori_stoi_mean = np.mean(np.array(ori_stoi_total))
    ori_lsd_mean = np.mean(np.array(ori_lsd_total))

    print('processed total {} files'.format(count))
    print("ori PESQ {}, STOI {}, LSD {}".format(ori_pesq_mean, ori_stoi_mean, ori_lsd_mean))
    print("crn PESQ {}, STOI {}, LSD {}".format(crn_pesq_mean, crn_stoi_mean, crn_lsd_mean))


