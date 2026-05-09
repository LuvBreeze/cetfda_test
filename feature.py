# feature.py(特征提取)
import numpy as np


def extract_features(sig):

    feats = []

    mean_val = np.mean(sig)
    std_val = np.std(sig)
    rms = np.sqrt(np.mean(sig ** 2))

    feats.extend([
        mean_val,
        std_val,
        np.max(sig),
        np.min(sig),
        np.max(sig) - np.min(sig),
        rms,
        np.mean(np.abs(sig)),
        np.var(sig),
        np.sum(sig ** 2)
    ])

    # 偏度
    skewness = np.mean(sig ** 3) / (std_val ** 3 + 1e-10)

    # 峭度
    kurtosis = np.mean(sig ** 4) / (std_val ** 4 + 1e-10)

    feats.append(skewness)
    feats.append(kurtosis)

    # FFT
    fft_vals = np.abs(np.fft.fft(sig))
    fft_half = fft_vals[:len(fft_vals)//2]

    feats.extend([
        np.mean(fft_half),
        np.std(fft_half),
        np.max(fft_half),
        np.sum(fft_half)
    ])

    return np.array(feats)