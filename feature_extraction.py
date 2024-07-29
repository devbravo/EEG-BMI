import time
import numpy as np
from scipy import stats
from scipy.signal import welch, stft
from scipy.signal import butter, lfilter
from mne import EpochsArray
from mne.decoding import CSP

from mne._fiff.meas_info import Info

import pywt
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from mne.epochs import Epochs

class FeatureExtractionStrategy(ABC):
    @abstractmethod
    def extract_features(self, epoch, methods: List[str], sfreq):
        pass

class TimeFeatureExtraction(FeatureExtractionStrategy):
    def __init__(self, methods: List[str]):
        self.methods = methods
        self.feature_methods = {
            'mean': np.mean,
            # 'median': np.median,
            'max': np.max,
            'min': np.min,
            'std': np.std,
            'var': np.var,
            'skew': stats.skew,
            'kurt': stats.kurtosis,
            'rms': lambda epoch: np.sqrt(np.mean(epoch**2, axis=1)),
            'ptp': np.ptp,
            'zero_crossings': lambda epoch: np.sum(np.diff(np.signbit(epoch), axis=1)),
            'mean_absolute_first_diff': self.mean_absolute_first_diff,
            'mean_absolute_first_diff_normalized': self.mean_absolute_first_diff_normalized,
            'mean_absolute_second_diff': self.mean_absolute_second_diff,
            'mean_absolute_second_diff_normalized': self.mean_absolute_second_diff_normalized
        }

    def extract_features(self, epoch, sfreq=None):
        epoch = np.asarray(epoch)
        features = []

        print('Extracting time domain features:')

        for method in self.methods:
            if method in self.feature_methods:
                start_time = time.time()

                print(f' - Extracting {method.capitalize()} feature...')

                feature = self.feature_methods[method](epoch, axis=1)
                features.append(feature)

                elapsed_time = time.time() - start_time

                print(f' Time elapsed: {elapsed_time:.4f} seconds')

        print('Feature extraction completed.')
        return np.concatenate(features, axis=1)

    def mean_absolute_first_diff(self, epoch, axis):
        return np.mean(np.abs(np.diff(epoch, n=1, axis=axis)), axis=axis)

    def mean_absolute_first_diff_normalized(self, epoch, axis):
        std = np.std(epoch, axis=axis, keepdims=True)
        return np.mean(np.abs(np.diff(epoch / std, n=1, axis=axis)), axis=axis)

    def mean_absolute_second_diff(self, epoch, axis):
        return np.mean(np.abs(np.diff(epoch, n=2, axis=axis)), axis=axis)

    def mean_absolute_second_diff_normalized(self, epoch, axis):
        std = np.std(epoch, axis=axis, keepdims=True)
        return np.mean(np.abs(np.diff(epoch / std, n=2, axis=axis)), axis=axis)
      

class FrequencyFeatureExtraction(FeatureExtractionStrategy):
    def fft_features(self, epoch, sfreq): 
        n_samples, n_channels = epoch.shape
        n_fft = n_samples 
        
        freqs = np.fft.fftfreq(n_fft, d=1/sfreq)
        fft_feats = np.zeros((n_channels, len(freqs))) 
        
        for ch in range(n_channels): 
            fft_result = np.fft.fft(epoch[:, ch])
            fft_magnitude = np.abs(fft_result)
            
            fft_magnitude = fft_magnitude[:n_fft // 2]
            fft_feats[ch, :] = fft_magnitude
        
        return fft_feats
    
    def band_power_features(self, epoch, sfreq):
        freqs, psd = welch(epoch, sfreq, axis=0)  # axis=0 for single epoch
        
        delta_power = np.sum(psd[(freqs >= 0.5) & (freqs < 4)], axis=0)
        theta_power = np.sum(psd[(freqs >= 4) & (freqs < 8)], axis=0)
        alpha_power = np.sum(psd[(freqs >= 8) & (freqs < 13)], axis=0)
        beta_power = np.sum(psd[(freqs >= 13) & (freqs < 30)], axis=0)
        
        band_power_feats = np.concatenate([delta_power, theta_power, alpha_power, beta_power])
        return band_power_feats
    
    def extract_features(self, epoch, sfreq):
        fft_feats = self.fft_features(epoch, sfreq)
        band_power_feats = self.band_power_features(epoch, sfreq)
        
        n_channels = fft_feats.shape[0]
        band_power_feats = band_power_feats.reshape((n_channels, -1))
        
        frequency_features = np.concatenate([fft_feats, band_power_feats], axis=1)
        return frequency_features.flatten().tolist()

class TimeFrequencyFeatureExtraction(FeatureExtractionStrategy):
  # def __init__(self, info, frequencies, n_cycles, event_id,
  #              exclude_wavelet=False, exclude_stft=False, exclude_tfr=False):
      # self.info = info
      # self.frequencies = frequencies
      # self.n_cycles = n_cycles
      # self.event_id = event_id
      # self.exclude_wavelet = exclude_wavelet
      # self.exclude_stft = exclude_stft
      # self.exclude_tfr = exclude_tfr
        
  def extract_stft_features(self, epoch, sfreq):
    if self.exclude_stft:
      return np.array([])
    
    f, t, Zxx = stft(epoch, sfreq, nperseg=128, noverlap=64)
    stft_features = np.abs(Zxx).flatten()
    return stft_features
      
  def extract_dwt_features(self, epoch):
    if self.exclude_wavelet:
      return np.array([])
    
    coeffs = pywt.wavedec(epoch, 'db4', level=3)
    dwt_feats = np.concatenate([coeff.flatten() for coeff in coeffs])   
    return dwt_feats
  
  def extract_features(self, epoch, sfreq):
    if sfreq is None:
            raise ValueError("Sampling frequency (sfreq) must be provided.")
    stft_feats = self.extract_stft_features(epoch, sfreq)
    dwt_feat = self.extract_dwt_features(epoch, sfreq)
    
    time_frequency_features = np.concatenate([stft_feats, dwt_feat], axis=1)
    return time_frequency_features.flatten().tolist()
  
  
class SpatialFeatureExtraction(FeatureExtractionStrategy):
    def __init__(self, info, sfreq, sub_band_ranges, n_components=4):
        self.info = info
        self.sfreq = sfreq
        self.sub_band_ranges = sub_band_ranges
        self.n_components = n_components
        self.sub_band_csp = SubBandCSP(sfreq, sub_band_ranges, n_components)

    def fit(self, X, y):
        self.sub_band_csp.fit(X, y)

    def extract_features(self, epoch, sfreq=None):
        if sfreq is None:
            sfreq = self.sfreq
        self.sub_band_csp.fit(epoch, y=None)  # Fit with no labels for transformation
        return self.sub_band_csp.transform(epoch)

class EEGFeatureExtractor:
    def __init__(self, info: Info, frequencies: np.ndarray[np.ndarray, np.ndarray], event_id: Dict[str, int]):
        self.info = info
        if frequencies is None:
          self.frequencies = np.concatenate([
          np.arange(1, 4, 0.5),   # Delta
          np.arange(4, 8, 0.5),   # Theta
          np.arange(8, 13, 0.5),  # Alpha
          np.arange(13, 30, 1),   # Beta
          np.arange(30, 50, 1)    # Gamma
        ])
        else:
            self.frequencies = frequencies
        self.n_cycles = self.frequencies / 2.0
        self.event_id = event_id
        
        self.time_extractor = TimeFeatureExtraction()
        self.frequency_extractor = FrequencyFeatureExtraction()
        self.time_frequency_extractor = TimeFrequencyFeatureExtraction(self.info, self.frequencies, self.n_cycles, self.event_id)
        self.spatial_extractor = SpatialFeatureExtraction(self.info)
    
    def extract_time_features(self, epoch: Epochs, methods: List[str], sfreq: float=None) -> np.ndarray:
        return self.time_extractor.extract_features(epoch, methods, sfreq)
    
    def extract_frequency_features(self, epoch, sfreq=None, methods=None):
        return self.frequency_extractor.extract_features(epoch, sfreq, methods)
    
    def extract_time_frequency_features(self, epoch, sfreq=None):
        return self.time_frequency_extractor.extract_features(epoch, sfreq)
    
    def extract_spatial_features(self, X, y, epoch, sfreq=None):
        self.spatial_extractor.fit(X, y)
        return self.spatial_extractor.extract_features(epoch, sfreq)
      
      
      
class SubBandCSP:
    def __init__(self, sfreq, sub_band_ranges, n_components=4):
        self.sfreq = sfreq
        self.sub_band_ranges = sub_band_ranges
        self.n_components = n_components
        self.csp_models = []

    def bandpass_filter(self, data, low, high, sfreq, order=5):
        nyquist = 0.5 * sfreq
        low = low / nyquist
        high = high / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_data = lfilter(b, a, data, axis=0)
        return filtered_data

    def fit(self, X, y):
        for low, high in self.sub_band_ranges:
            filtered_X = np.array([self.bandpass_filter(epoch, low, high, self.sfreq) for epoch in X])
            csp = CSP(n_components=self.n_components, reg=None, log=True)
            csp.fit(filtered_X, y)
            self.csp_models.append(csp)

    def transform(self, X):
        features = []
        for csp, (low, high) in zip(self.csp_models, self.sub_band_ranges):
            filtered_X = np.array([self.bandpass_filter(epoch, low, high, self.sfreq) for epoch in X])
            features.append(csp.transform(filtered_X))
        features = np.concatenate(features, axis=1)
        return features
      