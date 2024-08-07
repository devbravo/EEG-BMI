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
from typing import List, Tuple, Dict, Optional
from mne.epochs import Epochs


class FeatureExtractionStrategy(ABC):
    @abstractmethod
    def extract_features(self, epoch, methods: List[str], sfreq):
        pass

class TimeFeatureExtraction(FeatureExtractionStrategy):
    def __init__(self, methods: List[str], freq_bands: Optional[Dict[str, tuple]] = None):
        self.methods = methods
        self.freq_bands = freq_bands or {}
        self.calculate_baseline = False
        self.stats_methods = {
            'mean': lambda epoch: np.mean(epoch, axis=1),
            'max': lambda epoch: np.max(epoch, axis=1),
            'min': lambda epoch: np.min(epoch, axis=1),
            'std': lambda epoch: np.std(epoch, axis=1),
            'var': lambda epoch: np.var(epoch, axis=1),
            'skew': lambda epoch: stats.skew(epoch, axis=1),
            'kurt': lambda epoch: stats.kurtosis(epoch, axis=1),
            'rms': lambda epoch: np.sqrt(np.mean(epoch**2, axis=1)),
            'ptp': lambda epoch: np.ptp(epoch, axis=1),
            'zero_crossings': lambda epoch: np.sum(np.diff(np.signbit(epoch), axis=1), axis=1),
            'mafd': lambda epoch: np.mean(np.abs(np.diff(epoch, n=1, axis=1)), axis=1),
            'mafdn': lambda epoch: np.mean(np.abs(np.diff(epoch / np.std(epoch, axis=1, keepdims=True), n=1, axis=1)), axis=1),
            'masd': lambda epoch: np.mean(np.abs(np.diff(epoch, n=2, axis=1)), axis=1),
            'masdn': lambda epoch: np.mean(np.abs(np.diff(epoch / np.std(epoch, axis=1, keepdims=True), n=2, axis=1)), axis=1)
        }

    def statistical_features(self, epoch):
        statistical_features = {}
        for method in self.methods:
            if method in self.stats_methods:
              feature = self.stats_methods[method](epoch)
              statistical_features[method] = feature
        return statistical_features

    def ers_erd_features(self, epoch, sfreq):
        n_channels, n_samples = epoch.shape
        npgerseg = min(256, n_samples)
        freqs, psd = welch(epoch, sfreq, nperseg=npgerseg, axis=1)

        ers_erd_feats = []
        for band_name, (fmin, fmax) in self.freq_bands.items():
            band_mask = (freqs >= fmin) & (freqs <= fmax)
            psd_band = psd[:, band_mask]
            avg_power = np.mean(psd_band, axis=1)  
            ers_erd_feats.append(avg_power)
      
        return np.concatenate(ers_erd_feats, axis=0)
    
    def extract_features(self, X, sfreq=None):
        time_domain_features = {method: [] for method in self.methods}
        time_domain_features['ers_erd'] = []

        for epoch in X:
            stats_features = self.statistical_features(epoch)
            ers_erd_feats = self.ers_erd_features(epoch, sfreq)
            
            for key, value in stats_features.items():
                time_domain_features[key].append(value)

            time_domain_features['ers_erd'].append(ers_erd_feats)

        for key in time_domain_features:
            time_domain_features[key] = np.array(time_domain_features[key])

        return time_domain_features
      

class FrequencyFeatureExtraction(FeatureExtractionStrategy):
    def __init__(self, methods: List[str], freq_bands: Optional[Dict[str, tuple]] = None):
        self.methods = methods
        self.freq_bands = freq_bands or {}
      
    def fft_features(self, epoch, sfreq):
        n_channels, n_samples = epoch.shape
        n_fft = min(1024, n_samples)
        
        freqs = np.fft.rfftfreq(n_fft, d=1/sfreq)
        fft_feats = np.zeros((n_channels, len(freqs)))
        
        for ch in range(n_channels):
            fft_result = np.fft.rfft(epoch[ch, :], n=n_fft)
            fft_magnitude = np.abs(fft_result)
            fft_feats[ch, :len(fft_magnitude)] = fft_magnitude
        
        return fft_feats
    
    def power_band_features(self, epoch, sfreq):
        n_channels, n_samples = epoch.shape
        npgerseg = min(256, n_samples)
        
        freqs, psd = welch(epoch, sfreq, nperseg=npgerseg, axis=1)
        
        band_features = []
        for band_name, (fmin, fmax) in self.freq_bands.items():
            band_mask = (freqs >= fmin) & (freqs <= fmax)
            band_power = np.sum(psd[:, band_mask], axis=1, keepdims=True)
            band_features.append(band_power)
        
        return np.concatenate(band_features, axis=1)
      
    def extract_features(self, X, sfreq):
        frequency_domain_features = {'fft': [], 'power_band': []}
        for epoch in X:
            fft_feats = self.fft_features(epoch, sfreq)
            band_power_feats = self.power_band_features(epoch, sfreq)
            frequency_domain_features['fft'].append(fft_feats.flatten())
            frequency_domain_features['power_band'].append(band_power_feats.flatten())
        frequency_domain_features['fft'] = np.array(frequency_domain_features['fft'])
        frequency_domain_features['power_band'] = np.array(frequency_domain_features['power_band'])
        
        return frequency_domain_features

class TimeFrequencyFeatureExtraction(FeatureExtractionStrategy):
    def __init__(self, methods: List[str], freq_bands: Optional[Dict[str, tuple]] = None):
      self.methods = methods
      self.freq_bands = freq_bands or {}
    
    def stft_features(self, epoch, sfreq):
      n_channels, n_samples = epoch.shape
      stft_features = []
      
      for ch in range(n_channels):
          f, t, Zxx = stft(epoch[ch, :], sfreq, nperseg=64, noverlap=32)
          stft_features.append(np.abs(Zxx).flatten())
      
      return np.concatenate(stft_features)
    
    def dwt_features(self, epoch):
      n_channels, n_samples = epoch.shape
      dwt_features = []
      
      for ch in range(n_channels):
          time_series = epoch[ch, :]
          coeffs = pywt.wavedec(data=time_series, wavelet='db4', mode='periodization', level=3)
          channel_dwt_features = np.concatenate([coeff.flatten() for coeff in coeffs])
          dwt_features.append(channel_dwt_features)

      return np.concatenate(dwt_features)
    
    def extract_features(self, X, sfreq):
      time_frequency_domain_features = {
        'stft': [],
        'dwt': []
      }
    
      for epoch in X:
          stft_feats = self.stft_features(epoch, sfreq)
          dwt_feat = self.dwt_features(epoch)

          time_frequency_domain_features['stft'].append(stft_feats)
          time_frequency_domain_features['dwt'].append(dwt_feat)
       
      time_frequency_domain_features['stft'] = np.array(time_frequency_domain_features['stft'])
      time_frequency_domain_features['dwt'] = np.array(time_frequency_domain_features['dwt'])
      return time_frequency_domain_features
  
  
class SpatialFeatureExtraction(FeatureExtractionStrategy):
    def __init__(self, methods: List[str], freq_bands: Optional[Dict[str, tuple]] = None, n_components: int = 4):
        # self.info = info
        self.methods = methods
        self.freq_bands = freq_bands
        self.n_components = n_components

    def extract_features(self, X, y, sfreq):
      subband_csp = SubBandCSP(sfreq, self.freq_bands, self.n_components)
      spatial_domain_features = {'sb_csp': []}
      subband_csp.fit(X, y)
      
      for epoch in X:
        subBand_csp_feats = subband_csp.transform([epoch])[0]
        spatial_domain_features['sb_csp'].append(subBand_csp_feats.flatten())
        
      spatial_domain_features['sb_csp'] = np.array(spatial_domain_features['sb_csp'])
      return spatial_domain_features
    

class EEGFeatureExtractor:
    def __init__(self,  methods: List[str], freq_bands: Optional[Dict[str, tuple]] = None ):
        self.methods = methods
        self.freq_bands = freq_bands if freq_bands is not None else {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        self.time_extractor = TimeFeatureExtraction(methods=self.methods, freq_bands=self.freq_bands)
        self.frequency_extractor = FrequencyFeatureExtraction(methods=self.methods, freq_bands=self.freq_bands)
        self.time_frequency_extractor = TimeFrequencyFeatureExtraction(methods=self.methods, freq_bands=self.freq_bands)
        self.spatial_extractor = SpatialFeatureExtraction(methods=self.methods, freq_bands=self.freq_bands)
    
    def extract_time_features(self, X: Epochs, sfreq: float=None) -> np.ndarray:
        return self.time_extractor.extract_features(X, sfreq)
    
    def extract_frequency_features(self, X: Epochs, sfreq: float=None):
        return self.frequency_extractor.extract_features(X, sfreq)
    
    def extract_time_frequency_features(self, X: Epochs, sfreq: float=None):
        return self.time_frequency_extractor.extract_features(X, sfreq)
    
    def extract_spatial_features(self, X: Epochs, y, sfreq: float=None):
        return self.spatial_extractor.extract_features(X, y, sfreq)
      
      
      
class SubBandCSP:
    def __init__(self, sfreq, freq_bands, n_components=4):
        self.sfreq = sfreq
        self.freq_bands = freq_bands
        self.n_components = n_components
        self.csp_models = {}

    def bandpass_filter(self, data, low, high, order=5):
        nyquist = 0.5 * self.sfreq
        low = low / nyquist
        high = high / nyquist
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data, axis=0)

    def fit(self, X, y):
        for band_name, (low, high) in self.freq_bands.items():
            filtered_X = np.array([self.bandpass_filter(epoch, low, high) for epoch in X])
            csp = CSP(n_components=self.n_components, reg=0.1, log=True)
            csp.fit(filtered_X, y)
            self.csp_models[band_name] = csp

    def transform(self, X):
        transformed_X = []
        for band_name, (low, high) in self.freq_bands.items():
            csp = self.csp_models[band_name]
            filtered_X = np.array([self.bandpass_filter(epoch, low, high) for epoch in X])
            transformed_X.append(csp.transform(filtered_X))
        return np.concatenate(transformed_X, axis=1)
      