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
  

    def ers_erd_features(self, epoch, sfreq, ):
        n_channels, n_samples = epoch.shape
        npgerseg = min(256, n_samples)
        freqs, psd = welch(epoch, sfreq, nperseg=npgerseg, axis=1)

        ers_erd_feats = []
        for band_name, (fmin, fmax) in self.freq_bands.items():
            band_mask = (freqs >= fmin) & (freqs <= fmax)
            psd_band = psd[:, band_mask]
            psd_band = np.array(psd_band)

            if psd.ndim == 2:
                avg_power = np.mean(psd_band, axis=1)  
            elif psd.ndim == 3:
                avg_power = np.mean(psd_band, axis=1)
            else:
                raise ValueError("Unsupported PSD shape")
            ers_erd_feats.append(avg_power)
      
        return np.concatenate(ers_erd_feats, axis=0)
    
    def extract_features(self, X, sfreq=None):
      time_domain_features = {
          'statistical_features': {},
          'ers_erd': []
      }
      statistical_features_list = {method: [] for method in self.methods}

      for epoch in X:
          stats_features = self.statistical_features(epoch)
          ers_erd_feats = self.ers_erd_features(epoch, sfreq)
          
          for key, value in stats_features.items():
              statistical_features_list[key].append(value)

          time_domain_features['ers_erd'].append(ers_erd_feats)

      for key, feature_list in statistical_features_list.items():
          time_domain_features['statistical_features'][key] = np.array(feature_list)

      time_domain_features['ers_erd'] = np.array(time_domain_features['ers_erd'])

      return time_domain_features
      

class FrequencyFeatureExtraction(FeatureExtractionStrategy):
    def __init__(self, methods):
        self.methods = methods
      
    def fft_features(self, epoch, sfreq):
        n_channels, n_samples = epoch.shape
        n_fft = min(1024, n_samples)
        
        freqs = np.fft.rfftfreq(n_samples, d=1/sfreq)
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
        
        delta_power = np.sum(psd[:, (freqs >= 0.5) & (freqs < 4)], axis=1, keepdims=True)
        theta_power = np.sum(psd[:, (freqs >= 4) & (freqs < 8)], axis=1, keepdims=True)
        alpha_power = np.sum(psd[:, (freqs >= 8) & (freqs < 13)], axis=1, keepdims=True)
        beta_power = np.sum(psd[:, (freqs >= 13) & (freqs < 30)], axis=1, keepdims=True)
        
        band_power_feats = np.concatenate([delta_power, theta_power, alpha_power, beta_power], axis=1)
        return band_power_feats
      
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
    def extract_stft_features(self, epoch, sfreq):      
      f, t, Zxx = stft(epoch, sfreq, nperseg=128, noverlap=64)
      stft_features = np.abs(Zxx).flatten()
      return stft_features
        
    def extract_dwt_features(self, epoch):
      coeffs = pywt.wavedec(epoch, 'db4', level=3)
      dwt_feats = np.concatenate([coeff.flatten() for coeff in coeffs])   
      return dwt_feats
    
    def extract_features(self, X, sfreq):
      time_frequency_domain_features = []
      for epoch in X:
          stft_feats = self.extract_stft_features(epoch, sfreq)
          dwt_feat = self.extract_dwt_features(epoch)

          time_frequency_features = np.concatenate([stft_feats, dwt_feat])
          time_frequency_domain_features.append(time_frequency_features.flatten())
      return np.array(time_frequency_domain_features)
  
  
class SpatialFeatureExtraction(FeatureExtractionStrategy):
    def __init__(self, methods, sfreq, freq_bands: Optional[Dict[str, tuple]] = None, n_components: int = 4):
        # self.info = info
        self.methods = methods
        self.sfreq = sfreq
        self.freq_bands = freq_bands
        self.n_components = n_components
        self.subband_csp = SubBandCSP(self.sfreq, freq_bands, n_components)

    def extract_features(self, X, y):
      self.subband_csp.fit(X, y)
      
      spatial_domain_features = []
      for epoch in X:
        subBand_csp_feats = self.subband_csp.transform([epoch])[0]
        
        spatial_domain_features.append(subBand_csp_feats.flatten())
      return np.array(spatial_domain_features)
    

class EEGFeatureExtractor:
    def __init__(self,  methods: List[str], sfreq: str = None, freq_bands: Optional[Dict[str, tuple]] = None ):
        # if frequencies is []:
        #   self.frequencies = np.concatenate([
        #     np.arange(1, 4, 0.5),   # Delta
        #     np.arange(4, 8, 0.5),   # Theta
        #     np.arange(8, 13, 0.5),  # Alpha
        #     np.arange(13, 30, 1),   # Beta
        #     np.arange(30, 50, 1)    # Gamma
        # ])
        # else:
        #   self.frequencies = frequencies
          
        # self.n_cycles = self.frequencies / 2.0
        # self.event_id = event_id
        self.methods = methods
        self.sfreq = sfreq  # in Hz
        self.freq_bands = freq_bands if freq_bands is not None else {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
        }
        
        self.time_extractor = TimeFeatureExtraction(methods=self.methods, freq_bands=freq_bands)
        self.frequency_extractor = FrequencyFeatureExtraction(methods=self.methods)
        self.time_frequency_extractor = TimeFrequencyFeatureExtraction()
        self.spatial_extractor = SpatialFeatureExtraction(methods=self.methods, sfreq=self.sfreq, freq_bands=self.freq_bands)
    
    def extract_time_features(self, X: Epochs, sfreq: float=None) -> np.ndarray:
        return self.time_extractor.extract_features(X, sfreq)
    
    def extract_frequency_features(self, X, sfreq=None):
        return self.frequency_extractor.extract_features(X, sfreq)
    
    def extract_time_frequency_features(self, X, sfreq=None):
        return self.time_frequency_extractor.extract_features(X, sfreq)
    
    def extract_spatial_features(self, X, y, sfreq=None):
        return self.spatial_extractor.extract_features(X, sfreq)
      
      
      
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
        return lfilter(b, a, data, axis=0)

    def fit(self, X, y):
        for low, high in self.sub_band_ranges:
            filtered_X = np.array([self.bandpass_filter(epoch, low, high, self.sfreq) for epoch in X])
            csp = CSP(n_components=self.n_components, reg=None, log=True)
            csp.fit(filtered_X, y)
            
            self.csp_models.append(csp)

    def transform(self, X):
        features = []
        for csp, (low, high) in zip(self.csp_models, self.sub_band_ranges):
            band_features = [csp.transform(self.bandpass_filter(epoch, low, high, self.sfreq).reshape(1, -1))[0] for epoch in X]
            features.append(np.array(band_features))
        features = np.concatenate(features, axis=1)
        return features
      