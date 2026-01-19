"""
HMM Model Module
================
Hidden Markov Model implementation for market regime detection.

Supports:
- Gaussian HMM with configurable states
- Model training and prediction
- Regime probability estimation
- Model persistence
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from hmmlearn import hmm
import joblib
import warnings

warnings.filterwarnings('ignore')


@dataclass
class RegimeInfo:
    """
    Information about a detected regime.
    
    Attributes:
        regime_id: Regime identifier (0, 1, 2, ...)
        name: Human-readable name
        mean_return: Average return in this regime
        volatility: Standard deviation in this regime
        duration: Average duration in periods
        frequency: How often this regime occurs
    """
    regime_id: int
    name: str
    mean_return: float
    volatility: float
    duration: float
    frequency: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'regime_id': self.regime_id,
            'name': self.name,
            'mean_return': self.mean_return,
            'volatility': self.volatility,
            'duration': self.duration,
            'frequency': self.frequency
        }


@dataclass
class HMMResult:
    """
    Result of HMM fitting and prediction.
    
    Attributes:
        regimes: Array of regime labels
        regime_probs: Probability of each regime at each time
        log_likelihood: Model log-likelihood
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion
        regime_info: List of RegimeInfo for each regime
    """
    regimes: np.ndarray
    regime_probs: np.ndarray
    log_likelihood: float
    aic: float
    bic: float
    regime_info: List[RegimeInfo]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'n_samples': len(self.regimes),
            'log_likelihood': self.log_likelihood,
            'aic': self.aic,
            'bic': self.bic,
            'regime_info': [r.to_dict() for r in self.regime_info]
        }


class MarketRegimeHMM:
    """
    Hidden Markov Model for market regime detection.
    
    Features:
    - Gaussian emission HMM
    - Automatic regime labeling (Bull/Bear/Neutral)
    - Regime probability smoothing
    - Model selection via BIC
    """
    
    # Default regime names based on characteristics
    REGIME_NAMES = {
        'high_return_low_vol': 'Bull Market',
        'high_return_high_vol': 'Recovery',
        'low_return_low_vol': 'Quiet',
        'low_return_high_vol': 'Bear Market',
        'neutral': 'Neutral'
    }
    
    def __init__(
        self,
        n_regimes: int = 3,
        covariance_type: str = 'full',
        n_iter: int = 100,
        random_state: int = 42
    ):
        """
        Initialize HMM model.
        
        Parameters:
            n_regimes: Number of hidden states/regimes
            covariance_type: Type of covariance ('full', 'diag', 'spherical')
            n_iter: Maximum iterations for EM algorithm
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        
        self.model: Optional[hmm.GaussianHMM] = None
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.regime_info: List[RegimeInfo] = []
        
    def _create_model(self) -> hmm.GaussianHMM:
        """Create a new HMM model."""
        return hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state
        )
    
    def fit(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> 'MarketRegimeHMM':
        """
        Fit HMM to feature data.
        
        Parameters:
            features: 2D array of features (n_samples, n_features)
            feature_names: Names of features
        
        Returns:
            Self for method chaining
        """
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        self.feature_names = feature_names or [f'feature_{i}' for i in range(features.shape[1])]
        
        # Create and fit model
        self.model = self._create_model()
        self.model.fit(features)
        self.is_fitted = True
        
        # Label regimes based on characteristics
        self._label_regimes(features)
        
        return self
    
    def _label_regimes(self, features: np.ndarray) -> None:
        """
        Label regimes based on their characteristics.
        
        Assigns human-readable names based on mean return and volatility.
        """
        if not self.is_fitted or self.model is None:
            return
        
        # Get regime predictions
        regimes = self.model.predict(features)
        
        # Calculate statistics for each regime
        regime_stats = []
        for i in range(self.n_regimes):
            mask = regimes == i
            if mask.sum() > 0:
                regime_features = features[mask]
                mean_return = regime_features[:, 0].mean() if features.shape[1] > 0 else 0
                volatility = regime_features[:, 0].std() if features.shape[1] > 0 else 0
                
                # Calculate duration (average consecutive periods)
                transitions = np.diff(mask.astype(int))
                starts = np.where(transitions == 1)[0]
                ends = np.where(transitions == -1)[0]
                
                if len(starts) > 0 and len(ends) > 0:
                    if starts[0] > ends[0]:
                        ends = ends[1:]
                    if len(starts) > len(ends):
                        starts = starts[:len(ends)]
                    durations = ends - starts
                    avg_duration = durations.mean() if len(durations) > 0 else mask.sum()
                else:
                    avg_duration = mask.sum()
                
                frequency = mask.sum() / len(regimes)
                
                regime_stats.append({
                    'regime_id': i,
                    'mean_return': mean_return,
                    'volatility': volatility,
                    'duration': avg_duration,
                    'frequency': frequency
                })
        
        # Sort by mean return to assign names
        regime_stats.sort(key=lambda x: x['mean_return'])
        
        # Assign names based on ranking
        self.regime_info = []
        for idx, stats in enumerate(regime_stats):
            if self.n_regimes == 2:
                name = 'Bear Market' if idx == 0 else 'Bull Market'
            elif self.n_regimes == 3:
                names = ['Bear Market', 'Neutral', 'Bull Market']
                name = names[idx]
            else:
                name = f'Regime {stats["regime_id"]}'
            
            self.regime_info.append(RegimeInfo(
                regime_id=stats['regime_id'],
                name=name,
                mean_return=stats['mean_return'],
                volatility=stats['volatility'],
                duration=stats['duration'],
                frequency=stats['frequency']
            ))
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict regime labels for new data.
        
        Parameters:
            features: 2D array of features
        
        Returns:
            Array of regime labels
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        return self.model.predict(features)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Get regime probabilities for new data.
        
        Parameters:
            features: 2D array of features
        
        Returns:
            Array of shape (n_samples, n_regimes) with probabilities
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        return self.model.predict_proba(features)
    
    def score(self, features: np.ndarray) -> float:
        """
        Compute log-likelihood of the data.
        
        Parameters:
            features: 2D array of features
        
        Returns:
            Log-likelihood score
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before scoring")
        
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        return self.model.score(features)
    
    def get_regime_info(self, regime_id: int) -> Optional[RegimeInfo]:
        """Get information about a specific regime."""
        for info in self.regime_info:
            if info.regime_id == regime_id:
                return info
        return None
    
    def get_transition_matrix(self) -> np.ndarray:
        """
        Get the transition probability matrix.
        
        Returns:
            Matrix of shape (n_regimes, n_regimes)
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted first")
        
        return self.model.transmat_
    
    def get_stationary_distribution(self) -> np.ndarray:
        """
        Compute the stationary distribution of regimes.
        
        Returns:
            Array of stationary probabilities for each regime
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted first")
        
        # Solve π = π * P for stationary distribution
        transmat = self.model.transmat_
        n = transmat.shape[0]
        
        # Add constraint that probabilities sum to 1
        A = np.vstack([transmat.T - np.eye(n), np.ones(n)])
        b = np.zeros(n + 1)
        b[-1] = 1
        
        # Solve least squares
        stationary, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        return stationary
    
    def calculate_aic_bic(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Calculate AIC and BIC for model selection.
        
        Parameters:
            features: Feature data used for fitting
        
        Returns:
            Tuple of (AIC, BIC)
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted first")
        
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        n_samples = features.shape[0]
        n_features = features.shape[1]
        
        # Number of free parameters
        n_params = (
            self.n_regimes * n_features +  # Means
            self.n_regimes * n_features * (n_features + 1) // 2 +  # Covariances
            self.n_regimes * (self.n_regimes - 1)  # Transition probs
        )
        
        log_likelihood = self.score(features) * n_samples
        
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n_samples) - 2 * log_likelihood
        
        return aic, bic
    
    def fit_and_analyze(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> HMMResult:
        """
        Fit model and return comprehensive analysis.
        
        Parameters:
            features: Feature data
            feature_names: Names of features
        
        Returns:
            HMMResult with full analysis
        """
        self.fit(features, feature_names)
        
        regimes = self.predict(features)
        regime_probs = self.predict_proba(features)
        log_likelihood = self.score(features)
        aic, bic = self.calculate_aic_bic(features)
        
        return HMMResult(
            regimes=regimes,
            regime_probs=regime_probs,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            regime_info=self.regime_info
        )
    
    def save(self, filepath: str) -> None:
        """Save model to file."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self.model,
            'n_regimes': self.n_regimes,
            'covariance_type': self.covariance_type,
            'feature_names': self.feature_names,
            'regime_info': self.regime_info
        }
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'MarketRegimeHMM':
        """Load model from file."""
        model_data = joblib.load(filepath)
        
        instance = cls(
            n_regimes=model_data['n_regimes'],
            covariance_type=model_data['covariance_type']
        )
        instance.model = model_data['model']
        instance.is_fitted = True
        instance.feature_names = model_data['feature_names']
        instance.regime_info = model_data['regime_info']
        
        return instance


def select_optimal_regimes(
    features: np.ndarray,
    min_regimes: int = 2,
    max_regimes: int = 5,
    criterion: str = 'bic'
) -> Tuple[int, Dict[int, float]]:
    """
    Select optimal number of regimes using information criteria.
    
    Parameters:
        features: Feature data
        min_regimes: Minimum number of regimes to test
        max_regimes: Maximum number of regimes to test
        criterion: 'aic' or 'bic'
    
    Returns:
        Tuple of (optimal_n_regimes, scores_dict)
    """
    scores = {}
    
    for n in range(min_regimes, max_regimes + 1):
        model = MarketRegimeHMM(n_regimes=n)
        model.fit(features)
        aic, bic = model.calculate_aic_bic(features)
        scores[n] = bic if criterion == 'bic' else aic
    
    optimal = min(scores, key=scores.get)
    
    return optimal, scores


if __name__ == "__main__":
    print("Testing HMM Model...")
    
    # Generate synthetic data with 3 regimes
    np.random.seed(42)
    
    # Regime 1: Low volatility uptrend
    regime1 = np.random.normal(0.001, 0.01, 100)
    # Regime 2: High volatility
    regime2 = np.random.normal(-0.002, 0.03, 50)
    # Regime 3: Medium
    regime3 = np.random.normal(0.0005, 0.015, 100)
    
    returns = np.concatenate([regime1, regime2, regime3, regime1, regime2])
    
    # Fit HMM
    model = MarketRegimeHMM(n_regimes=3)
    result = model.fit_and_analyze(returns.reshape(-1, 1), feature_names=['returns'])
    
    print(f"\nModel Results:")
    print(f"  Log-Likelihood: {result.log_likelihood:.2f}")
    print(f"  AIC: {result.aic:.2f}")
    print(f"  BIC: {result.bic:.2f}")
    
    print(f"\nRegime Information:")
    for info in result.regime_info:
        print(f"  {info.name}: mean={info.mean_return:.4f}, vol={info.volatility:.4f}, freq={info.frequency:.1%}")
    
    print(f"\nTransition Matrix:")
    print(model.get_transition_matrix())
