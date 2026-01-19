"""
Regime Classifier Module
========================
High-level regime classification and analysis.

Supports:
- Regime detection pipeline
- Regime statistics
- Regime duration analysis
- Regime change detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .hmm_model import MarketRegimeHMM, HMMResult, RegimeInfo
from .feature_engineering import create_hmm_features


@dataclass
class RegimeAnalysis:
    """
    Comprehensive regime analysis results.
    
    Attributes:
        regimes: Series of regime labels indexed by date
        regime_probs: DataFrame of regime probabilities
        current_regime: Current detected regime
        regime_stats: Statistics for each regime
        transitions: List of regime transition dates
        duration_stats: Average duration by regime
    """
    regimes: pd.Series
    regime_probs: pd.DataFrame
    current_regime: int
    current_regime_name: str
    regime_stats: List[RegimeInfo]
    transitions: List[datetime]
    duration_stats: Dict[int, float]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'current_regime': self.current_regime,
            'current_regime_name': self.current_regime_name,
            'n_transitions': len(self.transitions),
            'regime_stats': [r.to_dict() for r in self.regime_stats],
            'duration_stats': self.duration_stats
        }


class RegimeClassifier:
    """
    Market regime classifier using HMM.
    
    Features:
    - Full regime detection pipeline
    - Regime probability smoothing
    - Transition detection
    - Regime persistence analysis
    """
    
    def __init__(
        self,
        n_regimes: int = 3,
        use_volatility: bool = True,
        smoothing_window: int = 5
    ):
        """
        Initialize classifier.
        
        Parameters:
            n_regimes: Number of regimes to detect
            use_volatility: Include volatility in features
            smoothing_window: Window for probability smoothing
        """
        self.n_regimes = n_regimes
        self.use_volatility = use_volatility
        self.smoothing_window = smoothing_window
        
        self.hmm = MarketRegimeHMM(n_regimes=n_regimes)
        self.is_fitted = False
        self.dates: Optional[pd.DatetimeIndex] = None
    
    def fit(
        self,
        prices: pd.DataFrame,
        returns: Optional[pd.Series] = None
    ) -> 'RegimeClassifier':
        """
        Fit classifier to market data.
        
        Parameters:
            prices: OHLCV DataFrame
            returns: Optional returns series
        
        Returns:
            Self for method chaining
        """
        # Create features
        features, feature_names, dates = create_hmm_features(
            prices, returns, 
            use_volatility=self.use_volatility,
            normalize=True
        )
        
        self.dates = dates
        
        # Fit HMM
        self.hmm.fit(features, feature_names)
        self.is_fitted = True
        
        return self
    
    def classify(
        self,
        prices: pd.DataFrame,
        returns: Optional[pd.Series] = None
    ) -> RegimeAnalysis:
        """
        Classify regimes in market data.
        
        Parameters:
            prices: OHLCV DataFrame
            returns: Optional returns series
        
        Returns:
            RegimeAnalysis with full results
        """
        if not self.is_fitted:
            self.fit(prices, returns)
        
        # Create features
        features, _, dates = create_hmm_features(
            prices, returns,
            use_volatility=self.use_volatility,
            normalize=True
        )
        
        # Get predictions
        regimes = self.hmm.predict(features)
        regime_probs = self.hmm.predict_proba(features)
        
        # Apply smoothing to probabilities
        smoothed_probs = self._smooth_probabilities(regime_probs)
        smoothed_regimes = np.argmax(smoothed_probs, axis=1)
        
        # Create output DataFrames/Series
        regime_series = pd.Series(smoothed_regimes, index=dates, name='regime')
        
        prob_columns = [f'regime_{i}_prob' for i in range(self.n_regimes)]
        regime_prob_df = pd.DataFrame(smoothed_probs, index=dates, columns=prob_columns)
        
        # Detect transitions
        transitions = self._detect_transitions(regime_series)
        
        # Calculate duration stats
        duration_stats = self._calculate_durations(regime_series)
        
        # Current regime
        current_regime = int(regime_series.iloc[-1])
        current_regime_info = self.hmm.get_regime_info(current_regime)
        current_name = current_regime_info.name if current_regime_info else f"Regime {current_regime}"
        
        return RegimeAnalysis(
            regimes=regime_series,
            regime_probs=regime_prob_df,
            current_regime=current_regime,
            current_regime_name=current_name,
            regime_stats=self.hmm.regime_info,
            transitions=transitions,
            duration_stats=duration_stats
        )
    
    def _smooth_probabilities(self, probs: np.ndarray) -> np.ndarray:
        """Apply moving average smoothing to probabilities."""
        if self.smoothing_window <= 1:
            return probs
        
        smoothed = np.zeros_like(probs)
        for i in range(probs.shape[1]):
            smoothed[:, i] = pd.Series(probs[:, i]).rolling(
                window=self.smoothing_window, min_periods=1
            ).mean().values
        
        # Renormalize
        row_sums = smoothed.sum(axis=1, keepdims=True)
        smoothed = smoothed / row_sums
        
        return smoothed
    
    def _detect_transitions(self, regimes: pd.Series) -> List[datetime]:
        """Detect regime transition points."""
        transitions = []
        prev_regime = regimes.iloc[0]
        
        for date, regime in regimes.items():
            if regime != prev_regime:
                transitions.append(date.to_pydatetime() if hasattr(date, 'to_pydatetime') else date)
                prev_regime = regime
        
        return transitions
    
    def _calculate_durations(self, regimes: pd.Series) -> Dict[int, float]:
        """Calculate average duration for each regime."""
        durations: Dict[int, List[int]] = {i: [] for i in range(self.n_regimes)}
        
        current_regime = regimes.iloc[0]
        current_duration = 1
        
        for regime in regimes.iloc[1:]:
            if regime == current_regime:
                current_duration += 1
            else:
                durations[current_regime].append(current_duration)
                current_regime = regime
                current_duration = 1
        
        # Add last period
        durations[current_regime].append(current_duration)
        
        # Calculate averages
        avg_durations = {}
        for regime_id, dur_list in durations.items():
            avg_durations[regime_id] = np.mean(dur_list) if dur_list else 0.0
        
        return avg_durations
    
    def get_regime_at_date(
        self,
        analysis: RegimeAnalysis,
        date: datetime
    ) -> Tuple[int, str, float]:
        """
        Get regime at a specific date.
        
        Returns:
            Tuple of (regime_id, regime_name, probability)
        """
        if date not in analysis.regimes.index:
            # Find closest date
            idx = analysis.regimes.index.get_indexer([date], method='nearest')[0]
            date = analysis.regimes.index[idx]
        
        regime_id = int(analysis.regimes.loc[date])
        regime_info = self.hmm.get_regime_info(regime_id)
        regime_name = regime_info.name if regime_info else f"Regime {regime_id}"
        
        prob_col = f'regime_{regime_id}_prob'
        probability = float(analysis.regime_probs.loc[date, prob_col])
        
        return regime_id, regime_name, probability
    
    def predict_next_regime(
        self,
        current_regime: int
    ) -> Tuple[int, float]:
        """
        Predict most likely next regime based on transition matrix.
        
        Returns:
            Tuple of (next_regime, probability)
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted first")
        
        trans_matrix = self.hmm.get_transition_matrix()
        next_probs = trans_matrix[current_regime]
        
        next_regime = int(np.argmax(next_probs))
        probability = float(next_probs[next_regime])
        
        return next_regime, probability
    
    def save(self, filepath: str) -> None:
        """Save classifier to file."""
        self.hmm.save(filepath)
    
    @classmethod
    def load(cls, filepath: str, n_regimes: int = 3) -> 'RegimeClassifier':
        """Load classifier from file."""
        instance = cls(n_regimes=n_regimes)
        instance.hmm = MarketRegimeHMM.load(filepath)
        instance.is_fitted = True
        return instance


def detect_regimes(
    prices: pd.DataFrame,
    n_regimes: int = 3,
    returns: Optional[pd.Series] = None
) -> RegimeAnalysis:
    """
    Convenience function for quick regime detection.
    
    Parameters:
        prices: OHLCV DataFrame
        n_regimes: Number of regimes
        returns: Optional returns series
    
    Returns:
        RegimeAnalysis results
    """
    classifier = RegimeClassifier(n_regimes=n_regimes)
    return classifier.classify(prices, returns)


if __name__ == "__main__":
    print("Testing Regime Classifier...")
    
    # Generate synthetic data
    from .data_loader import DataLoader
    
    loader = DataLoader()
    data = loader.generate_synthetic(n_samples=500, n_regimes=3)
    
    # Classify regimes
    classifier = RegimeClassifier(n_regimes=3)
    analysis = classifier.classify(data.prices, data.returns)
    
    print(f"\nRegime Analysis:")
    print(f"  Current Regime: {analysis.current_regime_name}")
    print(f"  Transitions: {len(analysis.transitions)}")
    
    print(f"\nDuration Statistics:")
    for regime_id, duration in analysis.duration_stats.items():
        info = classifier.hmm.get_regime_info(regime_id)
        name = info.name if info else f"Regime {regime_id}"
        print(f"  {name}: {duration:.1f} periods avg")
    
    print(f"\nRegime Distribution:")
    print(analysis.regimes.value_counts(normalize=True))
