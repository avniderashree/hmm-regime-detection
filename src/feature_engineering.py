"""
Feature Engineering Module
==========================
Creates features for HMM regime detection.

Supports:
- Return-based features
- Volatility features
- Technical indicators
- Feature normalization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FeatureSet:
    """
    Container for engineered features.
    
    Attributes:
        features: DataFrame of features
        feature_names: List of feature names
        dates: Date index
    """
    features: pd.DataFrame
    feature_names: List[str]
    dates: pd.DatetimeIndex
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return self.features.values
    
    def get_feature(self, name: str) -> pd.Series:
        """Get a single feature by name."""
        return self.features[name]


class FeatureEngineer:
    """
    Feature engineering for regime detection.
    
    Features:
    - Returns and log returns
    - Rolling volatility
    - Rolling mean
    - Momentum indicators
    - Volume features
    """
    
    def __init__(
        self,
        windows: Optional[List[int]] = None,
        include_volume: bool = True
    ):
        """
        Initialize feature engineer.
        
        Parameters:
            windows: Rolling window sizes
            include_volume: Whether to include volume features
        """
        self.windows = windows or [5, 10, 20]
        self.include_volume = include_volume
    
    def create_features(
        self,
        prices: pd.DataFrame,
        returns: Optional[pd.Series] = None
    ) -> FeatureSet:
        """
        Create full feature set from price data.
        
        Parameters:
            prices: DataFrame with OHLCV data
            returns: Optional pre-calculated returns
        
        Returns:
            FeatureSet object
        """
        features = pd.DataFrame(index=prices.index)
        
        # Get close prices
        close = prices['Close'] if 'Close' in prices.columns else prices.iloc[:, 0]
        
        # Calculate returns if not provided
        if returns is None:
            returns = close.pct_change()
        
        # 1. Basic returns
        features['returns'] = returns
        features['log_returns'] = np.log(close / close.shift(1))
        
        # 2. Rolling volatility
        for window in self.windows:
            features[f'volatility_{window}d'] = returns.rolling(window).std()
            features[f'mean_return_{window}d'] = returns.rolling(window).mean()
        
        # 3. Momentum features
        for window in self.windows:
            features[f'momentum_{window}d'] = close.pct_change(window)
        
        # 4. Relative strength
        features['rsi_14'] = self._calculate_rsi(close, 14)
        
        # 5. Price position
        for window in [20, 50]:
            rolling_high = close.rolling(window).max()
            rolling_low = close.rolling(window).min()
            features[f'price_position_{window}d'] = (close - rolling_low) / (rolling_high - rolling_low + 1e-10)
        
        # 6. Volume features
        if self.include_volume and 'Volume' in prices.columns:
            volume = prices['Volume']
            features['volume_change'] = volume.pct_change()
            features['volume_ma_ratio'] = volume / volume.rolling(20).mean()
        
        # 7. High-Low range
        if 'High' in prices.columns and 'Low' in prices.columns:
            features['range_pct'] = (prices['High'] - prices['Low']) / close
        
        # Drop NaN rows
        features = features.dropna()
        
        return FeatureSet(
            features=features,
            feature_names=list(features.columns),
            dates=features.index
        )
    
    def create_simple_features(
        self,
        returns: pd.Series,
        volatility_window: int = 20
    ) -> FeatureSet:
        """
        Create minimal feature set from returns only.
        
        Parameters:
            returns: Return series
            volatility_window: Window for volatility calculation
        
        Returns:
            FeatureSet object
        """
        features = pd.DataFrame(index=returns.index)
        
        features['returns'] = returns
        features['volatility'] = returns.rolling(volatility_window).std()
        
        # Drop NaN
        features = features.dropna()
        
        return FeatureSet(
            features=features,
            feature_names=list(features.columns),
            dates=features.index
        )
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def normalize_features(
        self,
        feature_set: FeatureSet,
        method: str = 'zscore'
    ) -> FeatureSet:
        """
        Normalize features.
        
        Parameters:
            feature_set: FeatureSet to normalize
            method: 'zscore' or 'minmax'
        
        Returns:
            Normalized FeatureSet
        """
        features = feature_set.features.copy()
        
        if method == 'zscore':
            features = (features - features.mean()) / features.std()
        elif method == 'minmax':
            features = (features - features.min()) / (features.max() - features.min() + 1e-10)
        
        return FeatureSet(
            features=features,
            feature_names=feature_set.feature_names,
            dates=feature_set.dates
        )
    
    def select_features(
        self,
        feature_set: FeatureSet,
        feature_names: List[str]
    ) -> FeatureSet:
        """
        Select subset of features.
        
        Parameters:
            feature_set: Original FeatureSet
            feature_names: Features to select
        
        Returns:
            FeatureSet with selected features
        """
        selected = feature_set.features[feature_names].copy()
        
        return FeatureSet(
            features=selected,
            feature_names=feature_names,
            dates=feature_set.dates
        )


def create_hmm_features(
    prices: pd.DataFrame,
    returns: Optional[pd.Series] = None,
    use_volatility: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, List[str], pd.DatetimeIndex]:
    """
    Create optimized features for HMM training.
    
    Parameters:
        prices: OHLCV DataFrame
        returns: Optional returns series
        use_volatility: Include volatility feature
        normalize: Normalize features
    
    Returns:
        Tuple of (feature_array, feature_names, dates)
    """
    engineer = FeatureEngineer(include_volume=False)
    
    if returns is None:
        close = prices['Close'] if 'Close' in prices.columns else prices.iloc[:, 0]
        returns = close.pct_change()
    
    if use_volatility:
        feature_set = engineer.create_simple_features(returns)
    else:
        features = pd.DataFrame({'returns': returns}).dropna()
        feature_set = FeatureSet(
            features=features,
            feature_names=['returns'],
            dates=features.index
        )
    
    if normalize:
        feature_set = engineer.normalize_features(feature_set)
    
    return feature_set.to_array(), feature_set.feature_names, feature_set.dates


if __name__ == "__main__":
    print("Testing Feature Engineering...")
    
    # Create sample price data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    prices = pd.DataFrame({
        'Open': 100 + np.cumsum(np.random.randn(500) * 0.5),
        'High': 101 + np.cumsum(np.random.randn(500) * 0.5),
        'Low': 99 + np.cumsum(np.random.randn(500) * 0.5),
        'Close': 100 + np.cumsum(np.random.randn(500) * 0.5),
        'Volume': np.random.randint(1000000, 5000000, 500)
    }, index=dates)
    prices['High'] = prices[['Open', 'High', 'Close']].max(axis=1)
    prices['Low'] = prices[['Open', 'Low', 'Close']].min(axis=1)
    
    # Create features
    engineer = FeatureEngineer()
    feature_set = engineer.create_features(prices)
    
    print(f"\nFeatures Created: {len(feature_set.feature_names)}")
    print(f"Feature Names: {feature_set.feature_names}")
    print(f"Shape: {feature_set.features.shape}")
    
    # Create HMM features
    X, names, dates = create_hmm_features(prices)
    print(f"\nHMM Features Shape: {X.shape}")
    print(f"HMM Feature Names: {names}")
