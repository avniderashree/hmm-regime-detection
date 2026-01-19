"""
Unit Tests for HMM Regime Detection
===================================
Comprehensive tests for all modules.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hmm_model import MarketRegimeHMM, RegimeInfo, HMMResult, select_optimal_regimes
from src.data_loader import DataLoader, MarketData
from src.feature_engineering import FeatureEngineer, FeatureSet, create_hmm_features
from src.regime_classifier import RegimeClassifier, RegimeAnalysis
from src.signal_generator import SignalGenerator, SignalType, TradingSignal


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_returns():
    """Generate sample return data."""
    np.random.seed(42)
    # Create 3-regime data
    regime1 = np.random.normal(0.001, 0.01, 100)
    regime2 = np.random.normal(-0.002, 0.03, 50)
    regime3 = np.random.normal(0.0005, 0.015, 100)
    return np.concatenate([regime1, regime2, regime3])


@pytest.fixture
def sample_prices():
    """Generate sample price DataFrame."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=250, freq='D')
    prices = pd.DataFrame({
        'Open': 100 + np.cumsum(np.random.randn(250) * 0.5),
        'High': 101 + np.cumsum(np.random.randn(250) * 0.5),
        'Low': 99 + np.cumsum(np.random.randn(250) * 0.5),
        'Close': 100 + np.cumsum(np.random.randn(250) * 0.5),
        'Volume': np.random.randint(1000000, 5000000, 250)
    }, index=dates)
    prices['High'] = prices[['Open', 'High', 'Close']].max(axis=1)
    prices['Low'] = prices[['Open', 'Low', 'Close']].min(axis=1)
    return prices


@pytest.fixture
def data_loader():
    """Create DataLoader instance."""
    return DataLoader()


@pytest.fixture
def feature_engineer():
    """Create FeatureEngineer instance."""
    return FeatureEngineer()


# ============================================================================
# Test HMM Model
# ============================================================================

class TestMarketRegimeHMM:
    """Tests for MarketRegimeHMM class."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = MarketRegimeHMM(n_regimes=3)
        assert model.n_regimes == 3
        assert model.is_fitted == False
        assert model.model is None
    
    def test_fit(self, sample_returns):
        """Test model fitting."""
        model = MarketRegimeHMM(n_regimes=3)
        features = sample_returns.reshape(-1, 1)
        
        model.fit(features)
        
        assert model.is_fitted == True
        assert model.model is not None
        assert len(model.regime_info) == 3
    
    def test_predict(self, sample_returns):
        """Test regime prediction."""
        model = MarketRegimeHMM(n_regimes=3)
        features = sample_returns.reshape(-1, 1)
        
        model.fit(features)
        regimes = model.predict(features)
        
        assert len(regimes) == len(sample_returns)
        assert set(regimes).issubset({0, 1, 2})
    
    def test_predict_proba(self, sample_returns):
        """Test probability prediction."""
        model = MarketRegimeHMM(n_regimes=3)
        features = sample_returns.reshape(-1, 1)
        
        model.fit(features)
        probs = model.predict_proba(features)
        
        assert probs.shape == (len(sample_returns), 3)
        assert np.allclose(probs.sum(axis=1), 1.0)
    
    def test_transition_matrix(self, sample_returns):
        """Test transition matrix extraction."""
        model = MarketRegimeHMM(n_regimes=3)
        features = sample_returns.reshape(-1, 1)
        
        model.fit(features)
        trans_matrix = model.get_transition_matrix()
        
        assert trans_matrix.shape == (3, 3)
        assert np.allclose(trans_matrix.sum(axis=1), 1.0)
    
    def test_aic_bic(self, sample_returns):
        """Test AIC/BIC calculation."""
        model = MarketRegimeHMM(n_regimes=3)
        features = sample_returns.reshape(-1, 1)
        
        model.fit(features)
        aic, bic = model.calculate_aic_bic(features)
        
        assert isinstance(aic, float)
        assert isinstance(bic, float)
    
    def test_fit_and_analyze(self, sample_returns):
        """Test complete analysis."""
        model = MarketRegimeHMM(n_regimes=3)
        features = sample_returns.reshape(-1, 1)
        
        result = model.fit_and_analyze(features)
        
        assert isinstance(result, HMMResult)
        assert len(result.regimes) == len(sample_returns)
        assert len(result.regime_info) == 3
    
    def test_save_load(self, sample_returns, tmp_path):
        """Test model persistence."""
        model = MarketRegimeHMM(n_regimes=3)
        features = sample_returns.reshape(-1, 1)
        model.fit(features)
        
        filepath = str(tmp_path / "model.pkl")
        model.save(filepath)
        
        loaded = MarketRegimeHMM.load(filepath)
        
        assert loaded.is_fitted == True
        assert loaded.n_regimes == 3
        
        # Predictions should match
        orig_pred = model.predict(features)
        loaded_pred = loaded.predict(features)
        np.testing.assert_array_equal(orig_pred, loaded_pred)


class TestModelSelection:
    """Tests for model selection functions."""
    
    def test_select_optimal_regimes(self):
        """Test optimal regime selection."""
        # Generate larger dataset for stability
        np.random.seed(42)
        regime1 = np.random.normal(0.001, 0.01, 150)
        regime2 = np.random.normal(-0.002, 0.025, 100)
        regime3 = np.random.normal(0.0005, 0.015, 150)
        features = np.concatenate([regime1, regime2, regime3]).reshape(-1, 1)
        
        optimal_n, scores = select_optimal_regimes(features, min_regimes=2, max_regimes=3)
        
        assert optimal_n in [2, 3]
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores.values())


# ============================================================================
# Test Data Loader
# ============================================================================

class TestDataLoader:
    """Tests for DataLoader class."""
    
    def test_generate_synthetic(self, data_loader):
        """Test synthetic data generation."""
        data = data_loader.generate_synthetic(n_samples=500, n_regimes=3)
        
        assert isinstance(data, MarketData)
        assert len(data.returns) == 500
        assert 'Close' in data.prices.columns
        assert 'True_Regime' in data.prices.columns
    
    def test_synthetic_regimes(self, data_loader):
        """Test synthetic data has correct regimes."""
        data = data_loader.generate_synthetic(n_samples=500, n_regimes=2)
        
        true_regimes = data.prices['True_Regime']
        assert set(true_regimes.unique()).issubset({0, 1})
    
    def test_calculate_returns(self, data_loader, sample_prices):
        """Test return calculation."""
        close = sample_prices['Close']
        
        simple_ret = data_loader.calculate_returns(close, method='simple')
        log_ret = data_loader.calculate_returns(close, method='log')
        
        assert len(simple_ret) == len(close) - 1
        assert len(log_ret) == len(close) - 1


# ============================================================================
# Test Feature Engineering
# ============================================================================

class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""
    
    def test_create_features(self, feature_engineer, sample_prices):
        """Test feature creation."""
        feature_set = feature_engineer.create_features(sample_prices)
        
        assert isinstance(feature_set, FeatureSet)
        assert 'returns' in feature_set.feature_names
        assert 'volatility_5d' in feature_set.feature_names
    
    def test_create_simple_features(self, feature_engineer):
        """Test simple feature creation."""
        returns = pd.Series(np.random.randn(100) * 0.01)
        
        feature_set = feature_engineer.create_simple_features(returns)
        
        assert 'returns' in feature_set.feature_names
        assert 'volatility' in feature_set.feature_names
    
    def test_normalize_features(self, feature_engineer, sample_prices):
        """Test feature normalization."""
        feature_set = feature_engineer.create_features(sample_prices)
        
        normalized = feature_engineer.normalize_features(feature_set, method='zscore')
        
        # Check mean ≈ 0 and std ≈ 1
        for col in normalized.features.columns:
            assert abs(normalized.features[col].mean()) < 0.1
            assert abs(normalized.features[col].std() - 1) < 0.1
    
    def test_select_features(self, feature_engineer, sample_prices):
        """Test feature selection."""
        feature_set = feature_engineer.create_features(sample_prices)
        
        selected = feature_engineer.select_features(feature_set, ['returns', 'volatility_5d'])
        
        assert len(selected.feature_names) == 2
        assert 'returns' in selected.feature_names


class TestCreateHMMFeatures:
    """Tests for create_hmm_features function."""
    
    def test_create_hmm_features(self, sample_prices):
        """Test HMM feature creation."""
        X, names, dates = create_hmm_features(sample_prices)
        
        assert X.ndim == 2
        assert len(names) > 0
        assert len(dates) == X.shape[0]
    
    def test_with_volatility(self, sample_prices):
        """Test with volatility feature."""
        X_with, names_with, _ = create_hmm_features(sample_prices, use_volatility=True)
        X_without, names_without, _ = create_hmm_features(sample_prices, use_volatility=False)
        
        assert X_with.shape[1] > X_without.shape[1]


# ============================================================================
# Test Regime Classifier
# ============================================================================

class TestRegimeClassifier:
    """Tests for RegimeClassifier class."""
    
    def test_initialization(self):
        """Test classifier initialization."""
        classifier = RegimeClassifier(n_regimes=3)
        
        assert classifier.n_regimes == 3
        assert classifier.is_fitted == False
    
    def test_fit(self, sample_prices):
        """Test classifier fitting."""
        classifier = RegimeClassifier(n_regimes=3)
        
        classifier.fit(sample_prices)
        
        assert classifier.is_fitted == True
    
    def test_classify(self, sample_prices):
        """Test regime classification."""
        classifier = RegimeClassifier(n_regimes=3)
        
        analysis = classifier.classify(sample_prices)
        
        assert isinstance(analysis, RegimeAnalysis)
        assert len(analysis.regimes) > 0
        assert analysis.current_regime in [0, 1, 2]
    
    def test_regime_probabilities(self, sample_prices):
        """Test regime probability output."""
        classifier = RegimeClassifier(n_regimes=3)
        analysis = classifier.classify(sample_prices)
        
        # Probabilities should sum to 1
        row_sums = analysis.regime_probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0)
    
    def test_transitions(self, sample_prices):
        """Test transition detection."""
        classifier = RegimeClassifier(n_regimes=3)
        analysis = classifier.classify(sample_prices)
        
        assert isinstance(analysis.transitions, list)
    
    def test_duration_stats(self, sample_prices):
        """Test duration statistics."""
        classifier = RegimeClassifier(n_regimes=3)
        analysis = classifier.classify(sample_prices)
        
        assert len(analysis.duration_stats) == 3
        assert all(d >= 0 for d in analysis.duration_stats.values())


# ============================================================================
# Test Signal Generator
# ============================================================================

class TestSignalGenerator:
    """Tests for SignalGenerator class."""
    
    def test_initialization(self):
        """Test generator initialization."""
        generator = SignalGenerator()
        
        assert generator.confidence_threshold == 0.6
        assert generator.strong_threshold == 0.8
    
    def test_generate_signals(self, sample_prices):
        """Test signal generation."""
        classifier = RegimeClassifier(n_regimes=3)
        analysis = classifier.classify(sample_prices)
        
        generator = SignalGenerator()
        signals = generator.generate_signals(analysis)
        
        assert len(signals.signals) == len(analysis.regimes)
        assert signals.current_signal is not None
    
    def test_signal_types(self, sample_prices):
        """Test different signal types are generated."""
        classifier = RegimeClassifier(n_regimes=3)
        analysis = classifier.classify(sample_prices)
        
        generator = SignalGenerator()
        signals = generator.generate_signals(analysis)
        
        signal_types = set(s.signal_type for s in signals.signals)
        assert len(signal_types) > 0
    
    def test_position_size(self, sample_prices):
        """Test position sizes are valid."""
        classifier = RegimeClassifier(n_regimes=3)
        analysis = classifier.classify(sample_prices)
        
        generator = SignalGenerator()
        signals = generator.generate_signals(analysis)
        
        for signal in signals.signals:
            assert -1.0 <= signal.position_size <= 1.0
    
    def test_calculate_performance(self, sample_prices):
        """Test performance calculation."""
        returns = sample_prices['Close'].pct_change().dropna()
        
        classifier = RegimeClassifier(n_regimes=3)
        analysis = classifier.classify(sample_prices)
        
        generator = SignalGenerator()
        signals = generator.generate_signals(analysis)
        
        perf = generator.calculate_signal_performance(signals, returns)
        
        assert 'total_return' in perf
        assert 'sharpe_ratio' in perf
        assert 'win_rate' in perf


class TestTradingSignal:
    """Tests for TradingSignal dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        signal = TradingSignal(
            date=datetime.now(),
            signal_type=SignalType.BUY,
            regime=1,
            regime_name='Bull Market',
            confidence=0.8,
            position_size=0.6,
            stop_loss_pct=0.02,
            take_profit_pct=0.04
        )
        
        d = signal.to_dict()
        
        assert d['signal_type'] == 'buy'
        assert d['regime'] == 1
        assert d['confidence'] == 0.8


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_full_pipeline(self, data_loader):
        """Test complete pipeline from data to signals."""
        # Generate data
        data = data_loader.generate_synthetic(n_samples=300, n_regimes=3)
        
        # Classify regimes
        classifier = RegimeClassifier(n_regimes=3)
        analysis = classifier.classify(data.prices, data.returns)
        
        # Generate signals
        generator = SignalGenerator()
        signals = generator.generate_signals(analysis)
        
        # Calculate performance
        perf = generator.calculate_signal_performance(signals, data.returns)
        
        # Assertions
        assert len(analysis.regimes) > 0
        assert len(signals.signals) > 0
        assert 'sharpe_ratio' in perf
    
    def test_model_persistence(self, data_loader, tmp_path):
        """Test saving and loading model."""
        data = data_loader.generate_synthetic(n_samples=300, n_regimes=3)
        
        # Train and save
        classifier = RegimeClassifier(n_regimes=3)
        classifier.classify(data.prices, data.returns)
        
        filepath = str(tmp_path / "classifier.pkl")
        classifier.save(filepath)
        
        # Load and predict
        loaded = RegimeClassifier.load(filepath)
        new_analysis = loaded.classify(data.prices, data.returns)
        
        assert len(new_analysis.regimes) > 0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, '-v'])
