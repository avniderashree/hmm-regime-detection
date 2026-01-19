"""
Signal Generator Module
=======================
Generates trading signals based on detected regimes.

Supports:
- Regime-based position sizing
- Signal confidence scoring
- Risk-adjusted signals
- Signal history tracking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .regime_classifier import RegimeAnalysis, RegimeClassifier


class SignalType(Enum):
    """Trading signal types."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class TradingSignal:
    """
    Trading signal with metadata.
    
    Attributes:
        date: Signal date
        signal_type: Type of signal
        regime: Current regime
        regime_name: Human-readable regime name
        confidence: Signal confidence (0-1)
        position_size: Suggested position size (-1 to 1)
        stop_loss_pct: Suggested stop loss percentage
        take_profit_pct: Suggested take profit percentage
    """
    date: datetime
    signal_type: SignalType
    regime: int
    regime_name: str
    confidence: float
    position_size: float
    stop_loss_pct: float
    take_profit_pct: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'date': self.date.isoformat() if hasattr(self.date, 'isoformat') else str(self.date),
            'signal_type': self.signal_type.value,
            'regime': self.regime,
            'regime_name': self.regime_name,
            'confidence': self.confidence,
            'position_size': self.position_size,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct
        }


@dataclass
class SignalSummary:
    """
    Summary of generated signals.
    
    Attributes:
        signals: List of all signals
        current_signal: Most recent signal
        signal_history: DataFrame of signal history
        regime_signal_map: Mapping of regimes to signals
    """
    signals: List[TradingSignal]
    current_signal: TradingSignal
    signal_history: pd.DataFrame
    regime_signal_map: Dict[int, SignalType]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'n_signals': len(self.signals),
            'current_signal': self.current_signal.to_dict(),
            'regime_signal_map': {k: v.value for k, v in self.regime_signal_map.items()}
        }


class SignalGenerator:
    """
    Generate trading signals from regime analysis.
    
    Features:
    - Regime-to-signal mapping
    - Confidence-based position sizing
    - Risk management parameters
    - Signal smoothing to reduce whipsaws
    """
    
    # Default regime-to-signal mapping
    DEFAULT_SIGNAL_MAP = {
        'Bull Market': SignalType.BUY,
        'Bear Market': SignalType.SELL,
        'Neutral': SignalType.HOLD,
        'Recovery': SignalType.BUY,
        'Quiet': SignalType.HOLD
    }
    
    # Position sizes by signal type
    DEFAULT_POSITION_SIZES = {
        SignalType.STRONG_BUY: 1.0,
        SignalType.BUY: 0.6,
        SignalType.HOLD: 0.0,
        SignalType.SELL: -0.6,
        SignalType.STRONG_SELL: -1.0
    }
    
    def __init__(
        self,
        confidence_threshold: float = 0.6,
        strong_threshold: float = 0.8,
        base_stop_loss: float = 0.02,
        base_take_profit: float = 0.04,
        regime_signal_map: Optional[Dict[str, SignalType]] = None
    ):
        """
        Initialize signal generator.
        
        Parameters:
            confidence_threshold: Minimum confidence for signal
            strong_threshold: Confidence for strong signals
            base_stop_loss: Base stop loss percentage
            base_take_profit: Base take profit percentage
            regime_signal_map: Custom regime-to-signal mapping
        """
        self.confidence_threshold = confidence_threshold
        self.strong_threshold = strong_threshold
        self.base_stop_loss = base_stop_loss
        self.base_take_profit = base_take_profit
        self.regime_signal_map = regime_signal_map or self.DEFAULT_SIGNAL_MAP.copy()
    
    def generate_signals(
        self,
        analysis: RegimeAnalysis,
        volatility: Optional[pd.Series] = None
    ) -> SignalSummary:
        """
        Generate trading signals from regime analysis.
        
        Parameters:
            analysis: RegimeAnalysis from classifier
            volatility: Optional volatility series for risk adjustment
        
        Returns:
            SignalSummary with all signals
        """
        signals = []
        regime_to_signal: Dict[int, SignalType] = {}
        
        # Map regime IDs to signal types
        for regime_info in analysis.regime_stats:
            signal_type = self.regime_signal_map.get(
                regime_info.name, 
                SignalType.HOLD
            )
            regime_to_signal[regime_info.regime_id] = signal_type
        
        # Generate signal for each date
        for date in analysis.regimes.index:
            regime = int(analysis.regimes.loc[date])
            
            # Get regime probability (confidence)
            prob_col = f'regime_{regime}_prob'
            confidence = float(analysis.regime_probs.loc[date, prob_col])
            
            # Determine signal type
            base_signal = regime_to_signal.get(regime, SignalType.HOLD)
            
            # Upgrade to strong signal if high confidence
            if confidence >= self.strong_threshold:
                if base_signal == SignalType.BUY:
                    signal_type = SignalType.STRONG_BUY
                elif base_signal == SignalType.SELL:
                    signal_type = SignalType.STRONG_SELL
                else:
                    signal_type = base_signal
            elif confidence >= self.confidence_threshold:
                signal_type = base_signal
            else:
                signal_type = SignalType.HOLD
            
            # Calculate position size
            position_size = self._calculate_position_size(signal_type, confidence)
            
            # Adjust stop/take profit based on regime volatility
            regime_info = next(
                (r for r in analysis.regime_stats if r.regime_id == regime),
                None
            )
            regime_vol = regime_info.volatility if regime_info else 0.02
            
            stop_loss = self._adjust_for_volatility(
                self.base_stop_loss, 
                regime_vol,
                volatility.loc[date] if volatility is not None and date in volatility.index else None
            )
            take_profit = self._adjust_for_volatility(
                self.base_take_profit,
                regime_vol,
                volatility.loc[date] if volatility is not None and date in volatility.index else None
            )
            
            # Get regime name
            regime_name = regime_info.name if regime_info else f"Regime {regime}"
            
            signal = TradingSignal(
                date=date.to_pydatetime() if hasattr(date, 'to_pydatetime') else date,
                signal_type=signal_type,
                regime=regime,
                regime_name=regime_name,
                confidence=confidence,
                position_size=position_size,
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit
            )
            signals.append(signal)
        
        # Create history DataFrame
        history_data = [s.to_dict() for s in signals]
        history_df = pd.DataFrame(history_data)
        history_df['date'] = pd.to_datetime(history_df['date'])
        history_df.set_index('date', inplace=True)
        
        return SignalSummary(
            signals=signals,
            current_signal=signals[-1],
            signal_history=history_df,
            regime_signal_map=regime_to_signal
        )
    
    def _calculate_position_size(
        self,
        signal_type: SignalType,
        confidence: float
    ) -> float:
        """Calculate position size based on signal and confidence."""
        base_size = self.DEFAULT_POSITION_SIZES.get(signal_type, 0.0)
        
        # Scale by confidence
        scaled_size = base_size * confidence
        
        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, scaled_size))
    
    def _adjust_for_volatility(
        self,
        base_value: float,
        regime_vol: float,
        current_vol: Optional[float] = None
    ) -> float:
        """Adjust stop/take profit for volatility."""
        # Use current volatility if available, else regime volatility
        vol = current_vol if current_vol is not None else regime_vol
        
        # Normalize (assume 0.02 is normal daily vol)
        vol_ratio = vol / 0.02 if vol > 0 else 1.0
        
        # Scale base value
        adjusted = base_value * vol_ratio
        
        # Clamp to reasonable range
        return max(0.005, min(0.10, adjusted))
    
    def get_signal_at_date(
        self,
        summary: SignalSummary,
        date: datetime
    ) -> Optional[TradingSignal]:
        """Get signal at a specific date."""
        for signal in summary.signals:
            signal_date = signal.date
            if hasattr(signal_date, 'date'):
                signal_date = signal_date.date()
            if hasattr(date, 'date'):
                date = date.date()
            if signal_date == date:
                return signal
        return None
    
    def calculate_signal_performance(
        self,
        summary: SignalSummary,
        returns: pd.Series
    ) -> Dict:
        """
        Calculate performance of signals.
        
        Returns:
            Dict with performance metrics
        """
        signal_df = summary.signal_history.copy()
        
        # Align returns with signals
        aligned_returns = returns.reindex(signal_df.index).fillna(0)
        
        # Calculate strategy returns
        signal_df['next_return'] = aligned_returns.shift(-1)
        signal_df['strategy_return'] = signal_df['position_size'] * signal_df['next_return']
        
        # Calculate metrics
        total_return = (1 + signal_df['strategy_return']).prod() - 1
        sharpe = signal_df['strategy_return'].mean() / (signal_df['strategy_return'].std() + 1e-10) * np.sqrt(252)
        
        # Win rate
        positive_returns = (signal_df['strategy_return'] > 0).sum()
        total_trades = (signal_df['position_size'] != 0).sum()
        win_rate = positive_returns / total_trades if total_trades > 0 else 0
        
        # By signal type
        by_signal = signal_df.groupby('signal_type')['strategy_return'].agg(['mean', 'count', 'sum'])
        
        return {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe),
            'win_rate': float(win_rate),
            'n_trades': int(total_trades),
            'by_signal': by_signal.to_dict()
        }


def generate_trading_signals(
    analysis: RegimeAnalysis,
    confidence_threshold: float = 0.6
) -> SignalSummary:
    """
    Convenience function for quick signal generation.
    
    Parameters:
        analysis: RegimeAnalysis from classifier
        confidence_threshold: Minimum confidence for signal
    
    Returns:
        SignalSummary with signals
    """
    generator = SignalGenerator(confidence_threshold=confidence_threshold)
    return generator.generate_signals(analysis)


if __name__ == "__main__":
    print("Testing Signal Generator...")
    
    # Generate synthetic data and classify
    from .data_loader import DataLoader
    from .regime_classifier import RegimeClassifier
    
    loader = DataLoader()
    data = loader.generate_synthetic(n_samples=500, n_regimes=3)
    
    classifier = RegimeClassifier(n_regimes=3)
    analysis = classifier.classify(data.prices, data.returns)
    
    # Generate signals
    generator = SignalGenerator()
    signals = generator.generate_signals(analysis)
    
    print(f"\nSignal Summary:")
    print(f"  Total Signals: {len(signals.signals)}")
    print(f"  Current Signal: {signals.current_signal.signal_type.value}")
    print(f"  Current Confidence: {signals.current_signal.confidence:.2%}")
    print(f"  Position Size: {signals.current_signal.position_size:.2f}")
    
    print(f"\nSignal Distribution:")
    print(signals.signal_history['signal_type'].value_counts())
    
    # Calculate performance
    perf = generator.calculate_signal_performance(signals, data.returns)
    print(f"\nPerformance:")
    print(f"  Total Return: {perf['total_return']:.2%}")
    print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
    print(f"  Win Rate: {perf['win_rate']:.2%}")
