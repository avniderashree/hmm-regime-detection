"""
HMM Regime Detection - Main Entry Point
========================================
Demonstrates the complete HMM regime detection pipeline.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import warnings

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import DataLoader, MarketData
from src.feature_engineering import FeatureEngineer, create_hmm_features
from src.hmm_model import MarketRegimeHMM, select_optimal_regimes
from src.regime_classifier import RegimeClassifier, detect_regimes
from src.signal_generator import SignalGenerator, generate_trading_signals
from src.visualization import (
    plot_regime_timeline,
    plot_regime_probabilities,
    plot_transition_matrix,
    plot_regime_statistics,
    plot_signal_performance,
    plot_regime_dashboard
)


def print_header(title: str) -> None:
    """Print formatted header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def print_section(title: str) -> None:
    """Print section header."""
    print(f"\n{title}")
    print(f"{'─' * 40}")


def main():
    """Main execution function."""
    print_header("HMM Regime Detection Pipeline")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print_section("Step 1: Loading Market Data")
    
    loader = DataLoader()
    
    # Try real data first, fall back to synthetic
    try:
        print("Attempting to fetch SPY data from Yahoo Finance...")
        data = loader.fetch_yahoo('SPY', '2018-01-01', '2023-12-31')
        print(f"✅ Loaded real SPY data")
    except Exception as e:
        print(f"Could not fetch real data: {e}")
        print("Generating synthetic data instead...")
        data = loader.generate_synthetic(n_samples=1260, n_regimes=3)  # ~5 years
    
    print(f"\nData Summary:")
    print(f"  Symbol: {data.symbol}")
    print(f"  Period: {data.start_date.date()} to {data.end_date.date()}")
    print(f"  Observations: {len(data.returns)}")
    print(f"  Mean Daily Return: {data.returns.mean():.4%}")
    print(f"  Daily Volatility: {data.returns.std():.4%}")
    print(f"  Annualized Vol: {data.returns.std() * np.sqrt(252):.2%}")
    
    # =========================================================================
    # Step 2: Feature Engineering
    # =========================================================================
    print_section("Step 2: Feature Engineering")
    
    engineer = FeatureEngineer(windows=[5, 10, 20])
    feature_set = engineer.create_features(data.prices, data.returns)
    
    print(f"\nFeatures Created: {len(feature_set.feature_names)}")
    print(f"Feature Names:")
    for name in feature_set.feature_names[:10]:
        print(f"  - {name}")
    if len(feature_set.feature_names) > 10:
        print(f"  ... and {len(feature_set.feature_names) - 10} more")
    
    # Create HMM-optimized features
    hmm_features, hmm_names, dates = create_hmm_features(
        data.prices, data.returns, 
        use_volatility=True, 
        normalize=True
    )
    
    print(f"\nHMM Features: {hmm_features.shape}")
    print(f"  Features: {hmm_names}")
    
    # =========================================================================
    # Step 3: Model Selection
    # =========================================================================
    print_section("Step 3: Model Selection (BIC)")
    
    optimal_n, scores = select_optimal_regimes(
        hmm_features, 
        min_regimes=2, 
        max_regimes=4,
        criterion='bic'
    )
    
    print(f"\nBIC Scores by Number of Regimes:")
    for n, score in sorted(scores.items()):
        marker = " ← Optimal" if n == optimal_n else ""
        print(f"  {n} regimes: {score:,.0f}{marker}")
    
    print(f"\n✅ Selected: {optimal_n} regimes")
    
    # =========================================================================
    # Step 4: Regime Detection
    # =========================================================================
    print_section("Step 4: Regime Detection")
    
    classifier = RegimeClassifier(n_regimes=optimal_n, smoothing_window=3)
    analysis = classifier.classify(data.prices, data.returns)
    
    print(f"\nCurrent Regime: {analysis.current_regime_name}")
    print(f"Regime Transitions: {len(analysis.transitions)}")
    
    print(f"\nRegime Statistics:")
    print(f"{'Regime':<15} {'Mean Ret':<12} {'Volatility':<12} {'Duration':<12} {'Frequency':<12}")
    print(f"{'─' * 63}")
    
    for info in analysis.regime_stats:
        print(f"{info.name:<15} {info.mean_return:>10.4%} {info.volatility:>10.4%} "
              f"{info.duration:>10.1f} {info.frequency:>10.1%}")
    
    # Transition matrix
    trans_matrix = classifier.hmm.get_transition_matrix()
    print(f"\nTransition Matrix:")
    regime_names = [info.name for info in analysis.regime_stats]
    
    print(f"{'From/To':<15}", end='')
    for name in regime_names:
        print(f"{name:<15}", end='')
    print()
    
    for i, name in enumerate(regime_names):
        print(f"{name:<15}", end='')
        for j in range(len(regime_names)):
            print(f"{trans_matrix[i, j]:>13.1%}", end='  ')
        print()
    
    # Regime distribution
    print(f"\nRegime Distribution:")
    dist = analysis.regimes.value_counts(normalize=True).sort_index()
    for regime_id, pct in dist.items():
        info = classifier.hmm.get_regime_info(regime_id)
        name = info.name if info else f"Regime {regime_id}"
        print(f"  {name}: {pct:.1%}")
    
    # =========================================================================
    # Step 5: Trading Signal Generation
    # =========================================================================
    print_section("Step 5: Trading Signal Generation")
    
    generator = SignalGenerator(
        confidence_threshold=0.6,
        strong_threshold=0.8,
        base_stop_loss=0.02,
        base_take_profit=0.04
    )
    
    signals = generator.generate_signals(analysis)
    
    print(f"\nCurrent Signal:")
    current = signals.current_signal
    print(f"  Type: {current.signal_type.value.upper()}")
    print(f"  Regime: {current.regime_name}")
    print(f"  Confidence: {current.confidence:.1%}")
    print(f"  Position Size: {current.position_size:+.2f}")
    print(f"  Stop Loss: {current.stop_loss_pct:.1%}")
    print(f"  Take Profit: {current.take_profit_pct:.1%}")
    
    print(f"\nSignal Distribution:")
    signal_dist = signals.signal_history['signal_type'].value_counts()
    for signal_type, count in signal_dist.items():
        print(f"  {signal_type}: {count} ({count/len(signals.signals):.1%})")
    
    # Calculate performance
    perf = generator.calculate_signal_performance(signals, data.returns)
    
    print(f"\nBacktest Performance:")
    print(f"  Total Return: {perf['total_return']:.2%}")
    print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
    print(f"  Win Rate: {perf['win_rate']:.1%}")
    print(f"  Total Trades: {perf['n_trades']}")
    
    # =========================================================================
    # Step 6: Visualization
    # =========================================================================
    print_section("Step 6: Generating Visualizations")
    
    # Prepare regime names dict
    regime_names_dict = {info.regime_id: info.name for info in analysis.regime_stats}
    
    # Get prices for plotting
    close_prices = data.prices['Close'] if 'Close' in data.prices.columns else data.prices.iloc[:, 0]
    close_prices = close_prices.reindex(analysis.regimes.index)
    
    # 1. Regime Dashboard
    print("  Creating regime dashboard...")
    plot_regime_dashboard(
        regimes=analysis.regimes,
        regime_probs=analysis.regime_probs,
        prices=close_prices,
        transition_matrix=trans_matrix,
        regime_names=regime_names_dict,
        save_path='output/regime_dashboard.png'
    )
    
    # 2. Regime Timeline
    print("  Creating regime timeline...")
    plot_regime_timeline(
        regimes=analysis.regimes,
        prices=close_prices,
        regime_names=regime_names_dict,
        save_path='output/regime_timeline.png'
    )
    
    # 3. Transition Matrix
    print("  Creating transition matrix...")
    plot_transition_matrix(
        transition_matrix=trans_matrix,
        regime_names=[info.name for info in analysis.regime_stats],
        save_path='output/transition_matrix.png'
    )
    
    # 4. Regime Statistics
    print("  Creating regime statistics...")
    plot_regime_statistics(
        regime_stats=[info.to_dict() for info in analysis.regime_stats],
        save_path='output/regime_statistics.png'
    )
    
    # 5. Signal Performance
    print("  Creating signal performance chart...")
    strategy_returns = signals.signal_history['position_size'].shift(1) * data.returns.reindex(signals.signal_history.index).fillna(0)
    cumulative_returns = (1 + strategy_returns.fillna(0)).cumprod() - 1
    
    plot_signal_performance(
        signal_history=signals.signal_history,
        cumulative_returns=cumulative_returns,
        save_path='output/signal_performance.png'
    )
    
    print("  ✅ All visualizations saved to ./output/")
    
    # =========================================================================
    # Step 7: Save Results
    # =========================================================================
    print_section("Step 7: Saving Results")
    
    # Save model
    classifier.save('models/hmm_regime_model.pkl')
    print("  ✅ Model saved to models/hmm_regime_model.pkl")
    
    # Save regime history
    regime_df = pd.DataFrame({
        'date': analysis.regimes.index,
        'regime': analysis.regimes.values,
        'regime_name': [regime_names_dict.get(r, f'Regime {r}') for r in analysis.regimes.values]
    })
    for i in range(optimal_n):
        regime_df[f'regime_{i}_prob'] = analysis.regime_probs.iloc[:, i].values
    
    regime_df.to_csv('output/regime_history.csv', index=False)
    print("  ✅ Regime history saved to output/regime_history.csv")
    
    # Save signals
    signals.signal_history.to_csv('output/signal_history.csv')
    print("  ✅ Signal history saved to output/signal_history.csv")
    
    # Save summary report
    summary = {
        'timestamp': datetime.now().isoformat(),
        'data': data.to_dict(),
        'model': {
            'n_regimes': optimal_n,
            'bic_score': scores[optimal_n],
            'log_likelihood': float(classifier.hmm.score(hmm_features))
        },
        'current_state': {
            'regime': analysis.current_regime,
            'regime_name': analysis.current_regime_name,
            'signal': current.signal_type.value,
            'confidence': current.confidence,
            'position_size': current.position_size
        },
        'performance': perf
    }
    
    import json
    with open('output/summary_report.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("  ✅ Summary report saved to output/summary_report.json")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_header("Pipeline Complete")
    
    print(f"""
Summary:
  - Data: {len(data.returns)} observations
  - Regimes: {optimal_n} detected
  - Current: {analysis.current_regime_name} ({current.confidence:.0%} confidence)
  - Signal: {current.signal_type.value.upper()} (position: {current.position_size:+.2f})
  - Sharpe: {perf['sharpe_ratio']:.2f}

Output Files:
  - output/regime_dashboard.png
  - output/regime_timeline.png
  - output/transition_matrix.png
  - output/regime_statistics.png
  - output/signal_performance.png
  - output/regime_history.csv
  - output/signal_history.csv
  - output/summary_report.json
  - models/hmm_regime_model.pkl
""")
    
    return analysis, signals


if __name__ == "__main__":
    analysis, signals = main()
