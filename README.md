# 📈 HMM Regime Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-32%20passed-brightgreen.svg)](tests/)
[![hmmlearn](https://img.shields.io/badge/hmmlearn-0.3+-orange.svg)](https://hmmlearn.readthedocs.io/)

A **production-grade Hidden Markov Model (HMM) system** for detecting market regimes and generating trading signals. This pipeline automatically identifies Bull, Bear, and Neutral market conditions using unsupervised machine learning—the same techniques used by quantitative hedge funds and systematic trading desks.

---

## 📋 Table of Contents

1. [What Is This Project?](#-what-is-this-project)
2. [Key Concepts Explained](#-key-concepts-explained)
3. [Features](#-features)
4. [Quick Start](#-quick-start)
5. [Detailed Installation](#-detailed-installation)
6. [How to Run](#-how-to-run)
7. [Understanding the Output](#-understanding-the-output)
8. [Project Architecture](#-project-architecture)
9. [API Reference](#-api-reference)
10. [Code Examples](#-code-examples)
11. [Testing](#-testing)
12. [Troubleshooting](#-troubleshooting)
13. [References](#-references)
14. [Author](#-author)

---

## 🎯 What Is This Project?

This project uses **Hidden Markov Models (HMM)** to detect underlying market regimes from price data. Markets don't behave consistently—sometimes they're trending up (Bull), sometimes crashing (Bear), and sometimes moving sideways (Neutral).

### Questions This Pipeline Answers

| Question | How We Answer It |
|----------|------------------|
| *"What regime is the market in right now?"* | HMM detects current state with probability |
| *"How confident is the regime prediction?"* | Regime probabilities (0-100%) |
| *"When did regime changes occur?"* | Transition detection with dates |
| *"Should I be long, short, or flat?"* | Trading signals based on regime |
| *"What's the expected duration of this regime?"* | Duration statistics from history |
| *"What happens in each regime?"* | Return/volatility characteristics |

### The Regime Model

```
+-------------+           +-------------+           +-------------+
|    BULL     |  <------> |   NEUTRAL   |  <------> |    BEAR     |
|-------------|           |-------------|           |-------------|
| High Return |           | Low Return  |           | Neg Return  |
| Low Vol     |           | Low Vol     |           | High Vol    |
| Signal: BUY |           | Signal: HOLD|           | Signal: SELL|
+------+------+           +------+------+           +------+------+
       |                         |                         |
       +-----------+-------------+-----------+-------------+
                   |                         |
             Transition Matrix          Transition Matrix
             P(Bull→Bear)               P(Bear→Bull)
```

### Real-World Applications

| Role | How They Use This |
|------|-------------------|
| **Quant Hedge Fund** | Regime-conditional trading strategies |
| **Asset Manager** | Dynamic asset allocation based on regime |
| **Risk Manager** | Adjust risk limits by market regime |
| **Algorithmic Trader** | Signal generation for systematic trading |
| **Portfolio Manager** | Regime-aware position sizing |

---

## 📚 Key Concepts Explained

### What is a Hidden Markov Model (HMM)?

An **HMM** is a statistical model that assumes the system being modeled is a Markov process with **hidden (unobserved) states**. In market terms:

```
                    OBSERVED                         HIDDEN
                    --------                         ------
Day 1:  Return = +0.5%   Volatility = 1.2%    -->   Bull Market
Day 2:  Return = +0.3%   Volatility = 1.1%    -->   Bull Market
Day 3:  Return = -2.1%   Volatility = 3.5%    -->   Bear Market  (TRANSITION!)
Day 4:  Return = -1.8%   Volatility = 3.2%    -->   Bear Market
```

We **observe** returns and volatility. The HMM **infers** which hidden state (regime) generated those observations.

### Key HMM Components

| Component | Description | In Market Terms |
|-----------|-------------|-----------------|
| **Hidden States** | Unobserved regimes | Bull, Bear, Neutral |
| **Observations** | Measured data | Returns, volatility |
| **Transition Matrix** | Probability of switching states | P(Bull→Bear), P(Bear→Bull) |
| **Emission Model** | How states generate observations | Each regime has mean/std |
| **Initial Distribution** | Starting state probabilities | Prior belief |

### Transition Matrix Example

```
             To:
          Bull   Neutral  Bear
From:  ┌─────────────────────┐
Bull   │  95%     4%     1%  │  <-- Bull tends to persist
Neutral│  10%    85%     5%  │  <-- Neutral is sticky
Bear   │   2%     8%    90%  │  <-- Bear markets last
       └─────────────────────┘
```

**Interpretation:** If we're in a Bull market today, there's a 95% chance we stay in Bull tomorrow, 4% chance we go to Neutral, and 1% chance we crash to Bear.

### Gaussian Emissions

Each regime generates returns from a **Gaussian (Normal) distribution**:

```
Bull Market:   μ = +0.08% daily,  σ = 1.0% daily
Neutral:       μ = +0.01% daily,  σ = 1.5% daily  
Bear Market:   μ = -0.15% daily,  σ = 2.5% daily
```

### The EM Algorithm

The HMM learns its parameters using the **Expectation-Maximization (EM)** algorithm:

1. **E-Step:** Estimate probability of each hidden state given observations
2. **M-Step:** Update parameters (means, variances, transition probs) to maximize likelihood
3. **Repeat** until convergence

---

## ✨ Features

### 1. Regime Detection

| Capability | Description |
|------------|-------------|
| **Gaussian HMM** | Detects regimes using returns + volatility |
| **Auto Labeling** | Names regimes (Bull/Bear/Neutral) automatically |
| **Model Selection** | Uses BIC to pick optimal number of regimes |
| **Probability Smoothing** | Reduces whipsaw with rolling average |

### 2. Signal Generation

| Signal Type | Position Size | Condition |
|-------------|--------------|-----------|
| STRONG_BUY | +1.0 | Bull regime, >80% confidence |
| BUY | +0.6 | Bull regime, >60% confidence |
| HOLD | 0.0 | Neutral or low confidence |
| SELL | -0.6 | Bear regime, >60% confidence |
| STRONG_SELL | -1.0 | Bear regime, >80% confidence |

### 3. Risk Management

- ✅ Volatility-adjusted stop losses
- ✅ Regime-aware take profits
- ✅ Confidence-based position sizing
- ✅ Transition probability for next regime

### 4. Visualization (6 Chart Types)

| Chart | Purpose |
|-------|---------|
| Regime Dashboard | All-in-one overview |
| Regime Timeline | Price with regime overlay |
| Probability Evolution | Stack plot of regime probs |
| Transition Matrix | Heatmap of transitions |
| Regime Statistics | Return/vol/frequency bars |
| Signal Performance | Cumulative strategy returns |

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/avniderashree/hmm-regime-detection.git
cd hmm-regime-detection

# Install dependencies
pip install -r requirements.txt

# Run the demo
python main.py
```

---

## 🛠️ Detailed Installation

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.8+ | 3.10+ |
| RAM | 1 GB | 2 GB+ |
| Disk | 50 MB | 100 MB |
| OS | Windows/macOS/Linux | Any |

### Step 1: Clone Repository

```bash
git clone https://github.com/avniderashree/hmm-regime-detection.git
cd hmm-regime-detection
```

### Step 2: Create Virtual Environment (Recommended)

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.21.0 | Numerical arrays |
| pandas | ≥1.3.0 | Data manipulation |
| scipy | ≥1.7.0 | Statistical functions |
| scikit-learn | ≥1.0.0 | ML utilities |
| hmmlearn | ≥0.3.0 | HMM implementation |
| matplotlib | ≥3.5.0 | Charting |
| seaborn | ≥0.11.0 | Chart styling |
| yfinance | ≥0.2.0 | Market data |
| joblib | ≥1.1.0 | Model serialization |
| pytest | ≥7.0.0 | Testing |

### Step 4: Verify Installation

```bash
python -c "from src.hmm_model import MarketRegimeHMM; print('✅ Installation successful!')"
```

---

## ▶️ How to Run

### Option 1: Full Demo (Recommended)

```bash
python main.py
```

This will:
1. ✅ Fetch SPY data (or generate synthetic if unavailable)
2. ✅ Engineer features (returns + volatility)
3. ✅ Select optimal number of regimes via BIC
4. ✅ Detect regimes with probability smoothing
5. ✅ Generate trading signals
6. ✅ Calculate backtest performance
7. ✅ Create 5 visualizations
8. ✅ Save reports to `./output/`

### Option 2: Run Individual Modules

```bash
# Test HMM model
python -m src.hmm_model

# Test data loader
python -m src.data_loader

# Test features
python -m src.feature_engineering

# Test classifier
python -m src.regime_classifier

# Test signals
python -m src.signal_generator
```

### Option 3: Interactive Python

```python
>>> from src.regime_classifier import RegimeClassifier
>>> from src.data_loader import DataLoader

>>> loader = DataLoader()
>>> data = loader.generate_synthetic(n_samples=500, n_regimes=3)

>>> classifier = RegimeClassifier(n_regimes=3)
>>> analysis = classifier.classify(data.prices, data.returns)

>>> print(f"Current Regime: {analysis.current_regime_name}")
Current Regime: Bull Market
```

### Option 4: Run Tests

```bash
pytest tests/ -v
```

---

## 📊 Understanding the Output

When you run `python main.py`, here's what each section means:

### Step 4: Regime Detection

```
Current Regime: Bull Market
Regime Transitions: 12

Regime Statistics:
Regime          Mean Ret     Volatility   Duration     Frequency
───────────────────────────────────────────────────────────────
Bear Market      -0.0800%      2.3500%        25.3         18.2%
Neutral           0.0200%      1.1200%        45.8         41.5%
Bull Market       0.1200%      0.8900%        38.2         40.3%
```

**Interpretation:**
- **Mean Ret**: Average daily return in that regime
- **Volatility**: Daily standard deviation
- **Duration**: Average consecutive days in regime
- **Frequency**: Percentage of time spent in regime

### Transition Matrix

```
             Bull Market    Neutral      Bear Market
Bull Market       93.2%        5.8%            1.0%
Neutral            8.5%       86.3%            5.2%
Bear Market        1.5%        6.8%           91.7%
```

**Interpretation:** 
- From Bull, 93.2% chance to stay Bull, 5.8% to Neutral, 1.0% to Bear
- Bear markets are "sticky" (91.7% persistence)

### Step 5: Trading Signals

```
Current Signal:
  Type: BUY
  Regime: Bull Market
  Confidence: 87.2%
  Position Size: +0.52
  Stop Loss: 2.1%
  Take Profit: 4.2%
```

**Interpretation:**
- **Type**: BUY (not STRONG_BUY because <80% scaled position)
- **Confidence**: 87.2% probability we're in Bull regime
- **Position Size**: Go 52% long (0.6 base * 0.87 confidence)
- **Stop Loss/Take Profit**: Volatility-adjusted levels

### Backtest Performance

```
Backtest Performance:
  Total Return: 23.45%
  Sharpe Ratio: 1.42
  Win Rate: 54.2%
  Total Trades: 423
```

---

## 🏗️ Project Architecture

### Directory Structure

```
hmm-regime-detection/
│
├── main.py                    # 🚀 Main entry point
├── requirements.txt           # 📦 Dependencies
├── README.md                  # 📖 Documentation
│
├── src/                       # 📁 Source code
│   ├── __init__.py
│   ├── hmm_model.py           # Core HMM (MarketRegimeHMM)
│   ├── data_loader.py         # Data fetching (Yahoo, synthetic)
│   ├── feature_engineering.py # Feature creation
│   ├── regime_classifier.py   # High-level classification
│   ├── signal_generator.py    # Trading signal generation
│   └── visualization.py       # Charts and dashboards
│
├── tests/                     # 🧪 Unit tests
│   └── test_hmm.py            # 32 comprehensive tests
│
├── output/                    # 📊 Generated reports
│   ├── regime_dashboard.png
│   ├── regime_timeline.png
│   ├── transition_matrix.png
│   ├── regime_statistics.png
│   ├── signal_performance.png
│   ├── regime_history.csv
│   ├── signal_history.csv
│   └── summary_report.json
│
└── models/                    # 💾 Saved models
    └── hmm_regime_model.pkl
```

### Module Relationships

```
main.py
    │
    ├── data_loader.py
    │       ├── MarketData (dataclass)
    │       └── DataLoader (class)
    │
    ├── feature_engineering.py
    │       ├── FeatureSet (dataclass)
    │       └── FeatureEngineer (class)
    │
    ├── hmm_model.py
    │       ├── RegimeInfo (dataclass)
    │       ├── HMMResult (dataclass)
    │       └── MarketRegimeHMM (class)  ← CORE MODEL
    │
    ├── regime_classifier.py (wraps hmm_model)
    │       ├── RegimeAnalysis (dataclass)
    │       └── RegimeClassifier (class)  ← MAIN ORCHESTRATOR
    │
    ├── signal_generator.py
    │       ├── TradingSignal (dataclass)
    │       ├── SignalSummary (dataclass)
    │       └── SignalGenerator (class)
    │
    └── visualization.py
            ├── plot_regime_timeline()
            ├── plot_regime_probabilities()
            ├── plot_transition_matrix()
            ├── plot_regime_statistics()
            ├── plot_signal_performance()
            └── plot_regime_dashboard()
```

---

## 📖 API Reference

### MarketRegimeHMM

```python
from src.hmm_model import MarketRegimeHMM

# Create model
model = MarketRegimeHMM(
    n_regimes=3,              # Number of hidden states
    covariance_type='full',   # 'full', 'diag', 'spherical'
    n_iter=100,               # EM iterations
    random_state=42           # Reproducibility
)

# Fit to data
features = returns.values.reshape(-1, 1)
model.fit(features, feature_names=['returns'])

# Predict regimes
regimes = model.predict(features)           # [0, 0, 1, 2, 2, ...]
probs = model.predict_proba(features)       # [[0.9, 0.08, 0.02], ...]

# Get transition matrix
trans_matrix = model.get_transition_matrix()

# Calculate model fit
aic, bic = model.calculate_aic_bic(features)

# Save/load
model.save('models/my_model.pkl')
loaded = MarketRegimeHMM.load('models/my_model.pkl')
```

### RegimeClassifier

```python
from src.regime_classifier import RegimeClassifier

classifier = RegimeClassifier(
    n_regimes=3,
    use_volatility=True,      # Include vol as feature
    smoothing_window=5        # Probability smoothing
)

# Classify regimes
analysis = classifier.classify(prices_df, returns_series)

# Access results
print(analysis.current_regime)       # 2
print(analysis.current_regime_name)  # "Bull Market"
print(analysis.transitions)          # [datetime, datetime, ...]
print(analysis.duration_stats)       # {0: 25.3, 1: 45.8, 2: 38.2}
```

### SignalGenerator

```python
from src.signal_generator import SignalGenerator

generator = SignalGenerator(
    confidence_threshold=0.6,  # Min confidence for signal
    strong_threshold=0.8,      # Threshold for STRONG signals
    base_stop_loss=0.02,       # 2% base stop
    base_take_profit=0.04      # 4% base take profit
)

# Generate signals
signals = generator.generate_signals(analysis)

# Current signal
current = signals.current_signal
print(current.signal_type)      # SignalType.BUY
print(current.position_size)    # 0.52
print(current.confidence)       # 0.87

# Performance
perf = generator.calculate_signal_performance(signals, returns)
print(perf['sharpe_ratio'])     # 1.42
```

### DataLoader

```python
from src.data_loader import DataLoader

loader = DataLoader(use_cache=True)

# Fetch real data
data = loader.fetch_yahoo('SPY', '2020-01-01', '2023-12-31')

# Generate synthetic (for testing)
data = loader.generate_synthetic(
    n_samples=1000,
    n_regimes=3,
    random_state=42
)

# Access data
print(data.prices)   # DataFrame with OHLCV
print(data.returns)  # Series of returns
print(data.symbol)   # 'SPY' or 'SYNTHETIC'
```

---

## 💻 Code Examples

### Example 1: Quick Regime Check

```python
from src.regime_classifier import RegimeClassifier
from src.data_loader import DataLoader

# Load data
loader = DataLoader()
data = loader.fetch_yahoo('SPY', '2020-01-01')

# Classify
classifier = RegimeClassifier(n_regimes=3)
analysis = classifier.classify(data.prices, data.returns)

# Print current state
print(f"Current Regime: {analysis.current_regime_name}")
print(f"Confidence: {analysis.regime_probs.iloc[-1].max():.1%}")
print(f"Recent Transitions: {len(analysis.transitions)}")
```

### Example 2: Generate Trading Signals

```python
from src.regime_classifier import RegimeClassifier
from src.signal_generator import SignalGenerator
from src.data_loader import DataLoader

loader = DataLoader()
data = loader.generate_synthetic(n_samples=500)

# Classify regimes
classifier = RegimeClassifier(n_regimes=3)
analysis = classifier.classify(data.prices, data.returns)

# Generate signals
generator = SignalGenerator(confidence_threshold=0.6)
signals = generator.generate_signals(analysis)

# Print current recommendation
current = signals.current_signal
print(f"Signal: {current.signal_type.value.upper()}")
print(f"Position: {current.position_size:+.2f}")
print(f"Stop Loss: {current.stop_loss_pct:.1%}")
print(f"Take Profit: {current.take_profit_pct:.1%}")
```

### Example 3: Model Selection with BIC

```python
from src.hmm_model import MarketRegimeHMM, select_optimal_regimes
from src.feature_engineering import create_hmm_features
from src.data_loader import DataLoader

loader = DataLoader()
data = loader.generate_synthetic(n_samples=1000)

# Create features
features, _, _ = create_hmm_features(data.prices, data.returns)

# Find optimal number of regimes
optimal_n, scores = select_optimal_regimes(
    features, 
    min_regimes=2, 
    max_regimes=5,
    criterion='bic'
)

print(f"Optimal regimes: {optimal_n}")
for n, score in scores.items():
    marker = " ← Best" if n == optimal_n else ""
    print(f"  {n} regimes: BIC = {score:,.0f}{marker}")
```

### Example 4: Visualize Results

```python
from src.regime_classifier import RegimeClassifier
from src.visualization import plot_regime_dashboard
from src.data_loader import DataLoader

loader = DataLoader()
data = loader.generate_synthetic(n_samples=500)

classifier = RegimeClassifier(n_regimes=3)
analysis = classifier.classify(data.prices, data.returns)

# Create dashboard
regime_names = {info.regime_id: info.name for info in analysis.regime_stats}

plot_regime_dashboard(
    regimes=analysis.regimes,
    regime_probs=analysis.regime_probs,
    prices=data.prices['Close'],
    transition_matrix=classifier.hmm.get_transition_matrix(),
    regime_names=regime_names,
    save_path='output/my_dashboard.png'
)
```

---

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Expected Output

```
tests/test_hmm.py::TestMarketRegimeHMM::test_initialization PASSED
tests/test_hmm.py::TestMarketRegimeHMM::test_fit PASSED
tests/test_hmm.py::TestMarketRegimeHMM::test_predict PASSED
tests/test_hmm.py::TestMarketRegimeHMM::test_predict_proba PASSED
tests/test_hmm.py::TestMarketRegimeHMM::test_transition_matrix PASSED
tests/test_hmm.py::TestMarketRegimeHMM::test_aic_bic PASSED
tests/test_hmm.py::TestMarketRegimeHMM::test_fit_and_analyze PASSED
tests/test_hmm.py::TestMarketRegimeHMM::test_save_load PASSED
...
============================== 32 passed in 2.39s ==============================
```

### Test Coverage by Module

| Test Class | Tests | Coverage |
|------------|-------|----------|
| TestMarketRegimeHMM | 8 | HMM fit/predict/save/load |
| TestModelSelection | 1 | Optimal regime selection |
| TestDataLoader | 3 | Data fetching |
| TestFeatureEngineer | 4 | Feature creation |
| TestRegimeClassifier | 6 | Classification pipeline |
| TestSignalGenerator | 5 | Signal generation |
| TestIntegration | 2 | End-to-end pipeline |

---

## 🔧 Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: hmmlearn` | Missing dependency | `pip install hmmlearn` |
| `ValueError: array must not contain NaNs` | Missing data | `dropna()` before fitting |
| `Model not converging` | Too few samples or too many regimes | Reduce `n_regimes` or add data |
| `yfinance returns empty` | Rate limit or invalid symbol | Use synthetic data as fallback |

### Model Not Converging

```python
# Increase iterations
model = MarketRegimeHMM(n_regimes=2, n_iter=200)  # Default is 100

# Or use fewer regimes
model = MarketRegimeHMM(n_regimes=2)  # Instead of 3 or 4
```

### HMM Numerical Issues

```python
# Use diagonal covariance for stability
model = MarketRegimeHMM(
    n_regimes=3,
    covariance_type='diag'  # Instead of 'full'
)
```

---

## 📚 References

### Academic Papers

| Paper | Topic |
|-------|-------|
| [Rabiner (1989)](https://www.cs.ubc.ca/~murphyk/Bayes/rabiner.pdf) | HMM Tutorial |
| [Hamilton (1989)](https://www.ssc.wisc.edu/~bhansen/718/Hamilton1989.pdf) | Markov Switching Models |
| [Guidolin & Timmermann (2007)](https://www.aeaweb.org/articles?id=10.1257/jep.21.2.163) | Regime Switching in Finance |

### Documentation

| Resource | Link |
|----------|------|
| hmmlearn | https://hmmlearn.readthedocs.io/ |
| scikit-learn | https://scikit-learn.org/ |
| yfinance | https://github.com/ranaroussi/yfinance |

---

## 🔗 Related Projects

| Project | Description | Link |
|---------|-------------|------|
| Portfolio VaR Calculator | Risk measurement | [Link](https://github.com/avniderashree/portfolio-var-calculator) |
| GARCH Volatility Forecaster | Vol prediction | [Link](https://github.com/avniderashree/garch-volatility-forecaster) |
| Credit Risk Model | PD/LGD estimation | [Link](https://github.com/avniderashree/credit-risk-model) |
| Monte Carlo Stress Testing | Scenario analysis | [Link](https://github.com/avniderashree/monte-carlo-stress-testing) |
| Liquidity Risk Predictor | ML liquidity model | [Link](https://github.com/avniderashree/liquidity-risk-predictor) |
| Derivatives MTM Dashboard | Options pricing | [Link](https://github.com/avniderashree/derivatives-mtm-dashboard) |
| Margin Call Automation | Margin pipeline | [Link](https://github.com/avniderashree/margin-call-automation) |

---

## 👤 Author

**Avni Derashree**

- GitHub: [@avniderashree](https://github.com/avniderashree)
- Email: avniderashree@gmail.com

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.
