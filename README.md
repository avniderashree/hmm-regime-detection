# 📈 HMM Regime Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-32%20passed-brightgreen.svg)](tests/)

A **Hidden Markov Model (HMM) system for detecting market regimes** and generating trading signals. Automatically identifies Bull, Bear, and Neutral market conditions using unsupervised learning.

---

## 🎯 What Is This Project?

This project uses **Hidden Markov Models** to detect underlying market regimes (states) from price data. Markets don't behave consistently—sometimes they're trending up (Bull), sometimes crashing (Bear), and sometimes moving sideways (Neutral). This system:

1. **Detects Regimes** - Identifies which regime the market is currently in
2. **Calculates Probabilities** - Shows confidence level for each regime
3. **Generates Signals** - Creates trading signals based on regime changes
4. **Backtests Performance** - Measures strategy performance

### The Regime Model

```
+-------------+     +-------------+     +-------------+
|    BULL     |<--->|   NEUTRAL   |<--->|    BEAR     |
| High Return |     | Low Vol     |     | High Vol    |
| Low Vol     |     | Sideways    |     | Negative    |
+------+------+     +------+------+     +------+------+
       |                   |                   |
       +-------------------+-------------------+
                           |
                     Transition Matrix
                     (probability of switching)
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Gaussian HMM** | Detects regimes using returns and volatility |
| **Auto Regime Labeling** | Automatically names regimes (Bull/Bear/Neutral) |
| **Model Selection** | Uses BIC to select optimal number of regimes |
| **Signal Generation** | Creates position sizes based on regime confidence |
| **Probability Smoothing** | Reduces whipsaw by smoothing transitions |
| **6 Visualizations** | Dashboards, timelines, transition matrices |

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

## 📁 Project Structure

```
hmm-regime-detection/
├── main.py                    # Main entry point
├── requirements.txt           # Dependencies
├── src/
│   ├── __init__.py
│   ├── hmm_model.py           # Core HMM implementation
│   ├── data_loader.py         # Data fetching (Yahoo, synthetic)
│   ├── feature_engineering.py # Feature creation
│   ├── regime_classifier.py   # High-level classification
│   ├── signal_generator.py    # Trading signal generation
│   └── visualization.py       # Charts and dashboards
├── tests/
│   └── test_hmm.py            # 32 comprehensive tests
├── output/                    # Generated charts/reports
└── models/                    # Saved models
```

---

## 📊 Understanding the Output

When you run `python main.py`:

### Regime Statistics
```
Regime Statistics:
Regime          Mean Ret     Volatility   Duration     Frequency
───────────────────────────────────────────────────────────────
Bear Market      -0.08%        2.35%        25.3         18.2%
Neutral           0.02%        1.12%        45.8         41.5%
Bull Market       0.12%        0.89%        38.2         40.3%
```

### Current Signal
```
Current Signal:
  Type: BUY
  Regime: Bull Market
  Confidence: 87.2%
  Position Size: +0.52
  Stop Loss: 2.1%
  Take Profit: 4.2%
```

---

## 📖 API Reference

### MarketRegimeHMM

```python
from src.hmm_model import MarketRegimeHMM

# Create and fit model
model = MarketRegimeHMM(n_regimes=3)
model.fit(features)

# Predict regimes
regimes = model.predict(features)
probabilities = model.predict_proba(features)

# Get transition matrix
trans_matrix = model.get_transition_matrix()
```

### RegimeClassifier

```python
from src.regime_classifier import RegimeClassifier

classifier = RegimeClassifier(n_regimes=3)
analysis = classifier.classify(prices, returns)

print(f"Current Regime: {analysis.current_regime_name}")
print(f"Transitions: {len(analysis.transitions)}")
```

### SignalGenerator

```python
from src.signal_generator import SignalGenerator

generator = SignalGenerator(confidence_threshold=0.6)
signals = generator.generate_signals(analysis)

current = signals.current_signal
print(f"Signal: {current.signal_type.value}")
print(f"Position: {current.position_size}")
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

**32 tests** covering:
- HMM model training and prediction
- Feature engineering
- Regime classification
- Signal generation
- Model persistence
- Full pipeline integration

---

## 📈 Output Files

| File | Description |
|------|-------------|
| `regime_dashboard.png` | Comprehensive dashboard |
| `regime_timeline.png` | Price with regime overlay |
| `transition_matrix.png` | Regime transition heatmap |
| `regime_statistics.png` | Return/vol/frequency comparison |
| `signal_performance.png` | Cumulative strategy returns |
| `regime_history.csv` | Full regime history |
| `signal_history.csv` | Trading signals |
| `summary_report.json` | JSON summary |

---

## 🔬 How HMM Works

1. **Hidden States**: Regimes (Bull/Bear/Neutral) are unobserved
2. **Observations**: Returns and volatility are observed
3. **Transition Matrix**: Probability of switching between regimes
4. **Emission Model**: Each regime has characteristic return/volatility
5. **EM Algorithm**: Learns parameters from historical data
6. **Viterbi**: Finds most likely sequence of regimes

---

## 👤 Author

**Avni Derashree**

- GitHub: [@avniderashree](https://github.com/avniderashree)

---

## 📄 License

MIT License
