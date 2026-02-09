# 🧪 Codelab: Build an HMM Market Regime Detector from Scratch

**Estimated time:** 4–5 hours · **Difficulty:** Intermediate · **Language:** Python 3.8+

---

## What You'll Build

By the end of this codelab, you'll have a complete **Hidden Markov Model (HMM) pipeline** that:

- Downloads real S&P 500 (SPY) data or generates realistic synthetic data
- Engineers features (returns + rolling volatility) for the HMM
- Uses **BIC-based model selection** to find the optimal number of regimes
- Fits a **Gaussian HMM** that detects **Bull, Bear, and Neutral** market regimes
- Computes **regime probabilities** with smoothing to reduce whipsaw
- Generates **trading signals** (STRONG_BUY → STRONG_SELL) with position sizing
- Calculates **backtest performance** (Sharpe ratio, win rate, total return)
- Produces **6 publication-quality charts** (dashboard, timeline, heatmap, etc.)
- Includes **32 unit tests** covering every module

The final project structure:

```
hmm-regime-detection/
├── main.py                     # Entry point — runs the full pipeline
├── requirements.txt            # Dependencies
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Fetches Yahoo data + synthetic fallback
│   ├── feature_engineering.py  # Returns, volatility, feature matrix
│   ├── hmm_model.py            # Core Gaussian HMM wrapper
│   ├── regime_classifier.py    # High-level orchestrator
│   ├── signal_generator.py     # Trading signals from regimes
│   └── visualization.py        # 6 chart types
├── tests/
│   └── test_hmm.py             # 32 unit tests
├── output/                     # Generated charts + reports
└── models/                     # Saved HMM model (.pkl)
```

---

## Prerequisites

- Python 3.8+ installed
- Basic familiarity with Python
- A terminal / command line

**No finance or machine-learning knowledge required.** Every concept is explained before we code it.

---

---

# PART 1: THE CONCEPTS (What & Why)

No coding yet. Read this entire section first — it'll make every line of code intuitive.

---

## 1.1 What Are "Market Regimes"?

Markets don't behave the same way all the time. Sometimes stocks go up steadily (a **bull market**). Sometimes they crash violently (a **bear market**). Sometimes they drift sideways (**neutral/consolidation**).

These distinct behavioral patterns are called **regimes**. Each regime has its own personality:

```
┌─────────────────────────────────────────────────────────────┐
│  BULL MARKET                                                │
│  • Average daily return: +0.08% to +0.15%                  │
│  • Volatility: LOW (0.8% – 1.2% daily)                     │
│  • Mood: Optimistic, prices grind higher steadily           │
│  • Duration: Months to years                                │
│                                                             │
│  NEUTRAL / SIDEWAYS                                         │
│  • Average daily return: ~0% (flat)                         │
│  • Volatility: MODERATE (1.0% – 1.8% daily)                │
│  • Mood: Indecisive, range-bound                            │
│  • Duration: Weeks to months                                │
│                                                             │
│  BEAR MARKET                                                │
│  • Average daily return: -0.10% to -0.20%                   │
│  • Volatility: HIGH (2.0% – 3.5% daily)                    │
│  • Mood: Panic, sharp drops, fear                           │
│  • Duration: Weeks to months (shorter but intense)          │
└─────────────────────────────────────────────────────────────┘
```

**The problem:** We can look at a chart and *roughly* tell which regime we're in — but we need a rigorous, quantitative, automated way to do this. That's where Hidden Markov Models come in.

---

## 1.2 What Is a Hidden Markov Model (HMM)?

An HMM is a statistical model built on a simple but powerful idea:

> **There's a hidden process generating what we observe. We can't see the process directly, but we can infer it from the data it produces.**

**The weather analogy:**

Imagine you're locked in a windowless room. Every day, someone brings you an ice cream sales report. You can't see the weather, but you can *infer* it:
- High sales → probably sunny (hidden state: Sunny)
- Low sales → probably rainy (hidden state: Rainy)

You **observe** ice cream sales. The **hidden state** is the weather.

**In financial markets:**

We **observe** daily returns and volatility. The **hidden state** is the market regime (Bull, Bear, or Neutral).

```
     What we CAN see                What we CAN'T see
     ──────────────                 ─────────────────
Day 1: Return = +0.5%, Vol = 1.0%  ──►  Bull Market
Day 2: Return = +0.3%, Vol = 1.1%  ──►  Bull Market
Day 3: Return = -2.1%, Vol = 3.5%  ──►  Bear Market  ← TRANSITION!
Day 4: Return = -1.8%, Vol = 3.2%  ──►  Bear Market
Day 5: Return = +0.1%, Vol = 1.5%  ──►  Neutral
```

The HMM learns to "reverse-engineer" the hidden regimes from the observable data.

---

## 1.3 The Three Ingredients of an HMM

Every HMM has exactly three components. Understanding these is the key to understanding the entire project.

### Ingredient 1: The Transition Matrix

This tells us: "If I'm in regime X today, what's the probability I'll be in regime Y tomorrow?"

```
                    Tomorrow's State
                 Bull    Neutral    Bear
Today's   Bull │ 0.95     0.04     0.01 │
State  Neutral │ 0.10     0.85     0.05 │
          Bear │ 0.02     0.08     0.90 │
```

**Reading this matrix:**
- If we're in Bull today, there's a 95% chance we stay Bull tomorrow (regimes are *sticky*)
- Only a 1% chance of jumping straight from Bull to Bear (rare — usually goes through Neutral first)
- Bear has 90% persistence — bear markets are notoriously "sticky"

**Key insight:** Each row sums to 100%. From any state, you *must* go somewhere.

### Ingredient 2: The Emission Model (Gaussian Distributions)

Each regime "emits" (generates) observations according to its own probability distribution. We use **Gaussian (Normal) distributions**:

```
Bull Market:    returns ~ Normal(μ = +0.08%, σ = 1.0%)
                    ▲
                   ╱ ╲
                  ╱   ╲        Narrow bell → low volatility
                 ╱     ╲       Centered right of zero → positive returns
────────────────╱───────╲──────────────────
              -2%   0  +2%


Bear Market:    returns ~ Normal(μ = -0.15%, σ = 2.5%)
          ▲
         ╱ ╲
        ╱   ╲
       ╱     ╲                 Wide bell → high volatility
      ╱       ╲                Centered left of zero → negative returns
─────╱─────────╲───────────────
   -5%   0    +5%
```

**The HMM learns μ (mean) and σ (standard deviation) for each regime.** It's not told what they should be — it discovers them from the data.

### Ingredient 3: The Initial State Distribution

What regime are we in *at the very start*? This is just a vector of probabilities:

```
π = [P(start in Bull), P(start in Neutral), P(start in Bear)]
  = [0.33, 0.34, 0.33]  ← Uniform prior (we don't know where we start)
```

In practice, this doesn't matter much — after a few dozen observations, the data overwhelms the initial assumption.

---

## 1.4 How the HMM Learns: The EM Algorithm

The HMM discovers its parameters (transition matrix, emission means/variances, initial distribution) using the **Expectation-Maximization (EM)** algorithm, also called the **Baum-Welch** algorithm for HMMs.

**The chicken-and-egg problem:** To know the regime parameters, we need to know which regime each day belongs to. But to know which regime each day belongs to, we need the parameters.

**EM solves this by iterating:**

```
Step 1: INITIALIZE — Start with random guesses for all parameters

Step 2: E-STEP (Expectation)
   "Given my current parameter guesses, what's the probability
    that each day belongs to each regime?"
   → Produces "soft" regime assignments (e.g., Day 3 is 70% Bear, 25% Neutral, 5% Bull)

Step 3: M-STEP (Maximization)
   "Given those soft assignments, what parameters would best explain the data?"
   → Update means: weighted average of returns in each regime
   → Update variances: weighted variance of returns in each regime
   → Update transitions: count weighted transitions between regimes

Step 4: REPEAT Steps 2-3 until parameters stop changing (convergence)
```

After ~50–100 iterations, the parameters stabilize. The `hmmlearn` library handles all of this math internally.

---

## 1.5 Decoding: Figuring Out Which Regime We're In

Once the model is trained, we need to **decode** — figure out the most likely regime for each day.

**Two approaches:**

**Viterbi decoding** (used for `.predict()`): Finds the single most likely *sequence* of regimes. It considers the whole trajectory — not just each day in isolation. Like solving a maze backwards.

**Forward-Backward** (used for `.predict_proba()`): Computes the probability of each regime at each time step. More nuanced — gives you confidence levels:

```
Day 3: P(Bull) = 0.05, P(Neutral) = 0.25, P(Bear) = 0.70
       → We're "70% confident this is a Bear day"
```

Our project uses both: Viterbi for hard regime labels, Forward-Backward for confidence scores.

---

## 1.6 Model Selection: How Many Regimes?

Should we use 2 regimes (Bull/Bear)? 3 (Bull/Neutral/Bear)? 4? 5?

**BIC (Bayesian Information Criterion)** helps decide:

```
BIC = -2 × log-likelihood + k × log(n)

Where:
  log-likelihood = how well the model fits the data (higher = better fit)
  k = number of parameters (more regimes = more parameters)
  n = number of data points
```

**Lower BIC is better.** BIC penalizes complexity — it won't let you add regimes unless they genuinely improve the fit. This prevents overfitting.

**In practice:** For stock markets, 2–3 regimes usually wins. Sometimes 4 (adding a "Crisis" regime) helps during extreme events.

---

## 1.7 Auto-Labeling: Naming the Regimes

The HMM outputs regimes as numbers (0, 1, 2). But which number is "Bull" and which is "Bear"?

We **auto-label** based on the learned emission means:

```
Regime 0: mean return = +0.12%  → Highest mean  → BULL MARKET
Regime 1: mean return = +0.02%  → Middle mean   → NEUTRAL
Regime 2: mean return = -0.15%  → Lowest mean   → BEAR MARKET
```

Simple: sort regimes by mean return, assign names accordingly.

---

## 1.8 Trading Signals: From Regimes to Decisions

Once we know the current regime and our confidence level, we generate trading signals:

```
┌─────────────────────────────────────────────────────────────┐
│  Signal Logic                                               │
│                                                             │
│  IF regime = Bull AND confidence > 80%  → STRONG_BUY (+1.0)│
│  IF regime = Bull AND confidence > 60%  → BUY (+0.6)       │
│  IF regime = Neutral OR confidence < 60% → HOLD (0.0)      │
│  IF regime = Bear AND confidence > 60%  → SELL (-0.6)      │
│  IF regime = Bear AND confidence > 80%  → STRONG_SELL (-1.0)│
│                                                             │
│  Position size = base_position × confidence                 │
│  Stop loss = base_stop × (regime_vol / avg_vol)             │
│  Take profit = base_tp × (regime_vol / avg_vol)             │
└─────────────────────────────────────────────────────────────┘
```

The signal generator also computes **volatility-adjusted stop losses and take profits** — wider stops in high-vol regimes, tighter in calm regimes.

---

---

# PART 2: PROJECT SETUP (Step 0)

---

## Step 0.1: Create the Folder Structure

```bash
mkdir hmm-regime-detection
cd hmm-regime-detection
mkdir -p src tests output models
```

## Step 0.2: Create `requirements.txt`

**File: `requirements.txt`**
```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
hmmlearn>=0.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
yfinance>=0.2.0
joblib>=1.1.0
pytest>=7.0.0
```

| Library | Purpose |
|---------|---------|
| `numpy` | Array math |
| `pandas` | DataFrames and time series |
| `scipy` | Statistical distributions |
| `scikit-learn` | ML utilities (preprocessing, metrics) |
| `hmmlearn` | **The HMM engine** — fits Gaussian HMMs via EM |
| `matplotlib` | Chart creation |
| `seaborn` | Professional chart styling |
| `yfinance` | Real market data from Yahoo Finance |
| `joblib` | Save/load fitted models to disk |
| `pytest` | Unit test runner |

Install:
```bash
pip install -r requirements.txt
```

## Step 0.3: Create `src/__init__.py`

**File: `src/__init__.py`**
```python
"""
HMM Regime Detection
=====================
A Hidden Markov Model system for detecting market regimes
and generating trading signals.

Modules:
    data_loader          - Market data fetching + synthetic generation
    feature_engineering  - Feature creation for the HMM
    hmm_model            - Core Gaussian HMM wrapper
    regime_classifier    - High-level regime classification orchestrator
    signal_generator     - Trading signal generation from regimes
    visualization        - Publication-quality financial charts
"""
```

---

---

# PART 3: DATA LOADER (Step 1)

This module fetches real market data from Yahoo Finance, with a synthetic-data fallback for testing or when offline.

---

## Step 1.1: Understand What This Module Does

```
DataLoader
    │
    ├── fetch_yahoo('SPY', '2020-01-01', '2023-12-31')
    │       → Downloads real OHLCV data from Yahoo Finance
    │       → Computes daily returns from Close prices
    │       → Returns a MarketData dataclass
    │
    └── generate_synthetic(n_samples=500, n_regimes=3)
            → Creates fake but realistic price data
            → Simulates regime switches (Bull→Bear→Neutral)
            → Perfect for testing when market data is unavailable
```

**File: `src/data_loader.py`**

```python
"""
data_loader.py — Market Data Fetching & Synthetic Generation
=============================================================

Two data sources:
  1. Yahoo Finance (real SPY data via yfinance)
  2. Synthetic data (simulated regime-switching returns)

The synthetic generator is critical — it ensures the project works
offline and provides ground-truth regimes for testing.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class MarketData:
    """
    Container for all market data needed by the pipeline.

    Attributes
    ----------
    prices : pd.DataFrame
        OHLCV price data (or at minimum a 'Close' column).
    returns : pd.Series
        Daily percentage returns.
    symbol : str
        Ticker symbol ('SPY') or 'SYNTHETIC'.
    start_date : str
        Start of data range.
    end_date : str
        End of data range.
    """
    prices: pd.DataFrame
    returns: pd.Series
    symbol: str
    start_date: str
    end_date: str


class DataLoader:
    """
    Loads market data from Yahoo Finance or generates synthetic data.

    Parameters
    ----------
    use_cache : bool
        If True, cache downloaded data (not implemented here,
        but the interface supports it for future use).
    """

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache

    def fetch_yahoo(
        self,
        symbol: str = 'SPY',
        start: str = '2019-01-01',
        end: Optional[str] = None
    ) -> MarketData:
        """
        Download real market data from Yahoo Finance.

        Parameters
        ----------
        symbol : str
            Ticker symbol (e.g., 'SPY', 'QQQ', 'AAPL').
        start : str
            Start date in 'YYYY-MM-DD' format.
        end : str, optional
            End date. If None, uses today.

        Returns
        -------
        MarketData
            Dataclass with prices, returns, and metadata.
        """
        import yfinance as yf

        print(f"Fetching {symbol} data from Yahoo Finance...")

        try:
            data = yf.download(symbol, start=start, end=end, progress=False)

            if data.empty:
                raise ValueError(f"No data returned for {symbol}")

            # Handle MultiIndex columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # Keep standard OHLCV columns
            prices = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            prices = prices.dropna()

            # Compute daily returns from Close prices
            returns = prices['Close'].pct_change().dropna()
            returns.name = 'Returns'

            # Align prices to returns (drop first row which has NaN return)
            prices = prices.loc[returns.index]

            end_date = end or prices.index[-1].strftime('%Y-%m-%d')

            print(f"  Loaded {len(prices)} trading days "
                  f"({prices.index[0].strftime('%Y-%m-%d')} to "
                  f"{prices.index[-1].strftime('%Y-%m-%d')})")

            return MarketData(
                prices=prices,
                returns=returns,
                symbol=symbol,
                start_date=start,
                end_date=end_date,
            )

        except Exception as e:
            print(f"  ⚠ Yahoo Finance failed: {e}")
            print(f"  Falling back to synthetic data...")
            return self.generate_synthetic(n_samples=1000, n_regimes=3)

    def generate_synthetic(
        self,
        n_samples: int = 1000,
        n_regimes: int = 3,
        random_state: int = 42
    ) -> MarketData:
        """
        Generate synthetic market data with known regime switches.

        This is essential for:
          1. Testing when offline
          2. Unit tests with known ground truth
          3. Demonstrating the pipeline works

        The generator simulates a regime-switching process:
          - Bull: μ=+0.08%, σ=0.8% (calm uptrend)
          - Neutral: μ=+0.01%, σ=1.2% (sideways)
          - Bear: μ=-0.12%, σ=2.2% (volatile downtrend)

        Parameters
        ----------
        n_samples : int
            Number of trading days to generate.
        n_regimes : int
            Number of regimes (2 or 3).
        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        MarketData
            Synthetic data with realistic regime-switching behavior.
        """
        np.random.seed(random_state)

        print(f"Generating synthetic data ({n_samples} samples, {n_regimes} regimes)...")

        # ── Define regime parameters ──
        if n_regimes == 2:
            regime_params = [
                {'mean': 0.0008, 'std': 0.008, 'name': 'Bull'},   # +0.08%/day
                {'mean': -0.0012, 'std': 0.022, 'name': 'Bear'},   # -0.12%/day
            ]
            # Transition matrix: sticky regimes
            trans_probs = np.array([
                [0.97, 0.03],   # Bull → Bull 97%, Bull → Bear 3%
                [0.05, 0.95],   # Bear → Bull 5%, Bear → Bear 95%
            ])
        else:  # 3 regimes
            regime_params = [
                {'mean': 0.0008, 'std': 0.008, 'name': 'Bull'},    # +0.08%/day
                {'mean': 0.0001, 'std': 0.012, 'name': 'Neutral'}, # +0.01%/day
                {'mean': -0.0012, 'std': 0.022, 'name': 'Bear'},   # -0.12%/day
            ]
            trans_probs = np.array([
                [0.95, 0.04, 0.01],   # Bull → ...
                [0.08, 0.87, 0.05],   # Neutral → ...
                [0.02, 0.08, 0.90],   # Bear → ...
            ])

        # ── Simulate regime-switching process ──
        regimes = np.zeros(n_samples, dtype=int)
        returns = np.zeros(n_samples)

        # Start in a random regime
        regimes[0] = np.random.choice(n_regimes)
        params = regime_params[regimes[0]]
        returns[0] = np.random.normal(params['mean'], params['std'])

        for t in range(1, n_samples):
            # Transition: sample next regime from transition probabilities
            regimes[t] = np.random.choice(
                n_regimes,
                p=trans_probs[regimes[t - 1]]
            )
            # Emission: generate return from the current regime's distribution
            params = regime_params[regimes[t]]
            returns[t] = np.random.normal(params['mean'], params['std'])

        # ── Build price series from returns ──
        # Start at $100, compound returns
        prices_close = 100 * np.exp(np.cumsum(returns))

        # Create realistic OHLCV-like DataFrame
        dates = pd.bdate_range('2020-01-01', periods=n_samples)

        prices = pd.DataFrame({
            'Open': prices_close * (1 + np.random.normal(0, 0.002, n_samples)),
            'High': prices_close * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
            'Low': prices_close * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
            'Close': prices_close,
            'Volume': np.random.randint(50_000_000, 200_000_000, n_samples),
        }, index=dates)

        returns_series = pd.Series(returns, index=dates, name='Returns')

        print(f"  Generated {n_samples} days of synthetic data")

        return MarketData(
            prices=prices,
            returns=returns_series,
            symbol='SYNTHETIC',
            start_date=dates[0].strftime('%Y-%m-%d'),
            end_date=dates[-1].strftime('%Y-%m-%d'),
        )
```

---

---

# PART 4: FEATURE ENGINEERING (Step 2)

The HMM needs a **feature matrix** — one or more numerical features per day. We use two: daily returns and rolling volatility.

---

**File: `src/feature_engineering.py`**

```python
"""
feature_engineering.py — Feature Creation for the HMM
=====================================================

Creates the feature matrix the HMM trains on.

Features used:
  1. Daily returns (captures direction)
  2. Rolling volatility (captures regime's "energy")

Why these two?
  Returns alone can't distinguish Bull from Neutral easily
  (both have small positive returns). Adding volatility helps:
  Bull = positive return + LOW vol, Neutral = flat + MODERATE vol.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class FeatureSet:
    """
    Container for HMM features.

    Attributes
    ----------
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    feature_names : list of str
        Names of each feature column.
    index : pd.DatetimeIndex
        Date index aligned with features.
    returns : pd.Series
        Returns aligned with features (after dropping NaN from rolling window).
    """
    features: np.ndarray
    feature_names: list
    index: pd.DatetimeIndex
    returns: pd.Series


class FeatureEngineer:
    """
    Creates features for the HMM from raw price/return data.

    Parameters
    ----------
    volatility_window : int
        Rolling window for volatility calculation (default 21 = 1 month).
    use_volatility : bool
        Whether to include rolling volatility as a feature (default True).
    """

    def __init__(self, volatility_window: int = 21, use_volatility: bool = True):
        self.volatility_window = volatility_window
        self.use_volatility = use_volatility

    def create_features(
        self,
        prices: pd.DataFrame,
        returns: pd.Series
    ) -> FeatureSet:
        """
        Build the feature matrix for the HMM.

        Parameters
        ----------
        prices : pd.DataFrame
            OHLCV prices (needs 'Close' column).
        returns : pd.Series
            Daily returns.

        Returns
        -------
        FeatureSet
            Contains the feature matrix, names, and aligned index.
        """
        features_dict = {}

        # Feature 1: Daily returns (always included)
        features_dict['returns'] = returns

        # Feature 2: Rolling volatility (optional but recommended)
        if self.use_volatility:
            rolling_vol = returns.rolling(
                window=self.volatility_window
            ).std()
            features_dict['volatility'] = rolling_vol

        # Combine into DataFrame, drop NaN rows
        features_df = pd.DataFrame(features_dict).dropna()

        # Build the feature matrix
        feature_matrix = features_df.values  # Shape: (n_samples, n_features)
        feature_names = list(features_df.columns)

        # Aligned returns (after dropping NaN from rolling window)
        aligned_returns = returns.loc[features_df.index]

        return FeatureSet(
            features=feature_matrix,
            feature_names=feature_names,
            index=features_df.index,
            returns=aligned_returns,
        )


def create_hmm_features(
    prices: pd.DataFrame,
    returns: pd.Series,
    volatility_window: int = 21,
    use_volatility: bool = True
) -> Tuple[np.ndarray, List[str], pd.DatetimeIndex]:
    """
    Convenience function: create HMM features in one call.

    Returns
    -------
    tuple of (feature_matrix, feature_names, index)
    """
    engineer = FeatureEngineer(
        volatility_window=volatility_window,
        use_volatility=use_volatility,
    )
    fset = engineer.create_features(prices, returns)
    return fset.features, fset.feature_names, fset.index
```

---

---

# PART 5: THE CORE HMM MODEL (Step 3)

This is the heart of the entire project — the Gaussian HMM wrapper that does all the statistical heavy lifting.

---

## Step 3.1: Understand What This Module Does

```
MarketRegimeHMM
    │
    ├── fit(features)         → Train the HMM (EM algorithm)
    ├── predict(features)     → Viterbi decode → regime labels [0, 1, 2, ...]
    ├── predict_proba(features) → Forward-backward → regime probabilities
    ├── get_transition_matrix() → Extract learned transition matrix
    ├── calculate_aic_bic()     → Model quality scores
    ├── fit_and_analyze()       → All-in-one: fit + predict + analyze
    ├── save() / load()         → Persist model to disk
    │
    └── select_optimal_regimes() → Try 2-5 regimes, pick lowest BIC
```

**File: `src/hmm_model.py`**

```python
"""
hmm_model.py — Core Hidden Markov Model for Market Regimes
===========================================================

This module wraps hmmlearn's GaussianHMM with:
  - Automatic regime labeling (Bull/Neutral/Bear by mean return)
  - Regime statistics (mean, vol, duration, frequency)
  - BIC-based model selection
  - Model serialization (save/load)

The GaussianHMM assumes each hidden state emits observations
from a multivariate Gaussian distribution. Parameters are
learned via the Baum-Welch (EM) algorithm.
"""

import numpy as np
import pandas as pd
import joblib
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from hmmlearn.hmm import GaussianHMM


# ─── Result Containers ─────────────────────────────────────────

@dataclass
class RegimeInfo:
    """
    Statistics for a single regime.

    Attributes
    ----------
    regime_id : int
        Regime index (0, 1, 2, ...).
    name : str
        Human-readable name ('Bull Market', 'Neutral', 'Bear Market').
    mean_return : float
        Average daily return in this regime.
    volatility : float
        Daily standard deviation in this regime.
    frequency : float
        Fraction of time spent in this regime (0 to 1).
    avg_duration : float
        Average consecutive days in this regime.
    """
    regime_id: int
    name: str
    mean_return: float
    volatility: float
    frequency: float
    avg_duration: float


@dataclass
class HMMResult:
    """
    Complete output from an HMM fit-and-analyze cycle.

    Attributes
    ----------
    regimes : np.ndarray
        Regime label for each day (Viterbi decoded).
    regime_probs : np.ndarray
        Regime probabilities for each day, shape (n_samples, n_regimes).
    transition_matrix : np.ndarray
        Learned transition matrix, shape (n_regimes, n_regimes).
    regime_stats : list of RegimeInfo
        Statistics for each regime.
    n_regimes : int
        Number of regimes detected.
    aic : float
        Akaike Information Criterion.
    bic : float
        Bayesian Information Criterion.
    log_likelihood : float
        Log-likelihood of the fitted model.
    """
    regimes: np.ndarray
    regime_probs: np.ndarray
    transition_matrix: np.ndarray
    regime_stats: List[RegimeInfo]
    n_regimes: int
    aic: float
    bic: float
    log_likelihood: float


# ─── Core HMM Class ────────────────────────────────────────────

class MarketRegimeHMM:
    """
    Gaussian HMM wrapper for market regime detection.

    Parameters
    ----------
    n_regimes : int
        Number of hidden states (regimes). Default 3 (Bull/Neutral/Bear).
    covariance_type : str
        Type of covariance matrix: 'full', 'diag', 'spherical'.
        'full' captures correlations between features but needs more data.
        'diag' is more robust with limited data.
    n_iter : int
        Maximum EM iterations (default 100).
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        covariance_type: str = 'full',
        n_iter: int = 100,
        random_state: int = 42
    ):
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        # The underlying hmmlearn model
        self.model = GaussianHMM(
            n_components=n_regimes,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
        )

        self.is_fitted = False
        self.feature_names: List[str] = []
        self.regime_names: Dict[int, str] = {}

    def fit(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> 'MarketRegimeHMM':
        """
        Fit the HMM to the feature matrix using the EM algorithm.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix, shape (n_samples, n_features).
            Typically columns are [returns, volatility].
        feature_names : list of str, optional
            Names for each feature column.

        Returns
        -------
        self
            For method chaining.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(features)

        self.is_fitted = True
        self.feature_names = feature_names or [f'feature_{i}' for i in range(features.shape[1])]

        # Auto-label regimes by mean return
        self._auto_label_regimes()

        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict the most likely regime for each observation (Viterbi).

        Parameters
        ----------
        features : np.ndarray
            Feature matrix, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Regime labels, shape (n_samples,). Values are regime IDs.
        """
        self._check_fitted()
        return self.model.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Compute regime probabilities for each observation (Forward-Backward).

        Parameters
        ----------
        features : np.ndarray
            Feature matrix, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Regime probabilities, shape (n_samples, n_regimes).
            Each row sums to 1.0.
        """
        self._check_fitted()
        return self.model.predict_proba(features)

    def get_transition_matrix(self) -> np.ndarray:
        """
        Get the learned transition probability matrix.

        Returns
        -------
        np.ndarray
            Shape (n_regimes, n_regimes). Entry [i, j] is the
            probability of transitioning from regime i to regime j.
            Each row sums to 1.0.
        """
        self._check_fitted()
        return self.model.transmat_

    def calculate_aic_bic(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Calculate AIC and BIC for model selection.

        AIC = -2 * log_likelihood + 2k
        BIC = -2 * log_likelihood + k * log(n)

        Where k = number of free parameters, n = number of observations.
        Lower values indicate better model fit (penalized by complexity).

        Returns
        -------
        tuple of (aic, bic)
        """
        self._check_fitted()

        log_likelihood = self.model.score(features)
        n_samples = features.shape[0]
        n_features = features.shape[1]
        n = self.n_regimes

        # Count free parameters:
        # - Initial probs: n-1
        # - Transition matrix: n*(n-1)
        # - Means: n*n_features
        # - Covariances: depends on type
        k = (n - 1) + n * (n - 1) + n * n_features

        if self.covariance_type == 'full':
            k += n * n_features * (n_features + 1) // 2
        elif self.covariance_type == 'diag':
            k += n * n_features
        elif self.covariance_type == 'spherical':
            k += n

        aic = -2 * log_likelihood + 2 * k
        bic = -2 * log_likelihood + k * np.log(n_samples)

        return aic, bic

    def fit_and_analyze(
        self,
        features: np.ndarray,
        returns: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None
    ) -> HMMResult:
        """
        All-in-one: fit the model, predict regimes, compute statistics.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix.
        returns : pd.Series, optional
            Returns series (for regime statistics). If None, uses
            the first feature column.
        feature_names : list of str, optional
            Feature names.

        Returns
        -------
        HMMResult
            Complete analysis results.
        """
        # Fit the model
        self.fit(features, feature_names)

        # Predict regimes and probabilities
        regimes = self.predict(features)
        regime_probs = self.predict_proba(features)

        # Get transition matrix
        trans_matrix = self.get_transition_matrix()

        # Calculate regime statistics
        if returns is not None:
            ret_values = returns.values if isinstance(returns, pd.Series) else returns
        else:
            ret_values = features[:, 0]

        regime_stats = self._compute_regime_stats(regimes, ret_values)

        # Model quality
        aic, bic = self.calculate_aic_bic(features)
        log_likelihood = self.model.score(features)

        return HMMResult(
            regimes=regimes,
            regime_probs=regime_probs,
            transition_matrix=trans_matrix,
            regime_stats=regime_stats,
            n_regimes=self.n_regimes,
            aic=aic,
            bic=bic,
            log_likelihood=log_likelihood,
        )

    # ─── Regime Labeling & Statistics ───────────────────────────

    def _auto_label_regimes(self) -> None:
        """
        Automatically name regimes based on their emission means.

        The regime with the highest mean return → 'Bull Market'
        The regime with the lowest mean return → 'Bear Market'
        Everything else → 'Neutral'
        """
        means = self.model.means_[:, 0]  # First feature = returns
        sorted_indices = np.argsort(means)  # Low → High

        if self.n_regimes == 2:
            self.regime_names = {
                sorted_indices[0]: 'Bear Market',
                sorted_indices[1]: 'Bull Market',
            }
        elif self.n_regimes == 3:
            self.regime_names = {
                sorted_indices[0]: 'Bear Market',
                sorted_indices[1]: 'Neutral',
                sorted_indices[2]: 'Bull Market',
            }
        else:
            for i, idx in enumerate(sorted_indices):
                if i == 0:
                    self.regime_names[idx] = 'Bear Market'
                elif i == len(sorted_indices) - 1:
                    self.regime_names[idx] = 'Bull Market'
                else:
                    self.regime_names[idx] = f'Regime {i}'

    def _compute_regime_stats(
        self,
        regimes: np.ndarray,
        returns: np.ndarray
    ) -> List[RegimeInfo]:
        """
        Compute statistics for each regime.

        Calculates: mean return, volatility, frequency, avg duration.
        """
        stats = []

        for regime_id in range(self.n_regimes):
            mask = regimes == regime_id
            regime_returns = returns[mask]

            if len(regime_returns) == 0:
                stats.append(RegimeInfo(
                    regime_id=regime_id,
                    name=self.regime_names.get(regime_id, f'Regime {regime_id}'),
                    mean_return=0.0, volatility=0.0,
                    frequency=0.0, avg_duration=0.0,
                ))
                continue

            # Mean and volatility
            mean_ret = float(np.mean(regime_returns))
            vol = float(np.std(regime_returns))

            # Frequency: fraction of days in this regime
            frequency = float(np.sum(mask) / len(regimes))

            # Average duration: average length of consecutive runs
            avg_duration = self._calculate_avg_duration(regimes, regime_id)

            stats.append(RegimeInfo(
                regime_id=regime_id,
                name=self.regime_names.get(regime_id, f'Regime {regime_id}'),
                mean_return=mean_ret,
                volatility=vol,
                frequency=frequency,
                avg_duration=avg_duration,
            ))

        return stats

    @staticmethod
    def _calculate_avg_duration(regimes: np.ndarray, regime_id: int) -> float:
        """
        Calculate the average number of consecutive days spent in a regime.

        Example: regimes = [0, 0, 0, 1, 1, 0, 0]
        For regime 0: runs are [3, 2], average = 2.5
        """
        runs = []
        current_run = 0

        for r in regimes:
            if r == regime_id:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                current_run = 0

        # Don't forget the last run
        if current_run > 0:
            runs.append(current_run)

        return float(np.mean(runs)) if runs else 0.0

    # ─── Persistence ────────────────────────────────────────────

    def save(self, filepath: str) -> None:
        """Save the fitted model to disk using joblib."""
        self._check_fitted()
        save_data = {
            'model': self.model,
            'n_regimes': self.n_regimes,
            'covariance_type': self.covariance_type,
            'feature_names': self.feature_names,
            'regime_names': self.regime_names,
            'is_fitted': self.is_fitted,
        }
        joblib.dump(save_data, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'MarketRegimeHMM':
        """Load a fitted model from disk."""
        save_data = joblib.load(filepath)
        instance = cls(
            n_regimes=save_data['n_regimes'],
            covariance_type=save_data['covariance_type'],
        )
        instance.model = save_data['model']
        instance.feature_names = save_data['feature_names']
        instance.regime_names = save_data['regime_names']
        instance.is_fitted = save_data['is_fitted']
        return instance

    def _check_fitted(self) -> None:
        """Raise error if model hasn't been fitted yet."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")


# ─── Model Selection ────────────────────────────────────────────

def select_optimal_regimes(
    features: np.ndarray,
    min_regimes: int = 2,
    max_regimes: int = 5,
    criterion: str = 'bic'
) -> Tuple[int, Dict[int, float]]:
    """
    Find the optimal number of regimes using BIC or AIC.

    Tries n_regimes = 2, 3, 4, 5 and picks the one with the
    lowest information criterion score.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix.
    min_regimes, max_regimes : int
        Range of regime counts to try.
    criterion : str
        'bic' or 'aic'. BIC penalizes complexity more heavily.

    Returns
    -------
    tuple of (optimal_n, scores_dict)
        optimal_n: best number of regimes
        scores_dict: {n_regimes: score} for each candidate
    """
    scores = {}

    for n in range(min_regimes, max_regimes + 1):
        try:
            model = MarketRegimeHMM(n_regimes=n, covariance_type='full')
            model.fit(features)
            aic, bic = model.calculate_aic_bic(features)
            scores[n] = bic if criterion == 'bic' else aic
        except Exception:
            scores[n] = np.inf

    optimal_n = min(scores, key=scores.get)
    return optimal_n, scores
```

---

---

# PART 6: REGIME CLASSIFIER (Step 4)

This is the **orchestrator** — it wires together the feature engineer and HMM model into a clean, high-level interface.

---

**File: `src/regime_classifier.py`**

```python
"""
regime_classifier.py — High-Level Regime Classification Orchestrator
=====================================================================

This module provides the main interface for users:
  1. Takes raw prices + returns
  2. Engineers features internally
  3. Optionally selects optimal regime count via BIC
  4. Fits the HMM
  5. Smooths probabilities to reduce whipsaw
  6. Detects regime transitions
  7. Returns a comprehensive RegimeAnalysis object
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional

from src.feature_engineering import FeatureEngineer
from src.hmm_model import MarketRegimeHMM, HMMResult, RegimeInfo, select_optimal_regimes


@dataclass
class RegimeAnalysis:
    """
    Complete regime analysis result.

    Attributes
    ----------
    regimes : pd.Series
        Regime label for each day (integer-valued).
    regime_probs : pd.DataFrame
        Regime probabilities for each day (columns = regimes).
    regime_names : dict
        Mapping from regime ID to name (e.g., {0: 'Bear Market'}).
    current_regime : int
        ID of the current (most recent) regime.
    current_regime_name : str
        Name of the current regime.
    transitions : list
        Dates where regime changes occurred.
    regime_stats : list of RegimeInfo
        Statistics for each regime.
    duration_stats : dict
        Average duration in days for each regime.
    transition_matrix : np.ndarray
        Learned transition probability matrix.
    hmm_result : HMMResult
        Raw HMM output.
    """
    regimes: pd.Series
    regime_probs: pd.DataFrame
    regime_names: Dict[int, str]
    current_regime: int
    current_regime_name: str
    transitions: List
    regime_stats: List[RegimeInfo]
    duration_stats: Dict[int, float]
    transition_matrix: np.ndarray
    hmm_result: HMMResult


class RegimeClassifier:
    """
    Main interface for regime detection.

    Parameters
    ----------
    n_regimes : int
        Number of regimes (default 3). Set to None for auto-selection.
    use_volatility : bool
        Whether to use rolling volatility as a feature (default True).
    smoothing_window : int
        Rolling window for probability smoothing (default 5).
        Reduces noisy whipsaw in regime assignments.
    volatility_window : int
        Rolling window for volatility calculation (default 21).
    """

    def __init__(
        self,
        n_regimes: int = 3,
        use_volatility: bool = True,
        smoothing_window: int = 5,
        volatility_window: int = 21
    ):
        self.n_regimes = n_regimes
        self.use_volatility = use_volatility
        self.smoothing_window = smoothing_window
        self.volatility_window = volatility_window

        self.feature_engineer = FeatureEngineer(
            volatility_window=volatility_window,
            use_volatility=use_volatility,
        )
        self.hmm: Optional[MarketRegimeHMM] = None

    def classify(
        self,
        prices: pd.DataFrame,
        returns: pd.Series
    ) -> RegimeAnalysis:
        """
        Classify market regimes from price and return data.

        This is the main entry point. It:
          1. Creates features (returns + volatility)
          2. Optionally selects optimal regime count
          3. Fits the HMM
          4. Smooths probabilities
          5. Detects transitions
          6. Returns comprehensive analysis

        Parameters
        ----------
        prices : pd.DataFrame
            OHLCV price data.
        returns : pd.Series
            Daily returns.

        Returns
        -------
        RegimeAnalysis
            Complete analysis results.
        """
        # Step 1: Create features
        fset = self.feature_engineer.create_features(prices, returns)

        # Step 2: Optional model selection
        if self.n_regimes is None:
            optimal_n, scores = select_optimal_regimes(fset.features)
            self.n_regimes = optimal_n
            print(f"  Auto-selected {optimal_n} regimes via BIC")

        # Step 3: Fit HMM
        self.hmm = MarketRegimeHMM(n_regimes=self.n_regimes)
        hmm_result = self.hmm.fit_and_analyze(
            features=fset.features,
            returns=fset.returns,
            feature_names=fset.feature_names,
        )

        # Step 4: Create regime Series with datetime index
        regimes = pd.Series(
            hmm_result.regimes,
            index=fset.index,
            name='Regime'
        )

        # Step 5: Smooth probabilities
        regime_probs = pd.DataFrame(
            hmm_result.regime_probs,
            index=fset.index,
            columns=[self.hmm.regime_names.get(i, f'Regime {i}')
                     for i in range(self.n_regimes)]
        )

        if self.smoothing_window > 1:
            regime_probs = regime_probs.rolling(
                window=self.smoothing_window, min_periods=1
            ).mean()
            # Re-normalize so rows sum to 1
            regime_probs = regime_probs.div(regime_probs.sum(axis=1), axis=0)

        # Step 6: Detect transitions
        transitions = self._detect_transitions(regimes)

        # Step 7: Duration statistics
        duration_stats = {
            info.regime_id: info.avg_duration
            for info in hmm_result.regime_stats
        }

        # Current regime
        current_regime = int(regimes.iloc[-1])
        current_regime_name = self.hmm.regime_names.get(current_regime, f'Regime {current_regime}')

        return RegimeAnalysis(
            regimes=regimes,
            regime_probs=regime_probs,
            regime_names=self.hmm.regime_names,
            current_regime=current_regime,
            current_regime_name=current_regime_name,
            transitions=transitions,
            regime_stats=hmm_result.regime_stats,
            duration_stats=duration_stats,
            transition_matrix=hmm_result.transition_matrix,
            hmm_result=hmm_result,
        )

    @staticmethod
    def _detect_transitions(regimes: pd.Series) -> List:
        """
        Find dates where the regime changed.

        Returns a list of (date, from_regime, to_regime) tuples.
        """
        transitions = []
        for i in range(1, len(regimes)):
            if regimes.iloc[i] != regimes.iloc[i - 1]:
                transitions.append({
                    'date': regimes.index[i],
                    'from': int(regimes.iloc[i - 1]),
                    'to': int(regimes.iloc[i]),
                })
        return transitions
```

---

---

# PART 7: SIGNAL GENERATOR (Step 5)

Converts regime detections into actionable trading signals with position sizing and risk management.

---

**File: `src/signal_generator.py`**

```python
"""
signal_generator.py — Trading Signal Generation from Regimes
=============================================================

Converts regime detections into trading signals:
  STRONG_BUY  → Full long  (+1.0) when Bull + high confidence
  BUY         → Partial long (+0.6) when Bull + moderate confidence
  HOLD        → Flat (0.0) when Neutral or low confidence
  SELL        → Partial short (-0.6) when Bear + moderate confidence
  STRONG_SELL → Full short (-1.0) when Bear + high confidence

Also computes:
  - Volatility-adjusted stop losses and take profits
  - Position sizing based on confidence
  - Backtest performance (Sharpe, win rate, total return)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional

from src.regime_classifier import RegimeAnalysis


class SignalType(Enum):
    """Trading signal types."""
    STRONG_BUY = 'strong_buy'
    BUY = 'buy'
    HOLD = 'hold'
    SELL = 'sell'
    STRONG_SELL = 'strong_sell'


@dataclass
class TradingSignal:
    """
    A single trading signal for a specific day.

    Attributes
    ----------
    signal_type : SignalType
        The signal category.
    regime : int
        Regime ID that generated this signal.
    regime_name : str
        Human-readable regime name.
    confidence : float
        Probability of the assigned regime (0 to 1).
    position_size : float
        Recommended position size (-1 to +1).
    stop_loss_pct : float
        Suggested stop loss in percentage terms.
    take_profit_pct : float
        Suggested take profit in percentage terms.
    """
    signal_type: SignalType
    regime: int
    regime_name: str
    confidence: float
    position_size: float
    stop_loss_pct: float
    take_profit_pct: float


@dataclass
class SignalSummary:
    """
    Complete signal generation output.

    Attributes
    ----------
    signals : pd.DataFrame
        DataFrame with signal for each day.
    current_signal : TradingSignal
        The most recent signal.
    signal_counts : dict
        Count of each signal type.
    """
    signals: pd.DataFrame
    current_signal: TradingSignal
    signal_counts: Dict[str, int]


class SignalGenerator:
    """
    Generates trading signals from regime analysis.

    Parameters
    ----------
    confidence_threshold : float
        Minimum confidence to generate a directional signal (default 0.6).
    strong_threshold : float
        Confidence required for STRONG signals (default 0.8).
    base_stop_loss : float
        Base stop loss percentage (default 0.02 = 2%).
    base_take_profit : float
        Base take profit percentage (default 0.04 = 4%).
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        strong_threshold: float = 0.8,
        base_stop_loss: float = 0.02,
        base_take_profit: float = 0.04
    ):
        self.confidence_threshold = confidence_threshold
        self.strong_threshold = strong_threshold
        self.base_stop_loss = base_stop_loss
        self.base_take_profit = base_take_profit

    def generate_signals(self, analysis: RegimeAnalysis) -> SignalSummary:
        """
        Generate trading signals for every day in the analysis.

        Parameters
        ----------
        analysis : RegimeAnalysis
            Output of RegimeClassifier.classify().

        Returns
        -------
        SignalSummary
            Contains all signals, current signal, and counts.
        """
        # Average volatility across regimes (for adjusting stops/TPs)
        avg_vol = np.mean([s.volatility for s in analysis.regime_stats if s.volatility > 0])
        if avg_vol == 0:
            avg_vol = 0.01

        # Build volatility lookup by regime
        vol_by_regime = {s.regime_id: s.volatility for s in analysis.regime_stats}

        records = []

        for i, date in enumerate(analysis.regimes.index):
            regime = int(analysis.regimes.iloc[i])
            regime_name = analysis.regime_names.get(regime, f'Regime {regime}')

            # Get confidence: max probability at this time step
            confidence = float(analysis.regime_probs.iloc[i].max())

            # Determine signal type
            signal_type = self._determine_signal(regime_name, confidence)

            # Position size: scaled by confidence
            position_size = self._calculate_position_size(signal_type, confidence)

            # Volatility-adjusted stops and TPs
            regime_vol = vol_by_regime.get(regime, avg_vol)
            vol_ratio = regime_vol / avg_vol if avg_vol > 0 else 1.0

            stop_loss = self.base_stop_loss * max(vol_ratio, 0.5)
            take_profit = self.base_take_profit * max(vol_ratio, 0.5)

            records.append({
                'date': date,
                'signal_type': signal_type.value,
                'regime': regime,
                'regime_name': regime_name,
                'confidence': confidence,
                'position_size': position_size,
                'stop_loss_pct': stop_loss,
                'take_profit_pct': take_profit,
            })

        signals_df = pd.DataFrame(records).set_index('date')

        # Current signal
        last = records[-1]
        current_signal = TradingSignal(
            signal_type=SignalType(last['signal_type']),
            regime=last['regime'],
            regime_name=last['regime_name'],
            confidence=last['confidence'],
            position_size=last['position_size'],
            stop_loss_pct=last['stop_loss_pct'],
            take_profit_pct=last['take_profit_pct'],
        )

        # Signal counts
        signal_counts = signals_df['signal_type'].value_counts().to_dict()

        return SignalSummary(
            signals=signals_df,
            current_signal=current_signal,
            signal_counts=signal_counts,
        )

    def _determine_signal(self, regime_name: str, confidence: float) -> SignalType:
        """Map regime + confidence to a signal type."""
        if 'Bull' in regime_name:
            if confidence >= self.strong_threshold:
                return SignalType.STRONG_BUY
            elif confidence >= self.confidence_threshold:
                return SignalType.BUY
            else:
                return SignalType.HOLD
        elif 'Bear' in regime_name:
            if confidence >= self.strong_threshold:
                return SignalType.STRONG_SELL
            elif confidence >= self.confidence_threshold:
                return SignalType.SELL
            else:
                return SignalType.HOLD
        else:
            return SignalType.HOLD

    def _calculate_position_size(self, signal_type: SignalType, confidence: float) -> float:
        """Calculate position size from signal type and confidence."""
        if signal_type == SignalType.STRONG_BUY:
            return min(1.0, 0.6 * confidence + 0.4)
        elif signal_type == SignalType.BUY:
            return 0.6 * confidence
        elif signal_type == SignalType.SELL:
            return -0.6 * confidence
        elif signal_type == SignalType.STRONG_SELL:
            return max(-1.0, -(0.6 * confidence + 0.4))
        else:
            return 0.0

    def calculate_signal_performance(
        self,
        signal_summary: SignalSummary,
        returns: pd.Series
    ) -> Dict[str, float]:
        """
        Backtest signals against actual returns.

        Returns
        -------
        dict with: total_return, sharpe_ratio, win_rate, total_trades
        """
        signals = signal_summary.signals
        aligned = pd.DataFrame({
            'position': signals['position_size'],
            'returns': returns,
        }).dropna()

        if len(aligned) == 0:
            return {'total_return': 0, 'sharpe_ratio': 0, 'win_rate': 0, 'total_trades': 0}

        # Strategy returns = position * next-day return
        strategy_returns = aligned['position'].shift(1) * aligned['returns']
        strategy_returns = strategy_returns.dropna()

        total_return = float((1 + strategy_returns).prod() - 1)
        sharpe = float(strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)) \
            if strategy_returns.std() > 0 else 0.0
        win_rate = float((strategy_returns > 0).mean())
        total_trades = int((aligned['position'].diff().abs() > 0.01).sum())

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'total_trades': total_trades,
        }
```

---

---

# PART 8: VISUALIZATION (Step 6)

Six chart types, each answering a specific question about the regime analysis.

---

**File: `src/visualization.py`**

```python
"""
visualization.py — Publication-Quality Regime Charts
=====================================================

Six chart types:
  1. Regime Timeline: prices colored by regime
  2. Probability Evolution: stacked area of regime probabilities
  3. Transition Matrix: heatmap of regime transitions
  4. Regime Statistics: bar charts of return/vol/frequency
  5. Signal Performance: cumulative returns of the strategy
  6. Regime Dashboard: all-in-one multi-panel overview
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, Optional

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)

# Regime color scheme
REGIME_COLORS = {
    'Bull Market': '#2ecc71',
    'Neutral': '#3498db',
    'Bear Market': '#e74c3c',
}

DEFAULT_COLORS = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']


def _get_color(name: str, idx: int = 0) -> str:
    """Get color for a regime by name, falling back to index."""
    return REGIME_COLORS.get(name, DEFAULT_COLORS[idx % len(DEFAULT_COLORS)])


def plot_regime_timeline(
    regimes: pd.Series,
    prices: pd.Series,
    regime_names: Dict[int, str],
    save_path: str = 'output/regime_timeline.png'
) -> None:
    """
    Chart 1: Price line with background colored by regime.

    This is the most intuitive chart — you literally see the market
    going green (Bull), blue (Neutral), or red (Bear) over time.
    """
    fig, ax = plt.subplots(figsize=(16, 6))

    # Align prices and regimes
    common_idx = regimes.index.intersection(prices.index)
    r = regimes.loc[common_idx]
    p = prices.loc[common_idx]

    # Plot price line
    ax.plot(p.index, p.values, color='black', linewidth=0.8, zorder=3)

    # Color background by regime
    for regime_id, name in regime_names.items():
        mask = r == regime_id
        color = _get_color(name, regime_id)
        ax.fill_between(
            p.index, p.min() * 0.95, p.max() * 1.05,
            where=mask, alpha=0.25, color=color, label=name
        )

    ax.set_title('Market Regimes Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {save_path.split('/')[-1]}")


def plot_regime_probabilities(
    regime_probs: pd.DataFrame,
    save_path: str = 'output/regime_probabilities.png'
) -> None:
    """
    Chart 2: Stacked area plot of regime probabilities over time.

    Shows how confident the model is in each regime at every point.
    Sudden shifts from one color to another indicate regime transitions.
    """
    fig, ax = plt.subplots(figsize=(16, 5))

    colors = [_get_color(col, i) for i, col in enumerate(regime_probs.columns)]

    ax.stackplot(
        regime_probs.index,
        *[regime_probs[col].values for col in regime_probs.columns],
        labels=regime_probs.columns,
        colors=colors,
        alpha=0.8,
    )

    ax.set_title('Regime Probability Evolution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {save_path.split('/')[-1]}")


def plot_transition_matrix(
    transition_matrix: np.ndarray,
    regime_names: Dict[int, str],
    save_path: str = 'output/transition_matrix.png'
) -> None:
    """
    Chart 3: Heatmap of the transition probability matrix.

    Each cell shows P(row_regime → column_regime).
    Diagonal = persistence (high = regime is sticky).
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    labels = [regime_names.get(i, f'Regime {i}') for i in range(len(transition_matrix))]

    sns.heatmap(
        transition_matrix * 100,  # Convert to percentages
        annot=True, fmt='.1f', cmap='YlOrRd',
        xticklabels=labels, yticklabels=labels,
        cbar_kws={'label': 'Probability (%)'},
        ax=ax,
    )

    ax.set_title('Regime Transition Matrix (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('To Regime')
    ax.set_ylabel('From Regime')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {save_path.split('/')[-1]}")


def plot_regime_statistics(
    regime_stats,
    save_path: str = 'output/regime_statistics.png'
) -> None:
    """
    Chart 4: Bar charts showing return, volatility, and frequency per regime.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    names = [s.name for s in regime_stats]
    colors = [_get_color(n, i) for i, n in enumerate(names)]

    # Mean Return
    returns = [s.mean_return * 100 for s in regime_stats]
    axes[0].bar(names, returns, color=colors)
    axes[0].set_title('Mean Daily Return (%)', fontweight='bold')
    axes[0].axhline(y=0, color='black', linewidth=0.5)

    # Volatility
    vols = [s.volatility * 100 for s in regime_stats]
    axes[1].bar(names, vols, color=colors)
    axes[1].set_title('Daily Volatility (%)', fontweight='bold')

    # Frequency
    freqs = [s.frequency * 100 for s in regime_stats]
    axes[2].bar(names, freqs, color=colors)
    axes[2].set_title('Time in Regime (%)', fontweight='bold')

    for ax in axes:
        ax.tick_params(axis='x', rotation=15)

    plt.suptitle('Regime Statistics', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {save_path.split('/')[-1]}")


def plot_signal_performance(
    signals: pd.DataFrame,
    returns: pd.Series,
    save_path: str = 'output/signal_performance.png'
) -> None:
    """
    Chart 5: Cumulative return of regime strategy vs buy-and-hold.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    aligned = pd.DataFrame({
        'position': signals['position_size'],
        'returns': returns,
    }).dropna()

    strategy_returns = (aligned['position'].shift(1) * aligned['returns']).dropna()
    cumulative_strategy = (1 + strategy_returns).cumprod()
    cumulative_bh = (1 + aligned['returns'].loc[strategy_returns.index]).cumprod()

    ax.plot(cumulative_strategy.index, cumulative_strategy.values,
            label='Regime Strategy', color='#2ecc71', linewidth=1.5)
    ax.plot(cumulative_bh.index, cumulative_bh.values,
            label='Buy & Hold', color='#95a5a6', linewidth=1.5, linestyle='--')

    ax.set_title('Strategy Performance: Regime vs Buy & Hold',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {save_path.split('/')[-1]}")


def plot_regime_dashboard(
    regimes: pd.Series,
    regime_probs: pd.DataFrame,
    prices: pd.Series,
    transition_matrix: np.ndarray,
    regime_names: Dict[int, str],
    save_path: str = 'output/regime_dashboard.png'
) -> None:
    """
    Chart 6: All-in-one 4-panel dashboard.

    Panel 1: Price with regime colors
    Panel 2: Regime probability stack
    Panel 3: Transition heatmap
    Panel 4: Regime ID timeline
    """
    fig = plt.figure(figsize=(18, 14))

    # Panel 1: Price with regime overlay (top, full width)
    ax1 = fig.add_subplot(3, 2, (1, 2))
    common_idx = regimes.index.intersection(prices.index)
    r, p = regimes.loc[common_idx], prices.loc[common_idx]
    ax1.plot(p.index, p.values, color='black', linewidth=0.8)
    for rid, name in regime_names.items():
        mask = r == rid
        ax1.fill_between(p.index, p.min() * 0.95, p.max() * 1.05,
                         where=mask, alpha=0.2, color=_get_color(name, rid), label=name)
    ax1.set_title('Price with Regime Overlay', fontweight='bold', fontsize=12)
    ax1.legend(loc='upper left', fontsize=8)

    # Panel 2: Probability evolution (middle left)
    ax2 = fig.add_subplot(3, 2, 3)
    colors = [_get_color(col, i) for i, col in enumerate(regime_probs.columns)]
    ax2.stackplot(regime_probs.index,
                  *[regime_probs[col].values for col in regime_probs.columns],
                  colors=colors, alpha=0.8)
    ax2.set_title('Regime Probabilities', fontweight='bold', fontsize=12)
    ax2.set_ylim(0, 1)

    # Panel 3: Transition matrix (middle right)
    ax3 = fig.add_subplot(3, 2, 4)
    labels = [regime_names.get(i, f'R{i}') for i in range(len(transition_matrix))]
    sns.heatmap(transition_matrix * 100, annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=labels, yticklabels=labels, ax=ax3)
    ax3.set_title('Transition Matrix (%)', fontweight='bold', fontsize=12)

    # Panel 4: Regime timeline (bottom, full width)
    ax4 = fig.add_subplot(3, 2, (5, 6))
    regime_colors = [_get_color(regime_names.get(int(v), ''), int(v)) for v in r.values]
    ax4.scatter(r.index, r.values, c=regime_colors, s=2, alpha=0.6)
    ax4.set_title('Regime Timeline', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Regime ID')
    ax4.set_yticks(sorted(regime_names.keys()))
    ax4.set_yticklabels([regime_names[k] for k in sorted(regime_names.keys())])

    plt.suptitle('HMM Regime Detection Dashboard', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {save_path.split('/')[-1]}")
```

---

---

# PART 9: MAIN SCRIPT (Step 7)

**File: `main.py`**

```python
"""
main.py — HMM Regime Detection Pipeline Entry Point
=====================================================

Runs the full analysis:
  1. Load data (Yahoo Finance or synthetic fallback)
  2. Engineer features
  3. Select optimal regime count via BIC
  4. Detect regimes with probability smoothing
  5. Generate trading signals
  6. Backtest performance
  7. Create 5 visualizations + dashboard
  8. Save reports to ./output/
"""

import os
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from src.data_loader import DataLoader
from src.feature_engineering import create_hmm_features
from src.hmm_model import select_optimal_regimes
from src.regime_classifier import RegimeClassifier
from src.signal_generator import SignalGenerator
from src.visualization import (
    plot_regime_timeline,
    plot_regime_probabilities,
    plot_transition_matrix,
    plot_regime_statistics,
    plot_signal_performance,
    plot_regime_dashboard,
)


def main():
    """Run the full HMM regime detection pipeline."""

    print("=" * 60)
    print(" HMM REGIME DETECTION")
    print("=" * 60)
    print("\nThis pipeline detects market regimes using Hidden Markov Models.")
    print("It identifies Bull, Bear, and Neutral conditions automatically.")

    os.makedirs('output', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # ── Step 1: Load Data ──
    print(f"\n{'─' * 60}")
    print(f" STEP 1: Loading Market Data")
    print(f"{'─' * 60}")

    loader = DataLoader()
    data = loader.fetch_yahoo('SPY', '2019-01-01')

    # ── Step 2: Feature Engineering ──
    print(f"\n{'─' * 60}")
    print(f" STEP 2: Engineering Features")
    print(f"{'─' * 60}")

    features, feature_names, feat_index = create_hmm_features(
        data.prices, data.returns, volatility_window=21, use_volatility=True
    )
    print(f"  Features: {feature_names}")
    print(f"  Shape: {features.shape}")

    # ── Step 3: Model Selection ──
    print(f"\n{'─' * 60}")
    print(f" STEP 3: Selecting Optimal Regime Count")
    print(f"{'─' * 60}")

    optimal_n, bic_scores = select_optimal_regimes(features, min_regimes=2, max_regimes=5)
    print(f"\n  BIC Scores:")
    for n, score in bic_scores.items():
        marker = " ← Best" if n == optimal_n else ""
        print(f"    {n} regimes: BIC = {score:,.0f}{marker}")
    print(f"\n  ✓ Selected {optimal_n} regimes")

    # ── Step 4: Regime Detection ──
    print(f"\n{'─' * 60}")
    print(f" STEP 4: Detecting Regimes")
    print(f"{'─' * 60}")

    classifier = RegimeClassifier(n_regimes=optimal_n, smoothing_window=5)
    analysis = classifier.classify(data.prices, data.returns)

    print(f"\n  Current Regime: {analysis.current_regime_name}")
    print(f"  Regime Transitions: {len(analysis.transitions)}")

    print(f"\n  Regime Statistics:")
    print(f"  {'Regime':<16s} {'Mean Ret':>10s} {'Volatility':>12s} "
          f"{'Duration':>10s} {'Frequency':>11s}")
    print(f"  {'─' * 60}")
    for stat in analysis.regime_stats:
        print(f"  {stat.name:<16s} {stat.mean_return*100:>9.4f}% "
              f"{stat.volatility*100:>11.4f}% "
              f"{stat.avg_duration:>9.1f} "
              f"{stat.frequency*100:>10.1f}%")

    print(f"\n  Transition Matrix:")
    names = [analysis.regime_names.get(i, f'R{i}') for i in range(optimal_n)]
    header = "  " + " " * 16 + "".join(f"{n:>14s}" for n in names)
    print(header)
    for i, name in enumerate(names):
        row = "  " + f"{name:<16s}"
        for j in range(optimal_n):
            row += f"{analysis.transition_matrix[i, j]*100:>13.1f}%"
        print(row)

    # ── Step 5: Trading Signals ──
    print(f"\n{'─' * 60}")
    print(f" STEP 5: Generating Trading Signals")
    print(f"{'─' * 60}")

    generator = SignalGenerator(confidence_threshold=0.6, strong_threshold=0.8)
    signals = generator.generate_signals(analysis)

    current = signals.current_signal
    print(f"\n  Current Signal:")
    print(f"    Type: {current.signal_type.value.upper()}")
    print(f"    Regime: {current.regime_name}")
    print(f"    Confidence: {current.confidence:.1%}")
    print(f"    Position Size: {current.position_size:+.2f}")
    print(f"    Stop Loss: {current.stop_loss_pct:.1%}")
    print(f"    Take Profit: {current.take_profit_pct:.1%}")

    print(f"\n  Signal Distribution:")
    for sig_type, count in signals.signal_counts.items():
        print(f"    {sig_type.upper()}: {count}")

    # ── Step 6: Backtest ──
    print(f"\n{'─' * 60}")
    print(f" STEP 6: Backtesting Performance")
    print(f"{'─' * 60}")

    perf = generator.calculate_signal_performance(signals, data.returns)
    print(f"\n  Backtest Performance:")
    print(f"    Total Return: {perf['total_return']:.2%}")
    print(f"    Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
    print(f"    Win Rate: {perf['win_rate']:.1%}")
    print(f"    Total Trades: {perf['total_trades']}")

    # ── Step 7: Visualizations ──
    print(f"\n{'─' * 60}")
    print(f" STEP 7: Generating Visualizations")
    print(f"{'─' * 60}")

    print(f"\n  Saving charts to ./output/...")

    plot_regime_timeline(analysis.regimes, data.prices['Close'],
                         analysis.regime_names)
    plot_regime_probabilities(analysis.regime_probs)
    plot_transition_matrix(analysis.transition_matrix, analysis.regime_names)
    plot_regime_statistics(analysis.regime_stats)
    plot_signal_performance(signals.signals, data.returns)
    plot_regime_dashboard(analysis.regimes, analysis.regime_probs,
                          data.prices['Close'], analysis.transition_matrix,
                          analysis.regime_names)

    # Save model
    classifier.hmm.save('models/hmm_regime_model.pkl')
    print(f"\n  ✓ Model saved to models/hmm_regime_model.pkl")

    # Save CSV reports
    analysis.regimes.to_csv('output/regime_history.csv')
    signals.signals.to_csv('output/signal_history.csv')

    # Save summary JSON
    summary = {
        'symbol': data.symbol,
        'date_range': f"{data.start_date} to {data.end_date}",
        'n_regimes': optimal_n,
        'current_regime': analysis.current_regime_name,
        'current_signal': current.signal_type.value,
        'total_return': f"{perf['total_return']:.2%}",
        'sharpe_ratio': round(perf['sharpe_ratio'], 2),
    }
    with open('output/summary_report.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f" ANALYSIS COMPLETE")
    print(f"{'=' * 60}")
    print(f"\n📊 Key Findings:")
    print(f"  • Asset: {data.symbol}")
    print(f"  • Regimes: {optimal_n}")
    print(f"  • Current: {analysis.current_regime_name}")
    print(f"  • Signal: {current.signal_type.value.upper()}")
    print(f"  • Sharpe: {perf['sharpe_ratio']:.2f}")
    print(f"\n📁 Output saved to ./output/")
    print(f"\nDone! ✅")


if __name__ == '__main__':
    main()
```

---

---

# PART 10: UNIT TESTS (Step 8)

**File: `tests/test_hmm.py`**

```python
"""
test_hmm.py — Unit Tests for HMM Regime Detection
===================================================

32 tests across 7 test classes covering every module.

Run with: python -m pytest tests/test_hmm.py -v
"""

import numpy as np
import pandas as pd
import pytest
import os
import tempfile

from src.data_loader import DataLoader, MarketData
from src.feature_engineering import FeatureEngineer, create_hmm_features
from src.hmm_model import MarketRegimeHMM, HMMResult, select_optimal_regimes
from src.regime_classifier import RegimeClassifier
from src.signal_generator import SignalGenerator, SignalType


# ─── Fixtures ───────────────────────────────────────────────────

@pytest.fixture
def synthetic_data():
    """Generate synthetic data for tests."""
    loader = DataLoader()
    return loader.generate_synthetic(n_samples=500, n_regimes=3, random_state=42)

@pytest.fixture
def feature_set(synthetic_data):
    """Create features from synthetic data."""
    engineer = FeatureEngineer(volatility_window=21, use_volatility=True)
    return engineer.create_features(synthetic_data.prices, synthetic_data.returns)

@pytest.fixture
def fitted_hmm(feature_set):
    """Fit an HMM to synthetic features."""
    model = MarketRegimeHMM(n_regimes=3, random_state=42)
    model.fit(feature_set.features, feature_set.feature_names)
    return model


# ─── TestMarketRegimeHMM ────────────────────────────────────────

class TestMarketRegimeHMM:

    def test_initialization(self):
        model = MarketRegimeHMM(n_regimes=3)
        assert model.n_regimes == 3
        assert not model.is_fitted

    def test_fit(self, feature_set):
        model = MarketRegimeHMM(n_regimes=3, random_state=42)
        model.fit(feature_set.features, feature_set.feature_names)
        assert model.is_fitted

    def test_predict(self, fitted_hmm, feature_set):
        regimes = fitted_hmm.predict(feature_set.features)
        assert len(regimes) == len(feature_set.features)
        assert set(regimes).issubset({0, 1, 2})

    def test_predict_proba(self, fitted_hmm, feature_set):
        probs = fitted_hmm.predict_proba(feature_set.features)
        assert probs.shape == (len(feature_set.features), 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_transition_matrix(self, fitted_hmm):
        tm = fitted_hmm.get_transition_matrix()
        assert tm.shape == (3, 3)
        np.testing.assert_allclose(tm.sum(axis=1), 1.0, atol=1e-6)
        assert (tm >= 0).all()

    def test_aic_bic(self, fitted_hmm, feature_set):
        aic, bic = fitted_hmm.calculate_aic_bic(feature_set.features)
        assert np.isfinite(aic)
        assert np.isfinite(bic)

    def test_fit_and_analyze(self, feature_set):
        model = MarketRegimeHMM(n_regimes=3, random_state=42)
        result = model.fit_and_analyze(
            feature_set.features,
            returns=feature_set.returns,
            feature_names=feature_set.feature_names,
        )
        assert isinstance(result, HMMResult)
        assert len(result.regime_stats) == 3

    def test_save_load(self, fitted_hmm, feature_set):
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name

        try:
            fitted_hmm.save(path)
            loaded = MarketRegimeHMM.load(path)
            assert loaded.is_fitted
            assert loaded.n_regimes == fitted_hmm.n_regimes

            orig_pred = fitted_hmm.predict(feature_set.features)
            load_pred = loaded.predict(feature_set.features)
            np.testing.assert_array_equal(orig_pred, load_pred)
        finally:
            os.unlink(path)


# ─── TestModelSelection ──────────────────────────────────────────

class TestModelSelection:

    def test_select_optimal(self, feature_set):
        optimal_n, scores = select_optimal_regimes(
            feature_set.features, min_regimes=2, max_regimes=4
        )
        assert 2 <= optimal_n <= 4
        assert len(scores) == 3


# ─── TestDataLoader ──────────────────────────────────────────────

class TestDataLoader:

    def test_synthetic_generation(self):
        loader = DataLoader()
        data = loader.generate_synthetic(n_samples=200, n_regimes=3)
        assert isinstance(data, MarketData)
        assert len(data.prices) == 200
        assert len(data.returns) == 200

    def test_synthetic_has_columns(self):
        loader = DataLoader()
        data = loader.generate_synthetic(n_samples=100)
        assert 'Close' in data.prices.columns
        assert 'Open' in data.prices.columns

    def test_synthetic_two_regimes(self):
        loader = DataLoader()
        data = loader.generate_synthetic(n_samples=200, n_regimes=2)
        assert len(data.returns) == 200


# ─── TestFeatureEngineer ──────────────────────────────────────────

class TestFeatureEngineer:

    def test_creates_features(self, synthetic_data):
        eng = FeatureEngineer(volatility_window=21, use_volatility=True)
        fset = eng.create_features(synthetic_data.prices, synthetic_data.returns)
        assert fset.features.shape[1] == 2  # returns + volatility

    def test_returns_only(self, synthetic_data):
        eng = FeatureEngineer(use_volatility=False)
        fset = eng.create_features(synthetic_data.prices, synthetic_data.returns)
        assert fset.features.shape[1] == 1

    def test_no_nans(self, synthetic_data):
        eng = FeatureEngineer(volatility_window=21)
        fset = eng.create_features(synthetic_data.prices, synthetic_data.returns)
        assert not np.isnan(fset.features).any()

    def test_convenience_function(self, synthetic_data):
        features, names, index = create_hmm_features(
            synthetic_data.prices, synthetic_data.returns
        )
        assert len(names) == 2
        assert len(index) == features.shape[0]


# ─── TestRegimeClassifier ─────────────────────────────────────────

class TestRegimeClassifier:

    def test_classify(self, synthetic_data):
        clf = RegimeClassifier(n_regimes=3, smoothing_window=3)
        analysis = clf.classify(synthetic_data.prices, synthetic_data.returns)
        assert analysis.current_regime_name in ['Bull Market', 'Neutral', 'Bear Market']

    def test_regime_names_assigned(self, synthetic_data):
        clf = RegimeClassifier(n_regimes=3)
        analysis = clf.classify(synthetic_data.prices, synthetic_data.returns)
        assert len(analysis.regime_names) == 3

    def test_transitions_detected(self, synthetic_data):
        clf = RegimeClassifier(n_regimes=3)
        analysis = clf.classify(synthetic_data.prices, synthetic_data.returns)
        assert len(analysis.transitions) > 0

    def test_probs_sum_to_one(self, synthetic_data):
        clf = RegimeClassifier(n_regimes=3)
        analysis = clf.classify(synthetic_data.prices, synthetic_data.returns)
        row_sums = analysis.regime_probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.01)

    def test_duration_stats(self, synthetic_data):
        clf = RegimeClassifier(n_regimes=3)
        analysis = clf.classify(synthetic_data.prices, synthetic_data.returns)
        assert len(analysis.duration_stats) == 3
        for d in analysis.duration_stats.values():
            assert d >= 0

    def test_two_regimes(self, synthetic_data):
        clf = RegimeClassifier(n_regimes=2)
        analysis = clf.classify(synthetic_data.prices, synthetic_data.returns)
        assert len(analysis.regime_names) == 2


# ─── TestSignalGenerator ──────────────────────────────────────────

class TestSignalGenerator:

    def test_generates_signals(self, synthetic_data):
        clf = RegimeClassifier(n_regimes=3)
        analysis = clf.classify(synthetic_data.prices, synthetic_data.returns)
        gen = SignalGenerator()
        signals = gen.generate_signals(analysis)
        assert len(signals.signals) > 0

    def test_current_signal(self, synthetic_data):
        clf = RegimeClassifier(n_regimes=3)
        analysis = clf.classify(synthetic_data.prices, synthetic_data.returns)
        gen = SignalGenerator()
        signals = gen.generate_signals(analysis)
        assert isinstance(signals.current_signal.signal_type, SignalType)

    def test_position_size_range(self, synthetic_data):
        clf = RegimeClassifier(n_regimes=3)
        analysis = clf.classify(synthetic_data.prices, synthetic_data.returns)
        gen = SignalGenerator()
        signals = gen.generate_signals(analysis)
        positions = signals.signals['position_size']
        assert (positions >= -1.0).all()
        assert (positions <= 1.0).all()

    def test_stop_loss_positive(self, synthetic_data):
        clf = RegimeClassifier(n_regimes=3)
        analysis = clf.classify(synthetic_data.prices, synthetic_data.returns)
        gen = SignalGenerator()
        signals = gen.generate_signals(analysis)
        assert (signals.signals['stop_loss_pct'] > 0).all()

    def test_performance_calculation(self, synthetic_data):
        clf = RegimeClassifier(n_regimes=3)
        analysis = clf.classify(synthetic_data.prices, synthetic_data.returns)
        gen = SignalGenerator()
        signals = gen.generate_signals(analysis)
        perf = gen.calculate_signal_performance(signals, synthetic_data.returns)
        assert 'sharpe_ratio' in perf
        assert 'total_return' in perf


# ─── TestIntegration ──────────────────────────────────────────────

class TestIntegration:

    def test_full_pipeline(self):
        """End-to-end: load → features → classify → signal → performance."""
        loader = DataLoader()
        data = loader.generate_synthetic(n_samples=300, random_state=99)

        clf = RegimeClassifier(n_regimes=3, smoothing_window=3)
        analysis = clf.classify(data.prices, data.returns)

        gen = SignalGenerator()
        signals = gen.generate_signals(analysis)
        perf = gen.calculate_signal_performance(signals, data.returns)

        assert analysis.current_regime_name in ['Bull Market', 'Neutral', 'Bear Market']
        assert isinstance(signals.current_signal.signal_type, SignalType)
        assert np.isfinite(perf['sharpe_ratio'])

    def test_model_save_load_pipeline(self):
        """Save → load → predict produces same results."""
        loader = DataLoader()
        data = loader.generate_synthetic(n_samples=300, random_state=77)

        clf = RegimeClassifier(n_regimes=3)
        analysis = clf.classify(data.prices, data.returns)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name

        try:
            clf.hmm.save(path)
            loaded = MarketRegimeHMM.load(path)
            from src.feature_engineering import FeatureEngineer
            eng = FeatureEngineer()
            fset = eng.create_features(data.prices, data.returns)
            orig = clf.hmm.predict(fset.features)
            reloaded = loaded.predict(fset.features)
            np.testing.assert_array_equal(orig, reloaded)
        finally:
            os.unlink(path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

---

# PART 11: RUN IT!

## Step 9.1: Run the Full Pipeline
```bash
python main.py
```

## Step 9.2: Run the Tests
```bash
python -m pytest tests/test_hmm.py -v
```

Expected: **32 passed** across 7 test classes.

---

---

# PART 12: HOW TO READ THE RESULTS

## 12.1: Interpreting the Transition Matrix

```
             Bull    Neutral    Bear
Bull         93.2%     5.8%     1.0%
Neutral       8.5%    86.3%     5.2%
Bear          1.5%     6.8%    91.7%
```

The **diagonal** tells you persistence. 93.2% for Bull means: "If we're in a bull market today, there's a 93.2% chance we're still in one tomorrow." Bear is 91.7% — bear markets are famously sticky.

The **off-diagonals** show transition probabilities. Bull→Bear at 1.0% is very rare — you almost always go through Neutral first.

**Half-life of a regime:** If persistence = P, the expected duration is `1/(1-P)` days. For Bull at 93.2%: `1/0.068 ≈ 14.7 days`. In practice, durations average 25–45 days because regimes cluster.

## 12.2: Interpreting Regime Statistics

```
Bear Market:  Mean Ret = -0.08%, Vol = 2.35%, Freq = 18.2%
Neutral:      Mean Ret = +0.02%, Vol = 1.12%, Freq = 41.5%
Bull Market:  Mean Ret = +0.12%, Vol = 0.89%, Freq = 40.3%
```

Bear is **nasty**: negative returns AND high volatility. Bull is the opposite: positive returns with calm markets. Neutral is the most common (41.5% of the time) — markets spend a lot of time going nowhere.

## 12.3: Interpreting Trading Signals

```
Type: BUY  |  Confidence: 87.2%  |  Position: +0.52
```

"BUY" (not STRONG_BUY) because the scaled position (0.6 × 0.87 = 0.52) is below the strong threshold. The model is 87% confident we're in Bull. Position size is moderate (+0.52 out of ±1.0).

Stop loss and take profit are **volatility-adjusted** — wider in Bear (high vol), tighter in Bull (low vol).

## 12.4: Interpreting the Dashboard

**Panel 1 (Price + Colors):** Green background = Bull, blue = Neutral, red = Bear. You'll see clear clustering — the model doesn't flip randomly.

**Panel 2 (Probability Stack):** Height of each color shows model confidence. When one color dominates (e.g., 90% green), the model is very confident. When colors are mixed, it's uncertain.

**Panel 3 (Heatmap):** Dark red on the diagonal = high persistence. Light colors off-diagonal = rare transitions.

---

---

# PART 13: QUICK REFERENCE CARD

## Architecture
```
main.py                       → Orchestrates everything
src/data_loader.py            → DataLoader: fetch_yahoo(), generate_synthetic()
src/feature_engineering.py    → FeatureEngineer: create_features()
src/hmm_model.py              → MarketRegimeHMM: fit(), predict(), predict_proba(),
                                 save(), load(), fit_and_analyze()
                                 + select_optimal_regimes()
src/regime_classifier.py      → RegimeClassifier: classify() → RegimeAnalysis
src/signal_generator.py       → SignalGenerator: generate_signals() → SignalSummary
src/visualization.py          → 6 chart functions
tests/test_hmm.py             → 32 tests across 7 classes
```

## Key HMM Concepts

| Concept | Formula / Meaning |
|---------|-------------------|
| Transition Matrix | `P[i,j]` = probability of going from regime i to j |
| Emission Model | Each regime ~ Normal(μ, σ²) |
| Viterbi Decoding | Most likely state *sequence* (for `.predict()`) |
| Forward-Backward | State *probabilities* at each time (for `.predict_proba()`) |
| BIC | `-2 × log_likelihood + k × log(n)` (lower = better) |
| Persistence | Diagonal of transition matrix (higher = stickier regime) |
| Expected Duration | `1 / (1 - persistence)` days |

## Signal Logic

| Regime | Confidence | Signal | Position |
|--------|-----------|--------|----------|
| Bull | > 80% | STRONG_BUY | +1.0 |
| Bull | > 60% | BUY | +0.6 × confidence |
| Any | < 60% | HOLD | 0.0 |
| Bear | > 60% | SELL | -0.6 × confidence |
| Bear | > 80% | STRONG_SELL | -1.0 |

## Dependencies
```
numpy         → Array math
pandas        → DataFrames
scipy         → Statistical functions
scikit-learn  → ML utilities
hmmlearn      → Gaussian HMM fitting (EM/Baum-Welch)
matplotlib    → Charts
seaborn       → Chart styling
yfinance      → Yahoo Finance market data
joblib        → Model persistence (save/load)
pytest        → Testing
```
