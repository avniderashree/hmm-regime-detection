"""
Data Loader Module
==================
Fetches and preprocesses market data for regime detection.

Supports:
- Yahoo Finance data fetching
- Synthetic data generation
- Return calculation
- Data validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class MarketData:
    """
    Container for market data.
    
    Attributes:
        prices: DataFrame with OHLCV data
        returns: Series of returns
        symbol: Ticker symbol
        start_date: Start of data
        end_date: End of data
    """
    prices: pd.DataFrame
    returns: pd.Series
    symbol: str
    start_date: datetime
    end_date: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'n_observations': len(self.returns),
            'mean_return': float(self.returns.mean()),
            'volatility': float(self.returns.std())
        }


class DataLoader:
    """
    Market data loader and preprocessor.
    
    Features:
    - Fetch data from Yahoo Finance
    - Generate synthetic data for testing
    - Calculate various return types
    - Handle missing data
    """
    
    def __init__(self, use_cache: bool = True):
        """
        Initialize data loader.
        
        Parameters:
            use_cache: Whether to cache fetched data
        """
        self.use_cache = use_cache
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def fetch_yahoo(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = '1d'
    ) -> MarketData:
        """
        Fetch data from Yahoo Finance.
        
        Parameters:
            symbol: Ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            interval: Data interval ('1d', '1wk', '1mo')
        
        Returns:
            MarketData object
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance required. Install with: pip install yfinance")
        
        cache_key = f"{symbol}_{start_date}_{end_date}_{interval}"
        
        if self.use_cache and cache_key in self._cache:
            prices = self._cache[cache_key]
        else:
            ticker = yf.Ticker(symbol)
            prices = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if self.use_cache:
                self._cache[cache_key] = prices
        
        if prices.empty:
            raise ValueError(f"No data found for {symbol}")
        
        # Calculate returns
        returns = prices['Close'].pct_change().dropna()
        
        return MarketData(
            prices=prices,
            returns=returns,
            symbol=symbol,
            start_date=prices.index[0].to_pydatetime() if hasattr(prices.index[0], 'to_pydatetime') else datetime.now(),
            end_date=prices.index[-1].to_pydatetime() if hasattr(prices.index[-1], 'to_pydatetime') else datetime.now()
        )
    
    def generate_synthetic(
        self,
        n_samples: int = 1000,
        n_regimes: int = 3,
        regime_params: Optional[List[Dict]] = None,
        random_state: int = 42
    ) -> MarketData:
        """
        Generate synthetic data with known regimes.
        
        Parameters:
            n_samples: Total number of samples
            n_regimes: Number of regimes
            regime_params: List of dicts with 'mean', 'std', 'duration'
            random_state: Random seed
        
        Returns:
            MarketData object with synthetic data
        """
        np.random.seed(random_state)
        
        if regime_params is None:
            # Default regime parameters
            if n_regimes == 2:
                regime_params = [
                    {'mean': -0.001, 'std': 0.025, 'duration': 50},  # Bear
                    {'mean': 0.001, 'std': 0.012, 'duration': 100}   # Bull
                ]
            else:
                regime_params = [
                    {'mean': -0.002, 'std': 0.030, 'duration': 40},   # Bear
                    {'mean': 0.0003, 'std': 0.015, 'duration': 80},   # Neutral
                    {'mean': 0.001, 'std': 0.010, 'duration': 100}    # Bull
                ]
        
        returns = []
        true_regimes = []
        current_idx = 0
        
        while current_idx < n_samples:
            for regime_id, params in enumerate(regime_params):
                if current_idx >= n_samples:
                    break
                
                # Vary duration slightly
                duration = int(params['duration'] * np.random.uniform(0.7, 1.3))
                duration = min(duration, n_samples - current_idx)
                
                # Generate returns for this regime
                regime_returns = np.random.normal(params['mean'], params['std'], duration)
                returns.extend(regime_returns)
                true_regimes.extend([regime_id] * duration)
                current_idx += duration
        
        returns = np.array(returns[:n_samples])
        true_regimes = np.array(true_regimes[:n_samples])
        
        # Create price series from returns
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Create DataFrame
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        price_df = pd.DataFrame({
            'Open': prices * 0.999,
            'High': prices * 1.005,
            'Low': prices * 0.995,
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, n_samples),
            'True_Regime': true_regimes
        }, index=dates)
        
        return_series = pd.Series(returns, index=dates, name='returns')
        
        return MarketData(
            prices=price_df,
            returns=return_series,
            symbol='SYNTHETIC',
            start_date=dates[0].to_pydatetime(),
            end_date=dates[-1].to_pydatetime()
        )
    
    def calculate_returns(
        self,
        prices: pd.Series,
        method: str = 'simple'
    ) -> pd.Series:
        """
        Calculate returns from prices.
        
        Parameters:
            prices: Price series
            method: 'simple' or 'log'
        
        Returns:
            Return series
        """
        if method == 'log':
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()
        
        return returns.dropna()
    
    def load_csv(
        self,
        filepath: str,
        date_column: str = 'Date',
        price_column: str = 'Close'
    ) -> MarketData:
        """
        Load data from CSV file.
        
        Parameters:
            filepath: Path to CSV file
            date_column: Name of date column
            price_column: Name of price column
        
        Returns:
            MarketData object
        """
        df = pd.read_csv(filepath, parse_dates=[date_column])
        df.set_index(date_column, inplace=True)
        df.sort_index(inplace=True)
        
        returns = df[price_column].pct_change().dropna()
        
        return MarketData(
            prices=df,
            returns=returns,
            symbol='CSV_DATA',
            start_date=df.index[0].to_pydatetime() if hasattr(df.index[0], 'to_pydatetime') else datetime.now(),
            end_date=df.index[-1].to_pydatetime() if hasattr(df.index[-1], 'to_pydatetime') else datetime.now()
        )


def get_sample_data(symbol: str = 'SPY', years: int = 5) -> MarketData:
    """
    Get sample market data for testing.
    
    Parameters:
        symbol: Ticker symbol
        years: Number of years of history
    
    Returns:
        MarketData object
    """
    loader = DataLoader()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    try:
        return loader.fetch_yahoo(
            symbol=symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
    except Exception:
        # Fall back to synthetic data
        return loader.generate_synthetic(n_samples=years * 252)


if __name__ == "__main__":
    print("Testing Data Loader...")
    
    loader = DataLoader()
    
    # Generate synthetic data
    data = loader.generate_synthetic(n_samples=500, n_regimes=3)
    
    print(f"\nSynthetic Data:")
    print(f"  Symbol: {data.symbol}")
    print(f"  Samples: {len(data.returns)}")
    print(f"  Mean Return: {data.returns.mean():.4f}")
    print(f"  Volatility: {data.returns.std():.4f}")
    print(f"  Date Range: {data.start_date.date()} to {data.end_date.date()}")
    
    # Try fetching real data
    try:
        real_data = loader.fetch_yahoo('SPY', '2020-01-01', '2023-12-31')
        print(f"\nReal Data (SPY):")
        print(f"  Samples: {len(real_data.returns)}")
        print(f"  Mean Return: {real_data.returns.mean():.4f}")
        print(f"  Volatility: {real_data.returns.std():.4f}")
    except Exception as e:
        print(f"\nCould not fetch real data: {e}")
