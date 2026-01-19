"""
Visualization Module
====================
Charts and dashboards for regime detection analysis.

Supports:
- Regime timeline plots
- Probability heatmaps
- Transition matrices
- Performance charts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def plot_regime_timeline(
    regimes: pd.Series,
    prices: Optional[pd.Series] = None,
    regime_names: Optional[Dict[int, str]] = None,
    figsize: tuple = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot regime timeline with optional price overlay.
    
    Parameters:
        regimes: Series of regime labels
        prices: Optional price series to overlay
        regime_names: Mapping of regime ID to name
        figsize: Figure size
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])
    
    # Regime colors
    n_regimes = regimes.nunique()
    colors = plt.cm.Set2(np.linspace(0, 1, n_regimes))
    regime_colors = {i: colors[i] for i in range(n_regimes)}
    
    # Price plot with regime background
    ax1 = axes[0]
    if prices is not None:
        ax1.plot(prices.index, prices.values, 'k-', linewidth=1, label='Price')
    
    # Add regime backgrounds
    unique_regimes = sorted(regimes.unique())
    for regime in unique_regimes:
        mask = regimes == regime
        regime_dates = regimes.index[mask]
        
        name = regime_names.get(regime, f'Regime {regime}') if regime_names else f'Regime {regime}'
        
        for i, date in enumerate(regime_dates):
            if i == 0:
                ax1.axvspan(date, date + pd.Timedelta(days=1), 
                           alpha=0.3, color=regime_colors[regime], label=name)
            else:
                ax1.axvspan(date, date + pd.Timedelta(days=1), 
                           alpha=0.3, color=regime_colors[regime])
    
    ax1.set_title('Price with Regime Overlay', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Regime timeline
    ax2 = axes[1]
    regime_numeric = regimes.values
    ax2.fill_between(regimes.index, 0, 1, alpha=0.3, color='gray')
    
    for regime in unique_regimes:
        mask = regimes == regime
        ax2.fill_between(regimes.index, 0, 1, where=mask, 
                        alpha=0.7, color=regime_colors[regime])
    
    ax2.set_title('Regime Timeline', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Regime', fontsize=11)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_regime_probabilities(
    regime_probs: pd.DataFrame,
    regime_names: Optional[Dict[int, str]] = None,
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot regime probability evolution over time.
    
    Parameters:
        regime_probs: DataFrame with regime probabilities
        regime_names: Mapping of regime ID to name
        figsize: Figure size
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_regimes = len(regime_probs.columns)
    colors = plt.cm.Set2(np.linspace(0, 1, n_regimes))
    
    # Stack plot
    bottom = np.zeros(len(regime_probs))
    
    for i, col in enumerate(regime_probs.columns):
        name = f'Regime {i}'
        if regime_names:
            name = regime_names.get(i, name)
        
        ax.fill_between(regime_probs.index, bottom, bottom + regime_probs[col].values,
                       alpha=0.7, color=colors[i], label=name)
        bottom += regime_probs[col].values
    
    ax.set_title('Regime Probability Evolution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Probability', fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_transition_matrix(
    transition_matrix: np.ndarray,
    regime_names: Optional[List[str]] = None,
    figsize: tuple = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot transition probability matrix as heatmap.
    
    Parameters:
        transition_matrix: Transition probability matrix
        regime_names: List of regime names
        figsize: Figure size
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_regimes = transition_matrix.shape[0]
    if regime_names is None:
        regime_names = [f'Regime {i}' for i in range(n_regimes)]
    
    # Heatmap
    sns.heatmap(
        transition_matrix,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=regime_names,
        yticklabels=regime_names,
        ax=ax,
        vmin=0,
        vmax=1
    )
    
    ax.set_title('Regime Transition Probabilities', fontsize=14, fontweight='bold')
    ax.set_xlabel('To Regime', fontsize=11)
    ax.set_ylabel('From Regime', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_regime_statistics(
    regime_stats: List[Dict],
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot regime statistics comparison.
    
    Parameters:
        regime_stats: List of regime statistics dicts
        figsize: Figure size
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    names = [s.get('name', f"Regime {s.get('regime_id', i)}") for i, s in enumerate(regime_stats)]
    colors = plt.cm.Set2(np.linspace(0, 1, len(regime_stats)))
    
    # Mean return
    ax1 = axes[0]
    returns = [s.get('mean_return', 0) * 100 for s in regime_stats]
    bars1 = ax1.bar(names, returns, color=colors, edgecolor='black', linewidth=0.5)
    ax1.axhline(0, color='black', linewidth=1)
    ax1.set_title('Mean Return (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Daily Return %', fontsize=10)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, val in zip(bars1, returns):
        ax1.annotate(f'{val:.3f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    
    # Volatility
    ax2 = axes[1]
    vols = [s.get('volatility', 0) * 100 for s in regime_stats]
    bars2 = ax2.bar(names, vols, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_title('Volatility (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Daily Vol %', fontsize=10)
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars2, vols):
        ax2.annotate(f'{val:.2f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    
    # Frequency
    ax3 = axes[2]
    freqs = [s.get('frequency', 0) * 100 for s in regime_stats]
    bars3 = ax3.bar(names, freqs, color=colors, edgecolor='black', linewidth=0.5)
    ax3.set_title('Frequency (%)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Time in Regime %', fontsize=10)
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars3, freqs):
        ax3.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_signal_performance(
    signal_history: pd.DataFrame,
    cumulative_returns: pd.Series,
    figsize: tuple = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot signal history and performance.
    
    Parameters:
        signal_history: DataFrame with signal history
        cumulative_returns: Cumulative strategy returns
        figsize: Figure size
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Cumulative returns
    ax1 = axes[0]
    ax1.plot(cumulative_returns.index, cumulative_returns.values, 'b-', linewidth=1.5)
    ax1.fill_between(cumulative_returns.index, 0, cumulative_returns.values, 
                    where=cumulative_returns.values >= 0, alpha=0.3, color='green')
    ax1.fill_between(cumulative_returns.index, 0, cumulative_returns.values,
                    where=cumulative_returns.values < 0, alpha=0.3, color='red')
    ax1.axhline(0, color='black', linewidth=1)
    ax1.set_title('Cumulative Strategy Returns', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Return', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Position sizes
    ax2 = axes[1]
    if 'position_size' in signal_history.columns:
        positions = signal_history['position_size']
        colors = ['green' if p > 0 else 'red' if p < 0 else 'gray' for p in positions]
        ax2.bar(positions.index, positions.values, color=colors, alpha=0.7, width=1)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_title('Position Sizes', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Position', fontsize=11)
    ax2.set_ylim(-1.2, 1.2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_regime_dashboard(
    regimes: pd.Series,
    regime_probs: pd.DataFrame,
    prices: pd.Series,
    transition_matrix: np.ndarray,
    regime_names: Optional[Dict[int, str]] = None,
    figsize: tuple = (16, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive regime analysis dashboard.
    
    Parameters:
        regimes: Series of regime labels
        regime_probs: DataFrame of regime probabilities
        prices: Price series
        transition_matrix: Transition matrix
        regime_names: Mapping of regime ID to name
        figsize: Figure size
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Grid layout
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.2)
    
    # Get regime info
    n_regimes = len(regime_probs.columns)
    colors = plt.cm.Set2(np.linspace(0, 1, n_regimes))
    names = regime_names or {i: f'Regime {i}' for i in range(n_regimes)}
    
    # 1. Price with regime overlay (top, full width)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(prices.index, prices.values, 'k-', linewidth=1, label='Price')
    
    for regime in sorted(regimes.unique()):
        mask = regimes == regime
        regime_dates = regimes.index[mask]
        name = names.get(regime, f'Regime {regime}')
        
        for i, date in enumerate(regime_dates):
            if i == 0:
                ax1.axvspan(date, date + pd.Timedelta(days=1),
                           alpha=0.3, color=colors[regime], label=name)
            else:
                ax1.axvspan(date, date + pd.Timedelta(days=1),
                           alpha=0.3, color=colors[regime])
    
    ax1.set_title('Price with Regime Detection', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Probability evolution (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    bottom = np.zeros(len(regime_probs))
    
    for i, col in enumerate(regime_probs.columns):
        name = names.get(i, f'Regime {i}')
        ax2.fill_between(regime_probs.index, bottom, bottom + regime_probs[col].values,
                        alpha=0.7, color=colors[i], label=name)
        bottom += regime_probs[col].values
    
    ax2.set_title('Regime Probabilities', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Probability', fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Transition matrix (middle right)
    ax3 = fig.add_subplot(gs[1, 1])
    name_list = [names.get(i, f'Regime {i}') for i in range(n_regimes)]
    
    sns.heatmap(
        transition_matrix,
        annot=True,
        fmt='.1%',
        cmap='Blues',
        xticklabels=name_list,
        yticklabels=name_list,
        ax=ax3,
        vmin=0,
        vmax=1
    )
    ax3.set_title('Transition Matrix', fontsize=12, fontweight='bold')
    ax3.set_xlabel('To', fontsize=10)
    ax3.set_ylabel('From', fontsize=10)
    
    # 4. Regime distribution (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    regime_counts = regimes.value_counts().sort_index()
    bar_colors = [colors[i] for i in regime_counts.index]
    bar_names = [names.get(i, f'Regime {i}') for i in regime_counts.index]
    
    ax4.pie(regime_counts.values, labels=bar_names, autopct='%1.1f%%',
           colors=bar_colors, explode=[0.02] * len(regime_counts))
    ax4.set_title('Time in Each Regime', fontsize=12, fontweight='bold')
    
    # 5. Current status (bottom right)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    current_regime = int(regimes.iloc[-1])
    current_name = names.get(current_regime, f'Regime {current_regime}')
    current_prob = regime_probs.iloc[-1, current_regime]
    
    status_text = (
        f"Current Status\n"
        f"{'─' * 30}\n\n"
        f"Current Regime: {current_name}\n"
        f"Confidence: {current_prob:.1%}\n"
        f"Total Periods: {len(regimes)}\n"
        f"Unique Regimes: {n_regimes}\n"
    )
    
    ax5.text(0.1, 0.9, status_text, transform=ax5.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    print("Visualization module loaded successfully.")
    print("\nAvailable functions:")
    print("  - plot_regime_timeline()")
    print("  - plot_regime_probabilities()")
    print("  - plot_transition_matrix()")
    print("  - plot_regime_statistics()")
    print("  - plot_signal_performance()")
    print("  - plot_regime_dashboard()")
