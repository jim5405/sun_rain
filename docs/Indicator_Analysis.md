# Technical Indicator Analysis and Improvement Plan

## 1. Current Indicator Logic

The current system uses a combination of Trend, Momentum, and Volatility indicators to determine market status ("Barometer") and trading signals ("Recovery").

### Indicators Used
1.  **Moving Averages (MA)**
    *   **Short MA** (default 50d): Short-term trend.
    *   **Long MA** (default 200d): Long-term trend.
    *   **Usage**: Alignment of Price, Short MA, and Long MA determines the "Weather" (Sunny, Rainy, etc.).
2.  **Relative Strength Index (RSI)**
    *   **Usage**: Measures momentum. Used to confirm "Sunny" days (RSI > 55) and identify "Rainy" days (RSI < 45).
3.  **MACD (Moving Average Convergence Divergence)**
    *   **Usage**: Momentum and trend reversal. Positive Histogram is a prerequisite for the "Recovery" signal.
4.  **Drawdown**
    *   **Usage**: Measures decline from peak. Used to identify "oversold" conditions relative to recent highs.
5.  **ADX (Average Directional Index)**
    *   **Usage**: Measures trend strength. Used to filter "Recovery" signals (ADX > 20) to ensure the recovery has strength.

### Decision Logic

#### Market Barometer (Status)
*   **☀️ Sunny**: Price > Short MA > Long MA AND RSI > Bull Threshold.
*   **🌥️ Cloudy**: Price > Short MA > Long MA (but RSI weak).
*   **☁️ Overcast**: Mixed signals (e.g., MA crossover pending).
*   **🌧️ Rainy**: Short MA > Price AND Long MA > Price.

#### Recovery Signal (Buy Signal)
*   **拨雲見日 (Clearing Up)**:
    *   Drawdown is significant (but not too deep).
    *   Drawdown is improving (Price rising).
    *   MACD Histogram > 0 (Momentum turning positive).
    *   ADX > Threshold (Trend is strong enough).
    *   DI+ > DI- (Direction is Up).

## 2. Analysis & Missing Elements

The current logic is robust for trend following and mean reversion, but it lacks **Volume Confirmation**.

*   **Weakness**: A price breakout or recovery on low volume is often a false signal (bull trap).
*   **Weakness**: "Sunny" days with declining volume may indicate trend exhaustion.

## 3. Improvement Plan

### Improvement 1: Volume Confirmation (Priority: High)
Integrate Volume Moving Average (VMA) to validate price moves.

*   **Logic Change**:
    *   **Sunny**: Requires Volume > VMA (or at least not significantly lower) to confirm strong trend.
    *   **Recovery**: Requires Volume spike (Volume > VMA * 1.2) or steady increase to confirm the reversal.
*   **Implementation**:
    *   Calculate `Vol_MA` (e.g., 20-day average volume).
    *   Update `get_recovery_status` to require `Volume > Vol_MA` for a high-confidence signal.

### Improvement 2: ATR (Average True Range) for Dynamic Stop Loss (Priority: Medium)
Currently, the system doesn't suggest explicit stop-loss levels.

*   **Logic Change**:
    *   Calculate ATR.
    *   Suggest Stop Loss at `Price - 2 * ATR`.

## 4. Execution Plan (Volume Confirmation)

1.  **Update `core/scan_module.py`**:
    *   Add `Vol_MA` calculation in `calculate_indicators`.
    *   Update `get_recovery_status` to check `Volume > Vol_MA`.
2.  **Update `tools/stock_analysis.py`**:
    *   Add Volume chart to the visualization.
3.  **Verify**:
    *   Run `judge_stock.py` on known stocks to see if signals change.
