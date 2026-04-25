"""
NIFTY Stock Portfolio Optimiser + Price Predictor
Full-featured Indian stock market analytics dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(
    page_title="NIFTY Portfolio Optimiser",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# NIFTY 50 const
# ---------------------------
NIFTY_50_SYMBOLS = [
    "^NSEI",  # NIFTY 50 index itself
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "BAJFINANCE.NS",
    "ITC.NS", "WIPRO.NS", "AXISBANK.NS", "HCLTECH.NS", "LT.NS",
    "SUNPHARMA.NS", "TITAN.NS", "MARUTI.NS", "ASIANPAINT.NS", "NESTLEIND.NS",
    "ULTRACEMCO.NS", "TECHM.NS", "POWERGRID.NS", "M&M.NS", "BAJAJFINSV.NS",
    "TATAMOTORS.NS", "GRASIM.NS", "NTPC.NS", "INDUSINDBK.NS", "ONGC.NS",
    "DIVISLAB.NS", "DRREDDY.NS", "HDFCLIFE.NS", "BRITANNIA.NS", "CIPLA.NS",
    "SBILIFE.NS", "JSWSTEEL.NS", "ADANIPORTS.NS", "COALINDIA.NS", "EICHERMOT.NS",
    "HEROMOTOCO.NS", "BPCL.NS", "TATASTEEL.NS", "HDFC.NS", "SHREECEM.NS",
    "UPL.NS", "BAJAJ-AUTO.NS", "TATACONSUM.NS", "HINDALCO.NS"
]

# ---------------------------
# Helper functions for indicators
# ---------------------------
def fetch_stock_data(symbols, period="1y"):
    """Fetch historical data for multiple symbols"""
    try:
        if len(symbols) == 1:
            stock = yf.Ticker(symbols[0])
            df = stock.history(period=period)
            if df.empty:
                st.error(f"No data found for {symbols[0]}")
                return None
            df.index = pd.to_datetime(df.index)
            return df
        else:
            data = {}
            for symbol in symbols:
                try:
                    stock = yf.Ticker(symbol)
                    df = stock.history(period=period)
                    if not df.empty:
                        df.index = pd.to_datetime(df.index)
                        data[symbol] = df['Close']
                except:
                    continue
            if not data:
                st.error("No data found for any symbol")
                return None
            df = pd.DataFrame(data)
            return df
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None


def compute_sma(data, window):
    """Simple Moving Average"""
    return data.rolling(window=window).mean()


def compute_ema(data, span):
    """Exponential Moving Average"""
    return data.ewm(span=span, adjust=False).mean()


def compute_macd(data, fast=12, slow=26, signal=9):
    """MACD line, signal line, histogram"""
    ema_fast = compute_ema(data, fast)
    ema_slow = compute_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_rsi(data, window=14):
    """Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_bollinger_bands(data, window=20, num_std=2):
    """Bollinger Bands"""
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return sma, upper, lower


def compute_atr(high, low, close, window=14):
    """Average True Range (Volatility)"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr


def compute_adx(high, low, close, window=14):
    """Average Directional Index (Trend Strength)"""
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    atr = compute_atr(high, low, close, window)
    
    plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
    minus_di = 100 * (abs(minus_dm).rolling(window=window).mean() / atr)
    
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=window).mean()
    
    return adx, plus_di, minus_di


def compute_obv(close, volume):
    """On-Balance Volume"""
    obv = np.zeros(len(close))
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv[i] = obv[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv[i] = obv[i-1] - volume.iloc[i]
        else:
            obv[i] = obv[i-1]
    return pd.Series(obv, index=close.index)


def compute_stochastic(high, low, close, k_window=14, d_window=3):
    """Stochastic Oscillator"""
    low_min = low.rolling(window=k_window).min()
    high_max = high.rolling(window=k_window).max()
    stoch_k = 100 * (close - low_min) / (high_max - low_min)
    stoch_d = stoch_k.rolling(window=d_window).mean()
    return stoch_k, stoch_d


def compute_ichimoku(high, low, close, conversion=9, base=26, lagging=52):
    """Ichimoku Cloud components"""
    conversion_line = (high.rolling(window=conversion).max() + low.rolling(window=conversion).min()) / 2
    base_line = (high.rolling(window=base).max() + low.rolling(window=base).min()) / 2
    leading_span_a = ((conversion_line + base_line) / 2).shift(base)
    leading_span_b = ((high.rolling(window=lagging).max() + low.rolling(window=lagging).min()) / 2).shift(base)
    lagging_span = close.shift(-base)
    return conversion_line, base_line, leading_span_a, leading_span_b, lagging_span


def compute_fibonacci_levels(high, low):
    """Fibonacci retracement levels"""
    diff = high - low
    levels = {
        '0%': low,
        '23.6%': low + diff * 0.236,
        '38.2%': low + diff * 0.382,
        '50%': low + diff * 0.5,
        '61.8%': low + diff * 0.618,
        '78.6%': low + diff * 0.786,
        '100%': high
    }
    return levels


def compute_risk_metrics(returns):
    """Compute comprehensive risk metrics"""
    # Sharpe ratio (assuming 6% risk-free rate for Indian market)
    risk_free_rate = 0.06
    excess_returns = returns - risk_free_rate / 252
    sharpe = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
    
    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    sortino = np.sqrt(252) * returns.mean() / downside_returns.std() if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
    
    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Value at Risk (95% confidence)
    var_95 = np.percentile(returns, 5) * np.sqrt(252)
    
    # Conditional Value at Risk (Expected Shortfall)
    cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * np.sqrt(252)
    
    return {
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_drawdown,
        'VaR (95%)': var_95,
        'CVaR (95%)': cvar_95,
        'Annualized Volatility': returns.std() * np.sqrt(252),
        'Annualized Return': (1 + returns.mean()) ** 252 - 1
    }


def compute_signal_score(data_dict):
    """Compute combined signal score from multiple indicators"""
    score = 0
    reasons = []
    
    # RSI Signal
    rsi = data_dict.get('rsi', 50)
    if rsi < 30:
        score += 25
        reasons.append(f"RSI {rsi:.1f} < 30 — Oversold (BUY Signal)")
    elif rsi > 70:
        score -= 25
        reasons.append(f"RSI {rsi:.1f} > 70 — Overbought (SELL Signal)")
    elif rsi < 40:
        score += 10
        reasons.append(f"RSI {rsi:.1f} < 40 — Approaching oversold")
    elif rsi > 60:
        score -= 10
        reasons.append(f"RSI {rsi:.1f} > 60 — Approaching overbought")
    
    # MACD Signal
    macd = data_dict.get('macd_hist', 0)
    macd_prev = data_dict.get('macd_hist_prev', 0)
    if macd > 0 and macd_prev <= 0:
        score += 30
        reasons.append("MACD Histogram turns positive — Bullish Crossover")
    elif macd < 0 and macd_prev >= 0:
        score -= 30
        reasons.append("MACD Histogram turns negative — Bearish Crossover")
    elif macd > 0:
        score += 10
        reasons.append("MACD Histogram positive — Bullish momentum")
    elif macd < 0:
        score -= 10
        reasons.append("MACD Histogram negative — Bearish momentum")
    
    # Bollinger Bands Signal
    bb_position = data_dict.get('bb_position', 0.5)
    if bb_position < 0:
        score += 20
        reasons.append("Price below Lower Bollinger Band — Oversold bounce possible")
    elif bb_position > 1:
        score -= 20
        reasons.append("Price above Upper Bollinger Band — Overbrought pullback possible")
    
    # Moving Average Crossover
    ma_cross = data_dict.get('ma_cross', 0)
    if ma_cross == 1:
        score += 15
        reasons.append("Golden Cross (50MA > 200MA) — Bullish trend")
    elif ma_cross == -1:
        score -= 15
        reasons.append("Death Cross (50MA < 200MA) — Bearish trend")
    
    # Volume confirmation
    volume_ratio = data_dict.get('volume_ratio', 1)
    if volume_ratio > 1.5:
        if score > 0:
            score += 15
            reasons.append(f"Volume {volume_ratio:.1f}x average — Volume confirmation")
        elif score < 0:
            score -= 10
            reasons.append(f"Volume {volume_ratio:.1f}x average — Selling pressure")
    
    # ATR Trend Strength
    atr = data_dict.get('atr', 0)
    if atr > data_dict.get('avg_atr', 0) * 1.5:
        if score > 0:
            score += 10
            reasons.append("High volatility — Strong trend")
    
    return score, reasons


# ---------------------------
# Initialize session state
# ---------------------------
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=['Symbol', 'Allocation %'])
if 'selected_stocks' not in st.session_state:
    st.session_state.selected_stocks = NIFTY_50_SYMBOLS[:10]
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = {}

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("📊 NIFTY Portfolio Optimiser")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Dashboard", "📈 Portfolio Builder", "🔬 Stock Analysis", "🎯 Signal Scanner", "📊 Risk Analytics"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Source**: Yahoo Finance (NSE)")
st.sidebar.markdown("**NIFTY 50 Universe**: 50 stocks")
st.sidebar.markdown("---")
st.sidebar.caption("Built for Indian Markets | Real-time Data")


# ---------------------------
# Page 1: Dashboard
# ---------------------------
if page == "🏠 Dashboard":
    st.title("🏠 NIFTY Market Dashboard")
    st.markdown("Real-time overview of NIFTY 50 and your portfolio performance")
    
    # Fetch NIFTY 50 Index Data
    nifty_data = fetch_stock_data(["^NSEI"], period="1mo")
    if nifty_data is not None and not nifty_data.empty:
        col1, col2, col3, col4 = st.columns(4)
        current_nifty = nifty_data['Close'].iloc[-1]
        prev_nifty = nifty_data['Close'].iloc[-2]
        nifty_change = ((current_nifty - prev_nifty) / prev_nifty) * 100
        
        col1.metric("NIFTY 50", f"{current_nifty:,.2f}", delta=f"{nifty_change:.2f}%")
        
        # Market breadth (Approximated)
        nifty_close = nifty_data['Close']
        nifty_ma20 = compute_sma(nifty_close, 20)
        nifty_ma50 = compute_sma(nifty_close, 50)
        
        if nifty_close.iloc[-1] > nifty_ma50.iloc[-1]:
            col2.metric("Market Trend", "Bullish", delta="Above 50 DMA")
            breadth = "Positive"
        else:
            col2.metric("Market Trend", "Bearish", delta="Below 50 DMA")
            breadth = "Negative"
        
        col3.metric("20-Day SMA", f"{nifty_ma20.iloc[-1]:,.2f}")
        col4.metric("50-Day SMA", f"{nifty_ma50.iloc[-1]:,.2f}")
        
        # NIFTY Price Chart
        st.subheader("NIFTY 50 Price Movement")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=nifty_data.index, y=nifty_data['Close'], mode='lines', name='NIFTY 50', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=nifty_data.index, y=nifty_ma20, mode='lines', name='20-Day SMA', line=dict(color='orange', dash='dash')))
        fig.add_trace(go.Scatter(x=nifty_data.index, y=nifty_ma50, mode='lines', name='50-Day SMA', line=dict(color='red', dash='dash')))
        fig.update_layout(title='NIFTY 50 Historical Prices with Moving Averages', xaxis_title='Date', yaxis_title='Price (INR)', hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Page 2: Portfolio Builder
# ---------------------------
elif page == "📈 Portfolio Builder":
    st.title("📈 Portfolio Builder & Optimisation")
    st.markdown("Build your portfolio using NIFTY 50 stocks and optimise allocation")
    
    # Stock selector
    selected_symbols = st.multiselect("Select Stocks for Portfolio", NIFTY_50_SYMBOLS, default=NIFTY_50_SYMBOLS[:10])
    
    if len(selected_symbols) >= 2:
        period = st.selectbox("Analysis Period", ["6mo", "1y", "2y"], index=1)
        
        if st.button("Optimise Portfolio"):
            with st.spinner("Fetching Data & Optimising Portfolio..."):
                # Fetch Returns data
                returns_data = fetch_stock_data(selected_symbols, period=period)
                
                if returns_data is not None and not returns_data.empty:
                    # Calculate Returns
                    returns = returns_data.pct_change().dropna()
                    
                    # Monte Carlo Simulation for Efficient Frontier
                    num_portfolios = 10000
                    results = np.zeros((4, num_portfolios))
                    weights_record = []
                    
                    for i in range(num_portfolios):
                        weights = np.random.random(len(selected_symbols))
                        weights /= np.sum(weights)
                        weights_record.append(weights)
                        
                        port_return = np.sum(returns.mean() * weights) * 252
                        port_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
                        sharpe = port_return / port_std
                        
                        results[0, i] = port_return
                        results[1, i] = port_std
                        results[2, i] = sharpe
                        results[3, i] = np.sum((port_return / port_std))
                    
                    # Find Maximum Sharpe Ratio
                    max_sharpe_idx = np.argmax(results[2])
                    optimal_weights = weights_record[max_sharpe_idx]
                    
                    # Display Results
                    st.subheader("📊 Optimal Portfolio Allocation")
                    opt_df = pd.DataFrame({
                        'Stock': selected_symbols,
                        'Optimal Weight %': optimal_weights * 100
                    }).sort_values('Optimal Weight %', ascending=False)
                    
                    st.dataframe(opt_df.style.format({'Optimal Weight %': '{:.2f}%'}), use_container_width=True)
                    
                    # Pie Chart
                    fig_pie = px.pie(opt_df, values='Optimal Weight %', names='Stock', title='Optimal Portfolio Allocation')
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Efficient Frontier Scatter
                    st.subheader("Efficient Frontier")
                    fig_frontier = go.Figure()
                    fig_frontier.add_trace(go.Scatter(
                        x=results[1,:], y=results[0,:],
                        mode='markers',
                        marker=dict(color=results[2,:], colorscale='RdYlGn', showscale=True),
                        text=[f'Sharpe: {sharpe:.2f}' for sharpe in results[2,:]],
                        name='Random Portfolios'
                    ))
                    fig_frontier.add_trace(go.Scatter(
                        x=[results[1, max_sharpe_idx]], y=[results[0, max_sharpe_idx]],
                        mode='markers',
                        marker=dict(color='red', size=15),
                        name='Max Sharpe Portfolio'
                    ))
                    fig_frontier.update_layout(title='Efficient Frontier (10,000 Simulations)', xaxis_title='Risk (Std Dev)', yaxis_title='Expected Return')
                    st.plotly_chart(fig_frontier, use_container_width=True)
                    
                    # Save to session state
                    st.session_state.portfolio = opt_df
     
    elif len(selected_symbols) == 1:
        st.warning("Please select at least 2 stocks for portfolio optimisation")
        
        if st.button("Add Single Stock to Portfolio"):
            st.session_state.portfolio = pd.DataFrame({
                'Stock': selected_symbols,
                'Optimal Weight %': 100
            })
            st.success(f"Added {selected_symbols[0]} to portfolio")
    else:
        st.info("Please select stocks from the NIFTY 50 universe")

# ---------------------------
# Page 3: Stock Analysis
# ---------------------------
elif page == "🔬 Stock Analysis":
    st.title("🔬 Advanced Stock Analysis")
    st.markdown("Comprehensive technical analysis for any NIFTY stock")
    
    selected_stock = st.selectbox("Select Stock for Analysis", NIFTY_50_SYMBOLS, index=1)
    period = st.selectbox("Analysis Period", ["3mo", "6mo", "1y", "2y"], index=2)
    
    if selected_stock:
        data = fetch_stock_data([selected_stock], period=period)
        
        if data is not None and not data.empty:
            # Calculate Indicators
            data['SMA20'] = compute_sma(data['Close'], 20)
            data['SMA50'] = compute_sma(data['Close'], 50)
            data['SMA200'] = compute_sma(data['Close'], 200)
            data['RSI'] = compute_rsi(data['Close'])
            macd_line, signal_line, macd_hist = compute_macd(data['Close'])
            data['MACD'] = macd_line
            data['MACD_Signal'] = signal_line
            data['MACD_Hist'] = macd_hist
            bb_middle, bb_upper, bb_lower = compute_bollinger_bands(data['Close'])
            data['BB_Upper'] = bb_upper
            data['BB_Lower'] = bb_lower
            data['ATR'] = compute_atr(data['High'], data['Low'], data['Close'])
            data['OBV'] = compute_obv(data['Close'], data['Volume'])
            stoch_k, stoch_d = compute_stochastic(data['High'], data['Low'], data['Close'])
            data['Stoch_K'] = stoch_k
            data['Stoch_D'] = stoch_d
            
            # Header Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"₹{data['Close'].iloc[-1]:.2f}")
            with col2:
                change = ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
                st.metric("Daily Change", f"{change:.2f}%", delta=f"{change:.2f}%")
            with col3:
                st.metric("RSI (14)", f"{data['RSI'].iloc[-1]:.1f}")
            with col4:
                st.metric("Volume (Today)", f"{data['Volume'].iloc[-1]:,.0f}")
            
            # Price Chart with BB and MAs
            st.subheader("📉 Price Chart with Bollinger Bands & Moving Averages")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Price'))
            fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], mode='lines', name='BB Upper', line=dict(color='gray', dash='dash')))
            fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], mode='lines', name='BB Lower', line=dict(color='gray', dash='dash')))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], mode='lines', name='SMA20', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], mode='lines', name='SMA50', line=dict(color='orange')))
            fig.update_layout(title=f'{selected_stock} - Price Analysis', xaxis_title='Date', yaxis_title='Price (₹)', hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
            
            # RSI Chart
            st.subheader("📊 Relative Strength Index (RSI)")
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='purple')))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            fig_rsi.update_layout(title='RSI (14-Day)', xaxis_title='Date', yaxis_title='RSI')
            st.plotly_chart(fig_rsi, use_container_width=True)
            
            # MACD Chart
            st.subheader("📈 MACD (Moving Average Convergence Divergence)")
            fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            fig_macd.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'), row=1, col=1)
            fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD', line=dict(color='blue')), row=2, col=1)
            fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], mode='lines', name='Signal', line=dict(color='red')), row=2, col=1)
            fig_macd.add_bar(x=data.index, y=data['MACD_Hist'], name='Histogram', row=2, col=1)
            fig_macd.update_layout(title=f'{selected_stock} - MACD Analysis', height=600)
            st.plotly_chart(fig_macd, use_container_width=True)
            
            # Volatility (ATR) Chart
            st.subheader("📉 Average True Range (ATR) - Market Volatility")
            fig_atr = go.Figure()
            fig_atr.add_trace(go.Scatter(x=data.index, y=data['ATR'], mode='lines', name='ATR', fill='tozeroy', line=dict(color='orange')))
            fig_atr.update_layout(title='ATR (14-Day) - Higher ATR indicates higher volatility', xaxis_title='Date', yaxis_title='ATR')
            st.plotly_chart(fig_atr, use_container_width=True)
            
            # On-Balance Volume Chart
            st.subheader("📊 On-Balance Volume (OBV)")
            fig_obv = go.Figure()
            fig_obv.add_trace(go.Scatter(x=data.index, y=data['OBV'], mode='lines', name='OBV', line=dict(color='green')))
            fig_obv.update_layout(title='OBV - Volume-based momentum indicator', xaxis_title='Date', yaxis_title='OBV')
            st.plotly_chart(fig_obv, use_container_width=True)
            
            # Stochastic Oscillator Chart
            st.subheader("🎲 Stochastic Oscillator")
            fig_stoch = go.Figure()
            fig_stoch.add_trace(go.Scatter(x=data.index, y=data['Stoch_K'], mode='lines', name='%K', line=dict(color='blue')))
            fig_stoch.add_trace(go.Scatter(x=data.index, y=data['Stoch_D'], mode='lines', name='%D', line=dict(color='red')))
            fig_stoch.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig_stoch.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Oversold")
            fig_stoch.update_layout(title='Stochastic Oscillator (14,3)', xaxis_title='Date', yaxis_title='Value')
            st.plotly_chart(fig_stoch, use_container_width=True)
            
            # Fibonacci Levels
            st.subheader("🔢 Fibonacci Retracement Levels (52-Week Range)")
            year_high = data['High'].max()
            year_low = data['Low'].min()
            fib_levels = compute_fibonacci_levels(year_high, year_low)
            fib_df = pd.DataFrame(list(fib_levels.items()), columns=['Level', 'Price (₹)'])
            st.dataframe(fib_df.style.format({'Price (₹)': '₹{:.2f}'}), use_container_width=True)
            
            # Store in session state for signal scanner
            st.session_state.analysis_data[selected_stock] = {
                'rsi': data['RSI'].iloc[-1],
                'macd_hist': data['MACD_Hist'].iloc[-1],
                'macd_hist_prev': data['MACD_Hist'].iloc[-2] if len(data) > 1 else 0,
                'close': data['Close'].iloc[-1],
                'bb_upper': data['BB_Upper'].iloc[-1],
                'bb_lower': data['BB_Lower'].iloc[-1],
                'sma50': data['SMA50'].iloc[-1],
                'sma200': data['SMA200'].iloc[-1] if 'SMA200' in data.columns else data['SMA50'].iloc[-1],
                'volume': data['Volume'].iloc[-1],
                'avg_volume': data['Volume'].rolling(window=20).mean().iloc[-1],
                'atr': data['ATR'].iloc[-1],
                'avg_atr': data['ATR'].rolling(window=20).mean().iloc[-1]
            }

# ---------------------------
# Page 4: Signal Scanner
# ---------------------------
elif page == "🎯 Signal Scanner":
    st.title("🎯 12-Factor High-Conviction Signal Scanner")
    st.markdown("Scores NIFTY 50 stocks based on 12 technical factors to identify top BUY/SELL opportunities")
    
    if st.button("🔄 Run Full Market Scan (50 Stocks)"):
        with st.spinner("Scanning NIFTY 50 stocks (This may take a minute)..."):
            all_scores = []
            
            for symbol in NIFTY_50_SYMBOLS:
                try:
                    # Fetch data for each symbol
                    data = fetch_stock_data([symbol], period="3mo")
                    if data is not None and not data.empty and len(data) > 50:
                        # Calculate indicators
                        rsi = compute_rsi(data['Close'])
                        macd_line, signal_line, macd_hist = compute_macd(data['Close'])
                        bb_middle, bb_upper, bb_lower = compute_bollinger_bands(data['Close'])
                        sma50 = compute_sma(data['Close'], 50)
                        sma200 = compute_sma(data['Close'], 200)
                        atr = compute_atr(data['High'], data['Low'], data['Close'])
                        
                        # BB Position
                        bb_position = (data['Close'].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) if (bb_upper.iloc[-1] - bb_lower.iloc[-1]) > 0 else 0.5
                        
                        # MA Crossover Signal
                        if len(sma50) > 1 and len(sma200) > 1:
                            if sma50.iloc[-1] > sma200.iloc[-1] and sma50.iloc[-2] <= sma200.iloc[-2]:
                                ma_cross = 1  # Golden Cross
                            elif sma50.iloc[-1] < sma200.iloc[-1] and sma50.iloc[-2] >= sma200.iloc[-2]:
                                ma_cross = -1  # Death Cross
                            else:
                                ma_cross = 0
                        else:
                            ma_cross = 0
                        
                        # Volume Ratio
                        avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1] if len(data) > 20 else data['Volume'].mean()
                        volume_ratio = data['Volume'].iloc[-1] / avg_volume if avg_volume > 0 else 1
                        
                        # Prepare data dictionary
                        data_dict = {
                            'rsi': rsi.iloc[-1],
                            'macd_hist': macd_hist.iloc[-1] if not macd_hist.empty else 0,
                            'macd_hist_prev': macd_hist.iloc[-2] if len(macd_hist) > 1 else 0,
                            'bb_position': bb_position - 1,  # Negative = below lower band, Positive = above upper band
                            'ma_cross': ma_cross,
                            'volume_ratio': volume_ratio,
                            'atr': atr.iloc[-1] if not atr.empty else 0,
                            'avg_atr': atr.rolling(window=20).mean().iloc[-1] if len(atr) > 20 else atr.mean()
                        }
                        
                        score, reasons = compute_signal_score(data_dict)
                        
                        all_scores.append({
                            'Symbol': symbol,
                            'Score': score,
                            'Signal': 'STRONG BUY' if score > 30 else ('BUY' if score > 15 else ('NEUTRAL' if -15 <= score <= 15 else ('SELL' if score < -15 else 'STRONG SELL'))),
                            'RSI': f"{rsi.iloc[-1]:.1f}",
                            'Volume Ratio': f"{volume_ratio:.2f}x"
                        })
                except Exception as e:
                    continue
            
            # Sort by score
            if all_scores:
                results_df = pd.DataFrame(all_scores)
                results_df = results_df.sort_values('Score', ascending=False)
                
                # Display Top 5 BUY and Top 5 SELL
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🟢 Top 5 BUY Signals")
                    top_buy = results_df[results_df['Score'] > 15].head(5)
                    if not top_buy.empty:
                        st.dataframe(top_buy.style.apply(lambda x: ['background: #1a3b32' if i < len(top_buy) else '' for i in range(len(x))], axis=0), use_container_width=True)
                    else:
                        st.info("No strong buy signals found")
                
                with col2:
                    st.subheader("🔴 Top 5 SELL Signals")
                    top_sell = results_df[results_df['Score'] < -15].tail(5).sort_values('Score', ascending=True)
                    if not top_sell.empty:
                        st.dataframe(top_sell.style.apply(lambda x: ['background: #3b1a1a' if i < len(top_sell) else '' for i in range(len(x))], axis=0), use_container_width=True)
                    else:
                        st.info("No strong sell signals found")
                
                # Full table
                st.subheader("📋 Complete Signal Scanner Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Market Breadth Gauge
                st.subheader("📊 Market Breadth")
                bullish_count = len(results_df[results_df['Score'] > 15])
                bearish_count = len(results_df[results_df['Score'] < -15])
                neutral_count = len(results_df) - bullish_count - bearish_count
                
                breadth_data = pd.DataFrame({
                    'Sentiment': ['Bullish', 'Bearish', 'Neutral'],
                    'Count': [bullish_count, bearish_count, neutral_count]
                })
                fig_breadth = px.pie(breadth_data, values='Count', names='Sentiment', title='Market Sentiment Distribution', color='Sentiment', color_discrete_map={'Bullish': 'green', 'Bearish': 'red', 'Neutral': 'gray'})
                st.plotly_chart(fig_breadth, use_container_width=True)
                
                st.success(f"Scan complete! Analysed {len(all_scores)} stocks")
            else:
                st.error("Failed to scan stocks. Please check your internet connection.")

# ---------------------------
# Page 5: Risk Analytics
# ---------------------------
elif page == "📊 Risk Analytics":
    st.title("📊 Advanced Portfolio Risk Analytics")
    st.markdown("Comprehensive risk assessment and portfolio optimisation metrics")
    
    if not st.session_state.portfolio.empty:
        portfolio_stocks = st.session_state.portfolio['Stock'].tolist()
        weights = st.session_state.portfolio['Optimal Weight %'].values / 100
        
        period = st.selectbox("Risk Analysis Period", ["1y", "2y", "3y"], index=0)
        
        if st.button("Calculate Risk Metrics"):
            with st.spinner("Fetching data and calculating risk metrics..."):
                # Fetch historical data
                returns_data = fetch_stock_data(portfolio_stocks, period=period)
                
                if returns_data is not None and not returns_data.empty:
                    # Calculate returns
                    returns = returns_data.pct_change().dropna()
                    
                    # Portfolio returns
                    if len(portfolio_stocks) > 1:
                        port_returns = returns.dot(weights)
                    else:
                        port_returns = returns
                    
                    # Calculate comprehensive risk metrics
                    metrics = compute_risk_metrics(port_returns)
                    
                    # Display metrics in columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}", help="Risk-adjusted return (>1 good, >2 excellent)")
                        st.metric("Sortino Ratio", f"{metrics['Sortino Ratio']:.2f}", help="Downside risk-adjusted return")
                        st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}", help="Worst peak-to-trough decline")
                    
                    with col2:
                        st.metric("Annualized Return", f"{metrics['Annualized Return']:.2%}", help="Average yearly return")
                        st.metric("Annualized Volatility", f"{metrics['Annualized Volatility']:.2%}", help="Standard deviation of returns")
                        st.metric("VaR (95%)", f"{metrics['VaR (95%)']:.2%}", help="Maximum expected loss at 95% confidence")
                    
                    with col3:
                        st.metric("CVaR (95%)", f"{metrics['CVaR (95%)']:.2%}", help="Expected loss in worst 5% of cases")
                        
                        # Risk Rating
                        if metrics['Sharpe Ratio'] > 1:
                            risk_rating = "Excellent"
                            rating_color = "🟢"
                        elif metrics['Sharpe Ratio'] > 0.5:
                            risk_rating = "Good"
                            rating_color = "🟡"
                        else:
                            risk_rating = "Poor"
                            rating_color = "🔴"
                        st.metric("Risk Rating", f"{rating_color} {risk_rating}")
                    
                    # Drawdown chart
                    st.subheader("📉 Portfolio Drawdown Analysis")
                    cumulative = (1 + port_returns).cumprod()
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / running_max
                    
                    fig_dd = go.Figure()
                    fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown, mode='lines', name='Drawdown', fill='tozeroy', line=dict(color='red')))
                    fig_dd.update_layout(title='Portfolio Drawdown Over Time', xaxis_title='Date', yaxis_title='Drawdown (%)')
                    st.plotly_chart(fig_dd, use_container_width=True)
                    
                    # Rolling Sharpe Ratio
                    st.subheader("📈 Rolling Sharpe Ratio (60-Day Window)")
                    rolling_sharpe = port_returns.rolling(window=60).apply(lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() > 0 else 0)
                    fig_rs = go.Figure()
                    fig_rs.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe, mode='lines', name='Rolling Sharpe', line=dict(color='green')))
                    fig_rs.add_hline(y=1, line_dash="dash", line_color="gray", annotation_text="Good Threshold")
                    fig_rs.update_layout(title='Rolling Sharpe Ratio (60 Days)', xaxis_title='Date', yaxis_title='Sharpe Ratio')
                    st.plotly_chart(fig_rs, use_container_width=True)
                    
                    # Correlation Matrix Heatmap
                    if len(portfolio_stocks) > 1:
                        st.subheader("📊 Correlation Heatmap")
                        corr_matrix = returns.corr()
                        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu', title='Portfolio Correlation Matrix')
                        st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Risk Contribution
                    st.subheader("🎯 Risk Contribution by Asset")
                    if len(portfolio_stocks) > 1:
                        # Calculate Marginal Risk Contribution
                        covariance = returns.cov() * 252
                        port_variance = np.dot(weights.T, np.dot(covariance, weights))
                        marginal_contrib = np.dot(covariance, weights) / port_variance
                        risk_contrib = marginal_contrib * weights
                        
                        risk_df = pd.DataFrame({
                            'Stock': portfolio_stocks,
                            'Risk Contribution %': risk_contrib * 100,
                            'Weight %': weights * 100
                        }).sort_values('Risk Contribution %', ascending=False)
                        
                        fig_rc = go.Figure()
                        fig_rc.add_trace(go.Bar(x=risk_df['Stock'], y=risk_df['Risk Contribution %'], name='Risk Contribution', marker_color='coral'))
                        fig_rc.add_trace(go.Bar(x=risk_df['Stock'], y=risk_df['Weight %'], name='Weight', marker_color='lightblue'))
                        fig_rc.update_layout(title='Risk Contribution vs Portfolio Weight', xaxis_title='Stock', yaxis_title='Percentage (%)', barmode='group')
                        st.plotly_chart(fig_rc, use_container_width=True)
                    
                    st.success("Risk analysis complete!")
    else:
        st.warning("Please build a portfolio first using the Portfolio Builder page.")
        st.info("Go to **Portfolio Builder** → Select stocks → Click **Optimise Portfolio** to create a portfolio")
