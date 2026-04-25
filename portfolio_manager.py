"""
NIFTY Portfolio Optimiser with Persistent Storage & Multiple Timeframes
Enhanced version with database persistence and expanded time periods
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sqlite3
import warnings
from scipy import stats
import os

warnings.filterwarnings('ignore')

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="NIFTY Portfolio Optimiser",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Enhanced NIFTY Universe (All Major Indices)
# ---------------------------
# Broad Market Benchmarks
BROAD_INDICES = ["^NSEI", "NIFTYNEXT50.NS", "NIFTY100.NS", "NIFTY200.NS", "NIFTY500.NS"]
# Market Cap Based
MARKET_CAP_INDICES = ["NIFTYMIDCAPSELECT.NS", "NIFTYMIDCAP100.NS", "NIFTYSMALLCAP250.NS", "NIFTYMICROCAP250.NS"]
# Sectoral Indices
SECTORAL_INDICES = ["NIFTYBANK.NS", "NIFTYFINANCIAL.NS", "NIFTYIT.NS", "NIFTYPHARMA.NS", 
                    "NIFTYAUTO.NS", "NIFTYFMCG.NS", "NIFTYMEDIA.NS", "NIFTYMETAL.NS", 
                    "NIFTYREALTY.NS", "NIFTYENERGY.NS", "NIFTYINFRA.NS", "NIFTYPSU.NS"]
# Thematic & Strategy Indices
THEMATIC_INDICES = ["NIFTYCPSE.NS", "NIFTYCOMMODITIES.NS", "NIFTYINDIACONSUMPTION.NS", 
                    "NIFTYMNC.NS", "NIFTYHEALTHCARE.NS", "NIFTYOILGAS.NS", "NIFTYPVTBANK.NS"]
# Individual Stock Universe (NIFTY 50)
NIFTY_50_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "BAJFINANCE.NS",
    "ITC.NS", "WIPRO.NS", "AXISBANK.NS", "HCLTECH.NS", "LT.NS",
    "SUNPHARMA.NS", "TITAN.NS", "MARUTI.NS", "ASIANPAINT.NS", "NESTLEIND.NS",
    "ULTRACEMCO.NS", "TECHM.NS", "POWERGRID.NS", "M&M.NS", "BAJAJFINSV.NS",
    "TATAMOTORS.NS", "GRASIM.NS", "NTPC.NS", "INDUSINDBK.NS", "ONGC.NS",
    "DIVISLAB.NS", "DRREDDY.NS", "HDFCLIFE.NS", "BRITANNIA.NS", "CIPLA.NS",
    "SBILIFE.NS", "JSWSTEEL.NS", "ADANIPORTS.NS", "COALINDIA.NS", "EICHERMOT.NS",
    "HEROMOTOCO.NS", "BPCL.NS", "TATASTEEL.NS", "SHREECEM.NS",
    "UPL.NS", "BAJAJ-AUTO.NS", "TATACONSUM.NS", "HINDALCO.NS"
]

ALL_AVAILABLE_INDICES = {
    "Broad Market Benchmarks": BROAD_INDICES,
    "Market Capitalization": MARKET_CAP_INDICES,
    "Sectoral Indices": SECTORAL_INDICES,
    "Thematic & Strategy": THEMATIC_INDICES,
    "Individual Stocks (NIFTY 50)": NIFTY_50_SYMBOLS
}

# Time periods for optimization (enhanced from 3 to 7 options)
OPTIMIZATION_PERIODS = {
    "Weekly (5 trading days)": 5,
    "Bi-Weekly (10 trading days)": 10,
    "Monthly (21 trading days)": 21,
    "Quarterly (63 trading days)": 63,
    "Bi-Annual (126 trading days)": 126,
    "Annual (252 trading days)": 252,
    "Multi-Year (504 trading days)": 504
}

# ---------------------------
# Database Setup for Persistent Portfolio Storage
# ---------------------------
DB_NAME = "portfolio_data.db"

def init_database():
    """Initialize SQLite database and create tables if they don't exist"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Create portfolio table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                allocation REAL NOT NULL,
                category TEXT DEFAULT 'Individual Stocks (NIFTY 50)',
                date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create optimization history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period TEXT NOT NULL,
                sharpe_ratio REAL,
                expected_return REAL,
                expected_risk REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Database initialization failed: {str(e)}")
        return False

def load_portfolio():
    """Load portfolio from database"""
    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query("SELECT symbol, allocation FROM portfolio ORDER BY symbol", conn)
        conn.close()
        return df if not df.empty else pd.DataFrame(columns=['symbol', 'allocation'])
    except Exception as e:
        st.error(f"Error loading portfolio: {str(e)}")
        return pd.DataFrame(columns=['symbol', 'allocation'])

def save_portfolio(symbol, allocation, category):
    """Save or update a portfolio entry"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Check if symbol already exists
        cursor.execute("SELECT id FROM portfolio WHERE symbol = ?", (symbol,))
        existing = cursor.fetchone()
        
        if existing:
            cursor.execute("UPDATE portfolio SET allocation = ?, category = ?, date_added = CURRENT_TIMESTAMP WHERE symbol = ?", 
                         (allocation, category, symbol))
        else:
            cursor.execute("INSERT INTO portfolio (symbol, allocation, category) VALUES (?, ?, ?)", 
                         (symbol, allocation, category))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error saving portfolio: {str(e)}")
        return False

def delete_from_portfolio(symbol):
    """Delete a holding from portfolio"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM portfolio WHERE symbol = ?", (symbol,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error deleting from portfolio: {str(e)}")
        return False

def update_allocation(symbol, new_allocation):
    """Update allocation percentage for a holding"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("UPDATE portfolio SET allocation = ? WHERE symbol = ?", (new_allocation, symbol))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error updating allocation: {str(e)}")
        return False

def clear_portfolio():
    """Clear all portfolio data"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM portfolio")
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error clearing portfolio: {str(e)}")
        return False

def save_optimization_result(period, sharpe_ratio, expected_return, expected_risk):
    """Save optimization results to history"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO optimization_history (period, sharpe_ratio, expected_return, expected_risk)
            VALUES (?, ?, ?, ?)
        ''', (period, sharpe_ratio, expected_return, expected_risk))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        return False

def load_optimization_history():
    """Load optimization history from database"""
    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query("SELECT * FROM optimization_history ORDER BY timestamp DESC LIMIT 50", conn)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame()

# Initialize database on startup
init_database()

# ---------------------------
# Helper Functions for Indicators (unchanged from original)
# ---------------------------
def fetch_stock_data(symbols, period="1y"):
    """Fetch historical data for multiple symbols"""
    try:
        if isinstance(symbols, str):
            symbols = [symbols]
        data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period)
                if not df.empty:
                    df.index = pd.to_datetime(df.index)
                    data[symbol] = df['Close']
            except Exception:
                continue
        if not data:
            return None
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def compute_risk_metrics(returns):
    """Compute comprehensive risk metrics"""
    risk_free_rate = 0.06
    excess_returns = returns - risk_free_rate / 252
    
    # Sharpe ratio
    sharpe = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
    
    # Sortino ratio
    downside_returns = returns[returns < 0]
    sortino = np.sqrt(252) * returns.mean() / downside_returns.std() if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
    
    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Value at Risk (95% confidence)
    var_95 = np.percentile(returns, 5) * np.sqrt(252)
    
    # Conditional Value at Risk
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

# ---------------------------
# Portfolio Optimization Function with Enhanced Periods
# ---------------------------
def optimize_portfolio(selected_symbols, period_days=252):
    """Optimize portfolio weights using Monte Carlo simulation"""
    try:
        # Fetch historical returns
        returns_data = fetch_stock_data(selected_symbols, period=f"{period_days}d")
        if returns_data is None or returns_data.empty:
            return None, None, None, None
        
        # Calculate daily returns
        returns = returns_data.pct_change().dropna()
        
        # Monte Carlo Simulation
        num_portfolios = 5000
        results = np.zeros((4, num_portfolios))
        weights_record = []
        
        for i in range(num_portfolios):
            weights = np.random.random(len(selected_symbols))
            weights /= np.sum(weights)
            weights_record.append(weights)
            
            port_return = np.sum(returns.mean() * weights) * 252
            port_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
            sharpe = port_return / port_std if port_std > 0 else 0
            
            results[0, i] = port_return
            results[1, i] = port_std
            results[2, i] = sharpe
        
        # Find maximum Sharpe ratio portfolio
        max_sharpe_idx = np.argmax(results[2])
        optimal_weights = weights_record[max_sharpe_idx]
        
        optimal_allocation = {
            'Symbol': selected_symbols,
            'Weight %': optimal_weights * 100
        }
        
        opt_df = pd.DataFrame(optimal_allocation)
        opt_df = opt_df.sort_values('Weight %', ascending=False)
        
        # Calculate portfolio metrics
        port_return = results[0, max_sharpe_idx]
        port_std = results[1, max_sharpe_idx]
        sharpe_ratio = results[2, max_sharpe_idx]
        
        return opt_df, port_return, port_std, sharpe_ratio
        
    except Exception as e:
        st.error(f"Portfolio optimization failed: {str(e)}")
        return None, None, None, None

# ---------------------------
# Find Optimal Rebalancing Window
# ---------------------------
def find_optimal_rebalancing_window(symbols, test_periods=None):
    """Identify the best rebalancing frequency based on historical performance"""
    if test_periods is None:
        test_periods = [5, 10, 21, 63, 126, 252]
    
    results = []
    for days in test_periods:
        try:
            opt_df, port_return, port_std, sharpe = optimize_portfolio(symbols, days)
            if port_return is not None:
                results.append({
                    'Rebalance Days': days,
                    'Rebalance Period': get_period_name(days),
                    'Expected Return': port_return,
                    'Expected Risk': port_std,
                    'Sharpe Ratio': sharpe
                })
        except:
            continue
    
    if results:
        results_df = pd.DataFrame(results)
        best_period = results_df.loc[results_df['Sharpe Ratio'].idxmax()]
        return results_df, best_period
    return None, None

def get_period_name(days):
    """Convert days to readable period name"""
    period_map = {5: "Weekly", 10: "Bi-Weekly", 21: "Monthly", 63: "Quarterly", 
                  126: "Bi-Annual", 252: "Annual", 504: "Multi-Year"}
    return period_map.get(days, f"{days} Days")

# ---------------------------
# Main UI Sections
# ---------------------------
st.sidebar.title("📊 NIFTY Portfolio Optimiser")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Dashboard", "📈 Portfolio Optimiser", "⚙️ Portfolio Management", "🎯 Optimal Rebalancing Window", "📊 Performance History", "🔬 Stock Analysis"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Portfolio Status**: {'Active' if not load_portfolio().empty else 'Empty'} | **Indices Available**: {len(BROAD_INDICES) + len(MARKET_CAP_INDICES) + len(SECTORAL_INDICES) + len(THEMATIC_INDICES)}")
st.sidebar.markdown("---")
st.sidebar.caption("Built for Indian Markets | Real-time Data")

# ---------------------------
# Page 1: Dashboard
# ---------------------------
if page == "🏠 Dashboard":
    st.title("🏠 NIFTY Market Dashboard")
    st.markdown("Real-time overview of NIFTY indices and your portfolio performance")
    
    # Fetch NIFTY 50 Index Data
    nifty_data = fetch_stock_data(["^NSEI"], period="1mo")
    if nifty_data is not None and not nifty_data.empty:
        col1, col2, col3, col4 = st.columns(4)
        current_nifty = nifty_data.iloc[-1, 0]
        prev_nifty = nifty_data.iloc[-2, 0]
        nifty_change = ((current_nifty - prev_nifty) / prev_nifty) * 100
        
        col1.metric("NIFTY 50", f"{current_nifty:,.2f}", delta=f"{nifty_change:.2f}%")
        
        # Display portfolio summary if available
        portfolio_df = load_portfolio()
        if not portfolio_df.empty:
            total_allocation = portfolio_df['allocation'].sum()
            col2.metric("Portfolio Stocks", len(portfolio_df))
            col3.metric("Total Allocation", f"{total_allocation:.1f}%")
            if total_allocation == 100:
                col4.success("⚖️ Fully Allocated")
            else:
                col4.warning(f"Adjust allocations to 100%")
        else:
            st.info("💡 Go to 'Portfolio Management' to start building your portfolio!")
        
        # NIFTY 50 Performance Chart
        st.subheader("📈 NIFTY 50 Price Movement")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=nifty_data.index, y=nifty_data.iloc[:, 0], mode='lines', name='NIFTY 50', line=dict(color='blue')))
        fig.update_layout(title='NIFTY 50 Historical Prices', xaxis_title='Date', yaxis_title='Price (INR)', hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Page 2: Portfolio Optimiser (New)
# ---------------------------
elif page == "📈 Portfolio Optimiser":
    st.title("📈 Portfolio Optimiser")
    st.markdown("Build and optimize your portfolio using Monte Carlo simulations")
    
    # Category selector
    category = st.selectbox("Select Category", list(ALL_AVAILABLE_INDICES.keys()))
    available_symbols = ALL_AVAILABLE_INDICES[category]
    
    selected_symbols = st.multiselect("Select Assets for Portfolio", available_symbols, 
                                      default=available_symbols[:3] if len(available_symbols) >= 3 else available_symbols)
    
    if len(selected_symbols) >= 2:
        # Period selector for optimization (enhanced to 7 options)
        selected_period = st.selectbox("Select Optimization Period", list(OPTIMIZATION_PERIODS.keys()), index=2)
        period_days = OPTIMIZATION_PERIODS[selected_period]
        
        if st.button("🚀 Run Portfolio Optimization", type="primary"):
            with st.spinner(f"Optimizing portfolio over {selected_period}..."):
                opt_df, port_return, port_std, sharpe = optimize_portfolio(selected_symbols, period_days)
                
                if opt_df is not None:
                    # Display results
                    st.subheader("📊 Optimal Portfolio Allocation")
                    st.dataframe(opt_df.style.format({'Weight %': '{:.2f}%'}), use_container_width=True)
                    
                    # Portfolio metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Expected Annual Return", f"{port_return:.2%}" if port_return else "N/A")
                    with col2:
                        st.metric("Expected Annual Risk", f"{port_std:.2%}" if port_std else "N/A")
                    with col3:
                        st.metric("Sharpe Ratio", f"{sharpe:.2f}" if sharpe else "N/A")
                    
                    # Pie chart
                    fig_pie = px.pie(opt_df, values='Weight %', names='Symbol', title='Portfolio Allocation')
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Save to portfolio table
                    st.subheader("💾 Save This Portfolio")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Save to Database", type="primary"):
                            # Clear existing portfolio first
                            clear_portfolio()
                            for _, row in opt_df.iterrows():
                                save_portfolio(row['Symbol'], row['Weight %'], category)
                            st.success(f"Portfolio with {len(opt_df)} assets saved to database!")
                            save_optimization_result(selected_period, sharpe, port_return, port_std)
                    
                    with col2:
                        # Export as CSV
                        csv = opt_df.to_csv(index=False)
                        st.download_button("📥 Download as CSV", csv, "portfolio_allocation.csv", "text/csv")
    else:
        if len(selected_symbols) == 1:
            st.warning("Please select at least 2 assets for portfolio optimization")
        else:
            st.info("Select assets from the categories above to begin")

# ---------------------------
# Page 3: Portfolio Management (Persistent CRUD)
# ---------------------------
elif page == "⚙️ Portfolio Management":
    st.title("⚙️ Portfolio Management")
    st.markdown("Manage your portfolio - data is **permanently saved** to database")
    
    # Display current portfolio
    current_portfolio = load_portfolio()
    
    if not current_portfolio.empty:
        st.subheader("📋 Current Portfolio Holdings")
        st.dataframe(current_portfolio, use_container_width=True)
        
        # Edit allocation
        st.subheader("✏️ Edit Holdings")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_edit = st.selectbox("Select Symbol", current_portfolio['symbol'].tolist())
        
        with col2:
            new_allocation = st.number_input("New Allocation (%)", 0.0, 100.0, 
                                           float(current_portfolio[current_portfolio['symbol'] == selected_edit]['allocation'].iloc[0]))
        
        with col3:
            if st.button("Update Allocation", type="primary"):
                update_allocation(selected_edit, new_allocation)
                st.rerun()
        
        # Delete holding
        st.subheader("🗑️ Remove Holding")
        col1, col2 = st.columns(2)
        with col1:
            selected_delete = st.selectbox("Select Symbol to Remove", current_portfolio['symbol'].tolist())
        with col2:
            if st.button("Delete", type="secondary"):
                delete_from_portfolio(selected_delete)
                st.rerun()
        
        # Clear all
        if st.button("⚠️ Clear All Portfolio Data", type="secondary"):
            clear_portfolio()
            st.rerun()
        
        # Export functionality
        st.subheader("📥 Export Data")
        csv = current_portfolio.to_csv(index=False)
        st.download_button("Download Portfolio CSV", csv, "my_portfolio.csv", "text/csv")
    
    else:
        st.info("No portfolio data found. Go to 'Portfolio Optimiser' to create and save one!")
    
    # Add individual holding manually
    with st.expander("➕ Add Single Holding Manually"):
        with st.form("add_manual_holding"):
            symbol = st.text_input("Symbol", placeholder="e.g., TCS.NS or RELIANCE.NS")
            allocation = st.number_input("Allocation (%)", 0.0, 100.0, 0.0)
            category = st.selectbox("Category", list(ALL_AVAILABLE_INDICES.keys()))
            submitted = st.form_submit_button("Add to Portfolio")
            
            if submitted:
                if symbol and allocation > 0:
                    save_portfolio(symbol, allocation, category)
                    st.success(f"Added {symbol} with {allocation}% allocation")
                    st.rerun()
                else:
                    st.error("Please provide both symbol and allocation")

# ---------------------------
# Page 4: Optimal Rebalancing Window (New Feature)
# ---------------------------
elif page == "🎯 Optimal Rebalancing Window":
    st.title("🎯 Optimal Rebalancing Window Analysis")
    st.markdown("Identifies the best rebalancing period for your portfolio")
    
    portfolio_data = load_portfolio()
    
    if not portfolio_data.empty:
        symbols = portfolio_data['symbol'].tolist()
        allocations = portfolio_data['allocation'].tolist()
        
        # Normalize allocations
        total_allocation = sum(allocations)
        if total_allocation > 0:
            allocations = [alloc/total_allocation for alloc in allocations]
        
        st.subheader("Current Portfolio for Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Symbols:**", ", ".join(symbols))
        with col2:
            st.write("**Weights:**", [f"{w:.1%}" for w in allocations])
        
        if st.button("🔍 Find Optimal Rebalancing Window", type="primary"):
            with st.spinner("Analyzing 7 different rebalancing periods (may take 30-60 seconds)..."):
                results_df, best_period = find_optimal_rebalancing_window(symbols)
                
                if results_df is not None and not results_df.empty:
                    st.subheader("📊 Historical Performance by Rebalancing Period")
                    st.dataframe(results_df.style.format({
                        'Expected Return': '{:.2%}',
                        'Expected Risk': '{:.2%}',
                        'Sharpe Ratio': '{:.3f}'
                    }), use_container_width=True)
                    
                    # Best period recommendation
                    st.subheader("⭐ Optimal Rebalancing Recommendation")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Recommended Period", best_period['Rebalance Period'])
                    with col2:
                        st.metric("Expected Sharpe Ratio", f"{best_period['Sharpe Ratio']:.3f}")
                    with col3:
                        st.metric("Risk-Adjusted Return", f"{(best_period['Expected Return'] - 0.06)/best_period['Expected Risk']:.2f}" if best_period['Expected Risk'] > 0 else "N/A")
                    
                    # Visualization
                    fig_perf = go.Figure()
                    fig_perf.add_trace(go.Bar(x=results_df['Rebalance Period'], y=results_df['Sharpe Ratio'], 
                                             marker_color=['green' if x == best_period['Rebalance Period'] else 'blue' for x in results_df['Rebalance Period']]))
                    fig_perf.update_layout(title='Sharpe Ratio by Rebalancing Period', xaxis_title='Rebalancing Frequency', yaxis_title='Sharpe Ratio')
                    st.plotly_chart(fig_perf, use_container_width=True)
                    
                    # Save recommendation
                    save_optimization_result(best_period['Rebalance Period'], best_period['Sharpe Ratio'], 
                                            best_period['Expected Return'], best_period['Expected Risk'])
                    st.success("Analysis complete! Rebalance recommendation saved to history.")
                else:
                    st.error("Analysis failed. Please check your portfolio data.")
    else:
        st.warning("No portfolio found. Please save a portfolio in 'Portfolio Optimiser' or 'Portfolio Management'")
        st.info("💡 Tip: Go to 'Portfolio Optimiser' → Select assets → Run Optimization → Save to Database")

# ---------------------------
# Page 5: Performance History (New Feature)
# ---------------------------
elif page == "📊 Performance History":
    st.title("📊 Portfolio Performance History")
    st.markdown("Track your optimization results over time")
    
    history_df = load_optimization_history()
    
    if not history_df.empty:
        st.subheader("Recent Optimization Results")
        st.dataframe(history_df, use_container_width=True)
        
        # Sharpe Ratio over time
        fig_sharpe = go.Figure()
        fig_sharpe.add_trace(go.Scatter(x=history_df['timestamp'], y=history_df['sharpe_ratio'], 
                                       mode='lines+markers', name='Sharpe Ratio'))
        fig_sharpe.update_layout(title='Sharpe Ratio Progression Over Time', xaxis_title='Date', yaxis_title='Sharpe Ratio')
        st.plotly_chart(fig_sharpe, use_container_width=True)
        
        # Return vs Risk scatter
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(x=history_df['expected_risk'], y=history_df['expected_return'], 
                                        mode='markers', text=history_df['period'], marker=dict(size=10)))
        fig_scatter.update_layout(title='Return vs Risk by Period', xaxis_title='Expected Risk', yaxis_title='Expected Return')
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("No historical data yet. Run some optimizations to see results here!")

# ---------------------------
# Page 6: Stock Analysis (simplified version of original)
# ---------------------------
elif page == "🔬 Stock Analysis":
    st.title("🔬 Individual Stock Analysis")
    
    # Get all available symbols
    all_symbols = []
    for indices in ALL_AVAILABLE_INDICES.values():
        all_symbols.extend(indices)
    all_symbols = list(set(all_symbols))
    
    selected_stock = st.selectbox("Select Stock/Index for Analysis", all_symbols, index=0)
    period = st.selectbox("Analysis Period", ["3mo", "6mo", "1y", "2y"], index=2)
    
    if selected_stock:
        data = fetch_stock_data(selected_stock, period=period)
        
        if data is not None and not data.empty:
            # Basic metrics
            col1, col2, col3, col4 = st.columns(4)
            current_price = data.iloc[-1, 0]
            price_change = ((current_price - data.iloc[-2, 0]) / data.iloc[-2, 0]) * 100
            
            with col1:
                st.metric("Current Price", f"₹{current_price:.2f}")
            with col2:
                st.metric("Period Change", f"{price_change:.2f}%", delta=f"{price_change:.2f}%")
            with col3:
                st.metric("Period High", f"₹{data.iloc[:, 0].max():.2f}")
            with col4:
                st.metric("Period Low", f"₹{data.iloc[:, 0].min():.2f}")
            
            # Price chart
            st.subheader("📉 Price Movement")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data.iloc[:, 0], mode='lines', name='Close Price', line=dict(color='blue')))
            fig.update_layout(title=f'{selected_stock} - Historical Prices', xaxis_title='Date', yaxis_title='Price (₹)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"No data found for {selected_stock}")

# ---------------------------
# Run the application
# ---------------------------
if __name__ == "__main__":
    pass
