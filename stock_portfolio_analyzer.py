import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import date, timedelta
import io

# Page configuration
st.set_page_config(
    page_title="Stock Portfolio Analyzer",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📊 Stock Portfolio Analyzer")
st.markdown("""
This application helps you analyze stock performance, calculate risk-return metrics, 
and optimize portfolio allocation using Modern Portfolio Theory.
""")

# Sidebar for inputs
st.sidebar.header("Portfolio Settings")

# Date range selector
today = date.today()
default_start_date = today - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=default_start_date)
end_date = st.sidebar.date_input("End Date", value=today)

# Convert dates to string for yfinance
start_date_str = start_date.strftime("%Y-%m-%d")
end_date_str = end_date.strftime("%Y-%m-%d")

# Default tickers
default_tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS']

# Stock selector
st.sidebar.subheader("Stock Selection")
ticker_input = st.sidebar.text_area(
    "Enter stock symbols (one per line):",
    "\n".join(default_tickers)
)

# Parse ticker input
tickers = [ticker.strip() for ticker in ticker_input.split("\n") if ticker.strip()]

# Risk-free rate input
risk_free_rate = st.sidebar.slider(
    "Risk-Free Rate (%)",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1
) / 100

# Number of simulated portfolios
num_portfolios = st.sidebar.slider(
    "Number of Portfolio Simulations",
    min_value=1000,
    max_value=20000,
    value=10000,
    step=1000
)

# Moving average settings
st.sidebar.subheader("Technical Analysis")
short_window = st.sidebar.slider("Short-term MA Window (days)", 10, 100, 50, 5)
long_window = st.sidebar.slider("Long-term MA Window (days)", 100, 300, 200, 10)

# Analysis type
st.sidebar.subheader("Analysis Options")
show_price_charts = st.sidebar.checkbox("Show Price Charts", value=True)
show_pct_change = st.sidebar.checkbox("Show Percentage Change", value=True)
show_moving_averages = st.sidebar.checkbox("Show Moving Averages", value=True)
show_volume = st.sidebar.checkbox("Show Volume Analysis", value=True)
show_returns_dist = st.sidebar.checkbox("Show Returns Distribution", value=True)
show_risk_return = st.sidebar.checkbox("Show Risk-Return Analysis", value=True)
show_correlation = st.sidebar.checkbox("Show Correlation Matrix", value=True)
show_efficient_frontier = st.sidebar.checkbox("Show Efficient Frontier", value=True)
show_optimal_allocation = st.sidebar.checkbox("Show Optimal Allocation", value=True)

# Button to run analysis
run_analysis = st.sidebar.button("Run Analysis", type="primary")

# Function to download stock data and perform analysis
@st.cache_data(ttl=3600)
def download_stock_data(tickers, start_date, end_date):
    try:
        # Validate tickers
        if not tickers:
            st.error("No stock tickers provided.")
            return None, False
        
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        # Check if data is empty
        if data.empty:
            st.error(f"No data downloaded for tickers: {', '.join(tickers)}. Check ticker symbols and date range.")
            return None, False
        
        # Handle multi-level column case
        if isinstance(data.columns, pd.MultiIndex):
            stock_dfs = []
            for ticker in tickers:
                # Safely extract data
                ticker_data = pd.DataFrame({
                    'Date': data.index,
                    'Open': data[('Open', ticker)] if ('Open', ticker) in data.columns else np.nan,
                    'High': data[('High', ticker)] if ('High', ticker) in data.columns else np.nan,
                    'Low': data[('Low', ticker)] if ('Low', ticker) in data.columns else np.nan,
                    'Close': data[('Close', ticker)] if ('Close', ticker) in data.columns else np.nan,
                    'Volume': data[('Volume', ticker)] if ('Volume', ticker) in data.columns else np.nan,
                    'Ticker': ticker
                })
                
                # Calculate daily returns
                ticker_data['Daily Return'] = ticker_data['Close'].pct_change()
                
                stock_dfs.append(ticker_data)
            
            stock_data = pd.concat(stock_dfs, ignore_index=True)
        else:
            # Single ticker case
            stock_data = data.reset_index()
            stock_data['Ticker'] = tickers[0]
            stock_data['Daily Return'] = stock_data['Close'].pct_change()
        
        # Clean and validate data
        stock_data = stock_data.dropna()
        
        # Ensure at least one stock has data
        if stock_data.empty:
            st.error(f"No valid data found for tickers: {', '.join(tickers)}. Check data availability.")
            return None, False
        
        return stock_data, True
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return None, False

# Function to calculate portfolio statistics
def calculate_portfolio_stats(stock_data, risk_free_rate, num_portfolios):
    # Calculate daily returns
    stock_data = stock_data.sort_values(['Ticker', 'Date'])
    stock_data['Daily Return'] = stock_data.groupby('Ticker')['Close'].pct_change()
    
    # Create a pivot table of daily returns
    daily_returns = stock_data.pivot_table(index='Date', columns='Ticker', values='Daily Return')
    daily_returns = daily_returns.dropna()
    
    # Calculate expected returns and volatility
    expected_returns = daily_returns.mean() * 252  # annualize
    volatility = daily_returns.std() * np.sqrt(252)  # annualize
    
    # Create stats DataFrame
    stock_stats = pd.DataFrame({
        'Expected Return': expected_returns,
        'Volatility': volatility,
        'Sharpe Ratio': (expected_returns - risk_free_rate) / volatility
    })
    
    # Portfolio optimization
    cov_matrix = daily_returns.cov() * 252  # annualized covariance
    
    # Arrays to store results
    results = np.zeros((3, num_portfolios))
    all_weights = np.zeros((num_portfolios, len(daily_returns.columns)))
    
    # Generate random portfolios
    np.random.seed(42)
    for i in range(num_portfolios):
        weights = np.random.random(len(daily_returns.columns))
        weights /= np.sum(weights)
        all_weights[i,:] = weights
        
        # Portfolio return and volatility
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Store results
        results[0,i] = portfolio_return
        results[1,i] = portfolio_volatility
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_volatility  # Sharpe ratio
    
    # Find optimal portfolio (max Sharpe ratio)
    max_sharpe_idx = np.argmax(results[2])
    max_sharpe_return = results[0, max_sharpe_idx]
    max_sharpe_volatility = results[1, max_sharpe_idx]
    max_sharpe_ratio = results[2, max_sharpe_idx]
    max_sharpe_weights = all_weights[max_sharpe_idx]
    
    # Create optimal allocation DataFrame
    optimal_allocation = pd.DataFrame({
        'Ticker': daily_returns.columns,
        'Weight': max_sharpe_weights
    })
    
    return stock_stats, daily_returns, results, optimal_allocation, max_sharpe_return, max_sharpe_volatility, max_sharpe_ratio

# Main analysis function
def run_stock_analysis():
    # Progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Download data
    status_text.text("Downloading stock data...")
    stock_data, success = download_stock_data(tickers, start_date_str, end_date_str)
    if not success:
        return
    progress_bar.progress(20)
    
    # Show raw data option
    if st.checkbox("Show raw data"):
        st.subheader("Raw Stock Data")
        st.dataframe(stock_data)
    
    # Step 2: Calculate percentage changes
    status_text.text("Calculating price changes...")
    pct_change_dfs = []
    for ticker in tickers:
        ticker_data = stock_data[stock_data['Ticker'] == ticker].copy()
        if not ticker_data.empty:  # Check if we have data for this ticker
            first_close = ticker_data['Close'].iloc[0]
            ticker_data['Pct_Change'] = (ticker_data['Close'] / first_close - 1) * 100
            pct_change_dfs.append(ticker_data)
    
    # Only proceed if we have some valid data
    if pct_change_dfs:
        pct_change_data = pd.concat(pct_change_dfs, ignore_index=True)
    else:
        st.error("No valid stock data available for analysis.")
        return
    
    progress_bar.progress(40)
    
    # Step 3: Calculate portfolio stats and optimization
    status_text.text("Calculating portfolio statistics...")
    stock_stats, daily_returns, results, optimal_allocation, max_sharpe_return, max_sharpe_volatility, max_sharpe_ratio = calculate_portfolio_stats(stock_data, risk_free_rate, num_portfolios)
    progress_bar.progress(60)
    
    # Step 4: Generate visualizations
    status_text.text("Creating visualizations...")
    
    # Price chart
    if show_price_charts:
        st.subheader("Stock Price Over Time")
        fig, ax = plt.subplots(figsize=(12, 6))
        for ticker in tickers:
            ticker_data = stock_data[stock_data['Ticker'] == ticker]
            ax.plot(ticker_data['Date'], ticker_data['Close'], label=ticker)
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price (₹)')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)
    
    # Percentage change chart
    if show_pct_change:
        st.subheader("Percentage Change in Stock Price")
        fig, ax = plt.subplots(figsize=(12, 6))
        for ticker in tickers:
            ticker_data = pct_change_data[pct_change_data['Ticker'] == ticker]
            ax.plot(ticker_data['Date'], ticker_data['Pct_Change'], label=ticker)
        ax.set_xlabel('Date')
        ax.set_ylabel('% Change from Start')
        ax.legend()
        ax.grid(True)
        ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    
    # Moving averages tabs
    if show_moving_averages:
        st.subheader("Moving Averages Analysis")
        
        # Create tabs for each ticker
        ticker_tabs = st.tabs(tickers)
        
        for i, ticker in enumerate(tickers):
            with ticker_tabs[i]:
                ticker_data = stock_data[stock_data['Ticker'] == ticker].copy()
                ticker_data = ticker_data.set_index('Date')
                
                # Calculate moving averages
                ticker_data['50_MA'] = ticker_data['Close'].rolling(window=short_window).mean()
                ticker_data['200_MA'] = ticker_data['Close'].rolling(window=long_window).mean()
                
                # Plot
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(ticker_data.index, ticker_data['Close'], label='Close Price')
                ax.plot(ticker_data.index, ticker_data['50_MA'], label=f'{short_window}-Day MA')
                ax.plot(ticker_data.index, ticker_data['200_MA'], label=f'{long_window}-Day MA')
                ax.set_title(f'{ticker} - Close Price and Moving Averages')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price (₹)')
                ax.legend()
                ax.grid(True)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show volume if selected
                if show_volume:
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.bar(ticker_data.index, ticker_data['Volume'], color='orange', alpha=0.7)
                    ax.set_title(f'{ticker} - Volume Traded')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Volume')
                    ax.grid(True)
                    plt.tight_layout()
                    st.pyplot(fig)
    
    progress_bar.progress(80)
    
    # Returns distribution
    if show_returns_dist:
        st.subheader("Distribution of Daily Returns")
        fig, ax = plt.subplots(figsize=(12, 6))
        for ticker in tickers:
            ticker_data = stock_data[stock_data['Ticker'] == ticker]
            returns = ticker_data['Daily Return'].dropna()
            sns.histplot(returns, bins=30, kde=True, label=ticker, alpha=0.5, ax=ax)
        ax.set_xlabel('Daily Return')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)
    
    # Risk-return analysis
    if show_risk_return:
        st.subheader("Risk-Return Analysis")
        
        # Show statistics
        st.write("Stock Risk-Return Statistics:")
        st.dataframe(stock_stats.style.format({
            'Expected Return': '{:.2%}',
            'Volatility': '{:.2%}',
            'Sharpe Ratio': '{:.2f}'
        }))
        
        # Risk-return scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(stock_stats['Volatility'], stock_stats['Expected Return'], s=100)
        for ticker in tickers:
            ax.annotate(ticker, 
                       (stock_stats.loc[ticker, 'Volatility'], 
                        stock_stats.loc[ticker, 'Expected Return']),
                       xytext=(5, 5),
                       textcoords='offset points')
        ax.set_title('Risk-Return Profile')
        ax.set_xlabel('Annualized Volatility (Risk)')
        ax.set_ylabel('Annualized Expected Return')
        ax.grid(True)
        ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    
    # Correlation matrix
    if show_correlation:
        st.subheader("Stock Correlation Matrix")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(daily_returns.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
    
    # Efficient frontier
    if show_efficient_frontier:
        st.subheader("Efficient Frontier")
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', marker='o', alpha=0.5)
        
        # Highlight the maximum Sharpe ratio portfolio
        ax.scatter(max_sharpe_volatility, max_sharpe_return, c='red', marker='*', s=300, label='Maximum Sharpe Ratio')
        
        # Add markers for individual stocks
        for i, ticker in enumerate(tickers):
            if ticker in stock_stats.index:
                ax.scatter(stock_stats.loc[ticker, 'Volatility'], 
                          stock_stats.loc[ticker, 'Expected Return'], 
                          marker='o', s=100, label=ticker)
        
        ax.set_title('Efficient Frontier')
        ax.set_xlabel('Volatility (Standard Deviation)')
        ax.set_ylabel('Expected Return')
        plt.colorbar(scatter, label='Sharpe Ratio')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display optimal portfolio stats
        st.write("**Optimal Portfolio (Maximum Sharpe Ratio):**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Return", f"{max_sharpe_return:.2%}")
        col2.metric("Volatility", f"{max_sharpe_volatility:.2%}")
        col3.metric("Sharpe Ratio", f"{max_sharpe_ratio:.2f}")
    
    # Optimal allocation
    if show_optimal_allocation:
        st.subheader("Optimal Portfolio Allocation")
        
        # Show weights table
        optimal_allocation_display = optimal_allocation.copy()
        optimal_allocation_display['Weight'] = optimal_allocation_display['Weight'].apply(lambda x: f"{x:.2%}")
        st.table(optimal_allocation_display)
        
        # Pie chart
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.pie(optimal_allocation['Weight'], labels=optimal_allocation['Ticker'], 
              autopct='%1.1f%%', startangle=90, shadow=True)
        ax.axis('equal')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Download allocation as CSV
        csv = optimal_allocation.to_csv(index=False)
        st.download_button(
            label="Download Optimal Allocation",
            data=csv,
            file_name="optimal_portfolio_allocation.csv",
            mime="text/csv",
        )
    
    progress_bar.progress(100)
    status_text.text("Analysis complete!")

# Check if tickers are provided
if not tickers:
    st.warning("Please enter at least one stock symbol in the sidebar.")
elif run_analysis:
    run_stock_analysis()
else:
    st.info("Enter your parameters in the sidebar and click 'Run Analysis' to start.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Stock data provided by Yahoo Finance. Portfolio optimization based on Modern Portfolio Theory.</p>
</div>
""", unsafe_allow_html=True)