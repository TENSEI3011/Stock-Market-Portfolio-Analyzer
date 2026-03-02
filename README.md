# 📊 Stock Portfolio Analyzer

A **Streamlit-based web application** that helps you analyze stock performance, calculate risk-return metrics, and optimize portfolio allocation using **Modern Portfolio Theory (MPT)**.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

| Feature | Description |
|---|---|
| **Stock Price Charts** | Visualize historical closing prices for multiple stocks |
| **Percentage Change** | Track cumulative returns from the start date |
| **Moving Averages** | Configurable short-term and long-term moving average analysis |
| **Volume Analysis** | View trading volume patterns per stock |
| **Returns Distribution** | Histogram with KDE of daily returns |
| **Risk-Return Profile** | Annualized expected return vs. volatility scatter plot |
| **Correlation Matrix** | Heatmap showing inter-stock correlations |
| **Efficient Frontier** | Monte Carlo simulation of random portfolio allocations |
| **Optimal Allocation** | Maximum Sharpe Ratio portfolio with downloadable CSV |

---

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Data:** [yfinance](https://pypi.org/project/yfinance/) (Yahoo Finance API)
- **Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/stock-portfolio-analyzer.git
   cd stock-portfolio-analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run "source file/stock_portfolio_analyzer.py"
   ```

4. Open your browser at `http://localhost:8501`

---

## 📖 Usage

1. **Set Date Range** — Pick a start and end date in the sidebar.
2. **Enter Stock Symbols** — Add ticker symbols (e.g., `RELIANCE.NS`, `TCS.NS`, `AAPL`). Use `.NS` suffix for NSE-listed Indian stocks.
3. **Configure Parameters** — Adjust the risk-free rate, number of simulations, and moving average windows.
4. **Select Analysis Options** — Toggle the charts and analyses you want to see.
5. **Click "Run Analysis"** — The app downloads data, computes metrics, and renders interactive visualizations.

---

## 📁 Project Structure

```
stock-portfolio-analyzer/
├── source file/
│   ├── stock_portfolio_analyzer.py   # Main application
│   └── requirements.txt              # Python dependencies
├── Stock portoflio optimisation Ds report.pdf
├── stock portfolio helpbook.pdf
└── README.md
```

---

## 📊 Key Concepts

### Modern Portfolio Theory (MPT)
The app uses Monte Carlo simulation to generate thousands of random portfolio weight combinations and identifies the **optimal portfolio** — the one with the highest **Sharpe Ratio** (best risk-adjusted return).

### Sharpe Ratio
```
Sharpe Ratio = (Portfolio Return − Risk-Free Rate) / Portfolio Volatility
```
A higher Sharpe Ratio indicates better risk-adjusted performance.

---

## 📸 Screenshots

| Stock Price Chart | Efficient Frontier |
|---|---|
| Historical closing prices for selected stocks | Simulated portfolios with optimal allocation highlighted |

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m "Add new feature"`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Stock data powered by [Yahoo Finance](https://finance.yahoo.com/)
- Built with [Streamlit](https://streamlit.io/)
- Portfolio optimization based on Modern Portfolio Theory by Harry Markowitz
