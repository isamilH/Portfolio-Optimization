import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib
from datetime import date, timedelta
from scipy.optimize import minimize
import io

# -----------------------------------------------------------
# 1) Portfolio Download & Returns
# -----------------------------------------------------------
@st.cache_data
def download_data(tickers, start_date, end_date):
    """
    Download stock data for multiple tickers using yfinance
    and return a dictionary of DataFrames keyed by ticker.
    """
    data_dict = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if not df.empty:
            # Compute daily returns
            df['Daily Return'] = df['Close'].pct_change()
            data_dict[ticker] = df
    return data_dict

def combine_returns(data_dict):
    """Combine daily returns of each ticker into a single DataFrame."""
    returns_df = pd.DataFrame()
    for ticker, df in data_dict.items():
        returns_df[ticker] = df['Daily Return']
    # Drop any rows with NaN
    returns_df.dropna(inplace=True)
    return returns_df

# -----------------------------------------------------------
# 2) Markowitz Portfolio Optimization
# -----------------------------------------------------------
def portfolio_performance(weights, mean_returns, cov_matrix, freq=252, risk_free_rate=0.0):
    """
    Given a set of weights, returns the portfolio annualized return and annualized volatility.
    freq=252 for daily data, 12 for monthly data, etc.
    """
    # Ensure weights is a numpy array
    weights = np.array(weights)

    # Portfolio return: dot product of weights and average returns
    port_return = np.sum(mean_returns * weights) * freq

    # Portfolio volatility (std dev)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * freq, weights)))

    # Portfolio Sharpe Ratio
    sharpe_ratio = (port_return - risk_free_rate) / port_volatility if port_volatility != 0 else 0

    return port_return, port_volatility, sharpe_ratio

def min_volatility(weights, mean_returns, cov_matrix, freq=252, risk_free_rate=0.0):
    """
    Objective function for minimum volatility (we'll just return the portfolio's volatility).
    """
    return portfolio_performance(weights, mean_returns, cov_matrix, freq, risk_free_rate)[1]

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, freq=252, risk_free_rate=0.0):
    """
    Objective function for maximizing Sharpe ratio => minimize negative Sharpe.
    """
    return -portfolio_performance(weights, mean_returns, cov_matrix, freq, risk_free_rate)[2]

def get_constraints_and_bounds(num_assets, allow_short=False):
    """
    For a standard Markowitz problem: sum of weights = 1
    If allow_short=True, weights can be negative (short selling).
    """
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    if allow_short:
        # e.g., weights can range from -1.0 to 1.0
        bounds = tuple((-1.0, 1.0) for _ in range(num_assets))
    else:
        # Long-only: 0.0 to 1.0
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    return constraints, bounds

def optimize_portfolio(mean_returns, cov_matrix, freq=252, risk_free_rate=0.0,
                       objective="sharpe", allow_short=False):
    """
    Optimize portfolio weights for either:
      - 'sharpe' => maximize Sharpe ratio
      - 'min_volatility' => minimize volatility
    Returns optimal weights.
    """
    num_assets = len(mean_returns)
    # Initial guess (equal weights)
    init_guess = num_assets * [1.0 / num_assets]

    constraints, bounds = get_constraints_and_bounds(num_assets, allow_short=allow_short)

    if objective == "sharpe":
        result = minimize(
            fun=neg_sharpe_ratio,
            x0=init_guess,
            args=(mean_returns, cov_matrix, freq, risk_free_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
    elif objective == "min_volatility":
        result = minimize(
            fun=min_volatility,
            x0=init_guess,
            args=(mean_returns, cov_matrix, freq, risk_free_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
    else:
        raise ValueError("Unknown objective. Choose 'sharpe' or 'min_volatility'.")

    return result.x if result.success else None

def compute_efficient_frontier(mean_returns, cov_matrix, freq=252, risk_free_rate=0.0,
                               n_points=50, allow_short=False):
    """
    Compute points on the efficient frontier by systematically varying target returns
    and minimizing volatility for each target.
    """
    num_assets = len(mean_returns)
    init_guess = num_assets * [1.0 / num_assets]
    constraints_base, bounds = get_constraints_and_bounds(num_assets, allow_short=allow_short)

    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), n_points)
    frontier = []

    for tr in target_returns:
        # constraints: sum of weights = 1, portfolio_return = tr
        # (plus short-selling bounds if allow_short=True)
        constraints = constraints_base + [{
            'type': 'eq',
            'fun': lambda w, tr=tr: np.sum(w * mean_returns) * freq - tr
        }]

        result = minimize(
            fun=min_volatility,
            x0=init_guess,
            args=(mean_returns, cov_matrix, freq, risk_free_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        if result.success:
            w = result.x
            port_ret, port_vol, _ = portfolio_performance(w, mean_returns, cov_matrix, freq, risk_free_rate)
            frontier.append((port_vol, port_ret, w))
        else:
            # If optimization fails for that target, skip it
            pass

    return frontier

# -----------------------------------------------------------
# 3) Streamlit App
# -----------------------------------------------------------
def main():
    st.title("Portfolio Optimization ")
    st.write("Build a multi-asset portfolio from multiple markets and apply Modern Portfolio Theory.")

    # --- SIDEBAR ---
    st.sidebar.subheader("Selection Panel")
    default_tickers = ["AAPL", "TSLA", "BMW.DE", "7203.T"]  # Apple (US), Tesla (US), BMW (DE), Toyota (JP)
    tickers = st.sidebar.text_area(
        "Enter Tickers (comma separated):",
        value=", ".join(default_tickers)
    ).split(",")
    tickers = [t.strip() for t in tickers if t.strip()]

    today = date.today()
    default_start = today - timedelta(days=365)
    start_date = st.sidebar.date_input("Start Date", default_start)
    end_date = st.sidebar.date_input("End Date", today)

    # Risk-free rate
    risk_free_rate_input = st.sidebar.number_input(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.1
    )
    risk_free_rate = risk_free_rate_input / 100.0  # convert to decimal

    # Short selling
    allow_short = st.sidebar.checkbox("Allow Short Selling?", value=False)

    # Number of points on the Efficient Frontier
    n_points = st.sidebar.slider("Number of Frontier Points", min_value=10, max_value=200, value=50, step=10)

    # Button to trigger loading & optimization
    if st.sidebar.button("Load & Optimize"):
        if len(tickers) == 0:
            st.error("Please provide at least one ticker.")
            return

        with st.spinner("Downloading data & optimizing..."):
            data_dict = download_data(tickers, start_date, end_date)

            # If no data, end
            if len(data_dict) == 0:
                st.error("No data found for the specified tickers/date range.")
                return

            returns_df = combine_returns(data_dict)
            if returns_df.empty:
                st.error("No valid returns data. Check tickers or date range.")
                return

            # Calculate mean returns & covariance
            mean_returns = returns_df.mean()
            cov_matrix = returns_df.cov()

            # Frequency (daily = 252 trading days/year)
            freq = 252

            # --- Optimize for Max Sharpe ---
            weights_sharpe = optimize_portfolio(
                mean_returns,
                cov_matrix,
                freq,
                risk_free_rate,
                objective="sharpe",
                allow_short=allow_short
            )
            ret_sharpe, vol_sharpe, sr_sharpe = portfolio_performance(
                weights_sharpe,
                mean_returns,
                cov_matrix,
                freq,
                risk_free_rate
            )

            # --- Optimize for Min Volatility ---
            weights_min_vol = optimize_portfolio(
                mean_returns,
                cov_matrix,
                freq,
                risk_free_rate,
                objective="min_volatility",
                allow_short=allow_short
            )
            ret_min_vol, vol_min_vol, sr_min_vol = portfolio_performance(
                weights_min_vol,
                mean_returns,
                cov_matrix,
                freq,
                risk_free_rate
            )

            # --- Efficient Frontier ---
            frontier = compute_efficient_frontier(
                mean_returns,
                cov_matrix,
                freq,
                risk_free_rate,
                n_points=n_points,
                allow_short=allow_short
            )
            frontier_vol = [pt[0] for pt in frontier]  # vol
            frontier_ret = [pt[1] for pt in frontier]  # ret

        # ------------------------
        # Create Tabs to Display Results
        # ------------------------
        tab_summary, tab_frontier, tab_correlation, tab_prices = st.tabs([
            "Optimal Portfolios",
            "Efficient Frontier",
            "Correlation Matrix",
            "Historical Prices"
        ])

        # 1) Optimal Portfolios
        with tab_summary:
            st.subheader("Optimal Portfolio Weights")

            # Max Sharpe
            st.markdown("**Maximum Sharpe Ratio Portfolio**")
            df_sharpe = pd.DataFrame({
                'Ticker': tickers,
                'Weight': weights_sharpe
            })
            df_sharpe['Weight'] = df_sharpe['Weight'].apply(lambda x: f"{x:.2%}")
            st.dataframe(df_sharpe)

            st.write(f"Annual Return: **{ret_sharpe:.2%}**")
            st.write(f"Annual Volatility: **{vol_sharpe:.2%}**")
            st.write(f"Sharpe Ratio: **{sr_sharpe:.2f}**")

            st.write("---")

            # Min Vol
            st.markdown("**Minimum Volatility Portfolio**")
            df_min_vol = pd.DataFrame({
                'Ticker': tickers,
                'Weight': weights_min_vol
            })
            df_min_vol['Weight'] = df_min_vol['Weight'].apply(lambda x: f"{x:.2%}")
            st.dataframe(df_min_vol)

            st.write(f"Annual Return: **{ret_min_vol:.2%}**")
            st.write(f"Annual Volatility: **{vol_min_vol:.2%}**")
            st.write(f"Sharpe Ratio: **{sr_min_vol:.2f}**")

            st.write("---")

            # --- Download Button for Results ---
            st.subheader("Download Results")
            csv_buffer = io.StringIO()
            # Combine both optimal portfolios into a single CSV
            combined_df = pd.DataFrame({
                "Ticker": tickers,
                "Max Sharpe Weights": weights_sharpe,
                "Min Vol Weights": weights_min_vol
            })
            combined_df.to_csv(csv_buffer, index=False)

            st.download_button(
                label="Download Optimal Weights as CSV",
                data=csv_buffer.getvalue(),
                file_name="optimal_portfolios.csv",
                mime="text/csv"
            )

        # 2) Efficient Frontier
        with tab_frontier:
            st.subheader("Efficient Frontier Visualization")
            fig_frontier = go.Figure()

            # Plot frontier
            fig_frontier.add_trace(go.Scatter(
                x=frontier_vol,
                y=frontier_ret,
                mode='lines+markers',
                name='Efficient Frontier'
            ))

            # Mark Max Sharpe
            fig_frontier.add_trace(go.Scatter(
                x=[vol_sharpe],
                y=[ret_sharpe],
                mode='markers',
                marker=dict(color='red', size=10),
                name='Max Sharpe'
            ))

            # Mark Min Vol
            fig_frontier.add_trace(go.Scatter(
                x=[vol_min_vol],
                y=[ret_min_vol],
                mode='markers',
                marker=dict(color='green', size=10),
                name='Min Volatility'
            ))

            fig_frontier.update_layout(
                xaxis_title="Annual Volatility (Std Dev)",
                yaxis_title="Annual Return",
                legend_title="Portfolios",
                hovermode="x"
            )
            st.plotly_chart(fig_frontier, use_container_width=True)

        # 3) Correlation Matrix
        with tab_correlation:
            st.subheader("Correlation Matrix of Daily Returns")
            corr_matrix = returns_df.corr()
            st.dataframe(corr_matrix.style.background_gradient(cmap='RdBu', axis=None))

            # Optional Heatmap
            fig_corr = go.Figure(
                data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1
                )
            )
            fig_corr.update_layout(
                width=600, height=600,
                xaxis_title="Ticker",
                yaxis_title="Ticker"
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        # 4) Historical Prices
        with tab_prices:
            st.subheader("Historical Closing Prices")
            fig_prices = go.Figure()
            for ticker, df in data_dict.items():
                fig_prices.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    mode='lines',
                    name=ticker
                ))
            fig_prices.update_layout(
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                hovermode="x"
            )
            st.plotly_chart(fig_prices, use_container_width=True)

    else:
        st.info(
            "Enter tickers (including from different markets, e.g., 'AAPL, BMW.DE, 7203.T') and click **Load & Optimize**."
        )

if __name__ == "__main__":
    main()
