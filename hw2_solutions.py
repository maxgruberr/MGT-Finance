import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def load_and_clean_data(filepath):
    """
    Loads and cleans the ETF data from a CSV file.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A cleaned DataFrame with 'Date' as datetime objects
                      and 'Price' and 'Change %' as numeric types.
    """
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df['Price'] = pd.to_numeric(df['Price'])
    df['Change %'] = df['Change %'].str.rstrip('%').astype('float') / 100.0
    return df

def calculate_annualized_return(df):
    """
    Calculates the annualized return of an ETF.

    Args:
        df (pd.DataFrame): The cleaned DataFrame with 'Change %' data.

    Returns:
        float: The annualized return.
    """
    mean_weekly_return = df['Change %'].mean()
    annualized_return = (1 + mean_weekly_return)**52 - 1
    return annualized_return

def calculate_annualized_volatility(df):
    """
    Calculates the annualized volatility of an ETF.

    Args:
        df (pd.DataFrame): The cleaned DataFrame with 'Change %' data.

    Returns:
        float: The annualized volatility.
    """
    weekly_volatility = df['Change %'].std()
    annualized_volatility = weekly_volatility * np.sqrt(52)
    return annualized_volatility

def calculate_sharpe_ratio(annualized_return, annualized_volatility, risk_free_rate=0.015):
    """
    Calculates the Sharpe ratio of an ETF.

    Args:
        annualized_return (float): The annualized return.
        annualized_volatility (float): The annualized volatility.
        risk_free_rate (float): The risk-free rate.

    Returns:
        float: The Sharpe ratio.
    """
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    return sharpe_ratio

def calculate_tangency_portfolio(returns):
    """
    Calculates the tangency portfolio for a set of assets.
    """
    cov_matrix = returns.cov() * 52
    expected_returns = (returns.mean() + 1)**52 - 1
    risk_free_rate = 0.015

    def objective(weights):
        return -(expected_returns.dot(weights) - risk_free_rate) / np.sqrt(weights.T.dot(cov_matrix).dot(weights))

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(len(expected_returns)))
    initial_weights = np.array([1/len(expected_returns)] * len(expected_returns))

    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    weights = result.x

    mean = expected_returns.dot(weights)
    variance = weights.T.dot(cov_matrix).dot(weights)
    std_dev = np.sqrt(variance)
    sharpe = (mean - risk_free_rate) / std_dev

    return weights, mean, variance, std_dev, sharpe

def plot_frontier(returns, tangency_portfolio):
    """
    Plots the mean-standard deviation frontier and the Capital Allocation Line.
    """
    expected_returns = (returns.mean() + 1)**52 - 1
    cov_matrix = returns.cov() * 52
    risk_free_rate = 0.015

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot individual assets
    for i, txt in enumerate(expected_returns.index):
        ax.scatter(np.sqrt(cov_matrix.iloc[i, i]), expected_returns[i], marker='o', s=100, label=txt)

    # Plot efficient frontier
    port_returns = []
    port_vols = []
    for _ in range(2500):
        weights = np.random.random(len(expected_returns))
        weights /= np.sum(weights)
        port_returns.append(expected_returns.dot(weights))
        port_vols.append(np.sqrt(weights.T.dot(cov_matrix).dot(weights)))
    ax.scatter(port_vols, port_returns, marker='.', alpha=0.5)

    # Plot tangency portfolio and CAL
    tangency_std = tangency_portfolio[3]
    tangency_mean = tangency_portfolio[1]
    ax.scatter(tangency_std, tangency_mean, marker='*', s=200, color='r', label='Tangency Portfolio')
    cal_x = [0, tangency_std * 1.5]
    cal_y = [risk_free_rate, risk_free_rate + (tangency_mean - risk_free_rate) * 1.5]
    ax.plot(cal_x, cal_y, linestyle='--', color='r', label='Capital Allocation Line')

    ax.set_title('Mean-Standard Deviation Frontier')
    ax.set_xlabel('Annualized Volatility (Standard Deviation)')
    ax.set_ylabel('Annualized Expected Return')
    ax.legend()
    ax.grid(True)
    plt.savefig('efficient_frontier.png')

def optimize_for_target_volatility(tangency_portfolio, target_volatility=0.15):
    """
    Calculates the optimal weights for a target volatility and the implied risk-aversion coefficient.
    """
    tangency_weights, tangency_mean, _, tangency_std, _ = tangency_portfolio
    risk_free_rate = 0.015

    # Calculate the weight in the tangency portfolio
    weight_in_tangency = target_volatility / tangency_std

    # Calculate the optimal weights in the three risky ETFs
    optimal_weights = weight_in_tangency * tangency_weights

    # Calculate the implied risk-aversion coefficient
    implied_risk_aversion = (tangency_mean - risk_free_rate) / (weight_in_tangency * tangency_std**2)

    return optimal_weights, implied_risk_aversion

def main():
    etf_files = [
        'EWL ETF Stock Price History.csv',
        'IEF ETF Stock Price History.csv',
        'SPY ETF Stock Price History.csv'
    ]

    returns = pd.DataFrame()
    for etf_file in etf_files:
        etf_name = etf_file.split(' ')[0]
        df = load_and_clean_data(etf_file)
        returns[etf_name] = df['Change %']

    tangency_portfolio = calculate_tangency_portfolio(returns)
    weights, mean, variance, std_dev, sharpe = tangency_portfolio

    print("--- Tangency Portfolio ---")
    etf_names = [name.split(' ')[0] for name in etf_files]
    for i, etf_name in enumerate(etf_names):
        print(f"Weight for {etf_name}: {weights[i]:.2%}")
    print(f"Mean: {mean:.2%}")
    print(f"Variance: {variance:.4f}")
    print(f"Standard Deviation: {std_dev:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print()

    plot_frontier(returns, tangency_portfolio)

    optimal_weights, implied_risk_aversion = optimize_for_target_volatility(tangency_portfolio)

    print("--- Portfolio for 15% Target Volatility ---")
    for i, etf_name in enumerate(etf_names):
        print(f"Optimal Weight for {etf_name}: {optimal_weights[i]:.2%}")
    print(f"Implied Risk-Aversion Coefficient: {implied_risk_aversion:.2f}")
    print()

if __name__ == "__main__":
    main()
