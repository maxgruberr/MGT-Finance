import pandas as pd
import numpy as np

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

def calculate_sharpe_ratio(annualized_return, annualized_volatility, risk_free_rate=0.01):
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

def main():
    etf_files = [
        'EWL ETF Stock Price History.csv',
        'IEF ETF Stock Price History.csv',
        'SPY ETF Stock Price History.csv',
        'VOO ETF Stock Price History.csv'
    ]

    for etf_file in etf_files:
        etf_name = etf_file.split(' ')[0]
        df = load_and_clean_data(etf_file)
        annualized_return = calculate_annualized_return(df)
        annualized_volatility = calculate_annualized_volatility(df)
        sharpe_ratio = calculate_sharpe_ratio(annualized_return, annualized_volatility)

        print(f"--- {etf_name} ---")
        print(f"Annualized Return: {annualized_return:.2%}")
        print(f"Annualized Volatility: {annualized_volatility:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print()

if __name__ == "__main__":
    main()
