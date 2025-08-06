#!/usr/bin/env python3
"""
Financial Analysis Pipeline Example

This example demonstrates how to create comprehensive financial datasets
using TempData, including stock market data, banking transactions, and
financial analytics with realistic market patterns and correlations.
"""

import tempdata
import pandas as pd
from datetime import datetime, timedelta
import os
import numpy as np

def create_stock_market_pipeline():
    """
    Create a comprehensive stock market data pipeline
    """
    print("Creating Stock Market Data Pipeline...")
    print("=" * 40)
    
    # Create output directory
    output_dir = "stock_market_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Daily stock prices for major indices
    print("\n1. Creating daily stock market data...")
    
    # S&P 500 style data - daily for 5 years
    print("   - Generating daily stock prices (5 years)...")
    daily_stocks_path = tempdata.create_dataset(
        f'{output_dir}/daily_stock_prices.csv',
        rows=1260,  # ~5 years of trading days (252 * 5)
        time_series=True,
        start_date='2019-01-01',
        end_date='2024-12-31',
        interval='1day',
        country='united_states',
        seed=12345
    )
    print(f"   ✓ Daily stock prices: {daily_stocks_path}")
    
    # Step 2: High-frequency trading data
    print("\n2. Creating high-frequency trading data...")
    
    # Minute-by-minute data for one trading day
    print("   - Generating minute-level trading data...")
    minute_trading_path = tempdata.create_dataset(
        f'{output_dir}/minute_trading_data.csv',
        rows=390,  # 6.5 hours * 60 minutes (9:30 AM - 4:00 PM)
        time_series=True,
        start_date='2024-01-15 09:30:00',
        end_date='2024-01-15 16:00:00',
        interval='1min',
        country='united_states',
        seed=12345
    )
    print(f"   ✓ Minute trading data: {minute_trading_path}")
    
    # Step 3: Options and derivatives data
    print("\n3. Creating derivatives market data...")
    
    # Options trading data
    print("   - Generating options trading data...")
    options_path = tempdata.create_dataset(
        f'{output_dir}/options_trading.csv',
        rows=10000,
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        country='united_states',
        seed=12345
    )
    print(f"   ✓ Options trading: {options_path}")
    
    # Step 4: Market volatility and risk metrics
    print("\n4. Creating risk and volatility data...")
    
    # VIX-style volatility index
    print("   - Generating volatility index data...")
    volatility_path = tempdata.create_dataset(
        f'{output_dir}/volatility_index.csv',
        rows=252,  # Daily for one year
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='1day',
        seed=12345
    )
    print(f"   ✓ Volatility index: {volatility_path}")
    
    print(f"\nStock Market Pipeline complete in: {output_dir}/")
    return output_dir

def create_banking_system_pipeline():
    """
    Create a comprehensive banking system data pipeline
    """
    print("\nCreating Banking System Pipeline...")
    print("=" * 35)
    
    banking_dir = "banking_system_data"
    os.makedirs(banking_dir, exist_ok=True)
    
    # Step 1: Customer accounts and profiles
    print("\n1. Creating customer account data...")
    
    # Bank customer profiles
    print("   - Generating customer profiles...")
    customers_path = tempdata.create_dataset(
        f'{banking_dir}/bank_customers.csv',
        rows=10000,
        country='united_states',
        seed=12345
    )
    print(f"   ✓ Bank customers: {customers_path}")
    
    # Account information
    print("   - Generating account information...")
    accounts_path = tempdata.create_dataset(
        f'{banking_dir}/bank_accounts.csv',
        rows=15000,  # Some customers have multiple accounts
        country='united_states',
        seed=12345
    )
    print(f"   ✓ Bank accounts: {accounts_path}")
    
    # Step 2: Transaction data
    print("\n2. Creating transaction data...")
    
    # Daily banking transactions
    print("   - Generating daily transactions...")
    transactions_path = tempdata.create_dataset(
        f'{banking_dir}/daily_transactions.csv',
        rows=500000,  # Large volume of daily transactions
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='1hour',  # Transactions throughout the day
        country='united_states',
        use_streaming=True,  # Large dataset
        formats=['parquet', 'csv'],
        seed=12345
    )
    print(f"   ✓ Daily transactions: {transactions_path}")
    
    # ATM transactions
    print("   - Generating ATM transactions...")
    atm_transactions_path = tempdata.create_dataset(
        f'{banking_dir}/atm_transactions.csv',
        rows=50000,
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        country='united_states',
        seed=12345
    )
    print(f"   ✓ ATM transactions: {atm_transactions_path}")
    
    # Step 3: Credit and loan data
    print("\n3. Creating credit and loan data...")
    
    # Loan applications and approvals
    print("   - Generating loan data...")
    loans_path = tempdata.create_dataset(
        f'{banking_dir}/loan_applications.csv',
        rows=5000,
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        country='united_states',
        seed=12345
    )
    print(f"   ✓ Loan applications: {loans_path}")
    
    # Credit card transactions
    print("   - Generating credit card transactions...")
    credit_card_path = tempdata.create_dataset(
        f'{banking_dir}/credit_card_transactions.csv',
        rows=200000,
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        country='united_states',
        use_streaming=True,
        formats=['parquet'],
        seed=12345
    )
    print(f"   ✓ Credit card transactions: {credit_card_path}")
    
    # Step 4: Fraud detection data
    print("\n4. Creating fraud detection data...")
    
    # Suspicious activity reports
    print("   - Generating fraud detection data...")
    fraud_path = tempdata.create_dataset(
        f'{banking_dir}/fraud_detection.csv',
        rows=2000,  # Relatively rare events
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        seed=12345
    )
    print(f"   ✓ Fraud detection: {fraud_path}")
    
    print(f"\nBanking System Pipeline complete in: {banking_dir}/")
    return banking_dir

def create_cryptocurrency_pipeline():
    """
    Create cryptocurrency market data pipeline
    """
    print("\nCreating Cryptocurrency Pipeline...")
    print("=" * 32)
    
    crypto_dir = "cryptocurrency_data"
    os.makedirs(crypto_dir, exist_ok=True)
    
    # Step 1: Major cryptocurrency prices
    print("\n1. Creating cryptocurrency price data...")
    
    # Daily crypto prices
    print("   - Generating daily crypto prices...")
    crypto_daily_path = tempdata.create_dataset(
        f'{crypto_dir}/crypto_daily_prices.csv',
        rows=365,  # Daily for one year
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='1day',
        seed=12345
    )
    print(f"   ✓ Daily crypto prices: {crypto_daily_path}")
    
    # Hourly crypto data for volatility analysis
    print("   - Generating hourly crypto data...")
    crypto_hourly_path = tempdata.create_dataset(
        f'{crypto_dir}/crypto_hourly_prices.csv',
        rows=8760,  # Hourly for one year
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='1hour',
        use_streaming=True,
        formats=['parquet'],
        seed=12345
    )
    print(f"   ✓ Hourly crypto prices: {crypto_hourly_path}")
    
    # Step 2: DeFi and trading data
    print("\n2. Creating DeFi trading data...")
    
    # Decentralized exchange transactions
    print("   - Generating DEX transactions...")
    dex_transactions_path = tempdata.create_dataset(
        f'{crypto_dir}/dex_transactions.csv',
        rows=100000,
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        use_streaming=True,
        formats=['parquet'],
        seed=12345
    )
    print(f"   ✓ DEX transactions: {dex_transactions_path}")
    
    print(f"\nCryptocurrency Pipeline complete in: {crypto_dir}/")
    return crypto_dir

def create_financial_risk_pipeline():
    """
    Create financial risk management data pipeline
    """
    print("\nCreating Financial Risk Pipeline...")
    print("=" * 32)
    
    risk_dir = "financial_risk_data"
    os.makedirs(risk_dir, exist_ok=True)
    
    # Step 1: Portfolio risk metrics
    print("\n1. Creating portfolio risk data...")
    
    # Daily portfolio valuations
    print("   - Generating portfolio valuations...")
    portfolio_path = tempdata.create_dataset(
        f'{risk_dir}/portfolio_valuations.csv',
        rows=252,  # Daily for one trading year
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='1day',
        seed=12345
    )
    print(f"   ✓ Portfolio valuations: {portfolio_path}")
    
    # Step 2: Credit risk data
    print("\n2. Creating credit risk data...")
    
    # Credit ratings and defaults
    print("   - Generating credit risk data...")
    credit_risk_path = tempdata.create_dataset(
        f'{risk_dir}/credit_risk_assessments.csv',
        rows=5000,
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        seed=12345
    )
    print(f"   ✓ Credit risk assessments: {credit_risk_path}")
    
    # Step 3: Market risk scenarios
    print("\n3. Creating market risk scenarios...")
    
    # Stress test scenarios
    print("   - Generating stress test data...")
    stress_test_path = tempdata.create_dataset(
        f'{risk_dir}/stress_test_scenarios.csv',
        rows=1000,
        seed=12345
    )
    print(f"   ✓ Stress test scenarios: {stress_test_path}")
    
    print(f"\nFinancial Risk Pipeline complete in: {risk_dir}/")
    return risk_dir

def create_financial_batch_pipeline():
    """
    Create related financial datasets using batch generation
    """
    print("\nCreating Financial Batch Pipeline...")
    print("=" * 35)
    
    batch_dir = "financial_batch_data"
    os.makedirs(batch_dir, exist_ok=True)
    
    # Define related financial datasets
    financial_batch = [
        {
            'filename': f'{batch_dir}/financial_institutions.csv',
            'rows': 100  # Banks, brokers, etc.
        },
        {
            'filename': f'{batch_dir}/financial_products.csv',
            'rows': 500,  # Stocks, bonds, funds, etc.
            'relationships': ['financial_institutions']
        },
        {
            'filename': f'{batch_dir}/client_portfolios.csv',
            'rows': 2000,
            'relationships': ['financial_institutions']
        },
        {
            'filename': f'{batch_dir}/portfolio_holdings.csv',
            'rows': 20000,
            'relationships': ['client_portfolios', 'financial_products']
        },
        {
            'filename': f'{batch_dir}/trading_transactions.csv',
            'rows': 100000,
            'relationships': ['portfolio_holdings'],
            'time_series': True,
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'interval': '1hour'
        }
    ]
    
    # Generate batch with relationships
    batch_paths = tempdata.create_batch(
        financial_batch,
        country='united_states',
        formats=['parquet', 'csv'],
        seed=12345
    )
    
    for path in batch_paths:
        print(f"   ✓ {path}")
    
    print(f"\nFinancial Batch Pipeline complete in: {batch_dir}/")
    return batch_dir

def create_multi_country_financial_data():
    """
    Create financial data for multiple countries/markets
    """
    print("\nCreating Multi-Country Financial Data...")
    print("=" * 38)
    
    global_dir = "global_financial_data"
    os.makedirs(global_dir, exist_ok=True)
    
    # Major financial markets
    markets = {
        'united_states': 'NYSE/NASDAQ',
        'united_kingdom': 'LSE',
        'germany': 'DAX',
        'japan': 'Nikkei',
        'australia': 'ASX'
    }
    
    for country, market_name in markets.items():
        print(f"\n   Creating {market_name} data ({country})...")
        
        # Daily market data
        market_path = tempdata.create_dataset(
            f'{global_dir}/market_data_{country}.csv',
            rows=252,  # One trading year
            time_series=True,
            start_date='2024-01-01',
            end_date='2024-12-31',
            interval='1day',
            country=country,
            seed=12345
        )
        print(f"     ✓ Market data: {market_path}")
        
        # Banking data for each country
        banking_path = tempdata.create_dataset(
            f'{global_dir}/banking_data_{country}.csv',
            rows=10000,
            time_series=True,
            start_date='2024-01-01',
            end_date='2024-12-31',
            country=country,
            seed=12345
        )
        print(f"     ✓ Banking data: {banking_path}")
    
    print(f"\nGlobal Financial Data complete in: {global_dir}/")
    return global_dir

def analyze_financial_data(data_dirs):
    """
    Perform basic analysis on generated financial data
    """
    print("\nPerforming Financial Data Analysis...")
    print("-" * 35)
    
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            continue
            
        print(f"\nAnalyzing {data_dir}:")
        
        # Find CSV files in the directory
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        for csv_file in csv_files[:3]:  # Analyze first 3 files
            try:
                file_path = os.path.join(data_dir, csv_file)
                df = pd.read_csv(file_path)
                
                print(f"  {csv_file}:")
                print(f"    Records: {len(df):,}")
                print(f"    Columns: {len(df.columns)}")
                
                # Financial-specific analysis
                if 'amount' in df.columns:
                    total_amount = df['amount'].sum()
                    avg_amount = df['amount'].mean()
                    print(f"    Total amount: ${total_amount:,.2f}")
                    print(f"    Average amount: ${avg_amount:.2f}")
                
                if 'price' in df.columns or 'close_price' in df.columns:
                    price_col = 'price' if 'price' in df.columns else 'close_price'
                    price_range = df[price_col].max() - df[price_col].min()
                    print(f"    Price range: ${df[price_col].min():.2f} - ${df[price_col].max():.2f}")
                    print(f"    Price volatility: ${price_range:.2f}")
                
                if 'volume' in df.columns:
                    total_volume = df['volume'].sum()
                    print(f"    Total volume: {total_volume:,}")
                
                # Time series analysis
                if 'date' in df.columns or 'timestamp' in df.columns:
                    time_col = 'date' if 'date' in df.columns else 'timestamp'
                    print(f"    Time range: {df[time_col].min()} to {df[time_col].max()}")
                
            except Exception as e:
                print(f"    Error analyzing {csv_file}: {e}")

def create_algorithmic_trading_data():
    """
    Create data for algorithmic trading strategy testing
    """
    print("\nCreating Algorithmic Trading Data...")
    print("=" * 35)
    
    algo_dir = "algorithmic_trading_data"
    os.makedirs(algo_dir, exist_ok=True)
    
    # High-frequency data for backtesting
    print("   - Generating tick-level data...")
    tick_data_path = tempdata.create_dataset(
        f'{algo_dir}/tick_level_data.csv',
        rows=100000,  # 100k ticks
        time_series=True,
        start_date='2024-01-15 09:30:00',
        end_date='2024-01-15 16:00:00',
        interval='1min',
        use_streaming=True,
        formats=['parquet'],
        seed=12345
    )
    print(f"   ✓ Tick-level data: {tick_data_path}")
    
    # Order book data
    print("   - Generating order book data...")
    orderbook_path = tempdata.create_dataset(
        f'{algo_dir}/order_book_data.csv',
        rows=50000,
        time_series=True,
        start_date='2024-01-15 09:30:00',
        end_date='2024-01-15 16:00:00',
        interval='1min',
        formats=['parquet'],
        seed=12345
    )
    print(f"   ✓ Order book data: {orderbook_path}")
    
    print(f"\nAlgorithmic Trading Data complete in: {algo_dir}/")
    return algo_dir

def main():
    """
    Main function to run all financial pipeline examples
    """
    print("TempData Financial Analysis Pipeline Examples")
    print("=" * 50)
    
    # Create different financial pipelines
    stock_dir = create_stock_market_pipeline()
    banking_dir = create_banking_system_pipeline()
    crypto_dir = create_cryptocurrency_pipeline()
    risk_dir = create_financial_risk_pipeline()
    batch_dir = create_financial_batch_pipeline()
    global_dir = create_multi_country_financial_data()
    algo_dir = create_algorithmic_trading_data()
    
    # Analyze generated data
    all_dirs = [stock_dir, banking_dir, crypto_dir, risk_dir, batch_dir, global_dir, algo_dir]
    analyze_financial_data(all_dirs)
    
    print("\n" + "=" * 50)
    print("All Financial Pipeline Examples Complete!")
    print("\nGenerated directories:")
    print(f"  - Stock Market Data: {stock_dir}/")
    print(f"  - Banking System Data: {banking_dir}/")
    print(f"  - Cryptocurrency Data: {crypto_dir}/")
    print(f"  - Financial Risk Data: {risk_dir}/")
    print(f"  - Batch Financial Data: {batch_dir}/")
    print(f"  - Global Financial Data: {global_dir}/")
    print(f"  - Algorithmic Trading Data: {algo_dir}/")
    
    print("\nFinancial analysis use cases:")
    print("  1. Portfolio risk assessment and optimization")
    print("  2. Algorithmic trading strategy backtesting")
    print("  3. Fraud detection model training")
    print("  4. Credit risk modeling and scoring")
    print("  5. Market volatility analysis and forecasting")
    print("  6. Regulatory compliance testing")
    print("  7. High-frequency trading system testing")
    print("  8. Cryptocurrency market analysis")
    print("  9. Cross-market correlation studies")
    print("  10. Stress testing and scenario analysis")

if __name__ == "__main__":
    main()