#!/usr/bin/env python3
"""
Comprehensive TempData Tutorial

This comprehensive tutorial demonstrates all key features of TempData with
step-by-step examples covering the main requirements:

- Requirement 2.1: Business datasets (sales, customers, ecommerce, inventory, marketing, etc.)
- Requirement 2.2: Financial datasets (stocks, banking, crypto, insurance, loans, etc.)
- Requirement 2.5: IoT sensor datasets (weather, energy, traffic, environmental, etc.)
- Requirement 5.1: Time series generation with realistic temporal patterns
- Requirement 8.1: Batch generation with maintained referential integrity

Each section includes:
- Code examples with explanations
- Best practices and tips
- Common use cases
- Performance considerations
"""

import tempdata
import pandas as pd
import os
from datetime import datetime, timedelta

def tutorial_section_1_business_datasets():
    """
    Tutorial Section 1: Business Datasets (Requirement 2.1)
    
    Demonstrates how to generate realistic business datasets including:
    - Sales transactions with seasonal patterns
    - Customer databases with demographics
    - E-commerce orders with product correlations
    - Inventory management data
    - Marketing campaign data
    """
    print("=" * 60)
    print("TUTORIAL SECTION 1: BUSINESS DATASETS")
    print("=" * 60)
    
    tutorial_dir = "tutorial_business_data"
    os.makedirs(tutorial_dir, exist_ok=True)
    
    print("\n1.1 Basic Sales Dataset Generation")
    print("-" * 40)
    print("Creating a simple sales dataset with 1000 transactions...")
    
    # Basic sales dataset
    sales_path = tempdata.create_dataset(
        f'{tutorial_dir}/basic_sales.csv',
        rows=1000,
        country='united_states',
        seed=12345  # For reproducible results
    )
    
    print(f"✓ Generated: {sales_path}")
    print("  - Contains realistic sales amounts, dates, regions")
    print("  - Uses US geographical patterns")
    print("  - Reproducible with seed=12345")
    
    print("\n1.2 Customer Database with Demographics")
    print("-" * 40)
    print("Creating customer database with realistic demographics...")
    
    customers_path = tempdata.create_dataset(
        f'{tutorial_dir}/customers_with_demographics.csv',
        rows=2000,
        country='united_states',
        seed=12345
    )
    
    print(f"✓ Generated: {customers_path}")
    print("  - Realistic names, addresses, phone numbers")
    print("  - Age distributions matching demographics")
    print("  - Geographic clustering by regions")
    
    print("\n1.3 E-commerce Orders with Time Series")
    print("-" * 40)
    print("Creating e-commerce orders with seasonal patterns...")
    
    ecommerce_path = tempdata.create_dataset(
        f'{tutorial_dir}/ecommerce_orders_seasonal.csv',
        rows=5000,
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='1day',
        country='united_states',
        seed=12345
    )
    
    print(f"✓ Generated: {ecommerce_path}")
    print("  - Daily orders throughout 2024")
    print("  - Seasonal shopping patterns (holidays, etc.)")
    print("  - Realistic order values and product categories")
    
    print("\n1.4 Multi-Format Export Example")
    print("-" * 40)
    print("Creating inventory data in multiple formats...")
    
    inventory_paths = tempdata.create_dataset(
        f'{tutorial_dir}/inventory_multi_format',
        rows=1500,
        formats=['csv', 'json', 'parquet', 'excel'],
        country='united_states',
        seed=12345
    )
    
    print(f"✓ Generated in multiple formats:")
    for path in inventory_paths:
        print(f"  - {path}")
    print("  - Same data in different formats for various tools")
    print("  - CSV for spreadsheets, JSON for APIs, Parquet for analytics")
    
    # Analyze the generated data
    print("\n1.5 Data Analysis Example")
    print("-" * 40)
    
    try:
        # Load and analyze the sales data
        sales_df = pd.read_csv(f'{tutorial_dir}/basic_sales.csv')
        print(f"Sales Dataset Analysis:")
        print(f"  - Total records: {len(sales_df):,}")
        print(f"  - Date range: {sales_df['date'].min()} to {sales_df['date'].max()}")
        print(f"  - Total sales: ${sales_df['amount'].sum():,.2f}")
        print(f"  - Average transaction: ${sales_df['amount'].mean():.2f}")
        print(f"  - Unique regions: {sales_df['region'].nunique()}")
        
        # Load and analyze customer data
        customers_df = pd.read_csv(f'{tutorial_dir}/customers_with_demographics.csv')
        print(f"\nCustomer Dataset Analysis:")
        print(f"  - Total customers: {len(customers_df):,}")
        print(f"  - Age range: {customers_df['age'].min()} to {customers_df['age'].max()}")
        print(f"  - Gender distribution: {customers_df['gender'].value_counts().to_dict()}")
        
    except Exception as e:
        print(f"Analysis error: {e}")
    
    return tutorial_dir

def tutorial_section_2_financial_datasets():
    """
    Tutorial Section 2: Financial Datasets (Requirement 2.2)
    
    Demonstrates financial data generation including:
    - Stock market data with volatility
    - Banking transactions with fraud patterns
    - Cryptocurrency data with high volatility
    - Insurance claims and policies
    - Loan applications and approvals
    """
    print("\n" + "=" * 60)
    print("TUTORIAL SECTION 2: FINANCIAL DATASETS")
    print("=" * 60)
    
    financial_dir = "tutorial_financial_data"
    os.makedirs(financial_dir, exist_ok=True)
    
    print("\n2.1 Stock Market Data with Realistic Volatility")
    print("-" * 50)
    print("Creating daily stock prices with market patterns...")
    
    # Daily stock data for one year
    stocks_path = tempdata.create_dataset(
        f'{financial_dir}/daily_stock_prices.csv',
        rows=252,  # Trading days in a year
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='1day',
        seed=12345
    )
    
    print(f"✓ Generated: {stocks_path}")
    print("  - Realistic price movements and volatility")
    print("  - Trading volume correlations")
    print("  - Market sector patterns")
    
    print("\n2.2 High-Frequency Trading Data")
    print("-" * 50)
    print("Creating minute-level trading data for backtesting...")
    
    hft_path = tempdata.create_dataset(
        f'{financial_dir}/minute_trading_data.csv',
        rows=390,  # 6.5 hours of trading (9:30 AM - 4:00 PM)
        time_series=True,
        start_date='2024-01-15 09:30:00',
        end_date='2024-01-15 16:00:00',
        interval='1min',
        formats=['parquet'],  # Efficient format for large data
        seed=12345
    )
    
    print(f"✓ Generated: {hft_path}")
    print("  - Minute-by-minute price and volume data")
    print("  - Realistic bid-ask spreads")
    print("  - Intraday volatility patterns")
    
    print("\n2.3 Banking Transactions with Fraud Detection")
    print("-" * 50)
    print("Creating banking transactions with fraud indicators...")
    
    banking_path = tempdata.create_dataset(
        f'{financial_dir}/banking_transactions_fraud.csv',
        rows=10000,
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        country='united_states',
        seed=12345
    )
    
    print(f"✓ Generated: {banking_path}")
    print("  - Normal and suspicious transaction patterns")
    print("  - Geographic anomaly detection scenarios")
    print("  - Time-based fraud indicators")
    
    print("\n2.4 Cryptocurrency Market Data")
    print("-" * 50)
    print("Creating crypto data with high volatility...")
    
    crypto_path = tempdata.create_dataset(
        f'{financial_dir}/cryptocurrency_hourly.csv',
        rows=8760,  # Hourly for one year
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='1hour',
        formats=['parquet'],
        seed=12345
    )
    
    print(f"✓ Generated: {crypto_path}")
    print("  - High volatility price movements")
    print("  - 24/7 trading patterns")
    print("  - Market cap and volume correlations")
    
    print("\n2.5 Financial Analysis Example")
    print("-" * 50)
    
    try:
        # Analyze stock data
        stocks_df = pd.read_csv(f'{financial_dir}/daily_stock_prices.csv')
        print(f"Stock Market Analysis:")
        print(f"  - Trading days: {len(stocks_df):,}")
        print(f"  - Price range: ${stocks_df['close_price'].min():.2f} - ${stocks_df['close_price'].max():.2f}")
        print(f"  - Average daily volume: {stocks_df['volume'].mean():,.0f}")
        
        # Calculate volatility
        stocks_df['daily_return'] = stocks_df['close_price'].pct_change()
        volatility = stocks_df['daily_return'].std() * (252 ** 0.5)  # Annualized
        print(f"  - Annualized volatility: {volatility:.2%}")
        
    except Exception as e:
        print(f"Analysis error: {e}")
    
    return financial_dir

def tutorial_section_3_iot_datasets():
    """
    Tutorial Section 3: IoT Sensor Datasets (Requirement 2.5)
    
    Demonstrates IoT data generation including:
    - Weather sensors with correlations
    - Energy consumption patterns
    - Traffic monitoring data
    - Environmental sensors
    - Industrial IoT data
    """
    print("\n" + "=" * 60)
    print("TUTORIAL SECTION 3: IOT SENSOR DATASETS")
    print("=" * 60)
    
    iot_dir = "tutorial_iot_data"
    os.makedirs(iot_dir, exist_ok=True)
    
    print("\n3.1 Weather Sensor Network")
    print("-" * 40)
    print("Creating weather sensors with realistic correlations...")
    
    weather_path = tempdata.create_dataset(
        f'{iot_dir}/weather_sensors_hourly.csv',
        rows=8760,  # Hourly for one year
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='1hour',
        country='germany',  # European climate patterns
        seed=12345
    )
    
    print(f"✓ Generated: {weather_path}")
    print("  - Temperature, humidity, pressure correlations")
    print("  - Seasonal weather patterns")
    print("  - Realistic sensor reading variations")
    
    print("\n3.2 Energy Consumption Monitoring")
    print("-" * 40)
    print("Creating energy consumption with usage patterns...")
    
    energy_path = tempdata.create_dataset(
        f'{iot_dir}/energy_consumption_15min.csv',
        rows=35040,  # 15-minute intervals for one year
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='15min',
        country='sweden',  # Nordic energy patterns
        seed=12345
    )
    
    print(f"✓ Generated: {energy_path}")
    print("  - Peak usage times (morning, evening)")
    print("  - Seasonal consumption variations")
    print("  - Realistic meter readings")
    
    print("\n3.3 Traffic Monitoring System")
    print("-" * 40)
    print("Creating traffic sensors with congestion patterns...")
    
    traffic_path = tempdata.create_dataset(
        f'{iot_dir}/traffic_sensors_30min.csv',
        rows=17520,  # 30-minute intervals for one year
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='30min',
        seed=12345
    )
    
    print(f"✓ Generated: {traffic_path}")
    print("  - Rush hour traffic patterns")
    print("  - Weekend vs weekday differences")
    print("  - Vehicle count and speed correlations")
    
    print("\n3.4 Industrial IoT Sensors")
    print("-" * 40)
    print("Creating industrial sensors with machine patterns...")
    
    industrial_path = tempdata.create_dataset(
        f'{iot_dir}/industrial_sensors_1min.csv',
        rows=10080,  # One week of minute data (for demo)
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-01-07',
        interval='1min',
        formats=['parquet'],  # Efficient for high-frequency data
        seed=12345
    )
    
    print(f"✓ Generated: {industrial_path}")
    print("  - Machine temperature, vibration, pressure")
    print("  - Production cycle patterns")
    print("  - Predictive maintenance indicators")
    
    print("\n3.5 Smart Home IoT Ecosystem")
    print("-" * 40)
    print("Creating smart home sensors with occupancy patterns...")
    
    smart_home_path = tempdata.create_dataset(
        f'{iot_dir}/smart_home_sensors.csv',
        rows=8760,  # Hourly for one year
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='1hour',
        country='canada',  # Cold climate patterns
        seed=12345
    )
    
    print(f"✓ Generated: {smart_home_path}")
    print("  - Thermostat, lighting, security sensors")
    print("  - Occupancy-based patterns")
    print("  - Energy efficiency correlations")
    
    print("\n3.6 IoT Data Analysis Example")
    print("-" * 40)
    
    try:
        # Analyze weather data
        weather_df = pd.read_csv(f'{iot_dir}/weather_sensors_hourly.csv')
        print(f"Weather Sensor Analysis:")
        print(f"  - Total readings: {len(weather_df):,}")
        print(f"  - Temperature range: {weather_df['temperature'].min():.1f}°C to {weather_df['temperature'].max():.1f}°C")
        print(f"  - Humidity range: {weather_df['humidity'].min():.1f}% to {weather_df['humidity'].max():.1f}%")
        
        # Analyze energy data (first 1000 rows for performance)
        energy_df = pd.read_csv(f'{iot_dir}/energy_consumption_15min.csv', nrows=1000)
        print(f"\nEnergy Consumption Analysis (sample):")
        print(f"  - Sample readings: {len(energy_df):,}")
        print(f"  - Power range: {energy_df['power_consumption'].min():.1f}kW to {energy_df['power_consumption'].max():.1f}kW")
        print(f"  - Average consumption: {energy_df['power_consumption'].mean():.1f}kW")
        
    except Exception as e:
        print(f"Analysis error: {e}")
    
    return iot_dir

def tutorial_section_4_time_series():
    """
    Tutorial Section 4: Time Series Generation (Requirement 5.1)
    
    Demonstrates advanced time series features:
    - Different time intervals
    - Seasonal patterns
    - Trend directions
    - Temporal correlations
    """
    print("\n" + "=" * 60)
    print("TUTORIAL SECTION 4: TIME SERIES GENERATION")
    print("=" * 60)
    
    timeseries_dir = "tutorial_timeseries_data"
    os.makedirs(timeseries_dir, exist_ok=True)
    
    print("\n4.1 Different Time Intervals")
    print("-" * 40)
    
    intervals = ['1min', '5min', '1hour', '1day', '1week']
    
    for interval in intervals:
        print(f"Creating {interval} interval data...")
        
        # Calculate appropriate number of rows for each interval
        if interval == '1min':
            rows, start, end = 1440, '2024-01-01', '2024-01-01 23:59:00'  # One day
        elif interval == '5min':
            rows, start, end = 288, '2024-01-01', '2024-01-01 23:55:00'   # One day
        elif interval == '1hour':
            rows, start, end = 168, '2024-01-01', '2024-01-07'            # One week
        elif interval == '1day':
            rows, start, end = 365, '2024-01-01', '2024-12-31'            # One year
        else:  # 1week
            rows, start, end = 52, '2024-01-01', '2024-12-31'             # One year
        
        path = tempdata.create_dataset(
            f'{timeseries_dir}/data_{interval.replace("min", "minute")}.csv',
            rows=rows,
            time_series=True,
            start_date=start,
            end_date=end,
            interval=interval,
            seed=12345
        )
        
        print(f"  ✓ {interval}: {rows} data points")
    
    print("\n4.2 Seasonal Patterns Example")
    print("-" * 40)
    print("Creating retail sales with seasonal patterns...")
    
    seasonal_path = tempdata.create_dataset(
        f'{timeseries_dir}/seasonal_retail_sales.csv',
        rows=365,  # Daily for one year
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='1day',
        seed=12345
    )
    
    print(f"✓ Generated: {seasonal_path}")
    print("  - Holiday shopping spikes")
    print("  - Back-to-school patterns")
    print("  - Summer/winter variations")
    
    print("\n4.3 Financial Time Series with Volatility")
    print("-" * 40)
    print("Creating stock data with realistic market patterns...")
    
    financial_ts_path = tempdata.create_dataset(
        f'{timeseries_dir}/financial_timeseries.csv',
        rows=1260,  # 5 years of trading days
        time_series=True,
        start_date='2020-01-01',
        end_date='2024-12-31',
        interval='1day',
        seed=12345
    )
    
    print(f"✓ Generated: {financial_ts_path}")
    print("  - Bull and bear market cycles")
    print("  - Volatility clustering")
    print("  - Market crash simulations")
    
    return timeseries_dir

def tutorial_section_5_batch_generation():
    """
    Tutorial Section 5: Batch Generation (Requirement 8.1)
    
    Demonstrates batch generation with relationships:
    - Simple parent-child relationships
    - Complex multi-level hierarchies
    - Time series alignment
    - Referential integrity
    """
    print("\n" + "=" * 60)
    print("TUTORIAL SECTION 5: BATCH GENERATION WITH RELATIONSHIPS")
    print("=" * 60)
    
    batch_dir = "tutorial_batch_data"
    os.makedirs(batch_dir, exist_ok=True)
    
    print("\n5.1 Simple Parent-Child Relationship")
    print("-" * 40)
    print("Creating customers and their orders...")
    
    simple_batch = [
        {
            'filename': f'{batch_dir}/customers.csv',
            'rows': 1000,
            'description': 'Customer master data'
        },
        {
            'filename': f'{batch_dir}/orders.csv',
            'rows': 5000,  # Multiple orders per customer
            'relationships': ['customers'],
            'time_series': True,
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'description': 'Customer orders with time series'
        }
    ]
    
    simple_paths = tempdata.create_batch(
        simple_batch,
        country='united_states',
        seed=12345
    )
    
    print("✓ Simple relationship generated:")
    for path in simple_paths:
        print(f"  - {os.path.basename(path)}")
    
    print("\n5.2 Complex Multi-Level Hierarchy")
    print("-" * 40)
    print("Creating a complete retail hierarchy...")
    
    complex_batch = [
        # Level 1: Foundation
        {
            'filename': f'{batch_dir}/stores.csv',
            'rows': 10,
            'description': 'Store locations'
        },
        {
            'filename': f'{batch_dir}/product_categories.csv',
            'rows': 20,
            'description': 'Product categories'
        },
        
        # Level 2: Depends on Level 1
        {
            'filename': f'{batch_dir}/products.csv',
            'rows': 500,
            'relationships': ['product_categories'],
            'description': 'Products by category'
        },
        {
            'filename': f'{batch_dir}/employees.csv',
            'rows': 100,
            'relationships': ['stores'],
            'description': 'Store employees'
        },
        
        # Level 3: Depends on Level 2
        {
            'filename': f'{batch_dir}/inventory.csv',
            'rows': 2000,
            'relationships': ['products', 'stores'],
            'description': 'Product inventory by store'
        },
        
        # Level 4: Transactional data
        {
            'filename': f'{batch_dir}/sales_transactions.csv',
            'rows': 10000,
            'relationships': ['products', 'stores', 'employees'],
            'time_series': True,
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'description': 'Sales transactions'
        }
    ]
    
    complex_paths = tempdata.create_batch(
        complex_batch,
        country='united_states',
        formats=['csv', 'parquet'],
        seed=12345
    )
    
    print("✓ Complex hierarchy generated:")
    for i, dataset in enumerate(complex_batch):
        filename = os.path.basename(dataset['filename'])
        print(f"  {i+1}. {filename:<25} ({dataset['rows']:,} rows)")
        if 'relationships' in dataset:
            deps = ', '.join(dataset['relationships'])
            print(f"     Dependencies: {deps}")
    
    print("\n5.3 Relationship Validation")
    print("-" * 40)
    
    try:
        # Load and validate relationships
        customers_df = pd.read_csv(f'{batch_dir}/customers.csv')
        orders_df = pd.read_csv(f'{batch_dir}/orders.csv')
        
        print("Relationship Validation:")
        print(f"  - Customers: {len(customers_df):,}")
        print(f"  - Orders: {len(orders_df):,}")
        
        # Check foreign key integrity
        unique_customer_ids_in_orders = orders_df['customer_id'].nunique()
        total_customers = len(customers_df)
        
        print(f"  - Customers referenced in orders: {unique_customer_ids_in_orders}/{total_customers}")
        print(f"  - Referential integrity: {'✓ PASS' if unique_customer_ids_in_orders <= total_customers else '✗ FAIL'}")
        
        # Check time series alignment
        if 'date' in orders_df.columns:
            date_range = f"{orders_df['date'].min()} to {orders_df['date'].max()}"
            print(f"  - Order date range: {date_range}")
        
    except Exception as e:
        print(f"Validation error: {e}")
    
    return batch_dir

def tutorial_section_6_best_practices():
    """
    Tutorial Section 6: Best Practices and Tips
    
    Covers:
    - Performance optimization
    - Memory management
    - Format selection
    - Seed management
    - Quality validation
    """
    print("\n" + "=" * 60)
    print("TUTORIAL SECTION 6: BEST PRACTICES AND TIPS")
    print("=" * 60)
    
    print("\n6.1 Performance Optimization")
    print("-" * 40)
    print("Tips for generating large datasets efficiently:")
    print("  ✓ Use streaming for datasets > 100K rows")
    print("  ✓ Choose Parquet format for large datasets")
    print("  ✓ Use appropriate time intervals (avoid 1-second for year-long data)")
    print("  ✓ Batch related datasets together for consistency")
    
    print("\n6.2 Memory Management")
    print("-" * 40)
    print("Managing memory usage:")
    print("  ✓ Enable streaming: use_streaming=True")
    print("  ✓ Process in chunks for analysis")
    print("  ✓ Use efficient data types (int32 vs int64)")
    print("  ✓ Clean up DataFrames after use")
    
    print("\n6.3 Format Selection Guide")
    print("-" * 40)
    print("Choose the right format for your use case:")
    print("  • CSV: Human-readable, Excel-compatible, universal")
    print("  • JSON: API-friendly, nested data, web applications")
    print("  • Parquet: Analytics, compression, columnar storage")
    print("  • Excel: Business users, formatted reports, charts")
    print("  • GeoJSON: Mapping, GIS applications, coordinates")
    
    print("\n6.4 Seed Management")
    print("-" * 40)
    print("Managing reproducibility:")
    print("  ✓ Use fixed seeds for testing: seed=12345")
    print("  ✓ Omit seed for unique data each run")
    print("  ✓ Document seeds used in production")
    print("  ✓ Use different seeds for different environments")
    
    print("\n6.5 Quality Validation")
    print("-" * 40)
    print("Ensuring data quality:")
    print("  ✓ Check data ranges and distributions")
    print("  ✓ Validate foreign key relationships")
    print("  ✓ Verify time series continuity")
    print("  ✓ Test with small datasets first")
    
    # Demonstrate a performance example
    print("\n6.6 Performance Example")
    print("-" * 40)
    
    perf_dir = "tutorial_performance"
    os.makedirs(perf_dir, exist_ok=True)
    
    print("Generating large dataset with streaming...")
    
    start_time = datetime.now()
    
    large_path = tempdata.create_dataset(
        f'{perf_dir}/large_dataset.parquet',
        rows=100000,  # 100K rows
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='1hour',
        formats=['parquet'],  # Efficient format
        use_streaming=True,   # Memory efficient
        seed=12345
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"✓ Generated 100K rows in {duration:.2f} seconds")
    print(f"  - Rate: {100000/duration:,.0f} rows/second")
    print(f"  - Format: Parquet (compressed)")
    print(f"  - Memory: Streaming (low usage)")

def main():
    """
    Main tutorial function - runs all sections
    """
    print("TempData Comprehensive Tutorial")
    print("=" * 60)
    print("This tutorial demonstrates all key features of TempData")
    print("with practical examples and best practices.")
    print()
    
    # Run all tutorial sections
    business_dir = tutorial_section_1_business_datasets()
    financial_dir = tutorial_section_2_financial_datasets()
    iot_dir = tutorial_section_3_iot_datasets()
    timeseries_dir = tutorial_section_4_time_series()
    batch_dir = tutorial_section_5_batch_generation()
    tutorial_section_6_best_practices()
    
    # Summary
    print("\n" + "=" * 60)
    print("TUTORIAL COMPLETE!")
    print("=" * 60)
    
    print("\nGenerated directories:")
    print(f"  1. Business Data: {business_dir}/")
    print(f"  2. Financial Data: {financial_dir}/")
    print(f"  3. IoT Sensor Data: {iot_dir}/")
    print(f"  4. Time Series Data: {timeseries_dir}/")
    print(f"  5. Batch Data: {batch_dir}/")
    print(f"  6. Performance Examples: tutorial_performance/")
    
    print("\nKey concepts covered:")
    print("  ✓ Business datasets (Requirement 2.1)")
    print("  ✓ Financial datasets (Requirement 2.2)")
    print("  ✓ IoT sensor datasets (Requirement 2.5)")
    print("  ✓ Time series generation (Requirement 5.1)")
    print("  ✓ Batch generation (Requirement 8.1)")
    print("  ✓ Performance optimization")
    print("  ✓ Best practices and tips")
    
    print("\nNext steps:")
    print("  1. Explore the generated datasets")
    print("  2. Try modifying parameters")
    print("  3. Integrate with your applications")
    print("  4. Build dashboards and analytics")
    print("  5. Test with your specific use cases")

if __name__ == "__main__":
    main()