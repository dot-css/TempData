#!/usr/bin/env python3
"""
Business Intelligence Pipeline Example

This example demonstrates how to create a comprehensive business intelligence
dataset using TempData, including related datasets for customers, products,
sales, and analytics with realistic business patterns.
"""

import tempdata
import pandas as pd
from datetime import datetime, timedelta
import os

def create_bi_pipeline():
    """
    Create a complete business intelligence pipeline with related datasets
    """
    print("Creating Business Intelligence Pipeline...")
    print("=" * 50)
    
    # Create output directory
    output_dir = "bi_pipeline_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Create foundational datasets
    print("\n1. Creating foundational datasets...")
    
    # Product catalog
    print("   - Generating product catalog...")
    products_path = tempdata.create_dataset(
        f'{output_dir}/products.csv',
        rows=1000,
        country='united_states',
        seed=12345  # Reproducible data
    )
    print(f"   ✓ Products: {products_path}")
    
    # Customer database
    print("   - Generating customer database...")
    customers_path = tempdata.create_dataset(
        f'{output_dir}/customers.csv',
        rows=5000,
        country='united_states',
        seed=12345
    )
    print(f"   ✓ Customers: {customers_path}")
    
    # Step 2: Create transactional data with relationships
    print("\n2. Creating transactional datasets...")
    
    # Sales transactions with time series
    print("   - Generating sales transactions (1 year of data)...")
    sales_path = tempdata.create_dataset(
        f'{output_dir}/sales_transactions.csv',
        rows=50000,
        country='united_states',
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='1day',
        seed=12345
    )
    print(f"   ✓ Sales Transactions: {sales_path}")
    
    # E-commerce orders
    print("   - Generating e-commerce orders...")
    ecommerce_path = tempdata.create_dataset(
        f'{output_dir}/ecommerce_orders.csv',
        rows=25000,
        country='united_states',
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        seed=12345
    )
    print(f"   ✓ E-commerce Orders: {ecommerce_path}")
    
    # Step 3: Create analytical datasets
    print("\n3. Creating analytical datasets...")
    
    # Web analytics
    print("   - Generating web analytics data...")
    web_analytics_path = tempdata.create_dataset(
        f'{output_dir}/web_analytics.csv',
        rows=100000,
        country='united_states',
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='1hour',
        seed=12345
    )
    print(f"   ✓ Web Analytics: {web_analytics_path}")
    
    # Customer support data
    print("   - Generating customer support data...")
    support_path = tempdata.create_dataset(
        f'{output_dir}/customer_support.csv',
        rows=8000,
        country='united_states',
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        seed=12345
    )
    print(f"   ✓ Customer Support: {support_path}")
    
    # Step 4: Create batch datasets with relationships
    print("\n4. Creating related datasets batch...")
    
    batch_datasets = [
        {
            'filename': f'{output_dir}/regions.csv',
            'rows': 50
        },
        {
            'filename': f'{output_dir}/sales_reps.csv',
            'rows': 200,
            'relationships': ['regions']
        },
        {
            'filename': f'{output_dir}/detailed_sales.csv',
            'rows': 75000,
            'relationships': ['customers', 'products', 'sales_reps'],
            'time_series': True,
            'start_date': '2024-01-01',
            'end_date': '2024-12-31'
        }
    ]
    
    batch_paths = tempdata.create_batch(
        batch_datasets,
        country='united_states',
        seed=12345
    )
    
    for path in batch_paths:
        print(f"   ✓ Batch dataset: {path}")
    
    # Step 5: Create multi-format exports for different tools
    print("\n5. Creating multi-format exports...")
    
    # Analytics-ready formats
    analytics_formats = ['csv', 'parquet', 'json']
    
    # Key performance indicators dataset
    kpi_paths = tempdata.create_dataset(
        f'{output_dir}/kpi_metrics',
        rows=365,  # Daily KPIs for one year
        country='united_states',
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='1day',
        formats=analytics_formats,
        seed=12345
    )
    print(f"   ✓ KPI Metrics: {kpi_paths}")
    
    # Executive dashboard data
    dashboard_paths = tempdata.create_dataset(
        f'{output_dir}/executive_dashboard',
        rows=52,  # Weekly data for one year
        country='united_states',
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='1week',
        formats=['excel', 'json'],
        seed=12345
    )
    print(f"   ✓ Executive Dashboard: {dashboard_paths}")
    
    print("\n" + "=" * 50)
    print("Business Intelligence Pipeline Complete!")
    print(f"All datasets created in: {output_dir}/")
    
    return output_dir

def analyze_bi_data(data_dir):
    """
    Perform basic analysis on the generated BI data
    """
    print("\nPerforming Basic Analysis...")
    print("-" * 30)
    
    try:
        # Load and analyze sales data
        sales_df = pd.read_csv(f'{data_dir}/sales_transactions.csv')
        print(f"Sales Transactions: {len(sales_df):,} records")
        print(f"Date range: {sales_df['date'].min()} to {sales_df['date'].max()}")
        print(f"Total sales amount: ${sales_df['amount'].sum():,.2f}")
        print(f"Average transaction: ${sales_df['amount'].mean():.2f}")
        
        # Load and analyze customer data
        customers_df = pd.read_csv(f'{data_dir}/customers.csv')
        print(f"\nCustomers: {len(customers_df):,} records")
        print(f"Customer segments: {customers_df['customer_segment'].value_counts().to_dict()}")
        
        # Load and analyze web analytics
        web_df = pd.read_csv(f'{data_dir}/web_analytics.csv')
        print(f"\nWeb Analytics: {len(web_df):,} records")
        print(f"Unique sessions: {web_df['session_id'].nunique():,}")
        print(f"Top device types: {web_df['device_type'].value_counts().head(3).to_dict()}")
        
    except Exception as e:
        print(f"Analysis error: {e}")

def create_geographical_bi_pipeline():
    """
    Create a geographical business intelligence pipeline with multiple countries
    """
    print("\nCreating Geographical BI Pipeline...")
    print("=" * 40)
    
    countries = ['united_states', 'germany', 'japan', 'brazil']
    geo_dir = "geo_bi_pipeline"
    os.makedirs(geo_dir, exist_ok=True)
    
    for country in countries:
        print(f"\nGenerating data for {country}...")
        
        # Country-specific customer data
        customers_path = tempdata.create_dataset(
            f'{geo_dir}/customers_{country}.csv',
            rows=2000,
            country=country,
            seed=12345
        )
        print(f"   ✓ Customers: {customers_path}")
        
        # Country-specific sales data
        sales_path = tempdata.create_dataset(
            f'{geo_dir}/sales_{country}.csv',
            rows=10000,
            country=country,
            time_series=True,
            start_date='2024-01-01',
            end_date='2024-12-31',
            seed=12345
        )
        print(f"   ✓ Sales: {sales_path}")
        
        # Generate addresses for market analysis
        addresses = tempdata.geo.addresses(country, count=100)
        
        # Save addresses to CSV
        addresses_df = pd.DataFrame(addresses)
        addresses_path = f'{geo_dir}/addresses_{country}.csv'
        addresses_df.to_csv(addresses_path, index=False)
        print(f"   ✓ Addresses: {addresses_path}")
    
    print(f"\nGeographical BI Pipeline complete in: {geo_dir}/")
    return geo_dir

def create_time_series_analysis():
    """
    Create time series datasets for advanced analytics
    """
    print("\nCreating Time Series Analysis Datasets...")
    print("=" * 45)
    
    ts_dir = "time_series_analysis"
    os.makedirs(ts_dir, exist_ok=True)
    
    # High-frequency financial data
    print("1. Generating high-frequency stock data...")
    stock_path = tempdata.create_dataset(
        f'{ts_dir}/stock_prices_1min.csv',
        rows=1440,  # One day of minute data
        time_series=True,
        start_date='2024-01-01 09:00:00',
        end_date='2024-01-01 17:00:00',
        interval='1min',
        seed=12345
    )
    print(f"   ✓ Stock prices (1-minute): {stock_path}")
    
    # Daily business metrics
    print("2. Generating daily business metrics...")
    daily_metrics_path = tempdata.create_dataset(
        f'{ts_dir}/daily_business_metrics.csv',
        rows=365,
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='1day',
        seed=12345
    )
    print(f"   ✓ Daily metrics: {daily_metrics_path}")
    
    # Hourly IoT sensor data
    print("3. Generating hourly IoT sensor data...")
    iot_path = tempdata.create_dataset(
        f'{ts_dir}/iot_sensors_hourly.csv',
        rows=8760,  # One year of hourly data
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='1hour',
        country='germany',  # European climate patterns
        seed=12345
    )
    print(f"   ✓ IoT sensors: {iot_path}")
    
    # Weekly aggregated data
    print("4. Generating weekly aggregated data...")
    weekly_path = tempdata.create_dataset(
        f'{ts_dir}/weekly_aggregates.csv',
        rows=52,
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='1week',
        seed=12345
    )
    print(f"   ✓ Weekly aggregates: {weekly_path}")
    
    print(f"\nTime Series Analysis datasets complete in: {ts_dir}/")
    return ts_dir

def main():
    """
    Main function to run all BI pipeline examples
    """
    print("TempData Business Intelligence Pipeline Examples")
    print("=" * 55)
    
    # Create main BI pipeline
    bi_dir = create_bi_pipeline()
    
    # Analyze the generated data
    analyze_bi_data(bi_dir)
    
    # Create geographical BI pipeline
    geo_dir = create_geographical_bi_pipeline()
    
    # Create time series analysis datasets
    ts_dir = create_time_series_analysis()
    
    print("\n" + "=" * 55)
    print("All Business Intelligence Examples Complete!")
    print("\nGenerated directories:")
    print(f"  - Main BI Pipeline: {bi_dir}/")
    print(f"  - Geographical BI: {geo_dir}/")
    print(f"  - Time Series Analysis: {ts_dir}/")
    
    print("\nNext steps:")
    print("  1. Load datasets into your BI tool (Tableau, Power BI, etc.)")
    print("  2. Create dashboards and visualizations")
    print("  3. Perform advanced analytics and machine learning")
    print("  4. Use the data for testing ETL pipelines")

if __name__ == "__main__":
    main()