# Quick Start Guide

This guide will get you up and running with TempData in just a few minutes.

## Your First Dataset

Let's start by generating a simple sales dataset:

```python
import tempdata

# Generate 1000 sales records
sales_data = tempdata.create_dataset('sales.csv', rows=1000)
print(f"Generated: {sales_data}")
```

This creates a CSV file with realistic sales data including:
- Transaction dates and amounts
- Customer information
- Product categories
- Regional data
- Sales representatives

## Specifying Countries

Generate data for specific geographical regions:

```python
# US-based customer data
us_customers = tempdata.create_dataset(
    'us_customers.csv',
    rows=5000,
    country='united_states'
)

# German business data
de_business = tempdata.create_dataset(
    'german_sales.csv',
    rows=2000,
    country='germany'
)

# Pakistani market data
pk_data = tempdata.create_dataset(
    'pakistan_market.csv',
    rows=1500,
    country='pakistan'
)
```

## Reproducible Data with Seeds

Use seeds for consistent, reproducible datasets:

```python
# This will always generate the same data
consistent_data = tempdata.create_dataset(
    'reproducible.csv',
    rows=1000,
    seed=12345
)

# Different seed = different data
different_data = tempdata.create_dataset(
    'different.csv',
    rows=1000,
    seed=67890
)
```

## Multiple Export Formats

Export the same dataset in multiple formats:

```python
# Generate data in multiple formats
formats = tempdata.create_dataset(
    'multi_format_data',  # No extension needed
    rows=5000,
    formats=['csv', 'json', 'parquet', 'excel']
)

print("Generated files:")
for file_path in formats:
    print(f"  - {file_path}")
```

Output:
```
Generated files:
  - multi_format_data.csv
  - multi_format_data.json
  - multi_format_data.parquet
  - multi_format_data.xlsx
```

## Time Series Data

Create time-based datasets with realistic patterns:

```python
# Daily stock prices for one year
stock_data = tempdata.create_dataset(
    'stock_prices.csv',
    rows=252,  # Trading days in a year
    time_series=True,
    start_date='2024-01-01',
    end_date='2024-12-31',
    interval='1day'
)

# Hourly IoT sensor readings
sensor_data = tempdata.create_dataset(
    'sensor_readings.csv',
    rows=8760,  # Hours in a year
    time_series=True,
    start_date='2024-01-01',
    end_date='2024-12-31',
    interval='1hour',
    country='netherlands'  # European climate patterns
)

# High-frequency trading data
hft_data = tempdata.create_dataset(
    'minute_trading.csv',
    rows=390,  # 6.5 hours of trading
    time_series=True,
    start_date='2024-01-15 09:30:00',
    end_date='2024-01-15 16:00:00',
    interval='1min'
)
```

## Related Datasets (Batch Generation)

Create multiple related datasets with referential integrity:

```python
# Define related datasets
datasets = [
    {
        'filename': 'customers.csv',
        'rows': 1000
    },
    {
        'filename': 'orders.csv',
        'rows': 5000,
        'relationships': ['customers'],  # Orders reference customers
        'time_series': True,
        'start_date': '2024-01-01',
        'end_date': '2024-12-31'
    },
    {
        'filename': 'order_items.csv',
        'rows': 15000,
        'relationships': ['orders']  # Items reference orders
    }
]

# Generate all datasets with maintained relationships
batch_files = tempdata.create_batch(
    datasets,
    country='united_states',
    seed=12345
)

print("Generated related datasets:")
for file_path in batch_files:
    print(f"  - {file_path}")
```

## Dataset Types

TempData supports 40+ dataset types across different domains:

```python
# Business datasets
sales = tempdata.create_dataset('sales.csv', rows=1000)
customers = tempdata.create_dataset('customers.csv', rows=2000)
inventory = tempdata.create_dataset('inventory.csv', rows=500)

# Financial datasets
stocks = tempdata.create_dataset('stocks.csv', rows=252, time_series=True)
banking = tempdata.create_dataset('banking.csv', rows=10000)
crypto = tempdata.create_dataset('crypto.csv', rows=8760, time_series=True)

# IoT sensor datasets
weather = tempdata.create_dataset('weather.csv', rows=8760, time_series=True)
energy = tempdata.create_dataset('energy.csv', rows=35040, time_series=True)
traffic = tempdata.create_dataset('traffic.csv', rows=17520, time_series=True)

# Healthcare datasets
patients = tempdata.create_dataset('patients.csv', rows=5000)
appointments = tempdata.create_dataset('appointments.csv', rows=20000)
prescriptions = tempdata.create_dataset('prescriptions.csv', rows=15000)
```

## Performance for Large Datasets

For large datasets, enable streaming to manage memory usage:

```python
# Generate 1 million rows efficiently
large_dataset = tempdata.create_dataset(
    'large_data.parquet',
    rows=1000000,
    use_streaming=True,
    formats=['parquet']  # Efficient columnar format
)
```

## Command Line Interface

TempData also provides a CLI for quick data generation:

```bash
# Generate a simple dataset
tempdata generate sales.csv --rows 1000 --country united_states

# Time series data
tempdata generate stocks.csv --rows 252 --time-series \
    --start-date 2024-01-01 --end-date 2024-12-31 --interval 1day

# Multiple formats
tempdata generate analytics.csv --rows 10000 \
    --formats csv,json,parquet --seed 12345

# Show help
tempdata --help
```

## Analyzing Generated Data

Use pandas to analyze your generated data:

```python
import pandas as pd

# Load and analyze the data
df = pd.read_csv('sales.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Total sales: ${df['amount'].sum():,.2f}")
print(f"Average transaction: ${df['amount'].mean():.2f}")

# Show sample data
print("\nSample data:")
print(df.head())

# Basic statistics
print("\nSummary statistics:")
print(df.describe())
```

## Next Steps

Now that you've learned the basics, explore more advanced features:

- **[Dataset Types](dataset-types.md)**: Learn about all 40+ available dataset types
- **[Geographical Data](geographical-data.md)**: Generate location-specific data
- **[Time Series](time-series.md)**: Advanced time-based data generation
- **[Batch Generation](batch-generation.md)**: Create complex related datasets
- **[Examples](../examples/business-intelligence.md)**: Real-world use cases and tutorials

## Common Patterns

### Testing Database Schemas

```python
# Generate test data for database tables
users = tempdata.create_dataset('users.csv', rows=1000)
products = tempdata.create_dataset('products.csv', rows=500)
orders = tempdata.create_batch([
    {'filename': 'orders.csv', 'rows': 5000, 'relationships': ['users']},
    {'filename': 'order_items.csv', 'rows': 15000, 'relationships': ['orders', 'products']}
])
```

### API Testing

```python
# Generate JSON data for API testing
api_data = tempdata.create_dataset(
    'api_test_data.json',
    rows=100,
    formats=['json']
)
```

### Dashboard Development

```python
# Generate time series data for dashboards
dashboard_data = tempdata.create_dataset(
    'dashboard_metrics.csv',
    rows=365,
    time_series=True,
    start_date='2024-01-01',
    end_date='2024-12-31',
    interval='1day'
)
```

### Machine Learning Training

```python
# Generate training data for ML models
ml_data = tempdata.create_dataset(
    'ml_training_data.parquet',
    rows=100000,
    formats=['parquet'],
    use_streaming=True
)
```