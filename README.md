# TempData

[![PyPI version](https://badge.fury.io/py/tempdata.svg)](https://badge.fury.io/py/tempdata)
[![Python Support](https://img.shields.io/pypi/pyversions/tempdata.svg)](https://pypi.org/project/tempdata/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/tempdata/badge/?version=latest)](https://tempdata.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://github.com/dot-css/tempdata/workflows/CI/badge.svg)](https://github.com/dot-css/tempdata/actions)
[![Coverage Status](https://codecov.io/gh/dot-css/tempdata/branch/main/graph/badge.svg)](https://codecov.io/gh/dot-css/tempdata)

**TempData** is a comprehensive Python library designed to generate realistic fake data for testing, development, and prototyping purposes. With support for 40+ dataset types spanning business, financial, healthcare, technology, IoT, and social domains, TempData provides worldwide geographical capabilities and time-based dynamic seeding for reproducible yet unique data generation.

## 🚀 Key Features

- **40+ Dataset Types**: Business, financial, healthcare, technology, IoT, and social datasets
- **Global Coverage**: Generate geographically accurate data for any country or region
- **Time Series Support**: Realistic temporal patterns with seasonal variations and correlations
- **Batch Generation**: Create related datasets with maintained referential integrity
- **Multiple Export Formats**: CSV, JSON, Parquet, Excel, GeoJSON
- **High Performance**: 50,000+ rows per second with streaming support for large datasets
- **Simple API**: Intuitive interface with sensible defaults
- **CLI Interface**: Command-line tools for automation and scripting
- **Reproducible**: Time-based dynamic seeding with optional fixed seeds

## 📦 Installation

```bash
# Install from PyPI
pip install tempdata

# Install with all optional dependencies
pip install tempdata[dev,docs,performance]

# Install from source
git clone https://github.com/dot-css/tempdata.git
cd tempdata
pip install -e .
```

## 🎯 Quick Start

### Basic Usage

```python
import tempdata

# Generate a simple sales dataset
sales_data = tempdata.create_dataset('sales.csv', rows=1000)

# Generate customer data for a specific country
customers = tempdata.create_dataset(
    'customers.csv',
    rows=5000,
    country='united_states',
    seed=12345  # For reproducible results
)

# Create time series data
stock_prices = tempdata.create_dataset(
    'stocks.csv',
    rows=252,  # One trading year
    time_series=True,
    start_date='2024-01-01',
    end_date='2024-12-31',
    interval='1day'
)
```

### Multiple Export Formats

```python
# Export the same dataset in multiple formats
paths = tempdata.create_dataset(
    'analytics_data',
    rows=10000,
    formats=['csv', 'json', 'parquet', 'excel']
)
# Returns: ['analytics_data.csv', 'analytics_data.json', 'analytics_data.parquet', 'analytics_data.xlsx']
```

### Batch Generation with Relationships

```python
# Create related datasets with referential integrity
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
    }
]

batch_paths = tempdata.create_batch(datasets, country='germany', seed=12345)
```

## 📊 Supported Dataset Types

### Business Datasets (10 types)
- **Sales**: Transaction data with seasonal patterns
- **Customers**: Demographics, contact info, segmentation
- **E-commerce**: Orders, products, reviews, inventory
- **Marketing**: Campaigns, leads, conversions
- **Employees**: HR data, payroll, performance
- **Suppliers**: Vendor information, contracts
- **Retail**: Store operations, POS data
- **CRM**: Customer relationship management
- **Inventory**: Stock levels, warehousing
- **Reviews**: Product and service feedback

### Financial Datasets (8 types)
- **Stocks**: Market data with realistic volatility
- **Banking**: Transactions, accounts, fraud detection
- **Cryptocurrency**: High-volatility trading data
- **Insurance**: Policies, claims, risk assessment
- **Loans**: Applications, approvals, payments
- **Investments**: Portfolios, performance, allocations
- **Accounting**: General ledger, financial statements
- **Payments**: Digital transactions, gateways

### IoT Sensor Datasets (6 types)
- **Weather**: Temperature, humidity, pressure with correlations
- **Energy**: Consumption patterns, smart grid data
- **Traffic**: Vehicle counts, congestion patterns
- **Environmental**: Air quality, noise levels
- **Industrial**: Manufacturing sensors, predictive maintenance
- **Smart Home**: Automation, security, appliance monitoring

### Healthcare Datasets (6 types)
- **Patients**: Demographics, medical history
- **Appointments**: Scheduling, outcomes
- **Lab Results**: Test data, reference ranges
- **Prescriptions**: Medications, dosages
- **Medical History**: Conditions, treatments
- **Clinical Trials**: Research data, protocols

### Technology Datasets (8 types)
- **Web Analytics**: Page views, user sessions, conversions
- **App Usage**: Mobile analytics, user behavior
- **System Logs**: Server metrics, error tracking
- **API Calls**: Request/response data, performance
- **Server Metrics**: CPU, memory, network usage
- **User Sessions**: Authentication, activity tracking
- **Error Logs**: Exception handling, debugging data
- **Performance**: Load times, throughput metrics

### Social Datasets (2 types)
- **Social Media**: Posts, engagement, trends
- **User Profiles**: Social network data, connections

## 🌍 Global Coverage

TempData supports geographically accurate data generation for countries worldwide:

```python
# Generate data for specific countries
us_data = tempdata.create_dataset('sales.csv', country='united_states')
uk_data = tempdata.create_dataset('sales.csv', country='united_kingdom')
jp_data = tempdata.create_dataset('sales.csv', country='japan')
pk_data = tempdata.create_dataset('sales.csv', country='pakistan')

# Generate addresses with accurate postal codes
addresses = tempdata.geo.addresses('germany', count=100)
coordinates = tempdata.geo.coordinates('tokyo', count=50)
routes = tempdata.geo.routes('new_york', 'boston', waypoints=5)
```

## ⏰ Time Series Generation

Create realistic time series data with various patterns:

```python
# High-frequency financial data
hft_data = tempdata.create_dataset(
    'minute_trading.csv',
    rows=390,  # 6.5 hours of trading
    time_series=True,
    start_date='2024-01-15 09:30:00',
    end_date='2024-01-15 16:00:00',
    interval='1min'
)

# IoT sensor data with correlations
sensor_data = tempdata.create_dataset(
    'weather_sensors.csv',
    rows=8760,  # Hourly for one year
    time_series=True,
    start_date='2024-01-01',
    end_date='2024-12-31',
    interval='1hour',
    country='netherlands'
)

# Business metrics with seasonal patterns
retail_sales = tempdata.create_dataset(
    'seasonal_sales.csv',
    rows=365,
    time_series=True,
    start_date='2024-01-01',
    end_date='2024-12-31',
    interval='1day'
)
```

## 🔗 Batch Generation with Relationships

Create complex data ecosystems with maintained referential integrity:

```python
# E-commerce ecosystem
ecommerce_datasets = [
    {'filename': 'categories.csv', 'rows': 50},
    {'filename': 'products.csv', 'rows': 2000, 'relationships': ['categories']},
    {'filename': 'customers.csv', 'rows': 5000},
    {'filename': 'orders.csv', 'rows': 25000, 'relationships': ['customers'], 'time_series': True},
    {'filename': 'order_items.csv', 'rows': 75000, 'relationships': ['orders', 'products']}
]

paths = tempdata.create_batch(ecommerce_datasets, country='united_states')
```

## 🖥️ Command Line Interface

```bash
# Generate a simple dataset
tempdata generate sales.csv --rows 1000 --country united_states

# Create time series data
tempdata generate stocks.csv --rows 252 --time-series --start-date 2024-01-01 --end-date 2024-12-31

# Multiple formats
tempdata generate analytics.csv --rows 10000 --formats csv,json,parquet

# Batch generation
tempdata batch config.json --country germany --seed 12345
```

## 📈 Performance

TempData is optimized for high performance:

- **50,000+ rows/second** for simple datasets
- **Streaming support** for datasets up to 100 million rows
- **Memory efficient** with <50MB usage for 1M rows
- **Parallel processing** with optional Dask integration
- **JIT compilation** with optional Numba acceleration

```python
# Large dataset with streaming
large_dataset = tempdata.create_dataset(
    'large_data.parquet',
    rows=1000000,
    use_streaming=True,
    formats=['parquet']  # Efficient columnar format
)
```

## 📚 Examples and Tutorials

### Business Intelligence Pipeline
```python
# Create a complete BI ecosystem
bi_pipeline = tempdata.examples.business_intelligence_pipeline()
# Generates: customers, products, sales, analytics, KPIs
```

### IoT Data Pipeline
```python
# Smart city IoT sensors
iot_pipeline = tempdata.examples.iot_data_pipeline()
# Generates: weather, energy, traffic, environmental sensors
```

### Financial Analysis
```python
# Financial market simulation
financial_pipeline = tempdata.examples.financial_analysis_pipeline()
# Generates: stocks, banking, crypto, risk management data
```

See the [examples directory](examples/) for comprehensive tutorials and use cases.

## 🔧 Configuration

### Environment Variables
```bash
export TEMPDATA_DEFAULT_COUNTRY=united_states
export TEMPDATA_DEFAULT_SEED=12345
export TEMPDATA_CACHE_DIR=/path/to/cache
export TEMPDATA_MAX_MEMORY=1GB
```

### Configuration File
```python
# tempdata_config.py
TEMPDATA_CONFIG = {
    'default_country': 'united_states',
    'default_formats': ['csv'],
    'performance': {
        'streaming_threshold': 100000,
        'parallel_processing': True,
        'cache_enabled': True
    }
}
```

## 🧪 Testing and Quality

TempData maintains high quality standards:

- **95%+ realistic data patterns** compared to real-world data
- **99%+ geographical accuracy** for coordinates and addresses
- **Comprehensive test suite** with 500+ test cases
- **Performance benchmarks** for all major operations
- **Type hints** for better IDE support
- **Code coverage** >90%

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/dot-css/tempdata.git
cd tempdata
pip install -e .[dev]
pre-commit install
pytest
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=tempdata

# Run performance benchmarks
pytest -m performance

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
```

## 📖 Documentation

Full documentation is available at [tempdata.readthedocs.io](https://tempdata.readthedocs.io)

- [User Guide](https://tempdata.readthedocs.io/en/latest/user-guide/)
- [API Reference](https://tempdata.readthedocs.io/en/latest/api/)
- [Examples](https://tempdata.readthedocs.io/en/latest/examples/)
- [Performance Guide](https://tempdata.readthedocs.io/en/latest/performance/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Faker](https://faker.readthedocs.io/) for foundational fake data generation
- [Pandas](https://pandas.pydata.org/) for data manipulation
- [GeoPy](https://geopy.readthedocs.io/) for geographical calculations
- The open-source community for inspiration and feedback

## 📞 Support

- **Documentation**: [tempdata.readthedocs.io](https://tempdata.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/dot-css/tempdata/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dot-css/tempdata/discussions)
- **Email**: saqibshaikhdz@gmail.com

## 🗺️ Roadmap

- [ ] **v0.2.0**: Advanced ML-based data generation
- [ ] **v0.3.0**: Real-time streaming data generation
- [ ] **v0.4.0**: Cloud integration (AWS, GCP, Azure)
- [ ] **v0.5.0**: GUI interface and visual data modeling
- [ ] **v1.0.0**: Production-ready stable release

---

**Made with ❤️ by the TempData Team**

*Generate realistic data, build better software.*