# TempData Library Documentation

## Table of Contents

1. [Quick Start](#quick-start)
2. [API Reference](#api-reference)
3. [Dataset Types](#dataset-types)
4. [Geographical Data](#geographical-data)
5. [Time Series Generation](#time-series-generation)
6. [Batch Generation](#batch-generation)
7. [Export Formats](#export-formats)
8. [Performance Tuning](#performance-tuning)
9. [Best Practices](#best-practices)
10. [Examples](#examples)

## Quick Start

### Installation

```bash
pip install tempdata
```

### Basic Usage

```python
import tempdata

# Generate a simple dataset
tempdata.create_dataset('sales.csv', rows=1000)

# Generate data for a specific country
tempdata.create_dataset('customers.json', rows=500, country='germany')

# Generate time series data
tempdata.create_dataset('stocks.csv', rows=1000, 
                       time_series=True, 
                       start_date='2024-01-01', 
                       end_date='2024-12-31')
```

### Command Line Usage

```bash
# Generate sales data
tempdata generate sales.csv --rows 1000 --country united_states

# Generate multiple formats
tempdata generate customers --rows 500 --formats csv,json,parquet

# Generate time series data
tempdata generate stocks.csv --rows 1000 --time-series --start-date 2024-01-01
```

## API Reference

### Core Functions

#### `create_dataset(filename, rows=500, **kwargs)`

Generate a single dataset with specified parameters.

**Parameters:**
- `filename` (str): Output filename that determines dataset type
- `rows` (int): Number of rows to generate (default: 500)
- `country` (str): Country code for geographical data (default: 'global')
- `seed` (int): Fixed seed for reproducible results (default: None)
- `formats` (List[str]): Export formats (default: ['csv'])
- `time_series` (bool): Enable time series generation (default: False)
- `start_date` (str/datetime): Start date for time series
- `end_date` (str/datetime): End date for time series
- `interval` (str): Time series interval ('1min', '5min', '1hour', '1day', etc.)
- `use_streaming` (bool): Force streaming generation (default: auto-detect)

**Returns:**
- `str`: Path to generated file(s)

**Examples:**

```python
# Basic dataset generation
path = tempdata.create_dataset('sales.csv', rows=1000)

# Multi-format export
paths = tempdata.create_dataset('customers', rows=500, 
                               formats=['csv', 'json', 'parquet'])

# Time series with custom parameters
tempdata.create_dataset('weather.csv', rows=8760,  # One year hourly
                       time_series=True,
                       interval='1hour',
                       start_date='2024-01-01',
                       end_date='2024-12-31',
                       country='germany')

# Reproducible generation with seed
tempdata.create_dataset('test_data.csv', rows=100, seed=12345)
```

#### `create_batch(datasets, **kwargs)`

Generate multiple related datasets with maintained relationships.

**Parameters:**
- `datasets` (List[Dict]): List of dataset specifications
- Global parameters applied to all datasets

**Dataset Specification:**
```python
{
    'filename': 'customers.csv',
    'rows': 1000,
    'relationships': ['other_dataset_name'],  # Optional
    # Additional dataset-specific parameters
}
```

**Examples:**

```python
# Related business datasets
datasets = [
    {'filename': 'customers.csv', 'rows': 1000},
    {'filename': 'products.csv', 'rows': 500},
    {'filename': 'sales.csv', 'rows': 5000, 'relationships': ['customers', 'products']}
]
paths = tempdata.create_batch(datasets, country='united_states')

# Healthcare system datasets
healthcare_datasets = [
    {'filename': 'patients.csv', 'rows': 2000},
    {'filename': 'appointments.csv', 'rows': 10000, 'relationships': ['patients']}
]
tempdata.create_batch(healthcare_datasets, country='canada')
```

### Geographical API

#### `tempdata.geo.addresses(country, count=1, **kwargs)`

Generate realistic addresses for specified country.

**Parameters:**
- `country` (str): Country code
- `count` (int): Number of addresses to generate
- `city` (str): Specific city (optional)
- `state_province` (str): Specific state/province (optional)
- `urban_bias` (float): Urban area bias 0.0-1.0 (default: 0.7)
- `address_type` (str): 'residential', 'commercial', or 'mixed'

**Examples:**

```python
# Generate addresses for different countries
us_addresses = tempdata.geo.addresses('united_states', count=10)
german_addresses = tempdata.geo.addresses('germany', count=5, city='Berlin')
pakistan_addresses = tempdata.geo.addresses('pakistan', count=3, 
                                           address_type='commercial')

# Access address components
address = us_addresses[0]
print(f"{address['street']}, {address['city']}, {address['state_province']}")
print(f"Coordinates: {address['coordinates']}")
```

#### `tempdata.geo.route(start_city, end_city, waypoints=0, **kwargs)`

Generate realistic route between cities.

**Parameters:**
- `start_city` (str): Starting city name
- `end_city` (str): Destination city name
- `waypoints` (int): Number of intermediate waypoints
- `transportation_mode` (str): 'driving', 'walking', 'cycling', 'public_transit'
- `route_type` (str): 'fastest', 'shortest', 'scenic'

**Examples:**

```python
# Simple route
route = tempdata.geo.route('Berlin', 'Munich')
print(f"Distance: {route['distance_km']} km")
print(f"Time: {route['estimated_time_minutes']} minutes")

# Scenic route with waypoints
scenic = tempdata.geo.route('Paris', 'Rome', waypoints=3, route_type='scenic')
print(f"Waypoints: {len(scenic['waypoints'])}")

# Walking route
walk = tempdata.geo.route('London', 'Oxford', transportation_mode='walking')
```

## Dataset Types

### Business Datasets

#### Sales Transactions (`sales`)
Realistic sales transaction data with seasonal trends and regional preferences.

**Columns:**
- `transaction_id`: Unique transaction identifier
- `date`: Transaction timestamp
- `customer_id`: Customer reference
- `product_id`: Product reference
- `amount`: Transaction amount (localized currency)
- `region`: Geographical region
- `payment_method`: Payment type (cash, card, digital)
- `sales_rep`: Sales representative
- `discount_applied`: Discount percentage
- `tax_amount`: Tax amount

**Example:**
```python
tempdata.create_dataset('sales.csv', rows=10000, 
                       country='united_states',
                       time_series=True,
                       start_date='2024-01-01')
```

#### Customer Database (`customers`)
Comprehensive customer profiles with demographic distributions.

**Columns:**
- `customer_id`: Unique customer identifier
- `first_name`, `last_name`: Customer names (localized)
- `email`: Email address
- `phone`: Phone number (country format)
- `address`: Full address
- `date_of_birth`: Birth date
- `registration_date`: Account creation date
- `customer_segment`: Segment classification
- `lifetime_value`: Customer lifetime value
- `preferred_contact`: Contact preference

#### E-commerce Orders (`ecommerce`)
Online order data with product correlations and shipping patterns.

**Columns:**
- `order_id`: Unique order identifier
- `customer_id`: Customer reference
- `order_date`: Order timestamp
- `products`: List of ordered products
- `total_amount`: Order total
- `shipping_address`: Delivery address
- `shipping_method`: Shipping option
- `payment_status`: Payment state
- `order_status`: Fulfillment status

### Financial Datasets

#### Stock Market Data (`stocks`)
Realistic stock price movements with market volatility patterns.

**Columns:**
- `symbol`: Stock ticker symbol
- `date`: Trading date
- `open_price`: Opening price
- `high_price`: Daily high
- `low_price`: Daily low
- `close_price`: Closing price
- `volume`: Trading volume
- `market_cap`: Market capitalization
- `sector`: Industry sector
- `volatility`: Price volatility measure

**Time Series Example:**
```python
# Generate one year of daily stock data
tempdata.create_dataset('stocks.csv', rows=252,  # Trading days
                       time_series=True,
                       interval='1day',
                       start_date='2024-01-01',
                       end_date='2024-12-31')
```

#### Banking Transactions (`banking`)
Bank transaction data with realistic patterns and fraud indicators.

**Columns:**
- `transaction_id`: Unique transaction ID
- `account_id`: Account identifier
- `date`: Transaction date
- `amount`: Transaction amount
- `transaction_type`: Type (debit, credit, transfer)
- `merchant`: Merchant name
- `category`: Transaction category
- `balance_after`: Account balance after transaction
- `fraud_indicator`: Fraud risk score
- `location`: Transaction location

### Healthcare Datasets

#### Patient Records (`patients`)
Medical patient data with demographic distributions and privacy compliance.

**Columns:**
- `patient_id`: Unique patient identifier
- `first_name`, `last_name`: Patient names
- `date_of_birth`: Birth date
- `gender`: Gender identity
- `address`: Patient address
- `phone`: Contact number
- `emergency_contact`: Emergency contact info
- `insurance_provider`: Insurance company
- `medical_conditions`: List of conditions
- `allergies`: Known allergies

#### Medical Appointments (`appointments`)
Healthcare appointment scheduling with realistic patterns.

**Columns:**
- `appointment_id`: Unique appointment ID
- `patient_id`: Patient reference
- `doctor_id`: Doctor identifier
- `appointment_date`: Scheduled date/time
- `department`: Medical department
- `appointment_type`: Type of visit
- `duration_minutes`: Appointment duration
- `status`: Appointment status
- `notes`: Appointment notes

### Technology Datasets

#### Web Analytics (`web_analytics`)
Website traffic data with user behavior patterns.

**Columns:**
- `session_id`: Unique session identifier
- `user_id`: User identifier
- `timestamp`: Event timestamp
- `page_url`: Page URL
- `event_type`: Event type (pageview, click, etc.)
- `device_type`: Device category
- `browser`: Browser type
- `operating_system`: OS type
- `referrer`: Traffic source
- `duration_seconds`: Session duration

#### System Logs (`system_logs`)
Application and system log data with realistic error patterns.

**Columns:**
- `timestamp`: Log timestamp
- `log_level`: Severity level
- `service`: Service name
- `message`: Log message
- `user_id`: Associated user
- `ip_address`: Source IP
- `response_time`: Response time (ms)
- `status_code`: HTTP status code
- `error_code`: Error identifier

### IoT Sensor Datasets

#### Weather Data (`weather`)
Environmental sensor readings with seasonal patterns.

**Columns:**
- `timestamp`: Reading timestamp
- `sensor_id`: Sensor identifier
- `location`: Sensor location
- `temperature`: Temperature (°C)
- `humidity`: Humidity percentage
- `pressure`: Atmospheric pressure
- `wind_speed`: Wind speed
- `wind_direction`: Wind direction
- `precipitation`: Rainfall amount
- `visibility`: Visibility distance

**Time Series Example:**
```python
# Generate hourly weather data for a year
tempdata.create_dataset('weather.csv', rows=8760,
                       time_series=True,
                       interval='1hour',
                       country='norway')  # Arctic weather patterns
```

#### Energy Consumption (`energy`)
Power usage data with consumption patterns.

**Columns:**
- `timestamp`: Reading timestamp
- `meter_id`: Meter identifier
- `location`: Installation location
- `consumption_kwh`: Energy consumption
- `peak_demand`: Peak demand
- `power_factor`: Power factor
- `voltage`: Voltage level
- `current`: Current draw
- `cost`: Energy cost
- `tariff_type`: Pricing tariff

### Social Datasets

#### Social Media Posts (`social_media`)
Social platform content with engagement patterns.

**Columns:**
- `post_id`: Unique post identifier
- `user_id`: Author identifier
- `timestamp`: Post timestamp
- `content`: Post content
- `platform`: Social platform
- `post_type`: Content type
- `likes`: Like count
- `shares`: Share count
- `comments`: Comment count
- `hashtags`: Associated hashtags

#### User Profiles (`user_profiles`)
Social media user profile data.

**Columns:**
- `user_id`: Unique user identifier
- `username`: Display username
- `display_name`: Full display name
- `bio`: Profile biography
- `location`: User location
- `follower_count`: Number of followers
- `following_count`: Number following
- `post_count`: Total posts
- `account_created`: Account creation date
- `verification_status`: Verification state

## Geographical Data

### Supported Countries

TempData supports 20+ countries with localized data patterns:

**Europe:**
- `united_kingdom` - UK addresses, postcodes, phone formats
- `germany` - German addresses, PLZ codes, cultural patterns
- `france` - French addresses, postal codes, regional data
- `spain` - Spanish addresses, regional variations
- `italy` - Italian addresses, regional patterns
- `netherlands` - Dutch addresses, postal codes
- `sweden`, `norway`, `denmark`, `finland` - Nordic countries
- `poland` - Polish addresses and cultural patterns
- `czech_republic` - Czech addresses and formats
- `austria` - Austrian addresses and patterns
- `switzerland` - Swiss addresses with multilingual support

**Americas:**
- `united_states` - US addresses, ZIP codes, state patterns
- `canada` - Canadian addresses, postal codes, provinces
- `brazil` - Brazilian addresses, CEP codes
- `mexico` - Mexican addresses and patterns
- `argentina` - Argentine addresses and formats
- `chile` - Chilean addresses and patterns

**Asia-Pacific:**
- `japan` - Japanese addresses with proper formatting
- `south_korea` - Korean addresses and patterns
- `india` - Indian addresses, PIN codes, regional data
- `pakistan` - Pakistani addresses and cultural patterns
- `australia` - Australian addresses, postcodes
- `new_zealand` - New Zealand addresses and patterns

**Africa:**
- `south_africa` - South African addresses and patterns
- `egypt` - Egyptian addresses and cultural data
- `nigeria` - Nigerian addresses and regional patterns
- `kenya` - Kenyan addresses and formats

### Address Generation Examples

```python
# Generate addresses for different countries
us_addresses = tempdata.geo.addresses('united_states', count=5)
german_addresses = tempdata.geo.addresses('germany', count=5, city='Berlin')
japanese_addresses = tempdata.geo.addresses('japan', count=5, urban_bias=0.9)

# Commercial addresses only
commercial = tempdata.geo.addresses('united_kingdom', count=10, 
                                   address_type='commercial')

# Specific region
california = tempdata.geo.addresses('united_states', count=20, 
                                   state_province='California')
```

### Route Generation Examples

```python
# International routes
europe_route = tempdata.geo.route('Berlin', 'Paris', waypoints=2)
us_route = tempdata.geo.route('New York', 'Los Angeles', waypoints=5)

# Different transportation modes
walking = tempdata.geo.route('London', 'Oxford', transportation_mode='walking')
cycling = tempdata.geo.route('Amsterdam', 'Utrecht', transportation_mode='cycling')

# Route types
scenic = tempdata.geo.route('San Francisco', 'Seattle', route_type='scenic')
fastest = tempdata.geo.route('Tokyo', 'Osaka', route_type='fastest')
```

## Time Series Generation

### Configuration Options

Time series generation supports various intervals and patterns:

**Intervals:**
- `1min`, `5min`, `15min`, `30min` - Sub-hourly data
- `1hour` - Hourly data
- `1day` - Daily data
- `1week` - Weekly data
- `1month` - Monthly data

**Parameters:**
- `start_date` - Start date (ISO format or datetime)
- `end_date` - End date (ISO format or datetime)
- `interval` - Time interval between points
- `seasonal_patterns` - Enable seasonal variations (default: True)
- `trend_direction` - Overall trend ('up', 'down', 'random')
- `volatility_level` - Data volatility (0.0-1.0)

### Examples

```python
# Daily stock data for one year
tempdata.create_dataset('stocks_daily.csv', rows=365,
                       time_series=True,
                       interval='1day',
                       start_date='2024-01-01',
                       end_date='2024-12-31')

# Hourly IoT sensor data
tempdata.create_dataset('sensors_hourly.csv', rows=8760,  # 365 * 24
                       time_series=True,
                       interval='1hour',
                       start_date='2024-01-01 00:00:00')

# High-frequency financial data
tempdata.create_dataset('trading_1min.csv', rows=1440,  # One day
                       time_series=True,
                       interval='1min',
                       start_date='2024-01-01 09:00:00',
                       end_date='2024-01-01 17:00:00')

# Weather data with seasonal patterns
tempdata.create_dataset('weather_seasonal.csv', rows=365,
                       time_series=True,
                       interval='1day',
                       country='canada',  # Cold climate patterns
                       seasonal_patterns=True)
```

### Time Series Patterns

Different dataset types include realistic temporal patterns:

**Financial Data:**
- Market volatility clustering
- Trading volume patterns
- Seasonal effects (January effect, etc.)
- Weekend gaps for stock data

**IoT Sensor Data:**
- Seasonal temperature variations
- Daily usage cycles for energy data
- Weather correlation patterns
- Equipment maintenance cycles

**Business Data:**
- Seasonal sales patterns
- Holiday effects
- Weekly business cycles
- Customer behavior patterns

## Batch Generation

### Relationship Types

Batch generation maintains referential integrity between datasets:

**One-to-Many:**
- One customer → Many sales transactions
- One product → Many inventory records
- One patient → Many appointments

**Many-to-Many:**
- Products ↔ Orders (through order items)
- Doctors ↔ Patients (through appointments)
- Users ↔ Social posts (through interactions)

### Examples

```python
# E-commerce system
ecommerce_system = [
    {'filename': 'customers.csv', 'rows': 1000},
    {'filename': 'products.csv', 'rows': 500},
    {'filename': 'orders.csv', 'rows': 5000, 'relationships': ['customers']},
    {'filename': 'order_items.csv', 'rows': 15000, 
     'relationships': ['orders', 'products']}
]
tempdata.create_batch(ecommerce_system, country='united_states')

# Healthcare system
healthcare_system = [
    {'filename': 'patients.csv', 'rows': 2000},
    {'filename': 'doctors.csv', 'rows': 50},
    {'filename': 'appointments.csv', 'rows': 10000, 
     'relationships': ['patients', 'doctors']},
    {'filename': 'prescriptions.csv', 'rows': 8000, 
     'relationships': ['appointments']}
]
tempdata.create_batch(healthcare_system, country='canada')

# Financial system with time series
financial_system = [
    {'filename': 'accounts.csv', 'rows': 1000},
    {'filename': 'transactions.csv', 'rows': 50000, 
     'relationships': ['accounts'], 'time_series': True,
     'start_date': '2024-01-01', 'interval': '1hour'}
]
tempdata.create_batch(financial_system, country='germany')
```

## Export Formats

### Supported Formats

**CSV (Comma-Separated Values)**
- Standard text format
- Configurable delimiters
- Proper header handling
- UTF-8 encoding

```python
tempdata.create_dataset('data.csv', rows=1000, formats=['csv'])
```

**JSON (JavaScript Object Notation)**
- Proper data type preservation
- Nested object support
- UTF-8 encoding
- Pretty printing option

```python
tempdata.create_dataset('data.json', rows=1000, formats=['json'])
```

**Parquet (Columnar Format)**
- Compressed binary format
- Optimized for analytics
- Schema preservation
- Fast read/write performance

```python
tempdata.create_dataset('data.parquet', rows=1000, formats=['parquet'])
```

**Excel (.xlsx)**
- Microsoft Excel format
- Multiple worksheet support
- Data type formatting
- Cell styling options

```python
tempdata.create_dataset('data.xlsx', rows=1000, formats=['excel'])
```

**GeoJSON (Geographical JSON)**
- Geographical data format
- Coordinate system support
- Feature collections
- Mapping application ready

```python
# Only for geographical datasets
tempdata.create_dataset('locations.geojson', rows=100, formats=['geojson'])
```

### Multi-Format Export

```python
# Export to multiple formats simultaneously
paths = tempdata.create_dataset('sales_data', rows=10000,
                               formats=['csv', 'json', 'parquet', 'excel'])
print(paths)  # Returns comma-separated list of file paths

# Batch export with consistent formats
datasets = [
    {'filename': 'customers', 'rows': 1000},
    {'filename': 'sales', 'rows': 5000}
]
tempdata.create_batch(datasets, formats=['csv', 'parquet'])
```

## Performance Tuning

### Memory Management

**Automatic Streaming:**
- Datasets ≥100K rows automatically use streaming
- Configurable memory thresholds
- Chunk-based processing
- Progress monitoring

```python
# Large dataset - automatically streams
tempdata.create_dataset('large_sales.csv', rows=1_000_000)

# Force streaming for smaller datasets
tempdata.create_dataset('medium_data.csv', rows=50_000, use_streaming=True)

# Custom streaming configuration
from tempdata.core.streaming import StreamingConfig
config = StreamingConfig(chunk_size=25000, max_memory_mb=100)
tempdata.create_dataset('custom_stream.csv', rows=500_000, 
                       streaming_config=config)
```

**Memory Optimization Tips:**
1. Use streaming for datasets >50K rows
2. Choose appropriate export formats (Parquet for analytics)
3. Limit concurrent batch generation
4. Use time series intervals appropriate for your use case

### Performance Benchmarks

**Generation Speed (rows/second):**
- Simple datasets: 50,000+ rows/sec
- Complex datasets: 25,000+ rows/sec
- Time series data: 15,000+ rows/sec
- Geographical data: 10,000+ rows/sec

**Memory Usage:**
- Standard generation: <50MB for 1M rows
- Streaming generation: <100MB regardless of size
- Export overhead: 10-20% of data size

**Optimization Examples:**

```python
# Fast generation for testing
tempdata.create_dataset('quick_test.csv', rows=1000)  # <0.1 seconds

# Optimized for large datasets
tempdata.create_dataset('big_data.parquet', rows=5_000_000,
                       formats=['parquet'],  # Fastest format
                       use_streaming=True)

# Batch optimization
datasets = [
    {'filename': f'partition_{i}.csv', 'rows': 100_000}
    for i in range(10)
]
tempdata.create_batch(datasets, formats=['parquet'])  # Parallel processing
```

## Best Practices

### Data Quality

**Reproducibility:**
```python
# Use seeds for consistent test data
tempdata.create_dataset('test_data.csv', rows=1000, seed=12345)

# Same seed produces identical results
data1 = tempdata.create_dataset('test1.csv', rows=100, seed=999)
data2 = tempdata.create_dataset('test2.csv', rows=100, seed=999)
# data1 and data2 are identical
```

**Realistic Patterns:**
```python
# Use appropriate countries for realistic data
tempdata.create_dataset('european_customers.csv', rows=5000, 
                       country='germany')

# Time series with seasonal patterns
tempdata.create_dataset('retail_sales.csv', rows=365,
                       time_series=True,
                       interval='1day',
                       seasonal_patterns=True)
```

### Development Workflow

**Testing:**
```python
# Small datasets for unit tests
tempdata.create_dataset('unit_test.csv', rows=10, seed=123)

# Medium datasets for integration tests
tempdata.create_dataset('integration_test.csv', rows=1000, seed=456)

# Large datasets for performance tests
tempdata.create_dataset('perf_test.csv', rows=100_000, seed=789)
```

**Staging Data:**
```python
# Production-like volumes for staging
staging_datasets = [
    {'filename': 'staging_customers.csv', 'rows': 50_000},
    {'filename': 'staging_orders.csv', 'rows': 200_000, 
     'relationships': ['customers']},
    {'filename': 'staging_products.csv', 'rows': 5_000}
]
tempdata.create_batch(staging_datasets, country='united_states')
```

### Security Considerations

**Data Privacy:**
- No real personal data is generated
- All data is synthetic and safe for development
- GDPR/CCPA compliant synthetic data
- No risk of data leakage

**Best Practices:**
```python
# Use appropriate localization
tempdata.create_dataset('eu_customers.csv', rows=10000, 
                       country='germany')  # GDPR-appropriate

# Avoid overly realistic patterns in sensitive domains
tempdata.create_dataset('medical_data.csv', rows=1000,
                       # Medical data is anonymized and synthetic
                       country='canada')
```

### File Organization

**Naming Conventions:**
```python
# Include metadata in filenames
tempdata.create_dataset('sales_2024_us_10k.csv', rows=10000,
                       country='united_states',
                       time_series=True,
                       start_date='2024-01-01')

# Batch with consistent naming
datasets = [
    {'filename': 'ecom_customers_2024.csv', 'rows': 5000},
    {'filename': 'ecom_orders_2024.csv', 'rows': 25000},
    {'filename': 'ecom_products_2024.csv', 'rows': 1000}
]
```

**Directory Structure:**
```
data/
├── raw/
│   ├── customers.csv
│   ├── orders.csv
│   └── products.csv
├── processed/
│   ├── customers.parquet
│   └── orders.parquet
└── test/
    ├── small_sample.csv
    └── integration_test.csv
```

This comprehensive documentation provides detailed information about all aspects of the TempData library, from basic usage to advanced features and best practices.