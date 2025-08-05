# TempData Batch Configuration Examples

This directory contains example configuration files for batch dataset generation using the TempData CLI.

## Usage

```bash
# Generate datasets from a configuration file
tempdata batch config.json

# Specify output directory
tempdata batch config.json --output-dir ./data

# Show what would be generated without creating files
tempdata batch config.json --dry-run

# Generate with progress indicators and verbose output
tempdata batch config.json --progress --verbose

# Generate quietly (minimal output)
tempdata batch config.json --quiet
```

## Configuration Files

### simple_batch.json
Basic batch generation with multiple independent datasets.
- Generates customers and sales data
- Uses global settings for country and formats
- No relationships between datasets

### related_datasets.json
Demonstrates datasets with relationships and referential integrity.
- Creates customers, sales, and products datasets
- Maintains foreign key relationships between customers and sales
- Uses Pakistani geographical data
- Exports in both CSV and JSON formats

### time_series_batch.yaml
Time series data generation with temporal patterns.
- Generates weather, energy, and stock data with time-based patterns
- Uses YAML configuration format
- Demonstrates different intervals for different datasets
- Covers a full year of hourly data

### large_dataset_batch.json
Large dataset generation with performance optimization.
- Generates datasets with 100K+ rows
- Uses Parquet format for efficient storage
- Demonstrates relationship handling with large datasets
- Includes web analytics data with millions of rows

## Configuration Structure

### JSON Format
```json
{
  "global": {
    "country": "global",
    "seed": 12345,
    "formats": ["csv", "json"]
  },
  "datasets": [
    {
      "filename": "output.csv",
      "rows": 1000,
      "type": "sales",
      "relationships": ["customers"]
    }
  ],
  "relationships": [
    {
      "source_dataset": "customers",
      "target_dataset": "sales", 
      "source_column": "customer_id",
      "target_column": "customer_id",
      "relationship_type": "one_to_many"
    }
  ]
}
```

### YAML Format
```yaml
global:
  country: pakistan
  seed: 42
  formats: [csv, parquet]

datasets:
  - filename: customers.csv
    rows: 1000
    type: customers
    
  - filename: sales.csv
    rows: 5000
    type: sales
    relationships: [customers]

relationships:
  - source_dataset: customers
    target_dataset: sales
    source_column: customer_id
    target_column: customer_id
    relationship_type: one_to_many
```

## Global Parameters

- `country`: Geographical region for data generation (default: "global")
- `seed`: Fixed seed for reproducible results (optional)
- `formats`: List of export formats ["csv", "json", "parquet", "excel", "geojson"]
- `time_series`: Enable time series generation (default: false)
- `start_date`: Start date for time series (ISO format)
- `end_date`: End date for time series (ISO format)
- `interval`: Time series interval ("1min", "1hour", "1day", etc.)

## Dataset Parameters

- `filename`: Output filename (determines dataset type if type not specified)
- `rows`: Number of rows to generate (default: 500)
- `type`: Explicit dataset type (optional, inferred from filename)
- `relationships`: List of dataset names this dataset depends on
- `formats`: Override global formats for this dataset (optional)

## Relationship Types

- `one_to_many`: Each source record can have multiple target records
- `many_to_one`: Multiple target records reference the same source record
- `one_to_one`: Each target record references a unique source record

## Performance Tips

1. Use Parquet format for large datasets (better compression and performance)
2. Set appropriate seeds for reproducible results
3. Use `--progress` flag for long-running generations
4. Consider memory usage when generating very large datasets
5. Use relationships to maintain data integrity across datasets

## Supported Dataset Types

- **Business**: sales, customers, ecommerce
- **Financial**: stocks, banking
- **Healthcare**: patients, appointments
- **Technology**: web_analytics, system_logs
- **IoT Sensors**: weather, energy
- **Social**: social_media, user_profiles