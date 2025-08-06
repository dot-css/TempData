"""
TempData - Realistic fake data generation library

A comprehensive Python library for generating realistic fake data for testing,
development, and prototyping purposes with worldwide geographical capabilities
and time-based dynamic seeding.

Key Features:
    - 40+ dataset types across business, financial, healthcare, technology, IoT, and social domains
    - Worldwide geographical data generation with country-specific accuracy
    - Time-based dynamic seeding for unique yet reproducible data
    - Multiple export formats (CSV, JSON, Parquet, Excel, GeoJSON)
    - Time series generation with realistic temporal patterns
    - Batch generation with maintained relationships between datasets
    - Performance optimization with streaming generation for large datasets
    - Command-line interface for automation and scripting

Quick Start:
    >>> import tempdata
    
    # Generate a simple sales dataset
    >>> tempdata.create_dataset('sales.csv', rows=1000)
    'sales.csv'
    
    # Generate data for a specific country
    >>> tempdata.create_dataset('customers.json', rows=500, country='pakistan')
    'customers.json'
    
    # Generate time series data
    >>> tempdata.create_dataset('stocks.csv', rows=1000, time_series=True, 
    ...                        start_date='2024-01-01', end_date='2024-12-31')
    'stocks.csv'
    
    # Generate geographical data
    >>> addresses = tempdata.geo.addresses('germany', count=10)
    >>> route = tempdata.geo.route('Berlin', 'Munich', waypoints=2)
    
    # Generate batch datasets with relationships
    >>> datasets = [
    ...     {'filename': 'customers.csv', 'rows': 1000},
    ...     {'filename': 'sales.csv', 'rows': 5000, 'relationships': ['customers']}
    ... ]
    >>> tempdata.create_batch(datasets, country='united_states')
    ['customers.csv', 'sales.csv']

Supported Dataset Types:
    Business: sales, customers, ecommerce, inventory, marketing, employees, 
              suppliers, retail, reviews, crm
    Financial: stocks, banking, crypto, insurance, loans, investments, 
               accounting, payments
    Healthcare: patients, medical_history, appointments, lab_results, 
                prescriptions, clinical_trials
    Technology: web_analytics, app_usage, system_logs, api_calls, 
                server_metrics, user_sessions, error_logs, performance
    IoT Sensors: weather, energy, traffic, environmental, industrial, smart_home
    Social: social_media, user_profiles

Supported Countries:
    Over 20 countries supported including: united_states, canada, united_kingdom,
    germany, france, spain, italy, netherlands, sweden, norway, denmark, 
    finland, poland, czech_republic, austria, switzerland, australia, 
    new_zealand, japan, south_korea, india, pakistan, brazil, mexico, 
    argentina, chile, south_africa, egypt, nigeria, kenya

Export Formats:
    - CSV: Standard comma-separated values
    - JSON: JavaScript Object Notation with proper data types
    - Parquet: Compressed columnar format for analytics
    - Excel: Microsoft Excel format (.xlsx)
    - GeoJSON: Geographical data in JSON format

Performance Guidelines:
    - Datasets up to 10K rows: Standard generation (< 1 second)
    - Datasets 10K-100K rows: Fast generation (1-10 seconds)
    - Datasets 100K+ rows: Automatic streaming generation (memory efficient)
    - Maximum supported: 10 million rows per dataset

For detailed documentation and examples, visit: https://tempdata.readthedocs.io
"""

__version__ = "0.1.0"
__author__ = "TempData Team"

# Core API imports
from .api import create_dataset, create_batch
from . import geo

__all__ = [
    "create_dataset",
    "create_batch", 
    "geo",
    "__version__"
]