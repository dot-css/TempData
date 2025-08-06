# TempData Performance Tuning Guide

## Overview

This guide provides detailed information on optimizing TempData performance for different use cases, from small test datasets to large-scale data generation scenarios.

## Performance Characteristics

### Generation Speed Benchmarks

| Dataset Type | Rows/Second | Memory Usage (1M rows) | Notes |
|--------------|-------------|-------------------------|-------|
| Simple Business | 50,000+ | 30MB | Basic sales, customers |
| Complex Business | 25,000+ | 45MB | E-commerce with relationships |
| Financial Time Series | 15,000+ | 40MB | Stock data with correlations |
| Geographical Data | 10,000+ | 50MB | Addresses with coordinates |
| IoT Sensor Data | 20,000+ | 35MB | Weather, energy readings |
| Healthcare Data | 12,000+ | 55MB | Patient records with privacy |

### Memory Usage Patterns

**Standard Generation:**
- Memory usage scales linearly with dataset size
- Peak memory: ~50MB per 1M rows
- Suitable for datasets up to 100K rows

**Streaming Generation:**
- Constant memory usage regardless of dataset size
- Peak memory: ~100MB maximum
- Automatically enabled for datasets ≥100K rows
- Can be forced for smaller datasets

## Optimization Strategies

### 1. Choose Appropriate Generation Mode

```python
import tempdata

# Small datasets (< 10K rows) - Standard generation
tempdata.create_dataset('small_test.csv', rows=5000)

# Medium datasets (10K-100K rows) - Standard generation
tempdata.create_dataset('medium_data.csv', rows=50000)

# Large datasets (≥ 100K rows) - Automatic streaming
tempdata.create_dataset('large_data.csv', rows=1_000_000)  # Auto-streams

# Force streaming for memory-constrained environments
tempdata.create_dataset('forced_stream.csv', rows=25000, use_streaming=True)
```

### 2. Optimize Export Formats

**Format Performance Comparison:**

| Format | Write Speed | File Size | Read Speed | Use Case |
|--------|-------------|-----------|------------|----------|
| CSV | Fast | Large | Medium | Human readable, universal |
| JSON | Medium | Large | Slow | Web APIs, nested data |
| Parquet | Very Fast | Small | Very Fast | Analytics, data science |
| Excel | Slow | Medium | Slow | Business reports, presentations |
| GeoJSON | Medium | Large | Medium | Mapping applications |

```python
# Fastest for large datasets
tempdata.create_dataset('analytics.parquet', rows=1_000_000, 
                       formats=['parquet'])

# Avoid Excel for large datasets
# tempdata.create_dataset('big_report.xlsx', rows=100_000)  # Slow!

# Multi-format optimization
tempdata.create_dataset('data', rows=50000,
                       formats=['parquet', 'csv'])  # Parquet first
```

### 3. Streaming Configuration

```python
from tempdata.core.streaming import StreamingConfig

# Memory-constrained environment
low_memory_config = StreamingConfig(
    chunk_size=10000,      # Smaller chunks
    max_memory_mb=50,      # Lower memory limit
    progress_callback=True  # Monitor progress
)

tempdata.create_dataset('constrained.csv', rows=500_000,
                       streaming_config=low_memory_config)

# High-performance environment
high_perf_config = StreamingConfig(
    chunk_size=100000,     # Larger chunks
    max_memory_mb=500,     # Higher memory limit
    parallel_chunks=4      # Parallel processing
)

tempdata.create_dataset('fast_gen.csv', rows=2_000_000,
                       streaming_config=high_perf_config)
```

### 4. Batch Generation Optimization

```python
# Optimize batch order - generate independent datasets first
optimized_batch = [
    # Independent datasets first (can be parallelized)
    {'filename': 'products.csv', 'rows': 1000},
    {'filename': 'categories.csv', 'rows': 100},
    {'filename': 'regions.csv', 'rows': 50},
    
    # Dependent datasets second
    {'filename': 'customers.csv', 'rows': 10000, 'relationships': ['regions']},
    {'filename': 'orders.csv', 'rows': 50000, 
     'relationships': ['customers', 'products']}
]

tempdata.create_batch(optimized_batch, formats=['parquet'])
```

### 5. Time Series Optimization

```python
# Optimize interval selection
# Daily data for long periods
tempdata.create_dataset('daily_stocks.csv', rows=252,  # One trading year
                       time_series=True, interval='1day')

# Hourly data for shorter periods
tempdata.create_dataset('hourly_sensors.csv', rows=168,  # One week
                       time_series=True, interval='1hour')

# Avoid very high frequency for large datasets
# tempdata.create_dataset('minute_data.csv', rows=525600,  # One year of minutes - Very slow!

# Use appropriate time ranges
tempdata.create_dataset('efficient_timeseries.csv', rows=8760,  # One year hourly
                       time_series=True,
                       interval='1hour',
                       start_date='2024-01-01',
                       end_date='2024-12-31')
```

## Memory Management

### Understanding Memory Usage

```python
import psutil
import tempdata

def monitor_memory_usage():
    """Monitor memory usage during generation"""
    process = psutil.Process()
    
    # Baseline memory
    baseline = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Baseline memory: {baseline:.1f} MB")
    
    # Generate dataset
    tempdata.create_dataset('memory_test.csv', rows=100000)
    
    # Peak memory
    peak = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Peak memory: {peak:.1f} MB")
    print(f"Memory increase: {peak - baseline:.1f} MB")

monitor_memory_usage()
```

### Memory-Efficient Patterns

```python
# Pattern 1: Use streaming for large datasets
def generate_large_dataset():
    return tempdata.create_dataset('large.parquet', rows=1_000_000,
                                  use_streaming=True,
                                  formats=['parquet'])

# Pattern 2: Generate in batches instead of single large dataset
def generate_in_batches():
    batch_datasets = [
        {'filename': f'batch_{i}.csv', 'rows': 100_000}
        for i in range(10)  # 10 batches of 100K = 1M total
    ]
    return tempdata.create_batch(batch_datasets)

# Pattern 3: Use appropriate data types
def optimize_data_types():
    # Parquet automatically optimizes data types
    return tempdata.create_dataset('optimized.parquet', rows=500_000,
                                  formats=['parquet'])
```

## Performance Monitoring

### Built-in Performance Metrics

```python
import time
import tempdata

def benchmark_generation():
    """Benchmark dataset generation performance"""
    
    test_cases = [
        (1000, 'Small dataset'),
        (10000, 'Medium dataset'),
        (100000, 'Large dataset'),
        (1000000, 'Very large dataset')
    ]
    
    for rows, description in test_cases:
        start_time = time.time()
        
        path = tempdata.create_dataset(f'benchmark_{rows}.csv', rows=rows)
        
        end_time = time.time()
        duration = end_time - start_time
        rate = rows / duration
        
        print(f"{description}: {rows:,} rows in {duration:.2f}s ({rate:,.0f} rows/sec)")

benchmark_generation()
```

### Custom Performance Monitoring

```python
import tempdata
from tempdata.core.streaming import StreamingConfig

class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.rows_processed = 0
    
    def progress_callback(self, rows_completed, total_rows):
        """Custom progress callback for streaming generation"""
        if self.start_time is None:
            self.start_time = time.time()
        
        self.rows_processed = rows_completed
        elapsed = time.time() - self.start_time
        rate = rows_completed / elapsed if elapsed > 0 else 0
        
        progress = (rows_completed / total_rows) * 100
        print(f"Progress: {progress:.1f}% ({rows_completed:,}/{total_rows:,}) "
              f"Rate: {rate:,.0f} rows/sec")

# Use custom monitoring
monitor = PerformanceMonitor()
config = StreamingConfig(
    chunk_size=50000,
    progress_callback=monitor.progress_callback
)

tempdata.create_dataset('monitored.csv', rows=500_000,
                       streaming_config=config)
```

## Scaling Strategies

### Horizontal Scaling

```python
import concurrent.futures
import tempdata

def generate_partition(partition_id, rows_per_partition):
    """Generate a single partition of data"""
    filename = f'partition_{partition_id}.parquet'
    return tempdata.create_dataset(filename, rows=rows_per_partition,
                                  formats=['parquet'])

def parallel_generation(total_rows, num_partitions=4):
    """Generate large dataset using parallel partitions"""
    rows_per_partition = total_rows // num_partitions
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_partitions) as executor:
        futures = [
            executor.submit(generate_partition, i, rows_per_partition)
            for i in range(num_partitions)
        ]
        
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    return results

# Generate 1M rows across 4 partitions
partitions = parallel_generation(1_000_000, num_partitions=4)
print(f"Generated {len(partitions)} partitions")
```

### Vertical Scaling

```python
# Optimize for high-memory environments
def high_memory_generation():
    config = StreamingConfig(
        chunk_size=500_000,    # Very large chunks
        max_memory_mb=2000,    # 2GB memory limit
        parallel_chunks=8      # High parallelism
    )
    
    return tempdata.create_dataset('high_memory.parquet', rows=10_000_000,
                                  streaming_config=config,
                                  formats=['parquet'])

# Optimize for low-memory environments
def low_memory_generation():
    config = StreamingConfig(
        chunk_size=5_000,      # Small chunks
        max_memory_mb=100,     # 100MB limit
        parallel_chunks=1      # No parallelism
    )
    
    return tempdata.create_dataset('low_memory.csv', rows=1_000_000,
                                  streaming_config=config)
```

## Environment-Specific Optimizations

### Development Environment

```python
# Fast iteration for development
def dev_dataset():
    return tempdata.create_dataset('dev_sample.csv', rows=1000,
                                  seed=12345,  # Consistent data
                                  formats=['csv'])  # Human readable

# Quick prototyping
def prototype_data():
    datasets = [
        {'filename': 'proto_users.csv', 'rows': 100},
        {'filename': 'proto_orders.csv', 'rows': 500, 'relationships': ['users']}
    ]
    return tempdata.create_batch(datasets, seed=999)
```

### Testing Environment

```python
# Unit test data
def unit_test_data():
    return tempdata.create_dataset('unit_test.csv', rows=10,
                                  seed=123,  # Deterministic
                                  formats=['csv'])

# Integration test data
def integration_test_data():
    return tempdata.create_dataset('integration_test.parquet', rows=10000,
                                  seed=456,
                                  formats=['parquet'])  # Fast I/O

# Load test data
def load_test_data():
    return tempdata.create_dataset('load_test.parquet', rows=1_000_000,
                                  use_streaming=True,
                                  formats=['parquet'])
```

### Production Environment

```python
# Production-scale data generation
def production_dataset():
    config = StreamingConfig(
        chunk_size=100_000,
        max_memory_mb=1000,
        parallel_chunks=4,
        progress_callback=True
    )
    
    return tempdata.create_dataset('production.parquet', rows=50_000_000,
                                  streaming_config=config,
                                  formats=['parquet'])

# Staging environment data
def staging_data():
    datasets = [
        {'filename': 'staging_customers.parquet', 'rows': 100_000},
        {'filename': 'staging_orders.parquet', 'rows': 1_000_000,
         'relationships': ['customers']},
        {'filename': 'staging_products.parquet', 'rows': 10_000}
    ]
    
    return tempdata.create_batch(datasets, 
                                formats=['parquet'],
                                country='united_states')
```

## Troubleshooting Performance Issues

### Common Performance Problems

**Problem 1: Slow generation for large datasets**
```python
# Bad: Using standard generation for large datasets
# tempdata.create_dataset('slow.csv', rows=1_000_000)

# Good: Use streaming generation
tempdata.create_dataset('fast.parquet', rows=1_000_000,
                       use_streaming=True,
                       formats=['parquet'])
```

**Problem 2: Memory errors**
```python
# Bad: Large dataset without streaming
# tempdata.create_dataset('memory_error.csv', rows=5_000_000)

# Good: Use streaming with memory limits
config = StreamingConfig(max_memory_mb=200)
tempdata.create_dataset('memory_safe.csv', rows=5_000_000,
                       streaming_config=config)
```

**Problem 3: Slow export formats**
```python
# Bad: Excel for large datasets
# tempdata.create_dataset('slow.xlsx', rows=100_000, formats=['excel'])

# Good: Use Parquet for large datasets
tempdata.create_dataset('fast.parquet', rows=100_000, formats=['parquet'])
```

### Performance Debugging

```python
import cProfile
import tempdata

def profile_generation():
    """Profile dataset generation performance"""
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Generate dataset
    tempdata.create_dataset('profile_test.csv', rows=50000)
    
    profiler.disable()
    profiler.print_stats(sort='cumulative')

# Run profiling
profile_generation()
```

### Resource Monitoring

```python
import psutil
import time
import tempdata

class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
    
    def start_monitoring(self):
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024
        print(f"Starting monitoring - Memory: {self.start_memory:.1f} MB")
    
    def stop_monitoring(self):
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024
        
        duration = end_time - self.start_time
        memory_increase = end_memory - self.start_memory
        
        print(f"Generation completed in {duration:.2f} seconds")
        print(f"Peak memory: {end_memory:.1f} MB")
        print(f"Memory increase: {memory_increase:.1f} MB")

# Use resource monitoring
monitor = ResourceMonitor()
monitor.start_monitoring()

tempdata.create_dataset('monitored.csv', rows=100_000)

monitor.stop_monitoring()
```

## Best Practices Summary

1. **Use streaming for datasets ≥100K rows**
2. **Choose Parquet format for large datasets and analytics**
3. **Use appropriate time series intervals**
4. **Monitor memory usage in production**
5. **Leverage parallel generation for very large datasets**
6. **Use seeds for reproducible performance testing**
7. **Profile generation code to identify bottlenecks**
8. **Consider horizontal scaling for massive datasets**

This performance guide provides comprehensive strategies for optimizing TempData usage across different environments and use cases.