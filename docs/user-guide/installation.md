# Installation

## Requirements

TempData requires Python 3.8 or higher and supports the following platforms:

- **Operating Systems**: Windows, macOS, Linux
- **Python Versions**: 3.8, 3.9, 3.10, 3.11, 3.12
- **Memory**: Minimum 512MB RAM (2GB+ recommended for large datasets)
- **Storage**: 100MB for installation, additional space for generated data

## Installation Methods

### PyPI Installation (Recommended)

Install the latest stable version from PyPI:

```bash
pip install tempdata
```

### Install with Optional Dependencies

For development and advanced features:

```bash
# Install with development tools
pip install tempdata[dev]

# Install with documentation tools
pip install tempdata[docs]

# Install with performance optimizations
pip install tempdata[performance]

# Install everything
pip install tempdata[dev,docs,performance]
```

### Development Installation

For contributing to TempData:

```bash
# Clone the repository
git clone https://github.com/dot-css/tempdata.git
cd tempdata

# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e .[dev]

# Set up pre-commit hooks
pre-commit install
```

### Conda Installation

TempData is also available via conda-forge:

```bash
conda install -c conda-forge tempdata
```

## Verification

Verify your installation:

```python
import tempdata
print(tempdata.__version__)

# Test basic functionality
data = tempdata.create_dataset('test.csv', rows=10)
print(f"Generated test dataset: {data}")
```

## Dependencies

### Core Dependencies

- **pandas** (>=1.5.0): Data manipulation and analysis
- **numpy** (>=1.21.0): Numerical computing
- **faker** (>=19.0.0): Basic fake data generation
- **pytz** (>=2023.3): Timezone handling
- **geopy** (>=2.3.0): Geographical calculations

### Export Format Dependencies

- **openpyxl** (>=3.1.0): Excel file support
- **pyarrow** (>=12.0.0): Parquet file support
- **geojson** (>=3.0.0): GeoJSON format support

### CLI Dependencies

- **click** (>=8.1.0): Command-line interface

### Utility Dependencies

- **psutil** (>=5.9.0): Memory monitoring
- **tqdm** (>=4.65.0): Progress bars

### Optional Dependencies

#### Development Tools (`[dev]`)

- **pytest** (>=7.4.0): Testing framework
- **pytest-cov** (>=4.1.0): Coverage reporting
- **pytest-benchmark** (>=4.0.0): Performance benchmarking
- **hypothesis** (>=6.82.0): Property-based testing
- **black** (>=23.7.0): Code formatting
- **flake8** (>=6.0.0): Linting
- **mypy** (>=1.5.0): Type checking
- **pre-commit** (>=3.3.0): Git hooks

#### Documentation Tools (`[docs]`)

- **sphinx** (>=7.1.0): Documentation generation
- **sphinx-rtd-theme** (>=1.3.0): ReadTheDocs theme
- **myst-parser** (>=2.0.0): Markdown support

#### Performance Optimizations (`[performance]`)

- **numba** (>=0.57.0): JIT compilation
- **dask** (>=2023.8.0): Parallel processing

## Troubleshooting

### Common Issues

#### ImportError: No module named 'tempdata'

Make sure TempData is installed in the correct Python environment:

```bash
python -m pip list | grep tempdata
```

#### Permission Errors on Windows

Run the command prompt as Administrator:

```bash
pip install --user tempdata
```

#### Memory Issues with Large Datasets

Enable streaming for large datasets:

```python
tempdata.create_dataset(
    'large_data.csv',
    rows=1000000,
    use_streaming=True
)
```

#### Slow Performance

Install performance optimizations:

```bash
pip install tempdata[performance]
```

### Getting Help

If you encounter issues:

1. Check the [FAQ](../faq.md)
2. Search [GitHub Issues](https://github.com/dot-css/tempdata/issues)
3. Create a new issue with:
   - Python version (`python --version`)
   - TempData version (`pip show tempdata`)
   - Operating system
   - Complete error message
   - Minimal code example

## Next Steps

- Read the [Quick Start Guide](quickstart.md)
- Explore [Dataset Types](dataset-types.md)
- Try the [Examples](../examples/business-intelligence.md)