# Contributing to TempData

Thank you for your interest in contributing to TempData! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues

Before creating an issue, please:

1. **Search existing issues** to avoid duplicates
2. **Use the issue templates** when available
3. **Provide detailed information** including:
   - Python version and operating system
   - TempData version
   - Complete error messages
   - Minimal code example to reproduce the issue
   - Expected vs actual behavior

### Suggesting Features

We welcome feature suggestions! Please:

1. **Check the roadmap** in README.md to see if it's already planned
2. **Open a discussion** first for major features
3. **Provide use cases** and examples
4. **Consider implementation complexity** and maintenance burden

### Contributing Code

1. **Fork the repository** and create a feature branch
2. **Follow the development setup** instructions below
3. **Write tests** for your changes
4. **Follow code style** guidelines
5. **Update documentation** if needed
6. **Submit a pull request** with a clear description

## 🛠️ Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- pip or conda

### Setup Instructions

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/tempdata.git
cd tempdata

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e .[dev,docs,performance]

# Install pre-commit hooks
pre-commit install

# Verify installation
python -c "import tempdata; print(tempdata.__version__)"
pytest --version
```

### Development Workflow

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ... edit code ...

# Run tests
pytest

# Run linting and formatting
black tempdata tests examples
flake8 tempdata tests examples
mypy tempdata

# Run pre-commit checks
pre-commit run --all-files

# Commit your changes
git add .
git commit -m "Add your descriptive commit message"

# Push to your fork
git push origin feature/your-feature-name

# Create a pull request on GitHub
```

## 📋 Code Style Guidelines

### Python Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting (line length: 88 characters)
- **Flake8**: Linting and style checking
- **MyPy**: Type checking
- **isort**: Import sorting

### Code Formatting

```bash
# Format code with Black
black tempdata tests examples

# Sort imports
isort tempdata tests examples

# Check linting
flake8 tempdata tests examples

# Type checking
mypy tempdata
```

### Naming Conventions

- **Functions and variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`
- **Files and modules**: `snake_case.py`

### Documentation Style

- Use **Google-style docstrings**
- Include **type hints** for all functions
- Add **examples** in docstrings when helpful
- Keep **line length** under 88 characters

Example:

```python
def create_dataset(
    filename: str,
    rows: int = 500,
    country: str = "united_states",
    seed: Optional[int] = None
) -> str:
    """Generate a realistic dataset and save to file.
    
    Args:
        filename: Output file path with extension
        rows: Number of rows to generate
        country: Country code for geographical data
        seed: Random seed for reproducible results
        
    Returns:
        Path to the generated file
        
    Raises:
        ValueError: If rows is negative or country is invalid
        
    Example:
        >>> path = create_dataset('sales.csv', rows=1000, seed=12345)
        >>> print(f"Generated: {path}")
        Generated: sales.csv
    """
```

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── test_api.py             # Main API tests
├── test_generators/        # Generator-specific tests
├── test_exporters/         # Export format tests
├── test_geographical/      # Geographical data tests
├── test_time_series/       # Time series tests
├── test_performance/       # Performance benchmarks
└── test_integration/       # End-to-end tests
```

### Writing Tests

```python
import pytest
import tempdata
from tempdata.core import DatasetGenerator

class TestDatasetGeneration:
    """Test dataset generation functionality."""
    
    def test_basic_generation(self):
        """Test basic dataset generation."""
        result = tempdata.create_dataset('test.csv', rows=10)
        assert result.endswith('test.csv')
        
    def test_with_seed(self):
        """Test reproducible generation with seed."""
        result1 = tempdata.create_dataset('test1.csv', rows=10, seed=12345)
        result2 = tempdata.create_dataset('test2.csv', rows=10, seed=12345)
        
        # Results should be identical
        with open(result1) as f1, open(result2) as f2:
            assert f1.read() == f2.read()
            
    @pytest.mark.parametrize("country", ["united_states", "germany", "japan"])
    def test_country_support(self, country):
        """Test generation for different countries."""
        result = tempdata.create_dataset(
            f'test_{country}.csv',
            rows=10,
            country=country
        )
        assert result is not None
        
    @pytest.mark.slow
    def test_large_dataset(self):
        """Test generation of large datasets."""
        result = tempdata.create_dataset(
            'large_test.csv',
            rows=100000,
            use_streaming=True
        )
        assert result is not None
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=tempdata --cov-report=html

# Run specific test categories
pytest -m "not slow"          # Skip slow tests
pytest -m integration         # Only integration tests
pytest -m performance         # Only performance tests

# Run specific test files
pytest tests/test_api.py
pytest tests/test_generators/

# Run with verbose output
pytest -v

# Run tests in parallel
pytest -n auto
```

### Test Categories

Use pytest markers to categorize tests:

- `@pytest.mark.slow`: Tests that take >1 second
- `@pytest.mark.integration`: End-to-end integration tests
- `@pytest.mark.performance`: Performance benchmarks
- `@pytest.mark.parametrize`: Parameterized tests

## 📚 Documentation Guidelines

### Documentation Structure

```
docs/
├── index.rst               # Main documentation index
├── user-guide/            # User-facing documentation
├── examples/              # Example tutorials
├── api/                   # API reference
├── advanced/              # Advanced topics
└── _static/               # Static assets
```

### Writing Documentation

- Use **Markdown** for user guides and examples
- Use **reStructuredText** for API documentation
- Include **code examples** that actually work
- Add **screenshots** for visual features
- Keep **language simple** and accessible

### Building Documentation

```bash
# Install documentation dependencies
pip install -e .[docs]

# Build documentation
cd docs
make html

# View documentation
open _build/html/index.html  # On macOS
# Or navigate to docs/_build/html/index.html in your browser

# Clean build
make clean html
```

### Documentation Standards

- **Code examples** must be tested and working
- **API documentation** is auto-generated from docstrings
- **User guides** should be task-oriented
- **Examples** should solve real-world problems

## 🚀 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md** with new features and fixes
3. **Run full test suite** including performance tests
4. **Update documentation** if needed
5. **Create release PR** and get approval
6. **Tag release** and push to GitHub
7. **Publish to PyPI** (automated via GitHub Actions)

## 🏗️ Project Structure

```
tempdata/
├── tempdata/              # Main package
│   ├── __init__.py       # Package initialization
│   ├── api.py            # Main API functions
│   ├── core/             # Core functionality
│   ├── generators/       # Data generators
│   ├── exporters/        # Export formats
│   ├── geographical/     # Geographical data
│   ├── time_series/      # Time series generation
│   ├── cli/              # Command-line interface
│   └── data/             # Data files and templates
├── tests/                # Test suite
├── examples/             # Example scripts
├── docs/                 # Documentation
├── .github/              # GitHub workflows
└── scripts/              # Development scripts
```

## 🎯 Areas for Contribution

### High Priority

- **New dataset types**: Healthcare, education, government
- **Performance optimizations**: Faster generation algorithms
- **Export formats**: New file formats and databases
- **Geographical coverage**: More countries and regions
- **Documentation**: Tutorials and examples

### Medium Priority

- **CLI enhancements**: Better command-line interface
- **Data quality**: More realistic patterns and correlations
- **Integration**: Plugins for popular tools
- **Visualization**: Data profiling and visualization tools

### Low Priority

- **GUI interface**: Desktop or web-based interface
- **Cloud integration**: AWS, GCP, Azure connectors
- **Real-time streaming**: Live data generation
- **Machine learning**: AI-powered data generation

## 🤔 Questions and Support

### Getting Help

- **Documentation**: [tempdata.readthedocs.io](https://tempdata.readthedocs.io)
- **GitHub Discussions**: For questions and ideas
- **GitHub Issues**: For bugs and feature requests
- **Email**: saqibshaikhdz@gmail.com for private matters

### Community Guidelines

- **Be respectful** and inclusive
- **Help others** learn and contribute
- **Share knowledge** and experiences
- **Follow the code of conduct**

## 📄 License

By contributing to TempData, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors are recognized in:

- **CONTRIBUTORS.md** file
- **GitHub contributors** page
- **Release notes** for significant contributions
- **Documentation** for major features

Thank you for contributing to TempData! 🎉