"""
Setup configuration for TempData library
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "TempData - Realistic fake data generation library"

# Read version from __init__.py
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'tempdata', '__init__.py')
    with open(version_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return "0.1.0"

setup(
    name="tempdata",
    version=get_version(),
    author="TempData Team",
    author_email="team@tempdata.dev",
    description="Realistic fake data generation library with worldwide geographical capabilities",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/dot-css/tempdata",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Testing",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Database",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "faker>=19.0.0",
        
        # Geographical and time handling
        "pytz>=2023.3",
        "geopy>=2.3.0",
        
        # Export formats
        "openpyxl>=3.1.0",  # Excel export
        "pyarrow>=12.0.0",  # Parquet export
        "geojson>=3.0.0",   # GeoJSON export
        
        # CLI interface
        "click>=8.1.0",
        
        # Performance and utilities
        "psutil>=5.9.0",    # Memory monitoring
        "tqdm>=4.65.0",     # Progress bars
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-benchmark>=4.0.0",
            "hypothesis>=6.82.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "performance": [
            "numba>=0.57.0",    # JIT compilation for performance
            "dask>=2023.8.0",   # Parallel processing
        ]
    },
    entry_points={
        "console_scripts": [
            "tempdata=tempdata.cli:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "tempdata": [
            "data/countries/*.json",
            "data/business/*.json", 
            "data/templates/*.json",
            "data/templates/*.txt",
        ],
    },
    zip_safe=False,
)