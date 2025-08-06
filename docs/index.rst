TempData Documentation
=====================

.. image:: https://badge.fury.io/py/tempdata.svg
   :target: https://badge.fury.io/py/tempdata
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/tempdata.svg
   :target: https://pypi.org/project/tempdata/
   :alt: Python Support

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://github.com/dot-css/tempdata/workflows/CI/badge.svg
   :target: https://github.com/dot-css/tempdata/actions
   :alt: Build Status

**TempData** is a comprehensive Python library designed to generate realistic fake data for testing, development, and prototyping purposes. With support for 40+ dataset types spanning business, financial, healthcare, technology, IoT, and social domains, TempData provides worldwide geographical capabilities and time-based dynamic seeding for reproducible yet unique data generation.

Key Features
------------

* **40+ Dataset Types**: Business, financial, healthcare, technology, IoT, and social datasets
* **Global Coverage**: Generate geographically accurate data for any country or region
* **Time Series Support**: Realistic temporal patterns with seasonal variations and correlations
* **Batch Generation**: Create related datasets with maintained referential integrity
* **Multiple Export Formats**: CSV, JSON, Parquet, Excel, GeoJSON
* **High Performance**: 50,000+ rows per second with streaming support for large datasets
* **Simple API**: Intuitive interface with sensible defaults
* **CLI Interface**: Command-line tools for automation and scripting
* **Reproducible**: Time-based dynamic seeding with optional fixed seeds

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   # Install from PyPI
   pip install tempdata

   # Install with all optional dependencies
   pip install tempdata[dev,docs,performance]

Basic Usage
~~~~~~~~~~~

.. code-block:: python

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

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user-guide/installation
   user-guide/quickstart
   user-guide/dataset-types
   user-guide/geographical-data
   user-guide/time-series
   user-guide/batch-generation
   user-guide/export-formats
   user-guide/cli

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/business-intelligence
   examples/iot-pipelines
   examples/financial-analysis
   examples/batch-relationships
   examples/performance-optimization

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/generators
   api/geographical
   api/time-series
   api/exporters
   api/cli

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   advanced/performance
   advanced/customization
   advanced/extending
   advanced/best-practices

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`