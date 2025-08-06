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

**TempData** is a comprehensive Python library designed to generate realistic fake data for testing, development, and prototyping purposes.

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

   pip install tempdata

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

API Reference
-------------

.. automodule:: tempdata
   :members:
   :undoc-members:
   :show-inheritance:

Core Functions
~~~~~~~~~~~~~~

.. autofunction:: tempdata.create_dataset

.. autofunction:: tempdata.create_batch

Geographical Utilities
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tempdata.geo
   :members:

Configuration
~~~~~~~~~~~~~

.. autoclass:: tempdata.config
   :members:

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tempdata.performance
   :members:

Examples
~~~~~~~~

.. autoclass:: tempdata.examples
   :members:

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user-guide/installation
   user-guide/quickstart

.. toctree::
   :maxdepth: 1
   :caption: Development

   README
   CONTRIBUTING
   CHANGELOG

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`