#!/usr/bin/env python3
"""
Setup script for TempData.

This file is maintained for backward compatibility.
The main configuration is in pyproject.toml.
"""

from setuptools import setup, find_packages

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tempdata",
    version="0.1.0",
    author="TempData Team",
    author_email="saqibshaikhdz@gmail.com",
    description="Realistic fake data generation library with worldwide geographical capabilities",
    long_description=long_description,
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
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "faker>=19.0.0",
        "pytz>=2023.3",
        "openpyxl>=3.1.0",
        "click>=8.1.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tempdata=tempdata.cli:cli",
        ],
    },
)