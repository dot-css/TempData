"""
TempData - Realistic fake data generation library

A comprehensive Python library for generating realistic fake data for testing,
development, and prototyping purposes with worldwide geographical capabilities
and time-based dynamic seeding.
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