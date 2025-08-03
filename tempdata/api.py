"""
Main API interface for TempData library

Provides the primary functions for dataset generation including
create_dataset() and create_batch() functions.
"""

from typing import List, Dict, Any, Optional
import os
from .core.seeding import MillisecondSeeder
from .exporters.export_manager import ExportManager


def create_dataset(filename: str, rows: int = 500, **kwargs) -> str:
    """
    Generate single dataset with specified parameters
    
    Args:
        filename: Output filename (determines dataset type from name)
        rows: Number of rows to generate (default: 500)
        **kwargs: Additional parameters (country, seed, formats, etc.)
    
    Returns:
        str: Path to generated file(s)
    """
    # This is a placeholder implementation
    # Will be fully implemented in later tasks
    raise NotImplementedError("create_dataset will be implemented in task 13.1")


def create_batch(datasets: List[Dict[str, Any]], **kwargs) -> List[str]:
    """
    Generate multiple related datasets with maintained relationships
    
    Args:
        datasets: List of dataset specifications
        **kwargs: Global parameters applied to all datasets
    
    Returns:
        List[str]: Paths to generated files
    """
    # This is a placeholder implementation  
    # Will be fully implemented in task 13.2
    raise NotImplementedError("create_batch will be implemented in task 13.2")