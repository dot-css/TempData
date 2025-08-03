"""
Base exporter class for all export formats

Provides common functionality for data export operations.
"""

import pandas as pd
from abc import ABC, abstractmethod
from typing import Any


class BaseExporter(ABC):
    """
    Abstract base class for all data exporters
    
    Provides common interface and functionality that all specific
    exporters must implement.
    """
    
    def __init__(self):
        """Initialize base exporter"""
        pass
    
    @abstractmethod
    def export(self, data: pd.DataFrame, filename: str, **kwargs) -> str:
        """
        Export data to specified format
        
        Args:
            data: DataFrame to export
            filename: Output filename
            **kwargs: Format-specific options
            
        Returns:
            str: Path to exported file
        """
        raise NotImplementedError("Subclasses must implement export() method")
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate data before export
        
        Args:
            data: DataFrame to validate
            
        Returns:
            bool: True if data is valid for export
        """
        return not data.empty and len(data) > 0