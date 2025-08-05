"""
Base exporter class for all export formats

Provides common functionality for data export operations.
"""

import os
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pathlib import Path


class BaseExporter(ABC):
    """
    Abstract base class for all data exporters
    
    Provides common interface and functionality that all specific
    exporters must implement.
    """
    
    def __init__(self):
        """Initialize base exporter"""
        self.supported_extensions = []
    
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
            
        Raises:
            ValueError: If data is invalid or filename is invalid
            IOError: If file cannot be written
        """
        raise NotImplementedError("Subclasses must implement export() method")
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate data before export
        
        Args:
            data: DataFrame to validate
            
        Returns:
            bool: True if data is valid for export
            
        Raises:
            ValueError: If data is invalid
        """
        if data is None:
            raise ValueError("Data cannot be None")
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
            
        if data.empty:
            raise ValueError("Data cannot be empty")
            
        if len(data) == 0:
            raise ValueError("Data must contain at least one row")
            
        return True
    
    def _validate_filename(self, filename: str, expected_extension: str = None) -> str:
        """
        Validate and normalize filename
        
        Args:
            filename: Input filename
            expected_extension: Expected file extension (e.g., '.csv')
            
        Returns:
            str: Validated and normalized filename
            
        Raises:
            ValueError: If filename is invalid
        """
        if not filename or not isinstance(filename, str):
            raise ValueError("Filename must be a non-empty string")
        
        # Remove any path separators that might cause issues
        filename = filename.strip()
        
        # Add extension if not present and expected_extension is provided
        if expected_extension and not filename.endswith(expected_extension):
            filename = f"{filename}{expected_extension}"
        
        return filename
    
    def _ensure_directory_exists(self, filepath: str) -> None:
        """
        Ensure the directory for the file exists
        
        Args:
            filepath: Full path to the file
        """
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
    
    def _optimize_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize data types for export to reduce file size
        
        Args:
            data: DataFrame to optimize
            
        Returns:
            pd.DataFrame: DataFrame with optimized data types
        """
        optimized_data = data.copy()
        
        for col in optimized_data.columns:
            col_data = optimized_data[col]
            
            # Skip if column contains mixed types or objects that aren't strings
            if col_data.dtype == 'object':
                # Try to convert string numbers to numeric
                if col_data.notna().all() and col_data.astype(str).str.match(r'^-?\d+\.?\d*$').all():
                    try:
                        if '.' in str(col_data.iloc[0]):
                            optimized_data[col] = pd.to_numeric(col_data, errors='ignore')
                        else:
                            optimized_data[col] = pd.to_numeric(col_data, errors='ignore', downcast='integer')
                    except (ValueError, TypeError):
                        pass
                continue
            
            # Optimize integer types
            if col_data.dtype.kind in 'iu':  # integer types
                col_min, col_max = col_data.min(), col_data.max()
                if col_min >= 0:  # unsigned integers
                    if col_max < 255:
                        optimized_data[col] = col_data.astype('uint8')
                    elif col_max < 65535:
                        optimized_data[col] = col_data.astype('uint16')
                    elif col_max < 4294967295:
                        optimized_data[col] = col_data.astype('uint32')
                else:  # signed integers
                    if col_min > -128 and col_max < 127:
                        optimized_data[col] = col_data.astype('int8')
                    elif col_min > -32768 and col_max < 32767:
                        optimized_data[col] = col_data.astype('int16')
                    elif col_min > -2147483648 and col_max < 2147483647:
                        optimized_data[col] = col_data.astype('int32')
            
            # Optimize float types
            elif col_data.dtype.kind == 'f':  # float types
                if col_data.dtype == 'float64':
                    # Check if we can safely downcast to float32
                    if col_data.between(-3.4e38, 3.4e38).all():
                        optimized_data[col] = col_data.astype('float32')
        
        return optimized_data
    
    def get_file_info(self, filepath: str) -> Dict[str, Any]:
        """
        Get information about the exported file
        
        Args:
            filepath: Path to the exported file
            
        Returns:
            Dict containing file information
        """
        if not os.path.exists(filepath):
            return {}
        
        stat = os.stat(filepath)
        return {
            'filepath': filepath,
            'size_bytes': stat.st_size,
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'created': stat.st_ctime,
            'modified': stat.st_mtime
        }