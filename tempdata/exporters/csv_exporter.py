"""
CSV export functionality

Exports data to CSV format with proper formatting and headers.
"""

import pandas as pd
import os
from typing import Optional, Union, Dict, Any
from .base_exporter import BaseExporter


class CSVExporter(BaseExporter):
    """
    Exporter for CSV format
    
    Handles CSV export with proper formatting, headers, and data type handling.
    Supports various CSV options for compatibility with different systems.
    """
    
    def __init__(self):
        """Initialize CSV exporter"""
        super().__init__()
        self.supported_extensions = ['.csv']
        
        # Default CSV export options
        self.default_options = {
            'index': False,
            'encoding': 'utf-8',
            'sep': ',',
            'quoting': 1,  # csv.QUOTE_ALL
            'quotechar': '"',
            'escapechar': None,
            'lineterminator': '\n',
            'header': True,
            'date_format': '%Y-%m-%d %H:%M:%S'
        }
    
    def export(self, data: pd.DataFrame, filename: str, **kwargs) -> str:
        """
        Export data to CSV format with proper formatting and headers
        
        Args:
            data: DataFrame to export
            filename: Output filename
            **kwargs: CSV-specific options that override defaults
                - sep: Field separator (default: ',')
                - encoding: File encoding (default: 'utf-8')
                - quoting: Quoting behavior (default: csv.QUOTE_ALL)
                - header: Include headers (default: True)
                - index: Include row index (default: False)
                - date_format: Date formatting string
                - na_rep: String representation of NaN values
                - float_format: Format string for floating point numbers
                
        Returns:
            str: Path to exported CSV file
            
        Raises:
            ValueError: If data is invalid or filename is invalid
            IOError: If file cannot be written
        """
        # Validate input data
        self._validate_data(data)
        
        # Validate and normalize filename
        csv_path = self._validate_filename(filename, '.csv')
        
        # Ensure output directory exists
        self._ensure_directory_exists(csv_path)
        
        # Prepare data for CSV export
        export_data = self._prepare_csv_data(data)
        
        # Merge default options with user-provided options
        export_options = {**self.default_options, **kwargs}
        
        try:
            # Export to CSV with proper error handling
            export_data.to_csv(csv_path, **export_options)
            
            # Verify the file was created successfully
            if not os.path.exists(csv_path):
                raise IOError(f"Failed to create CSV file: {csv_path}")
                
            return csv_path
            
        except Exception as e:
            # Clean up partial file if it exists
            if os.path.exists(csv_path):
                try:
                    os.remove(csv_path)
                except OSError:
                    pass
            raise IOError(f"Failed to export CSV file: {str(e)}")
    
    def _prepare_csv_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for CSV export with proper formatting
        
        Args:
            data: Original DataFrame
            
        Returns:
            pd.DataFrame: Prepared DataFrame for CSV export
        """
        # Create a copy to avoid modifying original data
        csv_data = data.copy()
        
        # Handle different data types appropriately for CSV
        for col in csv_data.columns:
            col_data = csv_data[col]
            
            # Handle datetime columns
            if pd.api.types.is_datetime64_any_dtype(col_data):
                # Convert to string format for better CSV compatibility
                csv_data[col] = col_data.dt.strftime('%Y-%m-%d %H:%M:%S')
                # Handle NaT values
                csv_data[col] = csv_data[col].replace('NaT', '')
            
            # Handle boolean columns
            elif pd.api.types.is_bool_dtype(col_data):
                # Convert to string for clarity
                csv_data[col] = col_data.map({True: 'True', False: 'False'})
            
            # Handle numeric columns with NaN
            elif pd.api.types.is_numeric_dtype(col_data):
                # Keep numeric types but ensure proper NaN handling
                pass
            
            # Handle object/string columns
            elif col_data.dtype == 'object':
                # Convert to string and handle None/NaN values
                csv_data[col] = col_data.astype(str)
                csv_data[col] = csv_data[col].replace(['None', 'nan', 'NaN'], '')
                
                # Handle potential CSV injection attacks
                csv_data[col] = csv_data[col].apply(self._sanitize_csv_value)
        
        return csv_data
    
    def _sanitize_csv_value(self, value: str) -> str:
        """
        Sanitize CSV values to prevent CSV injection attacks
        
        Args:
            value: String value to sanitize
            
        Returns:
            str: Sanitized value
        """
        if not isinstance(value, str):
            return str(value)
        
        # Check for potentially dangerous characters that could be interpreted as formulas
        dangerous_chars = ['=', '+', '-', '@']
        
        if value and value[0] in dangerous_chars:
            # Prefix with single quote to prevent formula interpretation
            return f"'{value}"
        
        return value
    
    def export_with_metadata(self, data: pd.DataFrame, filename: str, 
                           metadata: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        """
        Export CSV with optional metadata header
        
        Args:
            data: DataFrame to export
            filename: Output filename
            metadata: Optional metadata to include as comments
            **kwargs: CSV export options
            
        Returns:
            str: Path to exported CSV file
        """
        csv_path = self._validate_filename(filename, '.csv')
        self._ensure_directory_exists(csv_path)
        
        # If metadata is provided, write it as comments first
        if metadata:
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write("# CSV Export Metadata\n")
                for key, value in metadata.items():
                    f.write(f"# {key}: {value}\n")
                f.write("# \n")
        
        # Export the actual data
        export_options = {**self.default_options, **kwargs}
        if metadata:
            export_options['mode'] = 'a'  # Append mode to preserve metadata
        
        prepared_data = self._prepare_csv_data(data)
        prepared_data.to_csv(csv_path, **export_options)
        
        return csv_path
    
    def get_export_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get information about what the CSV export will contain
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dict with export information
        """
        info = {
            'rows': len(data),
            'columns': len(data.columns),
            'column_names': list(data.columns),
            'data_types': {},
            'memory_usage_mb': round(data.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            'estimated_csv_size_mb': 0
        }
        
        # Analyze data types
        for col in data.columns:
            dtype = str(data[col].dtype)
            info['data_types'][col] = dtype
        
        # Estimate CSV file size (rough approximation)
        # Average characters per cell + separators + newlines
        avg_chars_per_cell = 10
        total_cells = len(data) * len(data.columns)
        estimated_size = total_cells * avg_chars_per_cell
        info['estimated_csv_size_mb'] = round(estimated_size / (1024 * 1024), 2)
        
        return info