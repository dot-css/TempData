"""
JSON export functionality

Exports data to JSON format with proper data type preservation.
"""

import json
import pandas as pd
import numpy as np
import os
from datetime import datetime, date
from decimal import Decimal
from typing import Optional, Union, Dict, Any, List
from .base_exporter import BaseExporter


class JSONExporter(BaseExporter):
    """
    Exporter for JSON format
    
    Handles JSON export with proper data type preservation and formatting.
    Supports various JSON output formats including records, values, and index.
    """
    
    def __init__(self):
        """Initialize JSON exporter"""
        super().__init__()
        self.supported_extensions = ['.json']
        
        # Default JSON export options for pandas to_json()
        self.pandas_options = {
            'orient': 'records',  # records, values, index, columns, split, table
            'date_format': 'iso',  # iso, epoch
            'date_unit': 's',     # s, ms, us, ns
        }
        
        # Default JSON dump options for json.dump()
        self.json_dump_options = {
            'indent': 2,          # Pretty printing
            'ensure_ascii': False,
            'sort_keys': False,
            'separators': (',', ': ')
        }
    
    def export(self, data: pd.DataFrame, filename: str, **kwargs) -> str:
        """
        Export data to JSON format with proper data type preservation
        
        Args:
            data: DataFrame to export
            filename: Output filename
            **kwargs: JSON-specific options that override defaults
                - orient: JSON orientation ('records', 'values', 'index', 'columns', 'split', 'table')
                - date_format: Date format ('iso' or 'epoch')
                - date_unit: Date unit for epoch format ('s', 'ms', 'us', 'ns')
                - indent: Indentation for pretty printing (int or None)
                - ensure_ascii: Ensure ASCII encoding (bool)
                - sort_keys: Sort keys alphabetically (bool)
                - preserve_types: Preserve pandas data types in metadata (bool)
                - include_metadata: Include dataset metadata (bool)
                
        Returns:
            str: Path to exported JSON file
            
        Raises:
            ValueError: If data is invalid or filename is invalid
            IOError: If file cannot be written
        """
        # Validate input data
        self._validate_data(data)
        
        # Validate and normalize filename
        json_path = self._validate_filename(filename, '.json')
        
        # Ensure output directory exists
        self._ensure_directory_exists(json_path)
        
        # Separate pandas and json.dump options
        pandas_options = {**self.pandas_options}
        json_dump_options = {**self.json_dump_options}
        
        # Process user-provided options
        preserve_types = kwargs.pop('preserve_types', False)
        include_metadata = kwargs.pop('include_metadata', False)
        
        # Update options with user preferences
        for key, value in kwargs.items():
            if key in ['orient', 'date_format', 'date_unit']:
                pandas_options[key] = value
            elif key in ['indent', 'ensure_ascii', 'sort_keys', 'separators']:
                json_dump_options[key] = value
        
        try:
            # Prepare data for JSON export
            export_data = self._prepare_json_data(data)
            
            # Convert to JSON using pandas
            json_str = export_data.to_json(**pandas_options)
            
            # Parse back to dict for additional processing
            json_data = json.loads(json_str)
            
            # Add metadata if requested
            if include_metadata or preserve_types:
                json_data = self._add_metadata(json_data, data, preserve_types, include_metadata)
            
            # Write to file with custom JSON encoder
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, **json_dump_options, cls=CustomJSONEncoder)
            
            # Verify the file was created successfully
            if not os.path.exists(json_path):
                raise IOError(f"Failed to create JSON file: {json_path}")
                
            return json_path
            
        except Exception as e:
            # Clean up partial file if it exists
            if os.path.exists(json_path):
                try:
                    os.remove(json_path)
                except OSError:
                    pass
            raise IOError(f"Failed to export JSON file: {str(e)}")
    
    def _prepare_json_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for JSON export with proper type handling
        
        Args:
            data: Original DataFrame
            
        Returns:
            pd.DataFrame: Prepared DataFrame for JSON export
        """
        # Create a copy to avoid modifying original data
        json_data = data.copy()
        
        # Handle different data types appropriately for JSON
        for col in json_data.columns:
            col_data = json_data[col]
            
            # Handle datetime columns
            if pd.api.types.is_datetime64_any_dtype(col_data):
                # Keep as datetime - pandas will handle conversion based on date_format
                pass
            
            # Handle boolean columns - keep as boolean
            elif pd.api.types.is_bool_dtype(col_data):
                pass
            
            # Handle numeric columns
            elif pd.api.types.is_numeric_dtype(col_data):
                # Convert NaN to None for proper JSON null representation
                json_data[col] = col_data.where(pd.notna(col_data), None)
            
            # Handle object/string columns
            elif col_data.dtype == 'object':
                # Handle None/NaN values
                json_data[col] = col_data.where(pd.notna(col_data), None)
                
                # Try to preserve numeric strings as numbers if they're all numeric
                if col_data.notna().all() and self._is_numeric_string_column(col_data):
                    try:
                        json_data[col] = pd.to_numeric(col_data, errors='ignore')
                    except (ValueError, TypeError):
                        pass
        
        return json_data
    
    def _is_numeric_string_column(self, series: pd.Series) -> bool:
        """
        Check if a string column contains only numeric values
        
        Args:
            series: Pandas series to check
            
        Returns:
            bool: True if all non-null values are numeric strings
        """
        non_null_values = series.dropna()
        if len(non_null_values) == 0:
            return False
        
        try:
            # Try to convert all values to numeric
            pd.to_numeric(non_null_values, errors='raise')
            return True
        except (ValueError, TypeError):
            return False
    
    def _add_metadata(self, json_data: Dict[str, Any], original_data: pd.DataFrame, 
                     preserve_types: bool, include_metadata: bool) -> Dict[str, Any]:
        """
        Add metadata to JSON output
        
        Args:
            json_data: Original JSON data
            original_data: Original DataFrame
            preserve_types: Whether to preserve pandas data types
            include_metadata: Whether to include general metadata
            
        Returns:
            Dict with metadata added
        """
        result = {'data': json_data}
        
        if include_metadata:
            result['metadata'] = {
                'export_timestamp': datetime.now().isoformat(),
                'rows': len(original_data),
                'columns': len(original_data.columns),
                'column_names': list(original_data.columns),
                'export_format': 'json'
            }
        
        if preserve_types:
            type_info = {}
            for col in original_data.columns:
                dtype = str(original_data[col].dtype)
                type_info[col] = {
                    'pandas_dtype': dtype,
                    'python_type': str(type(original_data[col].iloc[0]).__name__) if len(original_data) > 0 else 'unknown'
                }
            
            if 'metadata' not in result:
                result['metadata'] = {}
            result['metadata']['column_types'] = type_info
        
        return result
    
    def export_nested(self, data: pd.DataFrame, filename: str, 
                     group_by: Optional[str] = None, **kwargs) -> str:
        """
        Export JSON with nested structure based on grouping
        
        Args:
            data: DataFrame to export
            filename: Output filename
            group_by: Column to group by for nested structure
            **kwargs: JSON export options
            
        Returns:
            str: Path to exported JSON file
        """
        json_path = self._validate_filename(filename, '.json')
        self._ensure_directory_exists(json_path)
        
        if group_by and group_by in data.columns:
            # Create nested structure
            nested_data = {}
            for group_value, group_data in data.groupby(group_by):
                # Remove the grouping column from the data
                group_records = group_data.drop(columns=[group_by]).to_dict('records')
                nested_data[str(group_value)] = group_records
            
            json_data = nested_data
        else:
            # Fall back to regular export
            prepared_data = self._prepare_json_data(data)
            json_data = json.loads(prepared_data.to_json(orient='records'))
        
        # Write to file
        json_dump_options = {**self.json_dump_options}
        for key, value in kwargs.items():
            if key in ['indent', 'ensure_ascii', 'sort_keys', 'separators']:
                json_dump_options[key] = value
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, **json_dump_options, cls=CustomJSONEncoder)
        
        return json_path
    
    def export_streaming(self, data: pd.DataFrame, filename: str, 
                        chunk_size: int = 10000, **kwargs) -> str:
        """
        Export large datasets in streaming fashion to manage memory
        
        Args:
            data: DataFrame to export
            filename: Output filename
            chunk_size: Number of records per chunk
            **kwargs: JSON export options
            
        Returns:
            str: Path to exported JSON file
        """
        json_path = self._validate_filename(filename, '.json')
        self._ensure_directory_exists(json_path)
        
        # Prepare options
        pandas_options = {'orient': 'records'}
        json_dump_options = {**self.json_dump_options}
        
        for key, value in kwargs.items():
            if key in ['indent', 'ensure_ascii', 'sort_keys', 'separators']:
                json_dump_options[key] = value
        
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write('[\n')
            
            total_rows = len(data)
            for i in range(0, total_rows, chunk_size):
                chunk = data.iloc[i:i + chunk_size]
                prepared_chunk = self._prepare_json_data(chunk)
                
                # Convert chunk to records
                chunk_records = json.loads(prepared_chunk.to_json(**pandas_options))
                
                # Write each record
                for j, record in enumerate(chunk_records):
                    json.dump(record, f, cls=CustomJSONEncoder, separators=json_dump_options.get('separators', (',', ':')))
                    
                    # Add comma if not the last record
                    if i + j + 1 < total_rows:
                        f.write(',')
                    f.write('\n')
            
            f.write(']')
        
        return json_path
    
    def get_export_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get information about what the JSON export will contain
        
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
            'estimated_json_size_mb': 0
        }
        
        # Analyze data types
        for col in data.columns:
            dtype = str(data[col].dtype)
            info['data_types'][col] = dtype
        
        # Estimate JSON file size (rough approximation)
        # JSON typically 2-3x larger than CSV due to key names and formatting
        avg_chars_per_cell = 15
        total_cells = len(data) * len(data.columns)
        estimated_size = total_cells * avg_chars_per_cell
        info['estimated_json_size_mb'] = round(estimated_size / (1024 * 1024), 2)
        
        return info


class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle special data types
    """
    
    def default(self, obj):
        """
        Handle special data types for JSON serialization
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON-serializable representation
        """
        # Handle pandas/numpy data types
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Handle datetime objects
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        
        # Handle Decimal objects
        elif isinstance(obj, Decimal):
            return float(obj)
        
        # Handle pandas Timestamp
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        
        # Handle NaN and infinity
        elif pd.isna(obj):
            return None
        elif obj == float('inf'):
            return "Infinity"
        elif obj == float('-inf'):
            return "-Infinity"
        
        # Fall back to default behavior
        return super().default(obj)