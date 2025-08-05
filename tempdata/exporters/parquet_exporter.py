"""
Parquet export functionality

Exports data to Parquet format with compression and data type optimization.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
from typing import Optional, Union, Dict, Any, List
from .base_exporter import BaseExporter


class ParquetExporter(BaseExporter):
    """
    Exporter for Parquet format
    
    Handles Parquet export with compression and data type optimization
    for efficient storage and fast analytics queries.
    """
    
    def __init__(self):
        """Initialize Parquet exporter"""
        super().__init__()
        self.supported_extensions = ['.parquet']
        
        # Default Parquet export options
        self.default_options = {
            'compression': 'snappy',  # snappy, gzip, brotli, lz4, zstd
            'index': False,
            'engine': 'pyarrow',
            'use_deprecated_int96_timestamps': False,
            'coerce_timestamps': 'ms',  # s, ms, us, ns
            'allow_truncated_timestamps': False
        }
        
        # Compression options with their characteristics
        self.compression_info = {
            'snappy': {'speed': 'fast', 'ratio': 'medium', 'cpu': 'low'},
            'gzip': {'speed': 'slow', 'ratio': 'high', 'cpu': 'high'},
            'brotli': {'speed': 'slow', 'ratio': 'very_high', 'cpu': 'high'},
            'lz4': {'speed': 'very_fast', 'ratio': 'low', 'cpu': 'very_low'},
            'zstd': {'speed': 'medium', 'ratio': 'high', 'cpu': 'medium'},
            'uncompressed': {'speed': 'very_fast', 'ratio': 'none', 'cpu': 'none'}
        }
    
    def export(self, data: pd.DataFrame, filename: str, **kwargs) -> str:
        """
        Export data to Parquet format with compression and optimization
        
        Args:
            data: DataFrame to export
            filename: Output filename
            **kwargs: Parquet-specific options that override defaults
                - compression: Compression algorithm ('snappy', 'gzip', 'brotli', 'lz4', 'zstd', 'uncompressed')
                - index: Include row index (bool)
                - engine: Parquet engine ('pyarrow' or 'fastparquet')
                - row_group_size: Number of rows per row group (int)
                - use_dictionary: Use dictionary encoding for string columns (bool)
                - write_statistics: Write column statistics (bool)
                - coerce_timestamps: Timestamp precision ('s', 'ms', 'us', 'ns')
                - optimize_types: Optimize data types for storage (bool)
                
        Returns:
            str: Path to exported Parquet file
            
        Raises:
            ValueError: If data is invalid or filename is invalid
            IOError: If file cannot be written
            ImportError: If required Parquet libraries are not available
        """
        # Check for required dependencies
        self._check_dependencies()
        
        # Validate input data
        self._validate_data(data)
        
        # Validate and normalize filename
        parquet_path = self._validate_filename(filename, '.parquet')
        
        # Ensure output directory exists
        self._ensure_directory_exists(parquet_path)
        
        # Merge default options with user-provided options
        export_options = {**self.default_options, **kwargs}
        
        # Extract custom options not used by pandas
        optimize_types = export_options.pop('optimize_types', True)
        row_group_size = export_options.pop('row_group_size', None)
        use_dictionary = export_options.pop('use_dictionary', True)
        write_statistics = export_options.pop('write_statistics', True)
        
        try:
            # Prepare data for Parquet export
            export_data = self._prepare_parquet_data(data, optimize_types)
            
            # Use PyArrow for more control over Parquet features
            if export_options.get('engine') == 'pyarrow' or export_options.get('engine') is None:
                return self._export_with_pyarrow(
                    export_data, parquet_path, export_options,
                    row_group_size, use_dictionary, write_statistics
                )
            else:
                # Fall back to pandas default
                export_data.to_parquet(parquet_path, **export_options)
                
                # Verify the file was created successfully
                if not os.path.exists(parquet_path):
                    raise IOError(f"Failed to create Parquet file: {parquet_path}")
                    
                return parquet_path
            
        except Exception as e:
            # Clean up partial file if it exists
            if os.path.exists(parquet_path):
                try:
                    os.remove(parquet_path)
                except OSError:
                    pass
            raise IOError(f"Failed to export Parquet file: {str(e)}")
    
    def _check_dependencies(self):
        """
        Check if required Parquet dependencies are available
        
        Raises:
            ImportError: If required libraries are not available
        """
        try:
            import pyarrow
            import pyarrow.parquet
        except ImportError:
            raise ImportError(
                "PyArrow is required for Parquet export. "
                "Install it with: pip install pyarrow"
            )
    
    def _prepare_parquet_data(self, data: pd.DataFrame, optimize_types: bool = True) -> pd.DataFrame:
        """
        Prepare data for Parquet export with type optimization
        
        Args:
            data: Original DataFrame
            optimize_types: Whether to optimize data types for storage
            
        Returns:
            pd.DataFrame: Prepared DataFrame for Parquet export
        """
        # Create a copy to avoid modifying original data
        parquet_data = data.copy()
        
        if optimize_types:
            parquet_data = self._optimize_data_types(parquet_data)
        
        # Handle different data types appropriately for Parquet
        for col in parquet_data.columns:
            col_data = parquet_data[col]
            
            # Handle datetime columns - Parquet handles these well natively
            if pd.api.types.is_datetime64_any_dtype(col_data):
                # Ensure timezone-naive datetimes for better compatibility
                if hasattr(col_data.dtype, 'tz') and col_data.dtype.tz is not None:
                    parquet_data[col] = col_data.dt.tz_convert('UTC').dt.tz_localize(None)
            
            # Handle boolean columns - keep as boolean
            elif pd.api.types.is_bool_dtype(col_data):
                pass
            
            # Handle categorical columns - Parquet supports these efficiently
            elif isinstance(col_data.dtype, pd.CategoricalDtype):
                pass
            
            # Handle object/string columns
            elif col_data.dtype == 'object':
                # Convert to category if it has low cardinality (good for compression)
                unique_ratio = col_data.nunique() / len(col_data) if len(col_data) > 0 else 1
                if unique_ratio < 0.5 and col_data.nunique() < 1000:
                    parquet_data[col] = col_data.astype('category')
        
        return parquet_data
    
    def _export_with_pyarrow(self, data: pd.DataFrame, parquet_path: str, 
                           export_options: Dict[str, Any], row_group_size: Optional[int],
                           use_dictionary: bool, write_statistics: bool) -> str:
        """
        Export using PyArrow for advanced Parquet features
        
        Args:
            data: DataFrame to export
            parquet_path: Output file path
            export_options: Export options
            row_group_size: Rows per row group
            use_dictionary: Use dictionary encoding
            write_statistics: Write column statistics
            
        Returns:
            str: Path to exported file
        """
        # Convert DataFrame to PyArrow Table
        table = pa.Table.from_pandas(data, preserve_index=export_options.get('index', False))
        
        # Configure compression
        compression = export_options.get('compression', 'snappy')
        if compression == 'uncompressed':
            compression = None
        
        # Set up Parquet writer options
        writer_options = {
            'compression': compression,
            'use_dictionary': use_dictionary,
            'write_statistics': write_statistics,
            'use_deprecated_int96_timestamps': export_options.get('use_deprecated_int96_timestamps', False),
            'coerce_timestamps': export_options.get('coerce_timestamps', 'ms'),
            'allow_truncated_timestamps': export_options.get('allow_truncated_timestamps', False)
        }
        
        if row_group_size:
            writer_options['row_group_size'] = row_group_size
        
        # Write the Parquet file
        pq.write_table(table, parquet_path, **writer_options)
        
        # Verify the file was created successfully
        if not os.path.exists(parquet_path):
            raise IOError(f"Failed to create Parquet file: {parquet_path}")
        
        return parquet_path
    
    def export_partitioned(self, data: pd.DataFrame, base_path: str, 
                          partition_cols: List[str], **kwargs) -> str:
        """
        Export data as partitioned Parquet dataset
        
        Args:
            data: DataFrame to export
            base_path: Base directory for partitioned dataset
            partition_cols: Columns to partition by
            **kwargs: Parquet export options
            
        Returns:
            str: Path to partitioned dataset directory
        """
        self._check_dependencies()
        self._validate_data(data)
        
        # Validate partition columns
        for col in partition_cols:
            if col not in data.columns:
                raise ValueError(f"Partition column '{col}' not found in data")
        
        # Ensure base directory exists
        os.makedirs(base_path, exist_ok=True)
        
        # Prepare data
        export_options = {**self.default_options, **kwargs}
        optimize_types = export_options.pop('optimize_types', True)
        export_data = self._prepare_parquet_data(data, optimize_types)
        
        # Convert to PyArrow Table
        table = pa.Table.from_pandas(export_data, preserve_index=export_options.get('index', False))
        
        # Configure compression
        compression = export_options.get('compression', 'snappy')
        if compression == 'uncompressed':
            compression = None
        
        # Write partitioned dataset
        pq.write_to_dataset(
            table,
            root_path=base_path,
            partition_cols=partition_cols,
            compression=compression,
            use_dictionary=export_options.get('use_dictionary', True),
            write_statistics=export_options.get('write_statistics', True)
        )
        
        return base_path
    
    def export_with_schema(self, data: pd.DataFrame, filename: str, 
                          schema: pa.Schema, **kwargs) -> str:
        """
        Export with explicit PyArrow schema for precise type control
        
        Args:
            data: DataFrame to export
            filename: Output filename
            schema: PyArrow schema to enforce
            **kwargs: Parquet export options
            
        Returns:
            str: Path to exported file
        """
        self._check_dependencies()
        self._validate_data(data)
        
        parquet_path = self._validate_filename(filename, '.parquet')
        self._ensure_directory_exists(parquet_path)
        
        # Convert DataFrame to PyArrow Table with explicit schema
        table = pa.Table.from_pandas(data, schema=schema, preserve_index=kwargs.get('index', False))
        
        # Export options
        export_options = {**self.default_options, **kwargs}
        compression = export_options.get('compression', 'snappy')
        if compression == 'uncompressed':
            compression = None
        
        # Write with schema
        pq.write_table(
            table, 
            parquet_path,
            compression=compression,
            use_dictionary=export_options.get('use_dictionary', True),
            write_statistics=export_options.get('write_statistics', True)
        )
        
        return parquet_path
    
    def get_compression_recommendation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Recommend compression algorithm based on data characteristics
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dict with compression recommendations
        """
        data_size_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
        num_string_cols = sum(1 for col in data.columns if data[col].dtype == 'object')
        num_numeric_cols = sum(1 for col in data.columns if pd.api.types.is_numeric_dtype(data[col]))
        
        # Analyze data characteristics
        has_many_strings = num_string_cols > len(data.columns) * 0.3
        is_large_dataset = data_size_mb > 100
        has_repetitive_data = any(
            data[col].nunique() / len(data) < 0.1 
            for col in data.columns 
            if len(data) > 0
        )
        
        # Make recommendations
        if is_large_dataset and has_repetitive_data:
            recommendation = 'brotli'
            reason = 'Large dataset with repetitive data benefits from high compression'
        elif has_many_strings and has_repetitive_data:
            recommendation = 'zstd'
            reason = 'String-heavy data with patterns compresses well with zstd'
        elif is_large_dataset:
            recommendation = 'gzip'
            reason = 'Large dataset benefits from good compression ratio'
        elif data_size_mb < 10:
            recommendation = 'lz4'
            reason = 'Small dataset prioritizes speed over compression'
        else:
            recommendation = 'snappy'
            reason = 'Balanced choice for general use cases'
        
        return {
            'recommended_compression': recommendation,
            'reason': reason,
            'data_size_mb': round(data_size_mb, 2),
            'string_columns': num_string_cols,
            'numeric_columns': num_numeric_cols,
            'compression_options': self.compression_info
        }
    
    def get_export_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get information about what the Parquet export will contain
        
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
            'estimated_parquet_size_mb': 0,
            'compression_recommendation': self.get_compression_recommendation(data)
        }
        
        # Analyze data types and their Parquet compatibility
        for col in data.columns:
            dtype = str(data[col].dtype)
            parquet_type = self._get_parquet_type(data[col])
            info['data_types'][col] = {
                'pandas_dtype': dtype,
                'parquet_type': parquet_type,
                'nullable': data[col].isna().any()
            }
        
        # Estimate Parquet file size (rough approximation)
        # Parquet is typically 50-80% smaller than CSV due to compression and columnar format
        base_size = data.memory_usage(deep=True).sum()
        compression_ratio = 0.3  # Assume 70% compression
        info['estimated_parquet_size_mb'] = round((base_size * compression_ratio) / (1024 * 1024), 2)
        
        return info
    
    def _get_parquet_type(self, series: pd.Series) -> str:
        """
        Determine the Parquet type for a pandas Series
        
        Args:
            series: Pandas Series to analyze
            
        Returns:
            str: Parquet type description
        """
        if pd.api.types.is_datetime64_any_dtype(series):
            return 'timestamp'
        elif pd.api.types.is_bool_dtype(series):
            return 'boolean'
        elif pd.api.types.is_integer_dtype(series):
            return f'int{series.dtype.itemsize * 8}'
        elif pd.api.types.is_float_dtype(series):
            return f'float{series.dtype.itemsize * 8}'
        elif isinstance(series.dtype, pd.CategoricalDtype):
            return 'category'
        elif series.dtype == 'object':
            return 'string'
        else:
            return 'unknown'