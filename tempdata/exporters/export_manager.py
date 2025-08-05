"""
Export manager for coordinating multiple format exports

Provides centralized management of data export operations across
multiple formats with validation, error handling, and performance optimization.
"""

import os
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path

from .base_exporter import BaseExporter
from .csv_exporter import CSVExporter
from .json_exporter import JSONExporter
from .parquet_exporter import ParquetExporter
from .excel_exporter import ExcelExporter
from .geojson_exporter import GeoJSONExporter


class ExportManager:
    """
    Manager class for coordinating multiple format exports
    
    Handles format validation, concurrent exports, error handling,
    and provides a unified interface for all export operations.
    """
    
    def __init__(self):
        """Initialize export manager with all available exporters"""
        self.exporters = {
            'csv': CSVExporter(),
            'json': JSONExporter(),
            'parquet': ParquetExporter(),
            'excel': ExcelExporter(),
            'xlsx': ExcelExporter(),  # Alias for excel
            'geojson': GeoJSONExporter()
        }
        
        # Supported format extensions
        self.format_extensions = {
            'csv': '.csv',
            'json': '.json',
            'parquet': '.parquet',
            'excel': '.xlsx',
            'xlsx': '.xlsx',
            'geojson': '.geojson'
        }
        
        # Thread lock for concurrent operations
        self._lock = threading.Lock()
        
        # Export results tracking
        self._export_results = {}
    
    def export_single(self, data: pd.DataFrame, filename: str, 
                     format_type: str, **kwargs) -> str:
        """
        Export data to a single format
        
        Args:
            data: DataFrame to export
            filename: Output filename (without extension)
            format_type: Export format ('csv', 'json', 'parquet', 'excel', 'geojson')
            **kwargs: Format-specific export options
            
        Returns:
            str: Path to exported file
            
        Raises:
            ValueError: If format is not supported or data is invalid
            IOError: If export fails
        """
        # Validate format
        if format_type not in self.exporters:
            raise ValueError(f"Unsupported format: {format_type}. "
                           f"Supported formats: {list(self.exporters.keys())}")
        
        # Get appropriate exporter
        exporter = self.exporters[format_type]
        
        # Add appropriate extension if not present
        if not any(filename.endswith(ext) for ext in ['.csv', '.json', '.parquet', '.xlsx', '.xls', '.geojson']):
            filename = f"{filename}{self.format_extensions[format_type]}"
        
        # Export using the specific exporter
        return exporter.export(data, filename, **kwargs)
    
    def export_multiple(self, data: pd.DataFrame, base_filename: str, 
                       formats: List[str], concurrent: bool = True, 
                       **kwargs) -> Dict[str, str]:
        """
        Export data to multiple formats
        
        Args:
            data: DataFrame to export
            base_filename: Base filename (without extension)
            formats: List of formats to export to
            concurrent: Whether to use concurrent export (default: True)
            **kwargs: Export options (applied to all formats)
            
        Returns:
            Dict mapping format names to exported file paths
            
        Raises:
            ValueError: If any format is not supported
            IOError: If any export fails
        """
        # Validate all formats first
        invalid_formats = [f for f in formats if f not in self.exporters]
        if invalid_formats:
            raise ValueError(f"Unsupported formats: {invalid_formats}. "
                           f"Supported formats: {list(self.exporters.keys())}")
        
        # Remove duplicates while preserving order
        formats = list(dict.fromkeys(formats))
        
        if concurrent and len(formats) > 1:
            return self._export_concurrent(data, base_filename, formats, **kwargs)
        else:
            return self._export_sequential(data, base_filename, formats, **kwargs)
    
    def _export_sequential(self, data: pd.DataFrame, base_filename: str, 
                          formats: List[str], **kwargs) -> Dict[str, str]:
        """
        Export data to multiple formats sequentially
        
        Args:
            data: DataFrame to export
            base_filename: Base filename
            formats: List of formats
            **kwargs: Export options
            
        Returns:
            Dict mapping format names to exported file paths
        """
        results = {}
        errors = {}
        
        for format_type in formats:
            try:
                # Create format-specific filename
                filename = f"{base_filename}{self.format_extensions[format_type]}"
                
                # Get format-specific options
                format_kwargs = self._get_format_options(format_type, kwargs)
                
                # Export
                result_path = self.exporters[format_type].export(data, filename, **format_kwargs)
                results[format_type] = result_path
                
            except Exception as e:
                errors[format_type] = str(e)
        
        # If there were errors, include them in the exception
        if errors:
            error_msg = "Export errors occurred:\n" + "\n".join([f"{fmt}: {err}" for fmt, err in errors.items()])
            if not results:  # All exports failed
                raise IOError(error_msg)
            else:  # Some exports succeeded
                # Log errors but return successful results
                print(f"Warning: {error_msg}")
        
        return results
    
    def _export_concurrent(self, data: pd.DataFrame, base_filename: str, 
                          formats: List[str], **kwargs) -> Dict[str, str]:
        """
        Export data to multiple formats concurrently
        
        Args:
            data: DataFrame to export
            base_filename: Base filename
            formats: List of formats
            **kwargs: Export options
            
        Returns:
            Dict mapping format names to exported file paths
        """
        results = {}
        errors = {}
        
        # Use ThreadPoolExecutor for concurrent exports
        max_workers = min(len(formats), 4)  # Limit concurrent threads
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all export tasks
            future_to_format = {}
            
            for format_type in formats:
                # Create format-specific filename
                filename = f"{base_filename}{self.format_extensions[format_type]}"
                
                # Get format-specific options
                format_kwargs = self._get_format_options(format_type, kwargs)
                
                # Submit export task
                future = executor.submit(
                    self._export_single_thread_safe,
                    data.copy(),  # Pass a copy to avoid threading issues
                    filename,
                    format_type,
                    format_kwargs
                )
                future_to_format[future] = format_type
            
            # Collect results as they complete
            for future in as_completed(future_to_format):
                format_type = future_to_format[future]
                try:
                    result_path = future.result()
                    results[format_type] = result_path
                except Exception as e:
                    errors[format_type] = str(e)
        
        # Handle errors
        if errors:
            error_msg = "Export errors occurred:\n" + "\n".join([f"{fmt}: {err}" for fmt, err in errors.items()])
            if not results:  # All exports failed
                raise IOError(error_msg)
            else:  # Some exports succeeded
                print(f"Warning: {error_msg}")
        
        return results
    
    def _export_single_thread_safe(self, data: pd.DataFrame, filename: str, 
                                  format_type: str, format_kwargs: Dict[str, Any]) -> str:
        """
        Thread-safe single export operation
        
        Args:
            data: DataFrame to export
            filename: Output filename
            format_type: Export format
            format_kwargs: Format-specific options
            
        Returns:
            str: Path to exported file
        """
        with self._lock:
            exporter = self.exporters[format_type]
        
        return exporter.export(data, filename, **format_kwargs)
    
    def _get_format_options(self, format_type: str, global_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract format-specific options from global kwargs
        
        Args:
            format_type: Export format
            global_kwargs: Global export options
            
        Returns:
            Dict with format-specific options
        """
        # Start with global options
        format_kwargs = global_kwargs.copy()
        
        # Remove format-specific prefixes and apply to appropriate formats
        format_specific_options = {}
        
        for key, value in global_kwargs.items():
            if key.startswith(f"{format_type}_"):
                # Remove format prefix
                clean_key = key[len(format_type) + 1:]
                format_specific_options[clean_key] = value
        
        # Update with format-specific options
        format_kwargs.update(format_specific_options)
        
        # Remove all format-specific options (including the current format's prefixed options)
        keys_to_remove = []
        for key in format_kwargs.keys():
            if '_' in key:
                prefix = key.split('_')[0]
                if prefix in self.exporters:
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            format_kwargs.pop(key, None)
        
        return format_kwargs
    
    def validate_formats(self, formats: List[str]) -> Dict[str, bool]:
        """
        Validate a list of export formats
        
        Args:
            formats: List of format names to validate
            
        Returns:
            Dict mapping format names to validation status
        """
        return {fmt: fmt in self.exporters for fmt in formats}
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of all supported export formats
        
        Returns:
            List of supported format names
        """
        return list(self.exporters.keys())
    
    def get_format_info(self, format_type: str) -> Dict[str, Any]:
        """
        Get information about a specific format
        
        Args:
            format_type: Format name
            
        Returns:
            Dict with format information
            
        Raises:
            ValueError: If format is not supported
        """
        if format_type not in self.exporters:
            raise ValueError(f"Unsupported format: {format_type}")
        
        exporter = self.exporters[format_type]
        
        return {
            'format': format_type,
            'extension': self.format_extensions[format_type],
            'exporter_class': exporter.__class__.__name__,
            'supported_extensions': getattr(exporter, 'supported_extensions', []),
            'description': exporter.__doc__.split('\n')[0] if exporter.__doc__ else 'No description available'
        }
    
    def get_export_info(self, data: pd.DataFrame, formats: List[str]) -> Dict[str, Any]:
        """
        Get information about what exports will contain
        
        Args:
            data: DataFrame to analyze
            formats: List of formats to get info for
            
        Returns:
            Dict with export information for each format
        """
        info = {
            'data_summary': {
                'rows': len(data),
                'columns': len(data.columns),
                'memory_usage_mb': round(data.memory_usage(deep=True).sum() / (1024 * 1024), 2)
            },
            'formats': {}
        }
        
        for format_type in formats:
            if format_type in self.exporters:
                exporter = self.exporters[format_type]
                
                # Get format-specific export info if available
                if hasattr(exporter, 'get_export_info'):
                    try:
                        format_info = exporter.get_export_info(data)
                        info['formats'][format_type] = format_info
                    except Exception as e:
                        info['formats'][format_type] = {'error': str(e)}
                else:
                    info['formats'][format_type] = {'info': 'No detailed info available'}
            else:
                info['formats'][format_type] = {'error': 'Unsupported format'}
        
        return info
    
    def cleanup_failed_exports(self, base_filename: str, formats: List[str]) -> None:
        """
        Clean up any partial files from failed exports
        
        Args:
            base_filename: Base filename used for exports
            formats: List of formats that were attempted
        """
        for format_type in formats:
            if format_type in self.format_extensions:
                filename = f"{base_filename}{self.format_extensions[format_type]}"
                if os.path.exists(filename):
                    try:
                        os.remove(filename)
                    except OSError:
                        pass  # Ignore cleanup errors
    
    def export_with_validation(self, data: pd.DataFrame, base_filename: str, 
                              formats: List[str], validate_data: bool = True, 
                              **kwargs) -> Dict[str, Union[str, Exception]]:
        """
        Export with comprehensive validation and error handling
        
        Args:
            data: DataFrame to export
            base_filename: Base filename
            formats: List of formats to export to
            validate_data: Whether to validate data before export
            **kwargs: Export options
            
        Returns:
            Dict mapping format names to either file paths (success) or exceptions (failure)
        """
        results = {}
        
        # Validate data if requested
        if validate_data:
            try:
                self._validate_export_data(data)
            except ValueError as e:
                # Return error for all formats
                return {fmt: e for fmt in formats}
        
        # Validate formats
        format_validation = self.validate_formats(formats)
        invalid_formats = [fmt for fmt, valid in format_validation.items() if not valid]
        
        if invalid_formats:
            error = ValueError(f"Unsupported formats: {invalid_formats}")
            for fmt in invalid_formats:
                results[fmt] = error
            # Remove invalid formats from processing
            formats = [fmt for fmt in formats if fmt not in invalid_formats]
        
        # Export valid formats
        if formats:
            try:
                export_results = self.export_multiple(data, base_filename, formats, **kwargs)
                results.update(export_results)
            except Exception as e:
                # If export_multiple fails completely, mark all remaining formats as failed
                for fmt in formats:
                    if fmt not in results:
                        results[fmt] = e
        
        return results
    
    def _validate_export_data(self, data: pd.DataFrame) -> None:
        """
        Validate data before export
        
        Args:
            data: DataFrame to validate
            
        Raises:
            ValueError: If data is invalid for export
        """
        if data is None:
            raise ValueError("Data cannot be None")
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        
        if len(data) == 0:
            raise ValueError("Data must contain at least one row")
        
        # Check for extremely large datasets that might cause issues
        memory_usage_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
        if memory_usage_mb > 1000:  # 1GB threshold
            print(f"Warning: Large dataset detected ({memory_usage_mb:.1f} MB). "
                  "Consider using streaming export for better performance.")