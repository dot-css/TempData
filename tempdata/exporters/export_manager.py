"""
Export manager for coordinating multiple format exports

Manages export operations across different formats with validation and error handling.
"""

import pandas as pd
from typing import List, Dict, Any
from .csv_exporter import CSVExporter
from .json_exporter import JSONExporter
from .parquet_exporter import ParquetExporter
from .excel_exporter import ExcelExporter
from .geojson_exporter import GeoJSONExporter


class ExportManager:
    """
    Manager for coordinating multiple format exports
    
    Handles export operations across different formats with format validation,
    error handling, and concurrent export support.
    """
    
    def __init__(self):
        """Initialize export manager with available exporters"""
        self.exporters = {
            'csv': CSVExporter(),
            'json': JSONExporter(),
            'parquet': ParquetExporter(),
            'excel': ExcelExporter(),
            'geojson': GeoJSONExporter()
        }
    
    def export(self, 
               data: pd.DataFrame, 
               formats: List[str], 
               filename: str, 
               **kwargs) -> List[str]:
        """
        Export data in specified formats
        
        Args:
            data: DataFrame to export
            formats: List of format names to export to
            filename: Base filename (without extension)
            **kwargs: Format-specific options
            
        Returns:
            List[str]: Paths to exported files
        """
        if data.empty:
            raise ValueError("Cannot export empty dataset")
        
        exported_files = []
        
        for format_type in formats:
            if format_type not in self.exporters:
                raise ValueError(f"Unsupported export format: {format_type}")
            
            try:
                exporter = self.exporters[format_type]
                exported_path = exporter.export(data, filename, **kwargs)
                exported_files.append(exported_path)
            except Exception as e:
                raise RuntimeError(f"Failed to export to {format_type}: {str(e)}")
        
        return exported_files
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported export formats
        
        Returns:
            List[str]: List of supported format names
        """
        return list(self.exporters.keys())