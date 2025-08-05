"""
Unit tests for CSV export functionality and data integrity
"""

import os
import pandas as pd
import pytest
import tempfile
import shutil
from datetime import datetime, date
from decimal import Decimal
from pathlib import Path

from tempdata.exporters.csv_exporter import CSVExporter
from tempdata.exporters.base_exporter import BaseExporter


class TestCSVExporter:
    """Test suite for CSV exporter functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.exporter = CSVExporter()
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_csv_exporter_inheritance(self):
        """Test that CSVExporter properly inherits from BaseExporter"""
        assert isinstance(self.exporter, BaseExporter)
        assert hasattr(self.exporter, 'export')
        assert hasattr(self.exporter, '_validate_data')
    
    def test_csv_exporter_initialization(self):
        """Test CSV exporter initialization"""
        assert self.exporter.supported_extensions == ['.csv']
        assert 'index' in self.exporter.default_options
        assert self.exporter.default_options['index'] is False
        assert self.exporter.default_options['encoding'] == 'utf-8'
        assert self.exporter.default_options['sep'] == ','
    
    def test_basic_csv_export(self):
        """Test basic CSV export functionality"""
        # Create test data
        data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'salary': [50000.0, 60000.0, 70000.0]
        })
        
        filename = os.path.join(self.temp_dir, 'test_basic.csv')
        result_path = self.exporter.export(data, filename)
        
        # Verify file was created
        assert os.path.exists(result_path)
        assert result_path.endswith('.csv')
        
        # Verify content
        exported_data = pd.read_csv(result_path)
        pd.testing.assert_frame_equal(data, exported_data)
    
    def test_csv_export_with_extension_auto_add(self):
        """Test that .csv extension is automatically added"""
        data = pd.DataFrame({'col1': [1, 2, 3]})
        filename = os.path.join(self.temp_dir, 'test_no_extension')
        
        result_path = self.exporter.export(data, filename)
        
        assert result_path.endswith('.csv')
        assert os.path.exists(result_path)
    
    def test_csv_export_with_datetime_data(self):
        """Test CSV export with datetime columns"""
        data = pd.DataFrame({
            'id': [1, 2, 3],
            'created_at': [
                datetime(2023, 1, 1, 12, 0, 0),
                datetime(2023, 1, 2, 13, 30, 0),
                datetime(2023, 1, 3, 14, 45, 0)
            ],
            'date_only': [
                date(2023, 1, 1),
                date(2023, 1, 2),
                date(2023, 1, 3)
            ]
        })
        
        filename = os.path.join(self.temp_dir, 'test_datetime.csv')
        result_path = self.exporter.export(data, filename)
        
        # Verify file was created
        assert os.path.exists(result_path)
        
        # Read back and verify datetime formatting
        exported_data = pd.read_csv(result_path)
        assert 'created_at' in exported_data.columns
        assert exported_data['created_at'].iloc[0] == '2023-01-01 12:00:00'
    
    def test_csv_export_with_boolean_data(self):
        """Test CSV export with boolean columns"""
        data = pd.DataFrame({
            'id': [1, 2, 3],
            'is_active': [True, False, True],
            'has_premium': [False, True, False]
        })
        
        filename = os.path.join(self.temp_dir, 'test_boolean.csv')
        result_path = self.exporter.export(data, filename)
        
        # Verify file was created
        assert os.path.exists(result_path)
        
        # Read back and verify boolean formatting
        exported_data = pd.read_csv(result_path)
        assert str(exported_data['is_active'].iloc[0]) == 'True'
        assert str(exported_data['is_active'].iloc[1]) == 'False'
    
    def test_csv_export_with_nan_values(self):
        """Test CSV export with NaN and None values"""
        data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', None, 'Charlie'],
            'age': [25, float('nan'), 35],
            'notes': ['Good', 'nan', None]
        })
        
        filename = os.path.join(self.temp_dir, 'test_nan.csv')
        result_path = self.exporter.export(data, filename)
        
        # Verify file was created
        assert os.path.exists(result_path)
        
        # Read back and verify NaN handling
        exported_data = pd.read_csv(result_path)
        assert len(exported_data) == 3
    
    def test_csv_export_with_special_characters(self):
        """Test CSV export with special characters and potential CSV injection"""
        data = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'formula': ['=SUM(A1:A10)', '+1+1', '-5', '@INDIRECT("A1")'],
            'normal': ['Alice', 'Bob', 'Charlie', 'David']
        })
        
        filename = os.path.join(self.temp_dir, 'test_special.csv')
        result_path = self.exporter.export(data, filename)
        
        # Verify file was created
        assert os.path.exists(result_path)
        
        # Read the raw file content to verify sanitization
        with open(result_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Verify that dangerous formulas are prefixed with single quote
        assert "'=SUM(A1:A10)" in content
        assert "'+1+1" in content
        assert "'-5" in content
        assert "'@INDIRECT" in content
    
    def test_csv_export_custom_options(self):
        """Test CSV export with custom options"""
        data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie']
        })
        
        filename = os.path.join(self.temp_dir, 'test_custom.csv')
        result_path = self.exporter.export(
            data, 
            filename,
            sep=';',
            encoding='utf-8',
            na_rep='NULL'
        )
        
        # Verify file was created
        assert os.path.exists(result_path)
        
        # Read raw content to verify separator
        with open(result_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert ';' in content  # Custom separator used
    
    def test_csv_export_with_metadata(self):
        """Test CSV export with metadata header"""
        data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie']
        })
        
        metadata = {
            'export_date': '2023-01-01',
            'source': 'test_data',
            'rows': len(data)
        }
        
        filename = os.path.join(self.temp_dir, 'test_metadata.csv')
        result_path = self.exporter.export_with_metadata(data, filename, metadata)
        
        # Verify file was created
        assert os.path.exists(result_path)
        
        # Read raw content to verify metadata
        with open(result_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert '# CSV Export Metadata' in content
        assert '# export_date: 2023-01-01' in content
        assert '# source: test_data' in content
    
    def test_csv_export_directory_creation(self):
        """Test that export creates necessary directories"""
        data = pd.DataFrame({'col1': [1, 2, 3]})
        
        # Use a nested directory that doesn't exist
        nested_dir = os.path.join(self.temp_dir, 'nested', 'deep', 'path')
        filename = os.path.join(nested_dir, 'test.csv')
        
        result_path = self.exporter.export(data, filename)
        
        # Verify directory was created and file exists
        assert os.path.exists(result_path)
        assert os.path.exists(nested_dir)
    
    def test_csv_export_info(self):
        """Test get_export_info method"""
        data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 40, 45],
            'salary': [50000.0, 60000.0, 70000.0, 80000.0, 90000.0]
        })
        
        info = self.exporter.get_export_info(data)
        
        assert info['rows'] == 5
        assert info['columns'] == 4
        assert 'id' in info['column_names']
        assert 'name' in info['column_names']
        assert 'data_types' in info
        assert 'memory_usage_mb' in info
        assert 'estimated_csv_size_mb' in info
    
    def test_csv_export_empty_data_validation(self):
        """Test validation with empty data"""
        empty_data = pd.DataFrame()
        filename = os.path.join(self.temp_dir, 'test_empty.csv')
        
        with pytest.raises(ValueError, match="Data cannot be empty"):
            self.exporter.export(empty_data, filename)
    
    def test_csv_export_none_data_validation(self):
        """Test validation with None data"""
        filename = os.path.join(self.temp_dir, 'test_none.csv')
        
        with pytest.raises(ValueError, match="Data cannot be None"):
            self.exporter.export(None, filename)
    
    def test_csv_export_invalid_filename_validation(self):
        """Test validation with invalid filename"""
        data = pd.DataFrame({'col1': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Filename must be a non-empty string"):
            self.exporter.export(data, "")
        
        with pytest.raises(ValueError, match="Filename must be a non-empty string"):
            self.exporter.export(data, None)
    
    def test_csv_export_file_cleanup_on_error(self):
        """Test that partial files are cleaned up on export errors"""
        data = pd.DataFrame({'col1': [1, 2, 3]})
        
        # Create a filename in a directory that will cause permission error
        if os.name == 'nt':  # Windows
            invalid_path = 'C:\\Windows\\System32\\test.csv'
        else:  # Unix-like
            invalid_path = '/root/test.csv'
        
        # This should raise an IOError and not leave partial files
        with pytest.raises(IOError):
            self.exporter.export(data, invalid_path)
    
    def test_csv_export_large_data_performance(self):
        """Test CSV export with larger dataset for performance"""
        # Create a larger dataset
        import numpy as np
        
        size = 10000
        data = pd.DataFrame({
            'id': range(size),
            'name': [f'User_{i}' for i in range(size)],
            'value': np.random.rand(size),
            'category': np.random.choice(['A', 'B', 'C'], size)
        })
        
        filename = os.path.join(self.temp_dir, 'test_large.csv')
        result_path = self.exporter.export(data, filename)
        
        # Verify file was created and has reasonable size
        assert os.path.exists(result_path)
        file_size = os.path.getsize(result_path)
        assert file_size > 0
        
        # Verify data integrity by reading back
        exported_data = pd.read_csv(result_path)
        assert len(exported_data) == size
        assert list(exported_data.columns) == ['id', 'name', 'value', 'category']
    
    def test_csv_sanitize_value_method(self):
        """Test the _sanitize_csv_value method directly"""
        # Test normal values
        assert self.exporter._sanitize_csv_value('normal_text') == 'normal_text'
        assert self.exporter._sanitize_csv_value('123') == '123'
        
        # Test dangerous values
        assert self.exporter._sanitize_csv_value('=SUM(A1:A10)') == "'=SUM(A1:A10)"
        assert self.exporter._sanitize_csv_value('+1+1') == "'+1+1"
        assert self.exporter._sanitize_csv_value('-5') == "'-5"
        assert self.exporter._sanitize_csv_value('@INDIRECT("A1")') == "'@INDIRECT(\"A1\")"
        
        # Test non-string values
        assert self.exporter._sanitize_csv_value(123) == '123'
        assert self.exporter._sanitize_csv_value(None) == 'None'
    
    def test_csv_prepare_data_method(self):
        """Test the _prepare_csv_data method"""
        # Create test data with various types
        data = pd.DataFrame({
            'datetime_col': [datetime(2023, 1, 1, 12, 0, 0)],
            'bool_col': [True],
            'numeric_col': [123.45],
            'string_col': ['test'],
            'formula_col': ['=SUM(A1:A10)']
        })
        
        prepared_data = self.exporter._prepare_csv_data(data)
        
        # Verify datetime conversion
        assert prepared_data['datetime_col'].iloc[0] == '2023-01-01 12:00:00'
        
        # Verify boolean conversion
        assert prepared_data['bool_col'].iloc[0] == 'True'
        
        # Verify numeric preservation
        assert prepared_data['numeric_col'].iloc[0] == 123.45
        
        # Verify formula sanitization
        assert prepared_data['formula_col'].iloc[0] == "'=SUM(A1:A10)"


class TestCSVExporterIntegration:
    """Integration tests for CSV exporter with real-world scenarios"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.exporter = CSVExporter()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_csv_export_sales_data_simulation(self):
        """Test CSV export with simulated sales data"""
        # Simulate realistic sales data
        sales_data = pd.DataFrame({
            'transaction_id': ['TXN_001', 'TXN_002', 'TXN_003'],
            'date': [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)],
            'customer_id': ['CUST_001', 'CUST_002', 'CUST_003'],
            'product': ['Widget A', 'Widget B', 'Widget C'],
            'amount': [99.99, 149.99, 199.99],
            'currency': ['USD', 'USD', 'USD'],
            'payment_method': ['Credit Card', 'PayPal', 'Bank Transfer'],
            'is_refunded': [False, False, True]
        })
        
        filename = os.path.join(self.temp_dir, 'sales_export.csv')
        result_path = self.exporter.export(sales_data, filename)
        
        # Verify export
        assert os.path.exists(result_path)
        
        # Read back and verify data integrity
        imported_data = pd.read_csv(result_path)
        assert len(imported_data) == 3
        assert 'transaction_id' in imported_data.columns
        assert imported_data['amount'].sum() == 449.97  # Verify numeric precision
    
    def test_csv_export_with_unicode_data(self):
        """Test CSV export with Unicode characters"""
        unicode_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['José', '北京', 'Müller'],
            'description': ['Café ☕', '数据库', 'Straße']
        })
        
        filename = os.path.join(self.temp_dir, 'unicode_test.csv')
        result_path = self.exporter.export(unicode_data, filename)
        
        # Verify export
        assert os.path.exists(result_path)
        
        # Read back with proper encoding
        imported_data = pd.read_csv(result_path, encoding='utf-8')
        assert imported_data['name'].iloc[0] == 'José'
        assert imported_data['name'].iloc[1] == '北京'
        assert imported_data['description'].iloc[0] == 'Café ☕'