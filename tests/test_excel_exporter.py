"""
Unit tests for Excel exporter functionality
"""

import pytest
import pandas as pd
import os
import tempfile
from datetime import datetime, date
from decimal import Decimal
from tempdata.exporters.excel_exporter import ExcelExporter


class TestExcelExporter:
    """Test cases for Excel exporter"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.exporter = ExcelExporter()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'age': [25, 30, 35, 28, 32],
            'salary': [50000.0, 60000.0, 70000.0, 55000.0, 65000.0],
            'is_active': [True, False, True, True, False],
            'hire_date': pd.to_datetime(['2020-01-15', '2019-03-20', '2018-07-10', '2021-02-28', '2020-11-05']),
            'department': ['Engineering', 'Sales', 'Marketing', 'Engineering', 'Sales']
        })
    
    def teardown_method(self):
        """Clean up test fixtures"""
        # Clean up temporary files
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_basic_excel_export(self):
        """Test basic Excel export functionality"""
        filename = os.path.join(self.temp_dir, 'test_basic.xlsx')
        
        result_path = self.exporter.export(self.sample_data, filename)
        
        assert os.path.exists(result_path)
        assert result_path.endswith('.xlsx')
        
        # Verify file can be read back
        read_data = pd.read_excel(result_path)
        assert len(read_data) == len(self.sample_data)
        assert list(read_data.columns) == list(self.sample_data.columns)
    
    def test_excel_export_with_styling(self):
        """Test Excel export with styling options"""
        filename = os.path.join(self.temp_dir, 'test_styled.xlsx')
        
        result_path = self.exporter.export(
            self.sample_data, 
            filename,
            apply_styling=True,
            auto_adjust_width=True,
            freeze_panes=(1, 0),
            add_filters=True
        )
        
        assert os.path.exists(result_path)
        
        # Verify file can be read back
        read_data = pd.read_excel(result_path)
        assert len(read_data) == len(self.sample_data)
    
    def test_excel_export_custom_sheet_name(self):
        """Test Excel export with custom sheet name"""
        filename = os.path.join(self.temp_dir, 'test_sheet.xlsx')
        
        result_path = self.exporter.export(
            self.sample_data, 
            filename,
            sheet_name='CustomSheet'
        )
        
        assert os.path.exists(result_path)
        
        # Verify sheet name
        excel_file = pd.ExcelFile(result_path)
        assert 'CustomSheet' in excel_file.sheet_names
    
    def test_excel_export_data_types(self):
        """Test Excel export handles different data types correctly"""
        # Create data with various types
        test_data = pd.DataFrame({
            'integers': [1, 2, 3],
            'floats': [1.1, 2.2, 3.3],
            'strings': ['a', 'b', 'c'],
            'booleans': [True, False, True],
            'dates': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'nulls': [None, 'value', None]
        })
        
        filename = os.path.join(self.temp_dir, 'test_types.xlsx')
        result_path = self.exporter.export(test_data, filename)
        
        assert os.path.exists(result_path)
        
        # Verify data can be read back
        read_data = pd.read_excel(result_path)
        assert len(read_data) == len(test_data)
    
    def test_excel_export_large_strings(self):
        """Test Excel export handles long strings correctly"""
        # Create data with very long strings
        long_string = 'x' * 40000  # Longer than Excel's 32,767 limit
        test_data = pd.DataFrame({
            'id': [1, 2],
            'long_text': [long_string, 'short']
        })
        
        filename = os.path.join(self.temp_dir, 'test_long_strings.xlsx')
        result_path = self.exporter.export(test_data, filename)
        
        assert os.path.exists(result_path)
        
        # Verify data was truncated appropriately
        read_data = pd.read_excel(result_path)
        assert len(read_data.iloc[0]['long_text']) <= 32767
    
    def test_excel_export_multiple_sheets(self):
        """Test Excel export with multiple sheets"""
        data_dict = {
            'Sheet1': self.sample_data,
            'Sheet2': self.sample_data.head(3),
            'Summary': pd.DataFrame({'total_rows': [len(self.sample_data)]})
        }
        
        filename = os.path.join(self.temp_dir, 'test_multi_sheet.xlsx')
        result_path = self.exporter.export_multiple_sheets(data_dict, filename)
        
        assert os.path.exists(result_path)
        
        # Verify all sheets exist
        excel_file = pd.ExcelFile(result_path)
        assert set(excel_file.sheet_names) == set(data_dict.keys())
        
        # Verify sheet contents
        for sheet_name, expected_data in data_dict.items():
            sheet_data = pd.read_excel(result_path, sheet_name=sheet_name)
            assert len(sheet_data) == len(expected_data)
    
    def test_excel_export_filename_validation(self):
        """Test filename validation and extension handling"""
        # Test without extension
        filename_no_ext = os.path.join(self.temp_dir, 'test_no_ext')
        result_path = self.exporter.export(self.sample_data, filename_no_ext)
        assert result_path.endswith('.xlsx')
        
        # Test with .xls extension
        filename_xls = os.path.join(self.temp_dir, 'test.xls')
        result_path = self.exporter.export(self.sample_data, filename_xls)
        assert result_path.endswith('.xls')
    
    def test_excel_export_empty_data_error(self):
        """Test Excel export with empty data raises error"""
        empty_data = pd.DataFrame()
        filename = os.path.join(self.temp_dir, 'test_empty.xlsx')
        
        with pytest.raises(ValueError, match="Data cannot be empty"):
            self.exporter.export(empty_data, filename)
    
    def test_excel_export_invalid_data_error(self):
        """Test Excel export with invalid data raises error"""
        filename = os.path.join(self.temp_dir, 'test_invalid.xlsx')
        
        with pytest.raises(ValueError, match="Data must be a pandas DataFrame"):
            self.exporter.export("not a dataframe", filename)
        
        with pytest.raises(ValueError, match="Data cannot be None"):
            self.exporter.export(None, filename)
    
    def test_excel_export_invalid_filename_error(self):
        """Test Excel export with invalid filename raises error"""
        with pytest.raises(ValueError, match="Filename must be a non-empty string"):
            self.exporter.export(self.sample_data, "")
        
        with pytest.raises(ValueError, match="Filename must be a non-empty string"):
            self.exporter.export(self.sample_data, None)
    
    def test_excel_export_directory_creation(self):
        """Test Excel export creates directories if they don't exist"""
        nested_dir = os.path.join(self.temp_dir, 'nested', 'directory')
        filename = os.path.join(nested_dir, 'test.xlsx')
        
        result_path = self.exporter.export(self.sample_data, filename)
        
        assert os.path.exists(result_path)
        assert os.path.exists(nested_dir)
    
    def test_excel_export_custom_options(self):
        """Test Excel export with custom options"""
        filename = os.path.join(self.temp_dir, 'test_custom.xlsx')
        
        result_path = self.exporter.export(
            self.sample_data,
            filename,
            index=True,  # Include index
            header=True,
            na_rep='N/A'  # Custom NaN representation
        )
        
        assert os.path.exists(result_path)
        
        # Verify options were applied
        read_data = pd.read_excel(result_path, index_col=0)
        assert len(read_data) == len(self.sample_data)
    
    def test_get_export_info(self):
        """Test getting export information"""
        info = self.exporter.get_export_info(self.sample_data)
        
        assert info['rows'] == len(self.sample_data)
        assert info['columns'] == len(self.sample_data.columns)
        assert info['column_names'] == list(self.sample_data.columns)
        assert 'data_types' in info
        assert 'memory_usage_mb' in info
        assert 'estimated_excel_size_mb' in info
        assert 'excel_limitations' in info
    
    def test_get_export_info_large_data_warning(self):
        """Test export info warns about Excel limitations"""
        # Create data that exceeds Excel row limit
        large_data = pd.DataFrame({
            'col1': range(2000000)  # More than Excel's 1,048,576 row limit
        })
        
        info = self.exporter.get_export_info(large_data)
        
        assert 'warnings' in info
        assert any('exceeding Excel\'s limit' in warning for warning in info['warnings'])
    
    def test_excel_export_timezone_handling(self):
        """Test Excel export handles timezone-aware datetimes"""
        # Create data with timezone-aware datetimes
        tz_data = pd.DataFrame({
            'id': [1, 2, 3],
            'timestamp': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-02 11:00:00', '2023-01-03 12:00:00']).tz_localize('UTC')
        })
        
        filename = os.path.join(self.temp_dir, 'test_timezone.xlsx')
        result_path = self.exporter.export(tz_data, filename)
        
        assert os.path.exists(result_path)
        
        # Verify data can be read back (timezone info will be lost in Excel)
        read_data = pd.read_excel(result_path)
        assert len(read_data) == len(tz_data)
    
    def test_excel_export_with_nan_values(self):
        """Test Excel export handles NaN values correctly"""
        nan_data = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [1.0, float('nan'), 3.0],
            'text': ['a', None, 'c']
        })
        
        filename = os.path.join(self.temp_dir, 'test_nan.xlsx')
        result_path = self.exporter.export(nan_data, filename)
        
        assert os.path.exists(result_path)
        
        # Verify NaN handling
        read_data = pd.read_excel(result_path)
        assert len(read_data) == len(nan_data)
        assert pd.isna(read_data.iloc[1]['value'])
    
    @pytest.mark.skipif(True, reason="Requires openpyxl dependency")
    def test_excel_dependencies_check(self):
        """Test Excel dependencies check"""
        # This test would check if ImportError is raised when openpyxl is not available
        # Skipped by default since we assume openpyxl is installed for testing
        pass