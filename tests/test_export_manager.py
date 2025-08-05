"""
Unit tests for ExportManager functionality
"""

import pytest
import pandas as pd
import os
import tempfile
from unittest.mock import Mock, patch
from tempdata.exporters.export_manager import ExportManager


class TestExportManager:
    """Test cases for ExportManager"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.manager = ExportManager()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'age': [25, 30, 35, 28, 32],
            'salary': [50000.0, 60000.0, 70000.0, 55000.0, 65000.0],
            'is_active': [True, False, True, True, False]
        })
        
        # Create geographical data for GeoJSON testing
        self.geo_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Location A', 'Location B', 'Location C'],
            'latitude': [40.7128, 34.0522, 41.8781],
            'longitude': [-74.0060, -118.2437, -87.6298]
        })
    
    def teardown_method(self):
        """Clean up test fixtures"""
        # Clean up temporary files
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test ExportManager initialization"""
        assert 'csv' in self.manager.exporters
        assert 'json' in self.manager.exporters
        assert 'parquet' in self.manager.exporters
        assert 'excel' in self.manager.exporters
        assert 'xlsx' in self.manager.exporters
        assert 'geojson' in self.manager.exporters
        
        # Check format extensions
        assert self.manager.format_extensions['csv'] == '.csv'
        assert self.manager.format_extensions['json'] == '.json'
        assert self.manager.format_extensions['excel'] == '.xlsx'
    
    def test_export_single_csv(self):
        """Test single format export - CSV"""
        filename = os.path.join(self.temp_dir, 'test_single')
        
        result_path = self.manager.export_single(self.sample_data, filename, 'csv')
        
        assert os.path.exists(result_path)
        assert result_path.endswith('.csv')
        
        # Verify content
        read_data = pd.read_csv(result_path)
        assert len(read_data) == len(self.sample_data)
    
    def test_export_single_json(self):
        """Test single format export - JSON"""
        filename = os.path.join(self.temp_dir, 'test_single')
        
        result_path = self.manager.export_single(self.sample_data, filename, 'json')
        
        assert os.path.exists(result_path)
        assert result_path.endswith('.json')
    
    def test_export_single_excel(self):
        """Test single format export - Excel"""
        filename = os.path.join(self.temp_dir, 'test_single')
        
        result_path = self.manager.export_single(self.sample_data, filename, 'excel')
        
        assert os.path.exists(result_path)
        assert result_path.endswith('.xlsx')
    
    def test_export_single_geojson(self):
        """Test single format export - GeoJSON"""
        filename = os.path.join(self.temp_dir, 'test_geo')
        
        result_path = self.manager.export_single(self.geo_data, filename, 'geojson')
        
        assert os.path.exists(result_path)
        assert result_path.endswith('.geojson')
    
    def test_export_single_unsupported_format(self):
        """Test single export with unsupported format"""
        filename = os.path.join(self.temp_dir, 'test_unsupported')
        
        with pytest.raises(ValueError, match="Unsupported format"):
            self.manager.export_single(self.sample_data, filename, 'xml')
    
    def test_export_multiple_sequential(self):
        """Test multiple format export - sequential"""
        base_filename = os.path.join(self.temp_dir, 'test_multi')
        formats = ['csv', 'json']
        
        results = self.manager.export_multiple(
            self.sample_data, 
            base_filename, 
            formats, 
            concurrent=False
        )
        
        assert len(results) == 2
        assert 'csv' in results
        assert 'json' in results
        
        # Verify files exist
        for format_type, path in results.items():
            assert os.path.exists(path)
            assert path.endswith(self.manager.format_extensions[format_type])
    
    def test_export_multiple_concurrent(self):
        """Test multiple format export - concurrent"""
        base_filename = os.path.join(self.temp_dir, 'test_concurrent')
        formats = ['csv', 'json', 'excel']
        
        results = self.manager.export_multiple(
            self.sample_data, 
            base_filename, 
            formats, 
            concurrent=True
        )
        
        assert len(results) == 3
        assert all(fmt in results for fmt in formats)
        
        # Verify files exist
        for format_type, path in results.items():
            assert os.path.exists(path)
    
    def test_export_multiple_with_duplicates(self):
        """Test multiple export with duplicate formats"""
        base_filename = os.path.join(self.temp_dir, 'test_duplicates')
        formats = ['csv', 'json', 'csv', 'json']  # Duplicates
        
        results = self.manager.export_multiple(
            self.sample_data, 
            base_filename, 
            formats
        )
        
        # Should only export unique formats
        assert len(results) == 2
        assert 'csv' in results
        assert 'json' in results
    
    def test_export_multiple_unsupported_format(self):
        """Test multiple export with unsupported format"""
        base_filename = os.path.join(self.temp_dir, 'test_unsupported')
        formats = ['csv', 'xml', 'json']
        
        with pytest.raises(ValueError, match="Unsupported formats"):
            self.manager.export_multiple(self.sample_data, base_filename, formats)
    
    def test_format_specific_options(self):
        """Test format-specific options"""
        base_filename = os.path.join(self.temp_dir, 'test_options')
        formats = ['csv', 'json']
        
        results = self.manager.export_multiple(
            self.sample_data,
            base_filename,
            formats,
            csv_sep=';',  # CSV-specific option
            json_indent=4  # JSON-specific option
        )
        
        assert len(results) == 2
        
        # Verify CSV uses semicolon separator
        csv_content = open(results['csv'], 'r').read()
        assert ';' in csv_content
    
    def test_validate_formats(self):
        """Test format validation"""
        formats = ['csv', 'json', 'xml', 'parquet']
        
        validation = self.manager.validate_formats(formats)
        
        assert validation['csv'] == True
        assert validation['json'] == True
        assert validation['xml'] == False
        assert validation['parquet'] == True
    
    def test_get_supported_formats(self):
        """Test getting supported formats"""
        formats = self.manager.get_supported_formats()
        
        assert isinstance(formats, list)
        assert 'csv' in formats
        assert 'json' in formats
        assert 'excel' in formats
        assert 'geojson' in formats
    
    def test_get_format_info(self):
        """Test getting format information"""
        info = self.manager.get_format_info('csv')
        
        assert info['format'] == 'csv'
        assert info['extension'] == '.csv'
        assert 'exporter_class' in info
        assert 'description' in info
    
    def test_get_format_info_unsupported(self):
        """Test getting info for unsupported format"""
        with pytest.raises(ValueError, match="Unsupported format"):
            self.manager.get_format_info('xml')
    
    def test_get_export_info(self):
        """Test getting export information"""
        formats = ['csv', 'json']
        
        info = self.manager.get_export_info(self.sample_data, formats)
        
        assert 'data_summary' in info
        assert info['data_summary']['rows'] == len(self.sample_data)
        assert info['data_summary']['columns'] == len(self.sample_data.columns)
        assert 'formats' in info
        assert 'csv' in info['formats']
        assert 'json' in info['formats']
    
    def test_cleanup_failed_exports(self):
        """Test cleanup of failed export files"""
        base_filename = os.path.join(self.temp_dir, 'test_cleanup')
        formats = ['csv', 'json']
        
        # Create some dummy files
        csv_file = f"{base_filename}.csv"
        json_file = f"{base_filename}.json"
        
        with open(csv_file, 'w') as f:
            f.write("dummy content")
        with open(json_file, 'w') as f:
            f.write("dummy content")
        
        assert os.path.exists(csv_file)
        assert os.path.exists(json_file)
        
        # Cleanup
        self.manager.cleanup_failed_exports(base_filename, formats)
        
        assert not os.path.exists(csv_file)
        assert not os.path.exists(json_file)
    
    def test_export_with_validation_valid_data(self):
        """Test export with validation - valid data"""
        base_filename = os.path.join(self.temp_dir, 'test_validation')
        formats = ['csv', 'json']
        
        results = self.manager.export_with_validation(
            self.sample_data, 
            base_filename, 
            formats
        )
        
        assert len(results) == 2
        assert all(isinstance(path, str) for path in results.values())
        assert all(os.path.exists(path) for path in results.values())
    
    def test_export_with_validation_invalid_data(self):
        """Test export with validation - invalid data"""
        base_filename = os.path.join(self.temp_dir, 'test_invalid')
        formats = ['csv', 'json']
        
        # Test with empty DataFrame
        empty_data = pd.DataFrame()
        
        results = self.manager.export_with_validation(
            empty_data, 
            base_filename, 
            formats
        )
        
        assert len(results) == 2
        assert all(isinstance(error, ValueError) for error in results.values())
    
    def test_export_with_validation_invalid_formats(self):
        """Test export with validation - invalid formats"""
        base_filename = os.path.join(self.temp_dir, 'test_invalid_formats')
        formats = ['csv', 'xml', 'json']
        
        results = self.manager.export_with_validation(
            self.sample_data, 
            base_filename, 
            formats
        )
        
        assert len(results) == 3
        assert isinstance(results['xml'], ValueError)
        assert isinstance(results['csv'], str)  # Should succeed
        assert isinstance(results['json'], str)  # Should succeed
    
    def test_validate_export_data_none(self):
        """Test data validation with None"""
        with pytest.raises(ValueError, match="Data cannot be None"):
            self.manager._validate_export_data(None)
    
    def test_validate_export_data_not_dataframe(self):
        """Test data validation with non-DataFrame"""
        with pytest.raises(ValueError, match="Data must be a pandas DataFrame"):
            self.manager._validate_export_data("not a dataframe")
    
    def test_validate_export_data_empty(self):
        """Test data validation with empty DataFrame"""
        empty_data = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Data must contain at least one row"):
            self.manager._validate_export_data(empty_data)
    
    def test_validate_export_data_no_rows(self):
        """Test data validation with DataFrame with no rows"""
        no_rows_data = pd.DataFrame(columns=['a', 'b', 'c'])
        
        with pytest.raises(ValueError, match="Data must contain at least one row"):
            self.manager._validate_export_data(no_rows_data)
    
    def test_validate_export_data_large_dataset_warning(self):
        """Test data validation with large dataset"""
        # Create a large dataset (this is a mock test)
        with patch('pandas.DataFrame.memory_usage') as mock_memory:
            # Mock memory usage to return > 1GB
            mock_memory.return_value = pd.Series([1100 * 1024 * 1024])  # 1.1GB
            
            with patch('builtins.print') as mock_print:
                self.manager._validate_export_data(self.sample_data)
                mock_print.assert_called_once()
                assert "Warning: Large dataset detected" in mock_print.call_args[0][0]
    
    def test_filename_with_extension(self):
        """Test export with filename that already has extension"""
        filename = os.path.join(self.temp_dir, 'test_with_ext.csv')
        
        result_path = self.manager.export_single(self.sample_data, filename, 'csv')
        
        assert result_path == filename
        assert os.path.exists(result_path)
    
    def test_concurrent_export_thread_safety(self):
        """Test thread safety of concurrent exports"""
        base_filename = os.path.join(self.temp_dir, 'test_thread_safety')
        formats = ['csv', 'json', 'excel']
        
        # Run multiple concurrent exports
        results1 = self.manager.export_multiple(
            self.sample_data, 
            f"{base_filename}_1", 
            formats, 
            concurrent=True
        )
        
        results2 = self.manager.export_multiple(
            self.sample_data, 
            f"{base_filename}_2", 
            formats, 
            concurrent=True
        )
        
        # Both should succeed
        assert len(results1) == 3
        assert len(results2) == 3
        
        # All files should exist
        for results in [results1, results2]:
            for path in results.values():
                assert os.path.exists(path)
    
    @patch('tempdata.exporters.export_manager.ExportManager._export_single_thread_safe')
    def test_concurrent_export_error_handling(self, mock_export):
        """Test error handling in concurrent exports"""
        # Mock one export to fail
        mock_export.side_effect = [
            "/path/to/success.csv",  # CSV succeeds
            Exception("JSON export failed"),  # JSON fails
            "/path/to/success.xlsx"  # Excel succeeds
        ]
        
        base_filename = os.path.join(self.temp_dir, 'test_error_handling')
        formats = ['csv', 'json', 'excel']
        
        # Should not raise exception, but print warning
        with patch('builtins.print') as mock_print:
            results = self.manager.export_multiple(
                self.sample_data, 
                base_filename, 
                formats, 
                concurrent=True
            )
            
            # Should have partial results
            assert 'csv' in results
            assert 'excel' in results
            assert 'json' not in results
            
            # Should print warning
            mock_print.assert_called_once()
            assert "Warning:" in mock_print.call_args[0][0]