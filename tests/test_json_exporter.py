"""
Unit tests for JSON exporter functionality
"""

import json
import os
import pandas as pd
import pytest
import tempfile
from datetime import datetime, date
from decimal import Decimal
from tempdata.exporters.json_exporter import JSONExporter, CustomJSONEncoder


class TestJSONExporter:
    """Test cases for JSONExporter class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.exporter = JSONExporter()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'age': [25, 30, 35, 28, 32],
            'salary': [50000.0, 60000.0, 70000.0, 55000.0, 65000.0],
            'active': [True, False, True, True, False],
            'join_date': pd.to_datetime(['2020-01-15', '2019-06-20', '2021-03-10', '2020-11-05', '2019-12-01'])
        })
    
    def teardown_method(self):
        """Clean up test fixtures"""
        # Clean up temporary files
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_basic_export(self):
        """Test basic JSON export functionality"""
        filename = os.path.join(self.temp_dir, 'test_basic.json')
        
        result_path = self.exporter.export(self.sample_data, filename)
        
        assert os.path.exists(result_path)
        assert result_path == filename
        
        # Verify JSON content
        with open(result_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        assert isinstance(json_data, list)
        assert len(json_data) == 5
        assert json_data[0]['name'] == 'Alice'
        assert json_data[0]['age'] == 25
        assert json_data[0]['active'] is True
    
    def test_export_with_different_orientations(self):
        """Test JSON export with different orientations"""
        orientations = ['records', 'values', 'index', 'columns']
        
        for orient in orientations:
            filename = os.path.join(self.temp_dir, f'test_{orient}.json')
            result_path = self.exporter.export(self.sample_data, filename, orient=orient)
            
            assert os.path.exists(result_path)
            
            with open(result_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Verify structure based on orientation
            if orient == 'records':
                assert isinstance(json_data, list)
                assert len(json_data) == 5
            elif orient == 'values':
                assert isinstance(json_data, list)
                assert len(json_data) == 5  # 5 rows
            elif orient == 'index':
                assert isinstance(json_data, dict)
                assert len(json_data) == 5  # 5 rows
            elif orient == 'columns':
                assert isinstance(json_data, dict)
                assert len(json_data) == 6  # 6 columns
    
    def test_export_with_metadata(self):
        """Test JSON export with metadata"""
        filename = os.path.join(self.temp_dir, 'test_metadata.json')
        
        result_path = self.exporter.export(
            self.sample_data, filename, 
            include_metadata=True, preserve_types=True
        )
        
        assert os.path.exists(result_path)
        
        with open(result_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        assert 'data' in json_data
        assert 'metadata' in json_data
        
        metadata = json_data['metadata']
        assert metadata['rows'] == 5
        assert metadata['columns'] == 6
        assert 'export_timestamp' in metadata
        assert 'column_types' in metadata
        
        # Verify data is still accessible
        assert isinstance(json_data['data'], list)
        assert len(json_data['data']) == 5
    
    def test_export_nested(self):
        """Test nested JSON export with grouping"""
        # Add a category column for grouping
        data_with_category = self.sample_data.copy()
        data_with_category['department'] = ['IT', 'HR', 'IT', 'Finance', 'HR']
        
        filename = os.path.join(self.temp_dir, 'test_nested.json')
        
        result_path = self.exporter.export_nested(
            data_with_category, filename, group_by='department'
        )
        
        assert os.path.exists(result_path)
        
        with open(result_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        assert isinstance(json_data, dict)
        assert 'IT' in json_data
        assert 'HR' in json_data
        assert 'Finance' in json_data
        
        # Verify IT department has 2 employees
        assert len(json_data['IT']) == 2
        # Verify HR department has 2 employees
        assert len(json_data['HR']) == 2
        # Verify Finance department has 1 employee
        assert len(json_data['Finance']) == 1
    
    def test_export_streaming(self):
        """Test streaming export for large datasets"""
        # Create larger dataset
        large_data = pd.concat([self.sample_data] * 100, ignore_index=True)
        
        filename = os.path.join(self.temp_dir, 'test_streaming.json')
        
        result_path = self.exporter.export_streaming(
            large_data, filename, chunk_size=50
        )
        
        assert os.path.exists(result_path)
        
        with open(result_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        assert isinstance(json_data, list)
        assert len(json_data) == 500  # 5 * 100
    
    def test_data_type_preservation(self):
        """Test that data types are properly preserved in JSON"""
        # Create data with various types
        mixed_data = pd.DataFrame({
            'integer': [1, 2, 3],
            'float': [1.1, 2.2, 3.3],
            'boolean': [True, False, True],
            'string': ['a', 'b', 'c'],
            'datetime': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']),
            'null_values': [1, None, 3]
        })
        
        filename = os.path.join(self.temp_dir, 'test_types.json')
        
        result_path = self.exporter.export(mixed_data, filename)
        
        with open(result_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        first_record = json_data[0]
        assert isinstance(first_record['integer'], int)
        assert isinstance(first_record['float'], float)
        assert isinstance(first_record['boolean'], bool)
        assert isinstance(first_record['string'], str)
        assert isinstance(first_record['datetime'], str)  # ISO format
        
        # Check null handling
        second_record = json_data[1]
        assert second_record['null_values'] is None
    
    def test_export_with_nan_values(self):
        """Test handling of NaN values in JSON export"""
        data_with_nan = pd.DataFrame({
            'values': [1.0, float('nan'), 3.0, None, 5.0],
            'strings': ['a', None, 'c', 'd', 'e']
        })
        
        filename = os.path.join(self.temp_dir, 'test_nan.json')
        
        result_path = self.exporter.export(data_with_nan, filename)
        
        with open(result_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # NaN and None should become null in JSON
        assert json_data[1]['values'] is None
        assert json_data[1]['strings'] is None
        assert json_data[3]['values'] is None
    
    def test_filename_validation(self):
        """Test filename validation and extension handling"""
        # Test without extension
        result_path = self.exporter.export(self.sample_data, 'test_no_ext')
        assert result_path.endswith('.json')
        
        # Test with correct extension
        result_path = self.exporter.export(self.sample_data, 'test_with_ext.json')
        assert result_path.endswith('.json')
        
        # Test invalid filename
        with pytest.raises(ValueError):
            self.exporter.export(self.sample_data, '')
        
        with pytest.raises(ValueError):
            self.exporter.export(self.sample_data, None)
    
    def test_invalid_data_handling(self):
        """Test handling of invalid data inputs"""
        filename = os.path.join(self.temp_dir, 'test_invalid.json')
        
        # Test None data
        with pytest.raises(ValueError):
            self.exporter.export(None, filename)
        
        # Test empty DataFrame
        with pytest.raises(ValueError):
            self.exporter.export(pd.DataFrame(), filename)
        
        # Test non-DataFrame input
        with pytest.raises(ValueError):
            self.exporter.export([1, 2, 3], filename)
    
    def test_export_info(self):
        """Test export information generation"""
        info = self.exporter.get_export_info(self.sample_data)
        
        assert info['rows'] == 5
        assert info['columns'] == 6
        assert len(info['column_names']) == 6
        assert 'data_types' in info
        assert 'memory_usage_mb' in info
        assert 'estimated_json_size_mb' in info
        
        # Verify column names
        expected_columns = ['id', 'name', 'age', 'salary', 'active', 'join_date']
        assert info['column_names'] == expected_columns
    
    def test_custom_json_encoder(self):
        """Test custom JSON encoder for special data types"""
        encoder = CustomJSONEncoder()
        
        # Test datetime encoding
        dt = datetime(2020, 1, 1, 12, 0, 0)
        encoded = encoder.default(dt)
        assert encoded == '2020-01-01T12:00:00'
        
        # Test date encoding
        d = date(2020, 1, 1)
        encoded = encoder.default(d)
        assert encoded == '2020-01-01'
        
        # Test Decimal encoding
        decimal_val = Decimal('123.45')
        encoded = encoder.default(decimal_val)
        assert encoded == 123.45
        
        # Test pandas NaN
        encoded = encoder.default(pd.NA)
        assert encoded is None
    
    def test_numeric_string_detection(self):
        """Test detection of numeric string columns"""
        # Create data with numeric strings
        numeric_strings = pd.Series(['1', '2', '3', '4', '5'])
        mixed_strings = pd.Series(['1', 'a', '3', '4', '5'])
        
        assert self.exporter._is_numeric_string_column(numeric_strings) is True
        assert self.exporter._is_numeric_string_column(mixed_strings) is False
    
    def test_concurrent_exports(self):
        """Test multiple concurrent exports don't interfere"""
        import threading
        
        def export_worker(worker_id):
            filename = os.path.join(self.temp_dir, f'concurrent_{worker_id}.json')
            result_path = self.exporter.export(self.sample_data, filename)
            assert os.path.exists(result_path)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=export_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all files were created
        for i in range(5):
            filename = os.path.join(self.temp_dir, f'concurrent_{i}.json')
            assert os.path.exists(filename)
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets"""
        # Create a larger dataset
        large_data = pd.DataFrame({
            'id': range(10000),
            'value': [f'value_{i}' for i in range(10000)],
            'random': pd.Series(range(10000)).apply(lambda x: x * 1.5)
        })
        
        filename = os.path.join(self.temp_dir, 'test_large.json')
        
        result_path = self.exporter.export(large_data, filename)
        
        assert os.path.exists(result_path)
        
        # Verify file size is reasonable
        file_size = os.path.getsize(result_path)
        assert file_size > 0
        
        # Verify content integrity by reading back
        with open(result_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        assert len(json_data) == 10000
        assert json_data[0]['id'] == 0
        assert json_data[-1]['id'] == 9999