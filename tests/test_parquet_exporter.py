"""
Unit tests for Parquet exporter functionality
"""

import os
import pandas as pd
import pytest
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
from tempdata.exporters.parquet_exporter import ParquetExporter


class TestParquetExporter:
    """Test cases for ParquetExporter class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.exporter = ParquetExporter()
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
        """Test basic Parquet export functionality"""
        filename = os.path.join(self.temp_dir, 'test_basic.parquet')
        
        result_path = self.exporter.export(self.sample_data, filename)
        
        assert os.path.exists(result_path)
        assert result_path == filename
        
        # Verify Parquet content by reading back
        df_read = pd.read_parquet(result_path)
        
        assert len(df_read) == 5
        assert list(df_read.columns) == list(self.sample_data.columns)
        assert df_read['name'].iloc[0] == 'Alice'
        assert df_read['age'].iloc[0] == 25
        assert df_read['active'].iloc[0] == True
    
    def test_export_with_different_compressions(self):
        """Test Parquet export with different compression algorithms"""
        compressions = ['snappy', 'gzip', 'brotli', 'lz4', 'zstd', 'uncompressed']
        
        for compression in compressions:
            filename = os.path.join(self.temp_dir, f'test_{compression}.parquet')
            
            try:
                result_path = self.exporter.export(
                    self.sample_data, filename, compression=compression
                )
                
                assert os.path.exists(result_path)
                
                # Verify content integrity
                df_read = pd.read_parquet(result_path)
                assert len(df_read) == 5
                assert df_read['name'].iloc[0] == 'Alice'
                
            except Exception as e:
                # Some compression algorithms might not be available
                if 'not available' in str(e).lower():
                    pytest.skip(f"Compression {compression} not available")
                else:
                    raise
    
    def test_data_type_optimization(self):
        """Test data type optimization for Parquet storage"""
        # Create data with various types that can be optimized
        data_to_optimize = pd.DataFrame({
            'small_int': [1, 2, 3, 4, 5],  # Can be int8
            'large_int': [1000000, 2000000, 3000000, 4000000, 5000000],  # Needs int32
            'float_data': [1.1, 2.2, 3.3, 4.4, 5.5],
            'category_data': ['A', 'B', 'A', 'C', 'B'],  # Good for categorical
            'string_data': ['unique1', 'unique2', 'unique3', 'unique4', 'unique5']
        })
        
        filename = os.path.join(self.temp_dir, 'test_optimized.parquet')
        
        result_path = self.exporter.export(data_to_optimize, filename, optimize_types=True)
        
        assert os.path.exists(result_path)
        
        # Read back and verify data integrity
        df_read = pd.read_parquet(result_path)
        assert len(df_read) == 5
        
        # Verify values are preserved
        assert df_read['small_int'].iloc[0] == 1
        assert df_read['large_int'].iloc[0] == 1000000
        assert abs(df_read['float_data'].iloc[0] - 1.1) < 0.001
    
    def test_export_with_pyarrow_options(self):
        """Test export with PyArrow-specific options"""
        filename = os.path.join(self.temp_dir, 'test_pyarrow.parquet')
        
        result_path = self.exporter.export(
            self.sample_data, filename,
            engine='pyarrow',
            row_group_size=2,
            use_dictionary=True,
            write_statistics=True
        )
        
        assert os.path.exists(result_path)
        
        # Verify content
        df_read = pd.read_parquet(result_path)
        assert len(df_read) == 5
        
        # Check Parquet metadata
        parquet_file = pq.ParquetFile(result_path)
        assert parquet_file.num_row_groups >= 1
    
    def test_export_partitioned(self):
        """Test partitioned Parquet export"""
        # Add partition columns
        data_with_partitions = self.sample_data.copy()
        data_with_partitions['department'] = ['IT', 'HR', 'IT', 'Finance', 'HR']
        data_with_partitions['year'] = [2020, 2019, 2021, 2020, 2019]
        
        base_path = os.path.join(self.temp_dir, 'partitioned_dataset')
        
        result_path = self.exporter.export_partitioned(
            data_with_partitions, base_path, partition_cols=['department', 'year']
        )
        
        assert os.path.exists(result_path)
        assert os.path.isdir(result_path)
        
        # Verify partitioned structure exists
        # Should have department=IT, department=HR, department=Finance subdirectories
        partition_dirs = os.listdir(result_path)
        assert any('department=IT' in d for d in partition_dirs)
        assert any('department=HR' in d for d in partition_dirs)
        assert any('department=Finance' in d for d in partition_dirs)
        
        # Read back the partitioned dataset
        df_read = pd.read_parquet(result_path)
        assert len(df_read) == 5
    
    def test_export_with_schema(self):
        """Test export with explicit PyArrow schema"""
        # Define explicit schema
        schema = pa.schema([
            ('id', pa.int32()),
            ('name', pa.string()),
            ('age', pa.int16()),
            ('salary', pa.float64()),
            ('active', pa.bool_()),
            ('join_date', pa.timestamp('ms'))
        ])
        
        filename = os.path.join(self.temp_dir, 'test_schema.parquet')
        
        result_path = self.exporter.export_with_schema(
            self.sample_data, filename, schema
        )
        
        assert os.path.exists(result_path)
        
        # Verify schema was applied
        parquet_file = pq.ParquetFile(result_path)
        file_schema = parquet_file.schema.to_arrow_schema()
        
        assert file_schema.field('id').type == pa.int32()
        assert file_schema.field('age').type == pa.int16()
        
        # Close the file to avoid locking issues
        parquet_file.close()
    
    def test_compression_recommendation(self):
        """Test compression algorithm recommendation"""
        # Test with different data characteristics
        
        # Small dataset
        small_data = self.sample_data.iloc[:2]
        rec = self.exporter.get_compression_recommendation(small_data)
        assert 'recommended_compression' in rec
        assert 'reason' in rec
        
        # Large dataset with repetitive data
        large_repetitive = pd.DataFrame({
            'category': ['A'] * 5000 + ['B'] * 5000,
            'value': list(range(10000))
        })
        rec = self.exporter.get_compression_recommendation(large_repetitive)
        assert rec['recommended_compression'] in ['brotli', 'zstd', 'gzip']
        
        # String-heavy data
        string_heavy = pd.DataFrame({
            'text1': ['repeated_text'] * 1000,
            'text2': ['another_text'] * 1000,
            'text3': ['third_text'] * 1000,
            'number': range(1000)
        })
        rec = self.exporter.get_compression_recommendation(string_heavy)
        assert rec['recommended_compression'] in ['zstd', 'brotli', 'gzip']
    
    def test_data_type_analysis(self):
        """Test Parquet data type analysis"""
        # Create data with various types
        mixed_data = pd.DataFrame({
            'integer': [1, 2, 3],
            'float': [1.1, 2.2, 3.3],
            'boolean': [True, False, True],
            'string': ['a', 'b', 'c'],
            'datetime': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']),
            'category': pd.Categorical(['X', 'Y', 'X'])
        })
        
        # Test internal type detection
        for col in mixed_data.columns:
            parquet_type = self.exporter._get_parquet_type(mixed_data[col])
            assert parquet_type is not None
            assert parquet_type != 'unknown'
    
    def test_export_info(self):
        """Test export information generation"""
        info = self.exporter.get_export_info(self.sample_data)
        
        assert info['rows'] == 5
        assert info['columns'] == 6
        assert len(info['column_names']) == 6
        assert 'data_types' in info
        assert 'memory_usage_mb' in info
        assert 'estimated_parquet_size_mb' in info
        assert 'compression_recommendation' in info
        
        # Verify column information
        for col in self.sample_data.columns:
            assert col in info['data_types']
            col_info = info['data_types'][col]
            assert 'pandas_dtype' in col_info
            assert 'parquet_type' in col_info
            assert 'nullable' in col_info
    
    def test_filename_validation(self):
        """Test filename validation and extension handling"""
        # Test without extension
        result_path = self.exporter.export(self.sample_data, 'test_no_ext')
        assert result_path.endswith('.parquet')
        
        # Test with correct extension
        result_path = self.exporter.export(self.sample_data, 'test_with_ext.parquet')
        assert result_path.endswith('.parquet')
        
        # Test invalid filename
        with pytest.raises(ValueError):
            self.exporter.export(self.sample_data, '')
        
        with pytest.raises(ValueError):
            self.exporter.export(self.sample_data, None)
    
    def test_invalid_data_handling(self):
        """Test handling of invalid data inputs"""
        filename = os.path.join(self.temp_dir, 'test_invalid.parquet')
        
        # Test None data
        with pytest.raises(ValueError):
            self.exporter.export(None, filename)
        
        # Test empty DataFrame
        with pytest.raises(ValueError):
            self.exporter.export(pd.DataFrame(), filename)
        
        # Test non-DataFrame input
        with pytest.raises(ValueError):
            self.exporter.export([1, 2, 3], filename)
    
    def test_datetime_handling(self):
        """Test proper handling of datetime columns"""
        # Create data with various datetime formats
        datetime_data = pd.DataFrame({
            'naive_datetime': pd.to_datetime(['2020-01-01 12:00:00', '2020-01-02 13:00:00']),
            'utc_datetime': pd.to_datetime(['2020-01-01 12:00:00', '2020-01-02 13:00:00']).tz_localize('UTC'),
            'date_only': pd.to_datetime(['2020-01-01', '2020-01-02']).date
        })
        
        filename = os.path.join(self.temp_dir, 'test_datetime.parquet')
        
        result_path = self.exporter.export(datetime_data, filename)
        
        assert os.path.exists(result_path)
        
        # Verify datetime handling
        df_read = pd.read_parquet(result_path)
        assert len(df_read) == 2
        
        # Naive datetime should be preserved
        assert pd.api.types.is_datetime64_any_dtype(df_read['naive_datetime'])
        
        # UTC datetime should be converted to naive
        assert pd.api.types.is_datetime64_any_dtype(df_read['utc_datetime'])
    
    def test_categorical_data_handling(self):
        """Test handling of categorical data"""
        categorical_data = pd.DataFrame({
            'category_col': pd.Categorical(['A', 'B', 'A', 'C', 'B']),
            'string_col': ['A', 'B', 'A', 'C', 'B'],  # Will be converted to category
            'high_cardinality': [f'item_{i}' for i in range(5)]  # Won't be converted
        })
        
        filename = os.path.join(self.temp_dir, 'test_categorical.parquet')
        
        result_path = self.exporter.export(categorical_data, filename)
        
        assert os.path.exists(result_path)
        
        # Verify categorical handling
        df_read = pd.read_parquet(result_path)
        assert len(df_read) == 5
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets"""
        # Create a larger dataset
        large_data = pd.DataFrame({
            'id': range(50000),
            'value': [f'value_{i}' for i in range(50000)],
            'random': pd.Series(range(50000)).apply(lambda x: x * 1.5)
        })
        
        filename = os.path.join(self.temp_dir, 'test_large.parquet')
        
        result_path = self.exporter.export(large_data, filename)
        
        assert os.path.exists(result_path)
        
        # Verify file size is reasonable (should be compressed)
        file_size = os.path.getsize(result_path)
        assert file_size > 0
        
        # Verify content integrity by reading back
        df_read = pd.read_parquet(result_path)
        assert len(df_read) == 50000
        assert df_read['id'].iloc[0] == 0
        assert df_read['id'].iloc[-1] == 49999
    
    def test_concurrent_exports(self):
        """Test multiple concurrent exports don't interfere"""
        import threading
        
        def export_worker(worker_id):
            filename = os.path.join(self.temp_dir, f'concurrent_{worker_id}.parquet')
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
            filename = os.path.join(self.temp_dir, f'concurrent_{i}.parquet')
            assert os.path.exists(filename)
    
    def test_null_value_handling(self):
        """Test proper handling of null values"""
        data_with_nulls = pd.DataFrame({
            'integers': [1, None, 3, 4, 5],
            'floats': [1.1, 2.2, None, 4.4, 5.5],
            'strings': ['a', None, 'c', 'd', 'e'],
            'booleans': [True, None, False, True, False]
        })
        
        filename = os.path.join(self.temp_dir, 'test_nulls.parquet')
        
        result_path = self.exporter.export(data_with_nulls, filename)
        
        assert os.path.exists(result_path)
        
        # Verify null handling
        df_read = pd.read_parquet(result_path)
        assert len(df_read) == 5
        
        # Check that nulls are preserved
        assert pd.isna(df_read['integers'].iloc[1])
        assert pd.isna(df_read['floats'].iloc[2])
        assert pd.isna(df_read['strings'].iloc[1])
        assert pd.isna(df_read['booleans'].iloc[1])
    
    def test_dependency_check(self):
        """Test dependency checking"""
        # This should not raise an error if pyarrow is installed
        self.exporter._check_dependencies()
        
        # Test would require mocking to simulate missing dependencies
        # For now, just verify the method exists and runs
        assert hasattr(self.exporter, '_check_dependencies')