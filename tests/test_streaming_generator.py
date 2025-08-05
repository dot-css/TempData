"""
Unit tests for streaming generation functionality

Tests the StreamingGenerator class, memory monitoring, and streaming configuration.
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import psutil

from tempdata.core.streaming import (
    StreamingGenerator, StreamingConfig, MemoryMonitor,
    create_streaming_generator, auto_detect_streaming_config
)
from tempdata.core.seeding import MillisecondSeeder
from tempdata.core.base_generator import BaseGenerator


class MockGenerator(BaseGenerator):
    """Mock generator for testing"""
    
    def generate(self, rows: int, **kwargs) -> pd.DataFrame:
        """Generate mock data"""
        # Support chunk offset for streaming consistency
        chunk_offset = kwargs.get('_chunk_offset', 0)
        start_id = chunk_offset + 1
        
        data = {
            'id': range(start_id, start_id + rows),
            'value': [f'value_{i}' for i in range(start_id, start_id + rows)],
            'amount': [i * 10.5 for i in range(start_id, start_id + rows)]
        }
        return pd.DataFrame(data)


class TestMemoryMonitor:
    """Test MemoryMonitor class"""
    
    def test_memory_monitor_initialization(self):
        """Test MemoryMonitor initialization"""
        monitor = MemoryMonitor(max_memory_mb=50)
        
        assert monitor.max_memory_mb == 50
        assert monitor.initial_memory > 0
        assert monitor.peak_memory >= monitor.initial_memory
    
    def test_get_current_memory_mb(self):
        """Test memory usage measurement"""
        monitor = MemoryMonitor()
        memory_mb = monitor.get_current_memory_mb()
        
        assert isinstance(memory_mb, float)
        assert memory_mb > 0
    
    def test_check_memory_usage(self):
        """Test memory usage statistics"""
        monitor = MemoryMonitor(max_memory_mb=100)
        stats = monitor.check_memory_usage()
        
        required_keys = ['current_mb', 'initial_mb', 'increase_mb', 'peak_mb', 
                        'max_allowed_mb', 'usage_percentage']
        
        for key in required_keys:
            assert key in stats
        
        assert stats['max_allowed_mb'] == 100
        assert stats['current_mb'] >= 0
        assert stats['usage_percentage'] >= 0
    
    def test_should_trigger_streaming(self):
        """Test streaming trigger logic"""
        # Mock memory usage to test trigger
        monitor = MemoryMonitor(max_memory_mb=100)
        
        with patch.object(monitor, 'get_current_memory_mb') as mock_memory:
            # Low memory usage - should not trigger
            mock_memory.return_value = monitor.initial_memory + 50  # 50MB increase
            assert not monitor.should_trigger_streaming()
            
            # High memory usage - should trigger
            mock_memory.return_value = monitor.initial_memory + 85  # 85MB increase (>80% of 100MB)
            assert monitor.should_trigger_streaming()
    
    def test_cleanup_memory(self):
        """Test memory cleanup functionality"""
        monitor = MemoryMonitor()
        
        # Should not raise any exceptions
        monitor.cleanup_memory()


class TestStreamingConfig:
    """Test StreamingConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = StreamingConfig()
        
        assert config.chunk_size == 50000
        assert config.max_memory_mb == 100
        assert config.auto_streaming_threshold == 100000
        assert config.memory_check_interval == 10
        assert config.enable_gc is True
        assert config.progress_callback is None
    
    def test_custom_config(self):
        """Test custom configuration values"""
        callback = Mock()
        config = StreamingConfig(
            chunk_size=25000,
            max_memory_mb=200,
            auto_streaming_threshold=50000,
            memory_check_interval=5,
            enable_gc=False,
            progress_callback=callback
        )
        
        assert config.chunk_size == 25000
        assert config.max_memory_mb == 200
        assert config.auto_streaming_threshold == 50000
        assert config.memory_check_interval == 5
        assert config.enable_gc is False
        assert config.progress_callback == callback


class TestStreamingGenerator:
    """Test StreamingGenerator class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.seeder = MillisecondSeeder(fixed_seed=12345)
        self.base_generator = MockGenerator(self.seeder)
        self.config = StreamingConfig(chunk_size=100, max_memory_mb=50)
        self.streaming_generator = StreamingGenerator(self.base_generator, self.config)
    
    def test_initialization(self):
        """Test StreamingGenerator initialization"""
        assert self.streaming_generator.base_generator == self.base_generator
        assert self.streaming_generator.config == self.config
        assert isinstance(self.streaming_generator.memory_monitor, MemoryMonitor)
        assert self.streaming_generator._chunk_count == 0
        assert self.streaming_generator._total_rows_generated == 0
    
    def test_should_use_streaming_row_threshold(self):
        """Test streaming decision based on row count"""
        # Below threshold - should not use streaming
        assert not self.streaming_generator.should_use_streaming(50000)
        
        # Above threshold - should use streaming
        assert self.streaming_generator.should_use_streaming(150000)
    
    def test_should_use_streaming_memory_threshold(self):
        """Test streaming decision based on memory usage"""
        with patch.object(self.streaming_generator.memory_monitor, 'should_trigger_streaming') as mock_trigger:
            # Memory trigger returns True
            mock_trigger.return_value = True
            assert self.streaming_generator.should_use_streaming(50000)
            
            # Memory trigger returns False
            mock_trigger.return_value = False
            assert not self.streaming_generator.should_use_streaming(50000)
    
    def test_generate_streaming(self):
        """Test streaming data generation"""
        rows = 250  # Should create 3 chunks of 100 rows each
        chunks = list(self.streaming_generator.generate_streaming(rows))
        
        assert len(chunks) == 3
        assert len(chunks[0]) == 100
        assert len(chunks[1]) == 100
        assert len(chunks[2]) == 50  # Last chunk has remaining rows
        
        # Verify data structure
        for chunk in chunks:
            assert 'id' in chunk.columns
            assert 'value' in chunk.columns
            assert 'amount' in chunk.columns
        
        # Verify total rows
        total_rows = sum(len(chunk) for chunk in chunks)
        assert total_rows == rows
    
    def test_generate_streaming_with_progress_callback(self):
        """Test streaming generation with progress callback"""
        callback = Mock()
        config = StreamingConfig(chunk_size=100, progress_callback=callback)
        streaming_gen = StreamingGenerator(self.base_generator, config)
        
        rows = 250
        list(streaming_gen.generate_streaming(rows))  # Consume all chunks
        
        # Verify callback was called
        assert callback.call_count == 3  # 3 chunks
        
        # Check callback arguments
        calls = callback.call_args_list
        assert calls[0][0] == (100, 250)  # (current_rows, total_rows)
        assert calls[1][0] == (200, 250)
        assert calls[2][0] == (250, 250)
    
    def test_generate_complete_small_dataset(self):
        """Test complete generation for small datasets (no streaming)"""
        with patch.object(self.streaming_generator, 'should_use_streaming', return_value=False):
            data = self.streaming_generator.generate_complete(100)
            
            assert len(data) == 100
            assert 'id' in data.columns
            assert 'value' in data.columns
            assert 'amount' in data.columns
    
    def test_generate_complete_large_dataset(self):
        """Test complete generation for large datasets (with streaming)"""
        with patch.object(self.streaming_generator, 'should_use_streaming', return_value=True):
            data = self.streaming_generator.generate_complete(250)
            
            assert len(data) == 250
            assert 'id' in data.columns
            assert 'value' in data.columns
            assert 'amount' in data.columns
            
            # Verify data continuity (IDs should be sequential)
            assert data['id'].tolist() == list(range(1, 251))
    
    def test_generate_to_file_csv(self):
        """Test streaming generation to CSV file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_output.csv"
            
            result_path = self.streaming_generator.generate_to_file(
                rows=250, 
                output_path=str(output_path), 
                format_type='csv'
            )
            
            assert result_path == str(output_path)
            assert output_path.exists()
            
            # Verify file content
            data = pd.read_csv(output_path)
            assert len(data) == 250
            assert 'id' in data.columns
            assert 'value' in data.columns
            assert 'amount' in data.columns
    
    def test_generate_to_file_json(self):
        """Test streaming generation to JSON Lines file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_output.jsonl"
            
            result_path = self.streaming_generator.generate_to_file(
                rows=250, 
                output_path=str(output_path), 
                format_type='json'
            )
            
            assert result_path == str(output_path)
            assert output_path.exists()
            
            # Verify file content (JSON Lines format)
            with open(output_path, 'r') as f:
                lines = f.readlines()
            
            assert len(lines) == 250
            
            # Verify first line is valid JSON
            import json
            first_record = json.loads(lines[0])
            assert 'id' in first_record
            assert 'value' in first_record
            assert 'amount' in first_record
    
    @pytest.mark.skipif(not pytest.importorskip("pyarrow", reason="pyarrow not available"), 
                       reason="pyarrow required for Parquet tests")
    def test_generate_to_file_parquet(self):
        """Test streaming generation to Parquet file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_output.parquet"
            
            result_path = self.streaming_generator.generate_to_file(
                rows=250, 
                output_path=str(output_path), 
                format_type='parquet'
            )
            
            assert result_path == str(output_path)
            assert output_path.exists()
            
            # Verify file content
            data = pd.read_parquet(output_path)
            assert len(data) == 250
            assert 'id' in data.columns
            assert 'value' in data.columns
            assert 'amount' in data.columns
    
    def test_generate_to_file_unsupported_format(self):
        """Test error handling for unsupported format"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_output.xml"
            
            with pytest.raises(ValueError, match="Unsupported streaming format: xml"):
                self.streaming_generator.generate_to_file(
                    rows=100, 
                    output_path=str(output_path), 
                    format_type='xml'
                )
    
    def test_get_memory_stats(self):
        """Test memory statistics reporting"""
        # Generate some data to update counters
        list(self.streaming_generator.generate_streaming(200))
        
        stats = self.streaming_generator.get_memory_stats()
        
        required_keys = ['current_mb', 'initial_mb', 'increase_mb', 'peak_mb', 
                        'max_allowed_mb', 'usage_percentage', 'chunks_generated',
                        'total_rows_generated', 'chunk_size', 'auto_streaming_threshold']
        
        for key in required_keys:
            assert key in stats
        
        assert stats['chunks_generated'] == 2  # 200 rows / 100 chunk_size = 2 chunks
        assert stats['total_rows_generated'] == 200
        assert stats['chunk_size'] == 100
    
    def test_chunk_generation_with_seeding(self):
        """Test that chunk generation maintains proper sequencing"""
        # Generate data in chunks
        chunks = list(self.streaming_generator.generate_streaming(200))
        
        # Combine chunks
        streamed_data = pd.concat(chunks, ignore_index=True)
        
        # Verify data continuity (IDs should be sequential)
        expected_ids = list(range(1, 201))
        assert streamed_data['id'].tolist() == expected_ids
        
        # Verify total length
        assert len(streamed_data) == 200
        
        # Verify data structure
        assert 'id' in streamed_data.columns
        assert 'value' in streamed_data.columns
        assert 'amount' in streamed_data.columns
    
    def test_memory_monitoring_during_generation(self):
        """Test memory monitoring during chunk generation"""
        config = StreamingConfig(chunk_size=50, memory_check_interval=1)  # Check every chunk
        streaming_gen = StreamingGenerator(self.base_generator, config)
        
        with patch.object(streaming_gen.memory_monitor, 'check_memory_usage') as mock_check:
            mock_check.return_value = {'usage_percentage': 50}
            
            list(streaming_gen.generate_streaming(150))  # 3 chunks
            
            # Memory should be checked for each chunk (3 times)
            assert mock_check.call_count == 3
    
    def test_memory_cleanup_trigger(self):
        """Test automatic memory cleanup when usage is high"""
        config = StreamingConfig(chunk_size=50, memory_check_interval=1, enable_gc=True)
        streaming_gen = StreamingGenerator(self.base_generator, config)
        
        with patch.object(streaming_gen.memory_monitor, 'check_memory_usage') as mock_check, \
             patch.object(streaming_gen.memory_monitor, 'cleanup_memory') as mock_cleanup:
            
            # Simulate high memory usage
            mock_check.return_value = {'usage_percentage': 95}
            
            list(streaming_gen.generate_streaming(150))  # 3 chunks
            
            # Cleanup should be called for each chunk due to high memory usage
            assert mock_cleanup.call_count == 3


class TestStreamingUtilityFunctions:
    """Test utility functions for streaming"""
    
    def test_create_streaming_generator(self):
        """Test streaming generator factory function"""
        seeder = MillisecondSeeder(fixed_seed=12345)
        config = StreamingConfig(chunk_size=1000)
        
        streaming_gen = create_streaming_generator(
            MockGenerator, seeder, 'en_US', config
        )
        
        assert isinstance(streaming_gen, StreamingGenerator)
        assert isinstance(streaming_gen.base_generator, MockGenerator)
        assert streaming_gen.config == config
    
    def test_auto_detect_streaming_config(self):
        """Test automatic streaming configuration detection"""
        # Test with specified available memory
        config = auto_detect_streaming_config(rows=1000000, available_memory_mb=1000)
        
        assert isinstance(config, StreamingConfig)
        assert config.chunk_size >= 1000
        assert config.chunk_size <= 100000
        assert config.max_memory_mb <= 200  # Capped at 200MB
        assert config.auto_streaming_threshold >= 50000
        assert config.enable_gc is True
    
    def test_auto_detect_streaming_config_system_memory(self):
        """Test automatic configuration with system memory detection"""
        config = auto_detect_streaming_config(rows=500000)
        
        assert isinstance(config, StreamingConfig)
        assert config.chunk_size >= 1000
        assert config.max_memory_mb > 0
        assert config.auto_streaming_threshold >= 50000
    
    def test_auto_detect_streaming_config_low_memory(self):
        """Test configuration for low memory systems"""
        config = auto_detect_streaming_config(rows=100000, available_memory_mb=100)
        
        # Should use smaller chunks for low memory systems
        assert config.chunk_size >= 1000
        assert config.max_memory_mb <= 10  # 10% of 100MB
        assert config.memory_check_interval >= 1


class TestStreamingIntegration:
    """Integration tests for streaming functionality"""
    
    def test_streaming_with_time_series(self):
        """Test streaming generation with time series data"""
        seeder = MillisecondSeeder(fixed_seed=12345)
        
        # Create a mock generator that supports time series
        class TimeSeriesMockGenerator(MockGenerator):
            def generate(self, rows: int, **kwargs):
                data = super().generate(rows, **kwargs)
                if kwargs.get('time_series'):
                    # Add timestamp column for time series
                    import datetime
                    start_date = datetime.datetime.now()
                    data['timestamp'] = [start_date + datetime.timedelta(hours=i) for i in range(rows)]
                return data
        
        base_generator = TimeSeriesMockGenerator(seeder)
        config = StreamingConfig(chunk_size=100)
        streaming_gen = StreamingGenerator(base_generator, config)
        
        # Generate time series data
        data = streaming_gen.generate_complete(250, time_series=True)
        
        assert len(data) == 250
        assert 'timestamp' in data.columns
        assert 'id' in data.columns
    
    def test_streaming_error_handling(self):
        """Test error handling in streaming generation"""
        # Create a generator that fails
        class FailingGenerator(BaseGenerator):
            def generate(self, rows: int, **kwargs):
                raise RuntimeError("Generation failed")
        
        seeder = MillisecondSeeder(fixed_seed=12345)
        failing_generator = FailingGenerator(seeder)
        streaming_gen = StreamingGenerator(failing_generator)
        
        with pytest.raises(RuntimeError, match="Failed to generate chunk"):
            list(streaming_gen.generate_streaming(100))
    
    def test_streaming_empty_dataset(self):
        """Test streaming generation with zero rows"""
        data = self.streaming_generator.generate_complete(0)
        
        # Should return empty DataFrame with correct structure
        assert len(data) == 0
        # Structure depends on the mock generator implementation
    
    def setup_method(self):
        """Set up test fixtures for integration tests"""
        self.seeder = MillisecondSeeder(fixed_seed=12345)
        self.base_generator = MockGenerator(self.seeder)
        self.streaming_generator = StreamingGenerator(self.base_generator)


if __name__ == '__main__':
    pytest.main([__file__])