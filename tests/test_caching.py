"""
Unit tests for caching and lazy loading system

Tests the enhanced caching functionality including memory cache,
disk cache, lazy data loader, and cache decorators.
"""

import unittest
import tempfile
import shutil
import time
import threading
import json
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

from tempdata.core.caching import (
    CacheConfig, CacheEntry, MemoryCache, DiskCache, 
    LazyDataLoader, cache_generator_result,
    get_global_loader, configure_global_cache, cleanup_global_cache
)


class TestCacheConfig(unittest.TestCase):
    """Test cache configuration"""
    
    def test_default_config(self):
        """Test default cache configuration"""
        config = CacheConfig()
        
        assert config.max_memory_mb == 100
        assert config.max_entries == 1000
        assert config.ttl_seconds == 3600
        assert config.enable_disk_cache is True
        assert config.disk_cache_dir is None
        assert config.enable_compression is True
        assert config.cleanup_interval == 300
    
    def test_custom_config(self):
        """Test custom cache configuration"""
        config = CacheConfig(
            max_memory_mb=50,
            max_entries=500,
            ttl_seconds=1800,
            enable_disk_cache=False,
            enable_compression=False,
            cleanup_interval=600
        )
        
        assert config.max_memory_mb == 50
        assert config.max_entries == 500
        assert config.ttl_seconds == 1800
        assert config.enable_disk_cache is False
        assert config.enable_compression is False
        assert config.cleanup_interval == 600


class TestCacheEntry(unittest.TestCase):
    """Test cache entry functionality"""
    
    def test_cache_entry_creation(self):
        """Test cache entry creation"""
        entry = CacheEntry(key="test_key", value="test_value", size_bytes=100)
        
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.size_bytes == 100
        assert entry.access_count == 0
        assert isinstance(entry.created_at, float)
        assert isinstance(entry.last_accessed, float)
    
    def test_cache_entry_expiration(self):
        """Test cache entry expiration"""
        entry = CacheEntry(key="test", value="data")
        
        # Should not be expired with long TTL
        assert not entry.is_expired(3600)
        
        # Should be expired with very short TTL
        time.sleep(0.01)
        assert entry.is_expired(0.001)
    
    def test_cache_entry_touch(self):
        """Test cache entry touch functionality"""
        entry = CacheEntry(key="test", value="data")
        original_access_time = entry.last_accessed
        original_count = entry.access_count
        
        time.sleep(0.01)
        entry.touch()
        
        assert entry.last_accessed > original_access_time
        assert entry.access_count == original_count + 1


class TestMemoryCache(unittest.TestCase):
    """Test memory cache functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = CacheConfig(
            max_memory_mb=1,  # Small cache for testing
            max_entries=10,
            ttl_seconds=1,
            cleanup_interval=0  # Disable background cleanup for tests
        )
        self.cache = MemoryCache(self.config)
    
    def tearDown(self):
        """Clean up after tests"""
        self.cache.shutdown()
    
    def test_cache_put_get(self):
        """Test basic cache put and get operations"""
        self.cache.put("key1", "value1")
        
        result = self.cache.get("key1")
        assert result == "value1"
        
        # Test non-existent key
        result = self.cache.get("nonexistent")
        assert result is None
    
    def test_cache_expiration(self):
        """Test cache entry expiration"""
        self.cache.put("key1", "value1")
        
        # Should be available immediately
        assert self.cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired
        assert self.cache.get("key1") is None
    
    def test_cache_size_calculation(self):
        """Test cache size calculation for different data types"""
        # Test string
        self.cache.put("string", "hello world")
        
        # Test DataFrame
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        self.cache.put("dataframe", df)
        
        # Test list
        self.cache.put("list", [1, 2, 3, 4, 5])
        
        # Test dict
        self.cache.put("dict", {"a": 1, "b": 2, "c": 3})
        
        stats = self.cache.get_stats()
        assert stats['memory_bytes'] > 0
        assert stats['entries'] == 4
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full"""
        # Create a cache with longer TTL for this test
        config = CacheConfig(
            max_memory_mb=1,
            max_entries=3,  # Even smaller for easier testing
            ttl_seconds=3600,  # Long TTL to avoid expiration during test
            cleanup_interval=0
        )
        cache = MemoryCache(config)
        
        try:
            # Fill cache to capacity
            for i in range(config.max_entries):
                cache.put(f"key{i}", f"value{i}")
            
            # Verify cache is full
            stats = cache.get_stats()
            assert stats['entries'] == config.max_entries
            
            # Sleep a bit to ensure different timestamps
            time.sleep(0.01)
            
            # Access first entry to make it recently used
            result = cache.get("key0")
            assert result == "value0"
            
            # Sleep a bit more
            time.sleep(0.01)
            
            # Add one more entry to trigger eviction
            cache.put("new_key", "new_value")
            
            # Cache should still be at max capacity
            stats = cache.get_stats()
            assert stats['entries'] <= config.max_entries
            
            # Recently accessed entry should still be there
            result = cache.get("key0")
            assert result == "value0", "Recently accessed key0 was evicted"
            
            # New entry should be there
            assert cache.get("new_key") == "new_value"
            
            # One of the older entries should be evicted
            # (either key1 or key2, depending on which was accessed less recently)
            evicted_count = 0
            if cache.get("key1") is None:
                evicted_count += 1
            if cache.get("key2") is None:
                evicted_count += 1
            
            assert evicted_count >= 1, "At least one old entry should be evicted"
            
        finally:
            cache.shutdown()
    
    def test_cache_invalidation(self):
        """Test cache invalidation"""
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")
        
        # Invalidate specific key
        result = self.cache.invalidate("key1")
        assert result is True
        assert self.cache.get("key1") is None
        assert self.cache.get("key2") == "value2"
        
        # Try to invalidate non-existent key
        result = self.cache.invalidate("nonexistent")
        assert result is False
    
    def test_cache_clear(self):
        """Test cache clearing"""
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")
        
        self.cache.clear()
        
        assert self.cache.get("key1") is None
        assert self.cache.get("key2") is None
        
        stats = self.cache.get_stats()
        assert stats['entries'] == 0
        assert stats['memory_bytes'] == 0
    
    def test_cache_stats(self):
        """Test cache statistics"""
        # Initially empty
        stats = self.cache.get_stats()
        assert stats['entries'] == 0
        assert stats['memory_bytes'] == 0
        
        # Add some entries
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")
        
        # Access entries to update hit rate
        self.cache.get("key1")
        self.cache.get("key2")
        
        stats = self.cache.get_stats()
        assert stats['entries'] == 2
        assert stats['memory_bytes'] > 0
        assert 'hit_rate' in stats
        assert 'config' in stats


class TestDiskCache(unittest.TestCase):
    """Test disk cache functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = CacheConfig(
            ttl_seconds=1,
            disk_cache_dir=Path(self.temp_dir),
            enable_compression=True
        )
        self.cache = DiskCache(self.config)
    
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_disk_cache_put_get(self):
        """Test basic disk cache operations"""
        test_data = {"key": "value", "number": 42}
        
        self.cache.put("test_key", test_data)
        result = self.cache.get("test_key")
        
        assert result == test_data
    
    def test_disk_cache_expiration(self):
        """Test disk cache expiration"""
        self.cache.put("key1", "value1")
        
        # Should be available immediately
        assert self.cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired and files cleaned up
        assert self.cache.get("key1") is None
    
    def test_disk_cache_compression(self):
        """Test disk cache compression"""
        large_data = {"data": "x" * 10000}  # Large string for compression
        
        self.cache.put("large_key", large_data)
        result = self.cache.get("large_key")
        
        assert result == large_data
        
        # Check that cache files exist
        cache_files = list(Path(self.temp_dir).glob("*.cache"))
        meta_files = list(Path(self.temp_dir).glob("*.meta"))
        
        assert len(cache_files) == 1
        assert len(meta_files) == 1
    
    def test_disk_cache_invalidation(self):
        """Test disk cache invalidation"""
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")
        
        # Invalidate specific key
        result = self.cache.invalidate("key1")
        assert result is True
        assert self.cache.get("key1") is None
        assert self.cache.get("key2") == "value2"
    
    def test_disk_cache_clear(self):
        """Test disk cache clearing"""
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")
        
        self.cache.clear()
        
        assert self.cache.get("key1") is None
        assert self.cache.get("key2") is None
        
        # Check that all files are removed
        cache_files = list(Path(self.temp_dir).glob("*.cache"))
        meta_files = list(Path(self.temp_dir).glob("*.meta"))
        
        assert len(cache_files) == 0
        assert len(meta_files) == 0
    
    def test_disk_cache_cleanup_expired(self):
        """Test cleanup of expired entries"""
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Cleanup expired entries
        removed_count = self.cache.cleanup_expired()
        
        assert removed_count == 2
        assert self.cache.get("key1") is None
        assert self.cache.get("key2") is None


class TestLazyDataLoader(unittest.TestCase):
    """Test enhanced lazy data loader"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data"
        self.data_dir.mkdir(parents=True)
        
        # Create test data files
        test_data = {
            "section1": {"key1": "value1", "key2": "value2"},
            "section2": {"key3": "value3", "key4": "value4"}
        }
        
        with open(self.data_dir / "test_data.json", "w") as f:
            json.dump(test_data, f)
        
        # Create cache config with short TTL for testing
        self.config = CacheConfig(
            ttl_seconds=1,
            max_memory_mb=1,
            max_entries=10,
            enable_disk_cache=True,
            disk_cache_dir=Path(self.temp_dir) / "cache",
            cleanup_interval=0
        )
        
        self.loader = LazyDataLoader(self.config, self.data_dir)
    
    def tearDown(self):
        """Clean up after tests"""
        self.loader.shutdown()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_loader_initialization(self):
        """Test loader initialization"""
        assert self.loader.data_root == self.data_dir
        assert self.loader.config == self.config
        assert self.loader.memory_cache is not None
        assert self.loader.disk_cache is not None
    
    def test_load_data_basic(self):
        """Test basic data loading"""
        data = self.loader.load_data("test_data.json")
        
        expected = {
            "section1": {"key1": "value1", "key2": "value2"},
            "section2": {"key3": "value3", "key4": "value4"}
        }
        
        assert data == expected
    
    def test_load_data_section(self):
        """Test loading specific section"""
        data = self.loader.load_data("test_data.json", section="section1")
        
        expected = {"key1": "value1", "key2": "value2"}
        assert data == expected
    
    def test_load_data_caching(self):
        """Test data loading with caching"""
        # First load
        data1 = self.loader.load_data("test_data.json")
        stats1 = self.loader.get_cache_stats()
        
        # Second load (should hit memory cache)
        data2 = self.loader.load_data("test_data.json")
        stats2 = self.loader.get_cache_stats()
        
        assert data1 == data2
        assert stats2['loader_stats']['memory_hits'] > stats1['loader_stats']['memory_hits']
    
    def test_load_data_force_reload(self):
        """Test force reload bypassing cache"""
        # First load
        data1 = self.loader.load_data("test_data.json")
        
        # Force reload
        data2 = self.loader.load_data("test_data.json", force_reload=True)
        
        assert data1 == data2
        
        stats = self.loader.get_cache_stats()
        assert stats['loader_stats']['loads'] >= 2
    
    def test_cache_invalidation(self):
        """Test cache invalidation"""
        # Load data
        self.loader.load_data("test_data.json")
        
        # Invalidate cache
        self.loader.invalidate_cache("test_data.json")
        
        # Next load should be a cache miss
        stats_before = self.loader.get_cache_stats()
        self.loader.load_data("test_data.json")
        stats_after = self.loader.get_cache_stats()
        
        assert stats_after['loader_stats']['loads'] > stats_before['loader_stats']['loads']
    
    def test_cache_stats(self):
        """Test cache statistics"""
        # Initially empty
        stats = self.loader.get_cache_stats()
        assert stats['loader_stats']['memory_hits'] == 0
        assert stats['loader_stats']['disk_hits'] == 0
        assert stats['loader_stats']['misses'] == 0
        
        # Load some data
        self.loader.load_data("test_data.json")
        self.loader.load_data("test_data.json", section="section1")
        
        # Access cached data
        self.loader.load_data("test_data.json")
        
        stats = self.loader.get_cache_stats()
        assert stats['loader_stats']['memory_hits'] > 0
        assert 'hit_rate' in stats
        assert 'memory_cache' in stats
    
    def test_cleanup(self):
        """Test cache cleanup"""
        # Load some data
        self.loader.load_data("test_data.json")
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Cleanup
        results = self.loader.cleanup()
        
        assert 'memory_entries_before' in results
        assert 'memory_entries_after' in results
        if self.loader.disk_cache:
            assert 'disk_entries_removed' in results
    
    def test_file_not_found(self):
        """Test handling of non-existent files"""
        with self.assertRaises(RuntimeError):
            self.loader.load_data("nonexistent.json")
    
    def test_invalid_section(self):
        """Test handling of invalid sections"""
        with self.assertRaises(RuntimeError):
            self.loader.load_data("test_data.json", section="nonexistent_section")


class TestCacheDecorator(unittest.TestCase):
    """Test cache decorator functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.call_count = 0
    
    def test_cache_decorator_basic(self):
        """Test basic cache decorator functionality"""
        @cache_generator_result(ttl_seconds=1, use_disk_cache=False)
        def test_function(x, y):
            self.call_count += 1
            return x + y
        
        # First call
        result1 = test_function(1, 2)
        assert result1 == 3
        assert self.call_count == 1
        
        # Second call with same arguments (should use cache)
        result2 = test_function(1, 2)
        assert result2 == 3
        assert self.call_count == 1  # Should not increment
        
        # Call with different arguments
        result3 = test_function(2, 3)
        assert result3 == 5
        assert self.call_count == 2
    
    def test_cache_decorator_expiration(self):
        """Test cache decorator expiration"""
        @cache_generator_result(ttl_seconds=0.1, use_disk_cache=False)
        def test_function(x):
            self.call_count += 1
            return x * 2
        
        # First call
        result1 = test_function(5)
        assert result1 == 10
        assert self.call_count == 1
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Second call (should not use cache)
        result2 = test_function(5)
        assert result2 == 10
        assert self.call_count == 2
    
    def test_cache_decorator_methods(self):
        """Test cache decorator utility methods"""
        @cache_generator_result(ttl_seconds=1, use_disk_cache=False)
        def test_function(x):
            return x
        
        # Test cache stats
        stats = test_function.get_cache_stats()
        assert 'memory' in stats
        assert 'disk_enabled' in stats
        
        # Test cache clearing
        test_function(1)
        test_function.clear_cache()
        
        # Should work without errors
        assert True


class TestGlobalCache(unittest.TestCase):
    """Test global cache functionality"""
    
    def test_global_loader(self):
        """Test global loader instance"""
        loader1 = get_global_loader()
        loader2 = get_global_loader()
        
        # Should be the same instance
        assert loader1 is loader2
    
    def test_configure_global_cache(self):
        """Test global cache configuration"""
        new_config = CacheConfig(max_memory_mb=50, max_entries=500)
        
        configure_global_cache(new_config)
        
        loader = get_global_loader()
        assert loader.config.max_memory_mb == 50
        assert loader.config.max_entries == 500
    
    def test_cleanup_global_cache(self):
        """Test global cache cleanup"""
        # This should work without errors
        results = cleanup_global_cache()
        assert isinstance(results, dict)


class TestCacheIntegration(unittest.TestCase):
    """Integration tests for caching system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data"
        self.data_dir.mkdir(parents=True)
        
        # Create test data
        large_data = {
            "countries": {f"country_{i}": {"name": f"Country {i}", "code": f"C{i:03d}"} 
                         for i in range(100)}
        }
        
        with open(self.data_dir / "large_data.json", "w") as f:
            json.dump(large_data, f)
        
        self.config = CacheConfig(
            max_memory_mb=1,
            max_entries=50,
            ttl_seconds=3600,
            enable_disk_cache=True,
            disk_cache_dir=Path(self.temp_dir) / "cache"
        )
        
        self.loader = LazyDataLoader(self.config, self.data_dir)
    
    def tearDown(self):
        """Clean up after tests"""
        self.loader.shutdown()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_memory_to_disk_cache_flow(self):
        """Test data flow from memory to disk cache"""
        # Load data (should go to memory cache)
        data1 = self.loader.load_data("large_data.json")
        
        # Clear memory cache but keep disk cache
        self.loader.memory_cache.clear()
        
        # Load again (should hit disk cache and populate memory cache)
        data2 = self.loader.load_data("large_data.json")
        
        assert data1 == data2
        
        stats = self.loader.get_cache_stats()
        assert stats['loader_stats']['disk_hits'] > 0
    
    def test_concurrent_access(self):
        """Test concurrent cache access"""
        results = []
        errors = []
        
        def load_data(thread_id):
            try:
                for i in range(10):
                    data = self.loader.load_data("large_data.json", 
                                               section="countries")
                    results.append((thread_id, len(data)))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=load_data, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 50  # 5 threads * 10 iterations
        
        # All results should be the same
        expected_length = results[0][1]
        for thread_id, length in results:
            assert length == expected_length
    
    def test_memory_pressure_handling(self):
        """Test cache behavior under memory pressure"""
        # Load multiple large datasets to trigger eviction
        for i in range(20):
            section_data = self.loader.load_data("large_data.json", 
                                               section="countries")
            # Modify the key to create different cache entries
            cache_key = f"large_data.json|section:countries|variant:{i}"
            self.loader.memory_cache.put(cache_key, section_data)
        
        # Check that cache enforced limits
        stats = self.loader.get_cache_stats()
        assert stats['memory_cache']['entries'] <= self.config.max_entries
        assert stats['memory_cache']['memory_mb'] <= self.config.max_memory_mb * 1.1  # Allow small overhead


if __name__ == '__main__':
    unittest.main()