"""
Caching and lazy loading system for TempData library

Provides LazyDataLoader for reference data caching and generator result caching
to improve performance and reduce memory usage.
"""

import json
import pickle
import hashlib
import time
import threading
from typing import Any, Dict, Optional, Callable, Union, List, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from functools import wraps
import weakref
import gc
import pandas as pd


@dataclass
class CacheConfig:
    """Configuration for caching system"""
    max_memory_mb: int = 100  # Maximum cache memory usage in MB
    max_entries: int = 1000   # Maximum number of cached entries
    ttl_seconds: int = 3600   # Time to live for cache entries (1 hour)
    enable_disk_cache: bool = True  # Enable disk-based caching
    disk_cache_dir: Optional[Path] = None  # Directory for disk cache
    enable_compression: bool = True  # Enable compression for disk cache
    cleanup_interval: int = 300  # Cleanup interval in seconds (5 minutes)


@dataclass
class CacheEntry:
    """Single cache entry with metadata"""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = 0
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if cache entry is expired"""
        return time.time() - self.created_at > ttl_seconds
    
    def touch(self) -> None:
        """Update last accessed time and increment access count"""
        self.last_accessed = time.time()
        self.access_count += 1


class MemoryCache:
    """
    In-memory cache with LRU eviction and memory management
    
    Provides fast access to frequently used data with automatic
    memory management and configurable eviction policies.
    """
    
    def __init__(self, config: CacheConfig):
        """
        Initialize memory cache
        
        Args:
            config: Cache configuration
        """
        self.config = config
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._current_memory_bytes = 0
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        
        # Start cleanup thread if cleanup interval is set
        if config.cleanup_interval > 0:
            self._start_cleanup_thread()
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread"""
        def cleanup_worker():
            while not self._stop_cleanup.wait(self.config.cleanup_interval):
                self._cleanup_expired()
                self._enforce_memory_limits()
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def _calculate_size(self, value: Any) -> int:
        """
        Estimate memory size of a value
        
        Args:
            value: Value to measure
            
        Returns:
            int: Estimated size in bytes
        """
        try:
            if isinstance(value, pd.DataFrame):
                return value.memory_usage(deep=True).sum()
            elif isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._calculate_size(k) + self._calculate_size(v) 
                          for k, v in value.items())
            else:
                # Fallback to pickle size estimation
                return len(pickle.dumps(value))
        except Exception:
            # If size calculation fails, use conservative estimate
            return 1024  # 1KB default
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries from cache"""
        with self._lock:
            expired_keys = []
            current_time = time.time()
            
            for key, entry in self._cache.items():
                if entry.is_expired(self.config.ttl_seconds):
                    expired_keys.append(key)
            
            for key in expired_keys:
                entry = self._cache.pop(key)
                self._current_memory_bytes -= entry.size_bytes
    
    def _enforce_memory_limits(self) -> None:
        """Enforce memory and entry count limits using LRU eviction"""
        with self._lock:
            max_memory_bytes = self.config.max_memory_mb * 1024 * 1024
            
            # Check if we need to evict entries
            if (len(self._cache) <= self.config.max_entries and 
                self._current_memory_bytes <= max_memory_bytes):
                return
            
            # Sort entries by last accessed time (LRU - oldest first)
            entries_by_access = sorted(
                self._cache.items(),
                key=lambda x: x[1].last_accessed
            )
            
            # Evict oldest entries until we're under limits
            keys_to_evict = []
            for key, entry in entries_by_access:
                # Check if we still need to evict
                current_entries = len(self._cache) - len(keys_to_evict)
                current_memory = self._current_memory_bytes - sum(
                    self._cache[k].size_bytes for k in keys_to_evict
                )
                
                if (current_entries <= self.config.max_entries and 
                    current_memory <= max_memory_bytes):
                    break
                
                keys_to_evict.append(key)
            
            # Actually evict the keys
            for key in keys_to_evict:
                entry = self._cache.pop(key)
                self._current_memory_bytes -= entry.size_bytes
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Optional[Any]: Cached value or None if not found/expired
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            
            # Check if expired
            if entry.is_expired(self.config.ttl_seconds):
                del self._cache[key]
                self._current_memory_bytes -= entry.size_bytes
                return None
            
            # Update access statistics
            entry.touch()
            return entry.value
    
    def put(self, key: str, value: Any) -> None:
        """
        Put value in cache
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache[key]
                self._current_memory_bytes -= old_entry.size_bytes
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size_bytes
            )
            
            self._cache[key] = entry
            self._current_memory_bytes += size_bytes
            
            # Enforce limits
            self._enforce_memory_limits()
    
    def invalidate(self, key: str) -> bool:
        """
        Remove specific key from cache
        
        Args:
            key: Cache key to remove
            
        Returns:
            bool: True if key was found and removed
        """
        with self._lock:
            entry = self._cache.pop(key, None)
            if entry:
                self._current_memory_bytes -= entry.size_bytes
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._current_memory_bytes = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dict[str, Any]: Cache statistics
        """
        with self._lock:
            return {
                'entries': len(self._cache),
                'memory_bytes': self._current_memory_bytes,
                'memory_mb': self._current_memory_bytes / (1024 * 1024),
                'hit_rate': self._calculate_hit_rate(),
                'config': {
                    'max_entries': self.config.max_entries,
                    'max_memory_mb': self.config.max_memory_mb,
                    'ttl_seconds': self.config.ttl_seconds
                }
            }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total_accesses = sum(entry.access_count for entry in self._cache.values())
        if total_accesses == 0:
            return 0.0
        
        # This is a simplified hit rate calculation
        # In a real implementation, you'd track hits vs misses
        return min(1.0, total_accesses / max(1, len(self._cache)))
    
    def shutdown(self) -> None:
        """Shutdown cache and cleanup resources"""
        if self._cleanup_thread:
            self._stop_cleanup.set()
            self._cleanup_thread.join(timeout=5.0)
        self.clear()


class DiskCache:
    """
    Disk-based cache for persistent storage of large datasets
    
    Provides persistent caching with compression and automatic
    cleanup of old entries.
    """
    
    def __init__(self, config: CacheConfig):
        """
        Initialize disk cache
        
        Args:
            config: Cache configuration
        """
        self.config = config
        
        # Set up cache directory
        if config.disk_cache_dir:
            self.cache_dir = Path(config.disk_cache_dir)
        else:
            self.cache_dir = Path.home() / '.tempdata_cache'
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
    
    def _get_cache_path(self, key: str) -> Path:
        """
        Get file path for cache key
        
        Args:
            key: Cache key
            
        Returns:
            Path: File path for cache entry
        """
        # Create safe filename from key
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.cache"
    
    def _get_metadata_path(self, key: str) -> Path:
        """
        Get metadata file path for cache key
        
        Args:
            key: Cache key
            
        Returns:
            Path: Metadata file path
        """
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.meta"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from disk cache
        
        Args:
            key: Cache key
            
        Returns:
            Optional[Any]: Cached value or None if not found/expired
        """
        with self._lock:
            cache_path = self._get_cache_path(key)
            meta_path = self._get_metadata_path(key)
            
            if not cache_path.exists() or not meta_path.exists():
                return None
            
            try:
                # Load metadata
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                
                # Check if expired
                if time.time() - metadata['created_at'] > self.config.ttl_seconds:
                    cache_path.unlink(missing_ok=True)
                    meta_path.unlink(missing_ok=True)
                    return None
                
                # Load data
                with open(cache_path, 'rb') as f:
                    if self.config.enable_compression:
                        import gzip
                        data = gzip.decompress(f.read())
                    else:
                        data = f.read()
                
                # Update access time
                metadata['last_accessed'] = time.time()
                metadata['access_count'] = metadata.get('access_count', 0) + 1
                
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f)
                
                return pickle.loads(data)
                
            except Exception as e:
                # If loading fails, remove corrupted files
                cache_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                return None
    
    def put(self, key: str, value: Any) -> None:
        """
        Put value in disk cache
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            cache_path = self._get_cache_path(key)
            meta_path = self._get_metadata_path(key)
            
            try:
                # Serialize data
                data = pickle.dumps(value)
                
                if self.config.enable_compression:
                    import gzip
                    data = gzip.compress(data)
                
                # Write data
                with open(cache_path, 'wb') as f:
                    f.write(data)
                
                # Write metadata
                metadata = {
                    'key': key,
                    'created_at': time.time(),
                    'last_accessed': time.time(),
                    'access_count': 1,
                    'size_bytes': len(data),
                    'compressed': self.config.enable_compression
                }
                
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f)
                    
            except Exception as e:
                # Clean up on failure
                cache_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                raise
    
    def invalidate(self, key: str) -> bool:
        """
        Remove specific key from disk cache
        
        Args:
            key: Cache key to remove
            
        Returns:
            bool: True if key was found and removed
        """
        with self._lock:
            cache_path = self._get_cache_path(key)
            meta_path = self._get_metadata_path(key)
            
            found = cache_path.exists() or meta_path.exists()
            
            cache_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            
            return found
    
    def clear(self) -> None:
        """Clear all disk cache entries"""
        with self._lock:
            for file_path in self.cache_dir.glob("*.cache"):
                file_path.unlink(missing_ok=True)
            for file_path in self.cache_dir.glob("*.meta"):
                file_path.unlink(missing_ok=True)
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired entries from disk cache
        
        Returns:
            int: Number of entries removed
        """
        with self._lock:
            removed_count = 0
            current_time = time.time()
            
            for meta_path in self.cache_dir.glob("*.meta"):
                try:
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                    
                    if current_time - metadata['created_at'] > self.config.ttl_seconds:
                        # Remove both cache and metadata files
                        cache_path = meta_path.with_suffix('.cache')
                        cache_path.unlink(missing_ok=True)
                        meta_path.unlink(missing_ok=True)
                        removed_count += 1
                        
                except Exception:
                    # Remove corrupted metadata files
                    cache_path = meta_path.with_suffix('.cache')
                    cache_path.unlink(missing_ok=True)
                    meta_path.unlink(missing_ok=True)
                    removed_count += 1
            
            return removed_count


class LazyDataLoader:
    """
    Enhanced lazy data loader with multi-level caching
    
    Provides memory and disk caching for reference data with
    automatic cache management and performance optimization.
    """
    
    def __init__(self, config: Optional[Union[CacheConfig, Path]] = None, data_root: Optional[Path] = None):
        """
        Initialize lazy data loader with caching
        
        Args:
            config: Cache configuration or data root path (for backward compatibility)
            data_root: Root path for data files (ignored if config is a Path)
        """
        # Handle backward compatibility - if config is a Path, treat it as data_root
        if isinstance(config, Path):
            data_root = config
            config = None
        
        self.config = config or CacheConfig()
        
        if data_root is None:
            data_root = Path(__file__).parent.parent / 'data'
        
        self.data_root = Path(data_root)
        
        # Initialize caches
        self.memory_cache = MemoryCache(self.config)
        self.disk_cache = DiskCache(self.config) if self.config.enable_disk_cache else None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'memory_hits': 0,
            'disk_hits': 0,
            'misses': 0,
            'loads': 0
        }
    
    def _generate_cache_key(self, file_path: str, section: Optional[str] = None, 
                          **kwargs) -> str:
        """
        Generate cache key for file and parameters
        
        Args:
            file_path: File path
            section: Optional section
            **kwargs: Additional parameters
            
        Returns:
            str: Cache key
        """
        key_parts = [file_path]
        if section:
            key_parts.append(f"section:{section}")
        
        # Add sorted kwargs for consistent keys
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            key_parts.extend(f"{k}:{v}" for k, v in sorted_kwargs)
        
        return "|".join(key_parts)
    
    def load_data(self, file_path: str, section: Optional[str] = None, 
                  force_reload: bool = False, **kwargs) -> Any:
        """
        Load data with multi-level caching
        
        Args:
            file_path: Relative path to data file
            section: Optional section to extract
            force_reload: Force reload bypassing cache
            **kwargs: Additional parameters for cache key
            
        Returns:
            Any: Loaded data
        """
        cache_key = self._generate_cache_key(file_path, section, **kwargs)
        
        with self._lock:
            # Skip cache if force reload
            if not force_reload:
                # Try memory cache first
                value = self.memory_cache.get(cache_key)
                if value is not None:
                    self._stats['memory_hits'] += 1
                    return value
                
                # Try disk cache if enabled
                if self.disk_cache:
                    value = self.disk_cache.get(cache_key)
                    if value is not None:
                        # Store in memory cache for faster future access
                        self.memory_cache.put(cache_key, value)
                        self._stats['disk_hits'] += 1
                        return value
            
            # Cache miss - load from file
            self._stats['misses'] += 1
            self._stats['loads'] += 1
            
            try:
                full_path = self.data_root / file_path
                
                if not full_path.exists():
                    raise FileNotFoundError(f"Data file not found: {full_path}")
                
                # Load JSON data
                with open(full_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract section if specified
                if section:
                    if section in data:
                        data = data[section]
                    else:
                        raise KeyError(f"Section '{section}' not found in {file_path}")
                
                # Cache the loaded data
                self.memory_cache.put(cache_key, data)
                if self.disk_cache:
                    self.disk_cache.put(cache_key, data)
                
                return data
                
            except Exception as e:
                raise RuntimeError(f"Failed to load data from {file_path}: {e}")
    
    def invalidate_cache(self, file_path: Optional[str] = None, 
                        section: Optional[str] = None) -> None:
        """
        Invalidate cache entries
        
        Args:
            file_path: Specific file to invalidate (None for all)
            section: Specific section to invalidate
        """
        with self._lock:
            if file_path is None:
                # Clear all caches
                self.memory_cache.clear()
                if self.disk_cache:
                    self.disk_cache.clear()
            else:
                # Invalidate specific entries
                cache_key = self._generate_cache_key(file_path, section)
                self.memory_cache.invalidate(cache_key)
                if self.disk_cache:
                    self.disk_cache.invalidate(cache_key)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics
        
        Returns:
            Dict[str, Any]: Cache statistics
        """
        with self._lock:
            stats = {
                'loader_stats': self._stats.copy(),
                'memory_cache': self.memory_cache.get_stats()
            }
            
            # Calculate hit rates
            total_requests = sum(self._stats.values())
            if total_requests > 0:
                stats['hit_rate'] = {
                    'memory': self._stats['memory_hits'] / total_requests,
                    'disk': self._stats['disk_hits'] / total_requests,
                    'total': (self._stats['memory_hits'] + self._stats['disk_hits']) / total_requests
                }
            else:
                stats['hit_rate'] = {'memory': 0.0, 'disk': 0.0, 'total': 0.0}
            
            return stats
    
    def cleanup(self) -> Dict[str, int]:
        """
        Perform cache cleanup
        
        Returns:
            Dict[str, int]: Cleanup statistics
        """
        with self._lock:
            results = {'memory_entries_before': len(self.memory_cache._cache)}
            
            # Memory cache cleanup is automatic, but we can force it
            self.memory_cache._cleanup_expired()
            self.memory_cache._enforce_memory_limits()
            
            results['memory_entries_after'] = len(self.memory_cache._cache)
            
            # Disk cache cleanup
            if self.disk_cache:
                results['disk_entries_removed'] = self.disk_cache.cleanup_expired()
            
            return results
    
    def shutdown(self) -> None:
        """Shutdown loader and cleanup resources"""
        self.memory_cache.shutdown()
        # Disk cache doesn't need explicit shutdown


def cache_generator_result(ttl_seconds: int = 3600, 
                          use_disk_cache: bool = False) -> Callable:
    """
    Decorator for caching generator results
    
    Args:
        ttl_seconds: Time to live for cached results
        use_disk_cache: Whether to use disk cache for persistence
        
    Returns:
        Callable: Decorator function
    """
    def decorator(func: Callable) -> Callable:
        # Create cache config for this decorator
        cache_config = CacheConfig(
            ttl_seconds=ttl_seconds,
            enable_disk_cache=use_disk_cache,
            max_memory_mb=50,  # Smaller cache for generator results
            max_entries=100
        )
        
        # Create dedicated cache instance
        memory_cache = MemoryCache(cache_config)
        disk_cache = DiskCache(cache_config) if use_disk_cache else None
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key_data = {
                'func': func.__name__,
                'args': args,
                'kwargs': sorted(kwargs.items())
            }
            cache_key = hashlib.md5(str(key_data).encode()).hexdigest()
            
            # Try memory cache first
            result = memory_cache.get(cache_key)
            if result is not None:
                return result
            
            # Try disk cache if enabled
            if disk_cache:
                result = disk_cache.get(cache_key)
                if result is not None:
                    memory_cache.put(cache_key, result)
                    return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            memory_cache.put(cache_key, result)
            if disk_cache:
                disk_cache.put(cache_key, result)
            
            return result
        
        # Add cache management methods to wrapper
        wrapper.clear_cache = lambda: (memory_cache.clear(), 
                                     disk_cache.clear() if disk_cache else None)
        wrapper.get_cache_stats = lambda: {
            'memory': memory_cache.get_stats(),
            'disk_enabled': disk_cache is not None
        }
        
        return wrapper
    
    return decorator


# Global cache instances
_global_config = CacheConfig()
_global_loader = None


def get_global_loader() -> LazyDataLoader:
    """Get global lazy data loader instance"""
    global _global_loader
    if _global_loader is None:
        _global_loader = LazyDataLoader(_global_config)
    return _global_loader


def configure_global_cache(config: CacheConfig) -> None:
    """
    Configure global cache settings
    
    Args:
        config: New cache configuration
    """
    global _global_config, _global_loader
    _global_config = config
    
    # Reset global loader to use new config
    if _global_loader:
        _global_loader.shutdown()
        _global_loader = None


def get_cache_stats() -> Dict[str, Any]:
    """Get global cache statistics"""
    loader = get_global_loader()
    return loader.get_cache_stats()


def cleanup_global_cache() -> Dict[str, int]:
    """Cleanup global cache"""
    loader = get_global_loader()
    return loader.cleanup()
 