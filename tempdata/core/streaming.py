"""
Streaming generation system for memory-efficient large dataset creation

Provides StreamingGenerator class for generating large datasets in chunks
to manage memory usage and enable processing of datasets that exceed available RAM.
"""

import psutil
import pandas as pd
from typing import Iterator, Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass
import gc
import time
from pathlib import Path

from .seeding import MillisecondSeeder
from .base_generator import BaseGenerator


@dataclass
class StreamingConfig:
    """Configuration for streaming generation"""
    chunk_size: int = 50000  # Default chunk size
    max_memory_mb: int = 100  # Maximum memory usage in MB
    auto_streaming_threshold: int = 100000  # Auto-enable streaming above this row count
    memory_check_interval: int = 10  # Check memory every N chunks
    enable_gc: bool = True  # Enable garbage collection between chunks
    progress_callback: Optional[Callable[[int, int], None]] = None  # Progress callback


class MemoryMonitor:
    """Monitor memory usage during generation"""
    
    def __init__(self, max_memory_mb: int = 100):
        """
        Initialize memory monitor
        
        Args:
            max_memory_mb: Maximum allowed memory usage in MB
        """
        self.max_memory_mb = max_memory_mb
        self.process = psutil.Process()
        self.initial_memory = self.get_current_memory_mb()
        self.peak_memory = self.initial_memory
    
    def get_current_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def check_memory_usage(self) -> Dict[str, float]:
        """
        Check current memory usage and return statistics
        
        Returns:
            Dictionary with memory statistics
        """
        current_memory = self.get_current_memory_mb()
        memory_increase = current_memory - self.initial_memory
        
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
        
        return {
            'current_mb': current_memory,
            'initial_mb': self.initial_memory,
            'increase_mb': memory_increase,
            'peak_mb': self.peak_memory,
            'max_allowed_mb': self.max_memory_mb,
            'usage_percentage': (memory_increase / self.max_memory_mb) * 100
        }
    
    def should_trigger_streaming(self) -> bool:
        """Check if memory usage should trigger streaming mode"""
        stats = self.check_memory_usage()
        return stats['increase_mb'] > self.max_memory_mb * 0.8  # 80% threshold
    
    def cleanup_memory(self) -> None:
        """Force garbage collection to free memory"""
        gc.collect()


class StreamingGenerator:
    """
    Memory-efficient generator for large datasets
    
    Generates data in configurable chunks to manage memory usage
    and enable processing of very large datasets.
    """
    
    def __init__(self, base_generator: BaseGenerator, config: Optional[StreamingConfig] = None):
        """
        Initialize streaming generator
        
        Args:
            base_generator: The base generator to use for data generation
            config: Streaming configuration (uses defaults if None)
        """
        self.base_generator = base_generator
        self.config = config or StreamingConfig()
        self.memory_monitor = MemoryMonitor(self.config.max_memory_mb)
        self._chunk_count = 0
        self._total_rows_generated = 0
    
    def should_use_streaming(self, rows: int) -> bool:
        """
        Determine if streaming should be used based on row count and memory
        
        Args:
            rows: Number of rows to generate
            
        Returns:
            bool: True if streaming should be used
        """
        # Check if rows exceed auto-streaming threshold
        if rows >= self.config.auto_streaming_threshold:
            return True
        
        # Check current memory usage
        if self.memory_monitor.should_trigger_streaming():
            return True
        
        return False
    
    def generate_streaming(self, rows: int, **kwargs) -> Iterator[pd.DataFrame]:
        """
        Generate data in streaming chunks
        
        Args:
            rows: Total number of rows to generate
            **kwargs: Additional parameters for generation
            
        Yields:
            pd.DataFrame: Chunks of generated data
        """
        chunk_size = min(self.config.chunk_size, rows)
        chunks_needed = (rows + chunk_size - 1) // chunk_size  # Ceiling division
        
        self._chunk_count = 0
        self._total_rows_generated = 0
        
        for chunk_idx in range(chunks_needed):
            # Calculate rows for this chunk
            start_row = chunk_idx * chunk_size
            remaining_rows = rows - start_row
            current_chunk_size = min(chunk_size, remaining_rows)
            
            # Generate chunk with adjusted seeding for consistency
            chunk_kwargs = kwargs.copy()
            chunk_kwargs['_chunk_offset'] = start_row  # Internal parameter for seeding
            
            try:
                chunk_data = self._generate_chunk(current_chunk_size, **chunk_kwargs)
                
                # Update counters
                self._chunk_count += 1
                self._total_rows_generated += len(chunk_data)
                
                # Report progress if callback provided
                if self.config.progress_callback:
                    self.config.progress_callback(self._total_rows_generated, rows)
                
                # Memory monitoring and cleanup
                if self._chunk_count % self.config.memory_check_interval == 0:
                    memory_stats = self.memory_monitor.check_memory_usage()
                    
                    # Force cleanup if memory usage is high
                    if memory_stats['usage_percentage'] > 90:
                        if self.config.enable_gc:
                            self.memory_monitor.cleanup_memory()
                
                yield chunk_data
                
            except Exception as e:
                raise RuntimeError(f"Failed to generate chunk {chunk_idx + 1}/{chunks_needed}: {str(e)}")
    
    def generate_to_file(self, rows: int, output_path: str, format_type: str = 'csv', **kwargs) -> str:
        """
        Generate large dataset directly to file using streaming
        
        Args:
            rows: Number of rows to generate
            output_path: Path to output file
            format_type: Export format ('csv', 'json', 'parquet')
            **kwargs: Additional generation parameters
            
        Returns:
            str: Path to generated file
        """
        output_path = Path(output_path)
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type == 'csv':
            return self._stream_to_csv(rows, output_path, **kwargs)
        elif format_type == 'json':
            return self._stream_to_json(rows, output_path, **kwargs)
        elif format_type == 'parquet':
            return self._stream_to_parquet(rows, output_path, **kwargs)
        else:
            raise ValueError(f"Unsupported streaming format: {format_type}")
    
    def generate_complete(self, rows: int, **kwargs) -> pd.DataFrame:
        """
        Generate complete dataset using streaming if necessary
        
        Args:
            rows: Number of rows to generate
            **kwargs: Additional generation parameters
            
        Returns:
            pd.DataFrame: Complete generated dataset
        """
        if not self.should_use_streaming(rows):
            # Use regular generation for smaller datasets
            return self.base_generator.generate(rows, **kwargs)
        
        # Use streaming for large datasets
        chunks = []
        for chunk in self.generate_streaming(rows, **kwargs):
            chunks.append(chunk)
        
        # Combine all chunks
        if chunks:
            result = pd.concat(chunks, ignore_index=True)
            
            # Final memory cleanup
            if self.config.enable_gc:
                self.memory_monitor.cleanup_memory()
            
            return result
        else:
            # Return empty DataFrame with correct structure
            return self.base_generator.generate(0, **kwargs)
    
    def _generate_chunk(self, chunk_size: int, **kwargs) -> pd.DataFrame:
        """
        Generate a single chunk of data
        
        Args:
            chunk_size: Number of rows in this chunk
            **kwargs: Generation parameters
            
        Returns:
            pd.DataFrame: Generated chunk
        """
        # Pass chunk offset to generator for proper sequencing
        chunk_offset = kwargs.get('_chunk_offset', 0)
        
        # Create a new seeder with offset for this chunk
        if hasattr(self.base_generator.seeder, 'seed'):
            chunk_seed = self.base_generator.seeder.seed + (chunk_offset // 1000)  # Vary seed less frequently
            chunk_seeder = MillisecondSeeder(fixed_seed=chunk_seed)
            
            # Temporarily replace the generator's seeder
            original_seeder = self.base_generator.seeder
            self.base_generator.seeder = chunk_seeder
            
            # Update faker instance with new seed
            self.base_generator.faker.seed_instance(
                chunk_seeder.get_contextual_seed(self.base_generator.__class__.__name__)
            )
            
            try:
                # Generate chunk data with offset information
                chunk_data = self.base_generator.generate(chunk_size, **kwargs)
                return chunk_data
            finally:
                # Restore original seeder
                self.base_generator.seeder = original_seeder
                self.base_generator.faker.seed_instance(
                    original_seeder.get_contextual_seed(self.base_generator.__class__.__name__)
                )
        else:
            # Fallback if seeder doesn't have seed attribute
            return self.base_generator.generate(chunk_size, **kwargs)
    
    def _stream_to_csv(self, rows: int, output_path: Path, **kwargs) -> str:
        """Stream generation directly to CSV file"""
        first_chunk = True
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            for chunk in self.generate_streaming(rows, **kwargs):
                # Write header only for first chunk
                chunk.to_csv(f, index=False, header=first_chunk, mode='a' if not first_chunk else 'w')
                first_chunk = False
        
        return str(output_path)
    
    def _stream_to_json(self, rows: int, output_path: Path, **kwargs) -> str:
        """Stream generation to JSON file (as JSON Lines format for efficiency)"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in self.generate_streaming(rows, **kwargs):
                # Write each row as a JSON line
                for _, row in chunk.iterrows():
                    f.write(row.to_json() + '\n')
        
        return str(output_path)
    
    def _stream_to_parquet(self, rows: int, output_path: Path, **kwargs) -> str:
        """Stream generation to Parquet file"""
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        writer = None
        schema = None
        
        try:
            for chunk in self.generate_streaming(rows, **kwargs):
                # Convert to Arrow table
                table = pa.Table.from_pandas(chunk)
                
                if writer is None:
                    # Initialize writer with schema from first chunk
                    schema = table.schema
                    writer = pq.ParquetWriter(output_path, schema)
                
                # Write chunk to file
                writer.write_table(table)
        
        finally:
            if writer:
                writer.close()
        
        return str(output_path)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get current memory statistics
        
        Returns:
            Dictionary with memory and generation statistics
        """
        memory_stats = self.memory_monitor.check_memory_usage()
        
        return {
            **memory_stats,
            'chunks_generated': self._chunk_count,
            'total_rows_generated': self._total_rows_generated,
            'chunk_size': self.config.chunk_size,
            'auto_streaming_threshold': self.config.auto_streaming_threshold
        }


def create_streaming_generator(generator_class: type, seeder: MillisecondSeeder, 
                             locale: str = 'en_US', 
                             config: Optional[StreamingConfig] = None) -> StreamingGenerator:
    """
    Factory function to create a streaming generator
    
    Args:
        generator_class: Class of the base generator to wrap
        seeder: Seeder instance for reproducible generation
        locale: Locale for data generation
        config: Streaming configuration
        
    Returns:
        StreamingGenerator: Configured streaming generator
    """
    base_generator = generator_class(seeder, locale)
    return StreamingGenerator(base_generator, config)


def auto_detect_streaming_config(rows: int, available_memory_mb: Optional[float] = None) -> StreamingConfig:
    """
    Automatically detect optimal streaming configuration based on system resources
    
    Args:
        rows: Number of rows to generate
        available_memory_mb: Available memory in MB (auto-detected if None)
        
    Returns:
        StreamingConfig: Optimized configuration
    """
    if available_memory_mb is None:
        # Get available memory
        memory = psutil.virtual_memory()
        available_memory_mb = memory.available / 1024 / 1024
    
    # Calculate optimal chunk size based on available memory
    # Assume roughly 1KB per row for estimation
    estimated_memory_per_row_kb = 1
    max_memory_mb = min(available_memory_mb * 0.1, 200)  # Use max 10% of available memory, cap at 200MB
    
    optimal_chunk_size = int((max_memory_mb * 1024) / estimated_memory_per_row_kb)
    optimal_chunk_size = max(1000, min(optimal_chunk_size, 100000))  # Between 1K and 100K rows
    
    # Auto-streaming threshold should be lower for systems with less memory
    auto_threshold = max(50000, optimal_chunk_size * 2)
    
    return StreamingConfig(
        chunk_size=optimal_chunk_size,
        max_memory_mb=int(max_memory_mb),
        auto_streaming_threshold=auto_threshold,
        memory_check_interval=max(1, optimal_chunk_size // 10000),  # Check more frequently for smaller chunks
        enable_gc=True
    )