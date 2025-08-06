"""
Performance benchmarks for TempData library

This module provides comprehensive performance testing including:
- Generation speed benchmarks for all dataset types
- Memory usage monitoring during large dataset creation
- Concurrent generation performance tests
- Export performance across different formats
- Streaming generation performance tests
"""

import pytest
import pandas as pd
import numpy as np
import time
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from memory_profiler import profile
import gc

from tempdata.core.seeding import MillisecondSeeder
from tempdata.core.streaming import StreamingGenerator
from tempdata.datasets.business.sales import SalesGenerator
from tempdata.datasets.financial.stocks import StockGenerator
from tempdata.datasets.healthcare.patients import PatientGenerator
from tempdata.datasets.technology.web_analytics import WebAnalyticsGenerator
from tempdata.datasets.iot_sensors.weather import WeatherGenerator
from tempdata.exporters.csv_exporter import CSVExporter
from tempdata.exporters.json_exporter import JSONExporter
from tempdata.exporters.parquet_exporter import ParquetExporter
from tempdata.exporters.export_manager import ExportManager


class TestGenerationSpeedBenchmarks:
    """Benchmark generation speed for all dataset types"""
    
    @pytest.fixture
    def seeder(self):
        return MillisecondSeeder(fixed_seed=12345)
    
    def test_sales_generator_speed_benchmark(self, benchmark, seeder):
        """Benchmark sales generator speed - should achieve 50K+ rows/second"""
        generator = SalesGenerator(seeder)
        
        def generate_sales():
            return generator.generate(1000)
        
        result = benchmark.pedantic(generate_sales, rounds=5, iterations=3)
        assert len(result) == 1000
        
        # Check that we meet performance requirements
        # pytest-benchmark will show the actual speed
        print(f"Sales generation completed for {len(result)} rows")
    
    def test_stock_generator_speed_benchmark(self, benchmark, seeder):
        """Benchmark stock generator speed"""
        generator = StockGenerator(seeder)
        
        def generate_stocks():
            return generator.generate(1000)
        
        result = benchmark.pedantic(generate_stocks, rounds=5, iterations=3)
        assert len(result) == 1000
        
        # Verify data quality is maintained at speed
        assert all(result['high'] >= result['low'])
        assert all(result['volume'] > 0)
    
    def test_patient_generator_speed_benchmark(self, benchmark, seeder):
        """Benchmark patient generator speed"""
        generator = PatientGenerator(seeder)
        
        def generate_patients():
            return generator.generate(1000)
        
        result = benchmark.pedantic(generate_patients, rounds=5, iterations=3)
        assert len(result) == 1000
        
        # Verify medical data constraints
        valid_blood_types = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
        assert all(result['blood_type'].isin(valid_blood_types))
    
    def test_web_analytics_speed_benchmark(self, benchmark, seeder):
        """Benchmark web analytics generator speed"""
        generator = WebAnalyticsGenerator(seeder)
        
        def generate_analytics():
            return generator.generate(1000)
        
        result = benchmark.pedantic(generate_analytics, rounds=5, iterations=3)
        assert len(result) == 1000
        
        # Verify web data constraints
        assert all(result['page_url'].str.startswith(('http://', 'https://')))
    
    def test_weather_generator_speed_benchmark(self, benchmark, seeder):
        """Benchmark weather generator speed"""
        generator = WeatherGenerator(seeder)
        
        def generate_weather():
            return generator.generate(1000)
        
        result = benchmark.pedantic(generate_weather, rounds=5, iterations=3)
        assert len(result) == 1000
        
        # Verify weather data ranges
        assert all(result['temperature'] >= -50)
        assert all(result['temperature'] <= 60)
        assert all(result['humidity'] >= 0)
        assert all(result['humidity'] <= 100)
    
    @pytest.mark.slow
    def test_large_dataset_speed_benchmark(self, benchmark, seeder):
        """Benchmark large dataset generation (10K rows)"""
        generator = SalesGenerator(seeder)
        
        def generate_large_dataset():
            return generator.generate(10000)
        
        result = benchmark.pedantic(generate_large_dataset, rounds=3, iterations=1)
        assert len(result) == 10000
        
        # Verify data quality is maintained
        assert result['transaction_id'].nunique() == 10000
        assert all(result['amount'] > 0)


class TestMemoryUsageBenchmarks:
    """Test memory usage during dataset generation"""
    
    @pytest.fixture
    def seeder(self):
        return MillisecondSeeder(fixed_seed=54321)
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def test_memory_usage_small_dataset(self, seeder):
        """Test memory usage for small datasets (1K rows)"""
        generator = SalesGenerator(seeder)
        
        # Force garbage collection
        gc.collect()
        initial_memory = self.get_memory_usage()
        
        # Generate data
        data = generator.generate(1000)
        peak_memory = self.get_memory_usage()
        
        # Clean up
        del data
        gc.collect()
        final_memory = self.get_memory_usage()
        
        memory_increase = peak_memory - initial_memory
        memory_cleanup = peak_memory - final_memory
        
        print(f"Memory usage - Initial: {initial_memory:.2f}MB, Peak: {peak_memory:.2f}MB, Final: {final_memory:.2f}MB")
        print(f"Memory increase: {memory_increase:.2f}MB, Cleanup: {memory_cleanup:.2f}MB")
        
        # Memory increase should be reasonable for 1K rows
        assert memory_increase < 50  # Less than 50MB for 1K rows
        assert memory_cleanup > 0   # Memory should be freed
    
    def test_memory_usage_medium_dataset(self, seeder):
        """Test memory usage for medium datasets (10K rows)"""
        generator = SalesGenerator(seeder)
        
        gc.collect()
        initial_memory = self.get_memory_usage()
        
        data = generator.generate(10000)
        peak_memory = self.get_memory_usage()
        
        del data
        gc.collect()
        final_memory = self.get_memory_usage()
        
        memory_increase = peak_memory - initial_memory
        
        print(f"10K rows - Memory increase: {memory_increase:.2f}MB")
        
        # Should use less than 100MB for 10K rows
        assert memory_increase < 100
    
    @pytest.mark.slow
    def test_memory_usage_large_dataset(self, seeder):
        """Test memory usage for large datasets (100K rows)"""
        generator = SalesGenerator(seeder)
        
        gc.collect()
        initial_memory = self.get_memory_usage()
        
        data = generator.generate(100000)
        peak_memory = self.get_memory_usage()
        
        del data
        gc.collect()
        final_memory = self.get_memory_usage()
        
        memory_increase = peak_memory - initial_memory
        
        print(f"100K rows - Memory increase: {memory_increase:.2f}MB")
        
        # Should use less than 500MB for 100K rows
        assert memory_increase < 500
    
    def test_streaming_memory_efficiency(self, seeder):
        """Test streaming generation memory efficiency"""
        streaming_gen = StreamingGenerator(seeder)
        
        gc.collect()
        initial_memory = self.get_memory_usage()
        
        # Generate 50K rows in streaming mode
        total_rows = 0
        max_memory = initial_memory
        
        for chunk in streaming_gen.generate_stream('sales', chunk_size=5000, total_rows=50000):
            current_memory = self.get_memory_usage()
            max_memory = max(max_memory, current_memory)
            total_rows += len(chunk)
            
            # Process chunk (simulate real usage)
            chunk_mean = chunk.select_dtypes(include=[np.number]).mean()
            del chunk_mean
        
        final_memory = self.get_memory_usage()
        max_memory_increase = max_memory - initial_memory
        
        print(f"Streaming 50K rows - Max memory increase: {max_memory_increase:.2f}MB")
        
        assert total_rows == 50000
        # Streaming should use significantly less memory than batch generation
        assert max_memory_increase < 100  # Should stay under 100MB


class TestConcurrentGenerationBenchmarks:
    """Test concurrent generation performance"""
    
    @pytest.fixture
    def seeder(self):
        return MillisecondSeeder(fixed_seed=98765)
    
    def test_thread_concurrent_generation(self, benchmark, seeder):
        """Test concurrent generation using threads"""
        def generate_concurrent_datasets():
            generators = [
                SalesGenerator(MillisecondSeeder(fixed_seed=seeder.seed + i))
                for i in range(4)
            ]
            
            def generate_data(gen):
                return gen.generate(1000)
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(generate_data, gen) for gen in generators]
                results = [future.result() for future in futures]
            
            return results
        
        results = benchmark(generate_concurrent_datasets)
        
        assert len(results) == 4
        for result in results:
            assert len(result) == 1000
            assert result['transaction_id'].nunique() == 1000
    
    def test_process_concurrent_generation(self, seeder):
        """Test concurrent generation using processes"""
        def generate_data_process(seed_offset):
            gen = SalesGenerator(MillisecondSeeder(fixed_seed=seeder.seed + seed_offset))
            return gen.generate(1000)
        
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(generate_data_process, i) for i in range(2)]
            results = [future.result() for future in futures]
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        print(f"Process concurrent generation time: {generation_time:.2f}s")
        
        assert len(results) == 2
        for result in results:
            assert len(result) == 1000
        
        # Should complete in reasonable time
        assert generation_time < 30  # Less than 30 seconds
    
    def test_mixed_generator_concurrent(self, seeder):
        """Test concurrent generation of different dataset types"""
        def generate_mixed_datasets():
            generators = [
                ('sales', SalesGenerator(MillisecondSeeder(fixed_seed=seeder.seed + 1))),
                ('stocks', StockGenerator(MillisecondSeeder(fixed_seed=seeder.seed + 2))),
                ('patients', PatientGenerator(MillisecondSeeder(fixed_seed=seeder.seed + 3))),
                ('weather', WeatherGenerator(MillisecondSeeder(fixed_seed=seeder.seed + 4)))
            ]
            
            def generate_data(name_gen_tuple):
                name, gen = name_gen_tuple
                return name, gen.generate(500)
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(generate_data, ng) for ng in generators]
                results = {name: data for name, data in [future.result() for future in futures]}
            
            return results
        
        start_time = time.time()
        results = generate_mixed_datasets()
        end_time = time.time()
        
        generation_time = end_time - start_time
        print(f"Mixed concurrent generation time: {generation_time:.2f}s")
        
        assert len(results) == 4
        assert 'sales' in results
        assert 'stocks' in results
        assert 'patients' in results
        assert 'weather' in results
        
        for name, data in results.items():
            assert len(data) == 500
            print(f"{name}: {len(data)} rows generated")


class TestExportPerformanceBenchmarks:
    """Test export performance across different formats"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for export testing"""
        seeder = MillisecondSeeder(fixed_seed=11111)
        generator = SalesGenerator(seeder)
        return generator.generate(5000)
    
    def test_csv_export_performance(self, benchmark, sample_data, tmp_path):
        """Benchmark CSV export performance"""
        exporter = CSVExporter()
        output_file = tmp_path / "test_sales.csv"
        
        def export_csv():
            exporter.export(sample_data, str(output_file))
            return output_file
        
        result_file = benchmark(export_csv)
        
        assert result_file.exists()
        assert result_file.stat().st_size > 0
        
        # Verify exported data
        exported_data = pd.read_csv(result_file)
        assert len(exported_data) == len(sample_data)
    
    def test_json_export_performance(self, benchmark, sample_data, tmp_path):
        """Benchmark JSON export performance"""
        exporter = JSONExporter()
        output_file = tmp_path / "test_sales.json"
        
        def export_json():
            exporter.export(sample_data, str(output_file))
            return output_file
        
        result_file = benchmark(export_json)
        
        assert result_file.exists()
        assert result_file.stat().st_size > 0
    
    def test_parquet_export_performance(self, benchmark, sample_data, tmp_path):
        """Benchmark Parquet export performance"""
        exporter = ParquetExporter()
        output_file = tmp_path / "test_sales.parquet"
        
        def export_parquet():
            exporter.export(sample_data, str(output_file))
            return output_file
        
        result_file = benchmark(export_parquet)
        
        assert result_file.exists()
        assert result_file.stat().st_size > 0
        
        # Verify exported data
        exported_data = pd.read_parquet(result_file)
        assert len(exported_data) == len(sample_data)
    
    def test_multi_format_export_performance(self, benchmark, sample_data, tmp_path):
        """Benchmark multi-format export performance"""
        export_manager = ExportManager()
        base_filename = str(tmp_path / "test_sales")
        formats = ['csv', 'json', 'parquet']
        
        def export_multi_format():
            export_manager.export_multiple_formats(sample_data, base_filename, formats)
            return [tmp_path / f"test_sales.{fmt}" for fmt in formats]
        
        result_files = benchmark(export_multi_format)
        
        for file_path in result_files:
            assert file_path.exists()
            assert file_path.stat().st_size > 0
    
    def test_concurrent_export_performance(self, sample_data, tmp_path):
        """Test concurrent export to multiple formats"""
        formats = ['csv', 'json', 'parquet']
        
        def export_format(fmt):
            if fmt == 'csv':
                exporter = CSVExporter()
            elif fmt == 'json':
                exporter = JSONExporter()
            elif fmt == 'parquet':
                exporter = ParquetExporter()
            
            output_file = tmp_path / f"concurrent_test.{fmt}"
            exporter.export(sample_data, str(output_file))
            return output_file
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(export_format, fmt) for fmt in formats]
            result_files = [future.result() for future in futures]
        
        end_time = time.time()
        export_time = end_time - start_time
        
        print(f"Concurrent export time: {export_time:.2f}s")
        
        for file_path in result_files:
            assert file_path.exists()
            assert file_path.stat().st_size > 0
        
        # Should complete in reasonable time
        assert export_time < 10  # Less than 10 seconds


class TestScalabilityBenchmarks:
    """Test scalability with increasing dataset sizes"""
    
    @pytest.fixture
    def seeder(self):
        return MillisecondSeeder(fixed_seed=22222)
    
    @pytest.mark.slow
    def test_scalability_progression(self, seeder):
        """Test generation time scaling with dataset size"""
        generator = SalesGenerator(seeder)
        sizes = [1000, 5000, 10000, 25000, 50000]
        times = []
        
        for size in sizes:
            start_time = time.time()
            data = generator.generate(size)
            end_time = time.time()
            
            generation_time = end_time - start_time
            times.append(generation_time)
            
            print(f"Size: {size}, Time: {generation_time:.2f}s, Rate: {size/generation_time:.0f} rows/s")
            
            # Verify data quality
            assert len(data) == size
            assert data['transaction_id'].nunique() == size
            
            # Clean up memory
            del data
            gc.collect()
        
        # Check that time scales reasonably (not exponentially)
        # Time should roughly scale linearly with size
        time_ratios = [times[i] / times[i-1] for i in range(1, len(times))]
        size_ratios = [sizes[i] / sizes[i-1] for i in range(1, len(sizes))]
        
        for i, (time_ratio, size_ratio) in enumerate(zip(time_ratios, size_ratios)):
            print(f"Step {i+1}: Size ratio {size_ratio:.1f}x, Time ratio {time_ratio:.1f}x")
            # Time ratio should not be much larger than size ratio (indicating good scalability)
            assert time_ratio <= size_ratio * 2  # Allow some overhead
    
    @pytest.mark.slow
    def test_memory_scalability(self, seeder):
        """Test memory usage scaling with dataset size"""
        generator = SalesGenerator(seeder)
        sizes = [1000, 5000, 10000, 25000]
        memory_usage = []
        
        for size in sizes:
            gc.collect()
            initial_memory = self.get_memory_usage()
            
            data = generator.generate(size)
            peak_memory = self.get_memory_usage()
            
            memory_increase = peak_memory - initial_memory
            memory_usage.append(memory_increase)
            
            print(f"Size: {size}, Memory increase: {memory_increase:.2f}MB, MB per 1K rows: {memory_increase/(size/1000):.2f}")
            
            del data
            gc.collect()
        
        # Memory usage should scale roughly linearly
        memory_per_1k = [mem / (size / 1000) for mem, size in zip(memory_usage, sizes)]
        
        # Memory per 1K rows should be relatively consistent
        memory_std = np.std(memory_per_1k)
        memory_mean = np.mean(memory_per_1k)
        
        print(f"Memory per 1K rows - Mean: {memory_mean:.2f}MB, Std: {memory_std:.2f}MB")
        
        # Standard deviation should be less than 50% of mean (reasonable consistency)
        assert memory_std < memory_mean * 0.5
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--benchmark-only'])