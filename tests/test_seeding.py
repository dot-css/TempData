"""
Unit tests for the MillisecondSeeder class

Tests cover seeding reproducibility, uniqueness, contextual seeds, and temporal seeds.
"""

import unittest
import time
import hashlib
from tempdata.core.seeding import MillisecondSeeder


class TestMillisecondSeeder(unittest.TestCase):
    """Test cases for MillisecondSeeder class"""
    
    def test_fixed_seed_reproducibility(self):
        """Test that fixed seeds produce identical results across multiple runs"""
        fixed_seed = 12345
        
        # Create two seeders with the same fixed seed
        seeder1 = MillisecondSeeder(fixed_seed=fixed_seed)
        seeder2 = MillisecondSeeder(fixed_seed=fixed_seed)
        
        # Both should have the same seed value
        self.assertEqual(seeder1.seed, fixed_seed)
        self.assertEqual(seeder2.seed, fixed_seed)
        
        # Contextual seeds should be identical
        context = "test_context"
        context_seed1 = seeder1.get_contextual_seed(context)
        context_seed2 = seeder2.get_contextual_seed(context)
        self.assertEqual(context_seed1, context_seed2)
        
        # Temporal seeds should be identical for same offset
        temporal_seed1 = seeder1.get_temporal_seed(100)
        temporal_seed2 = seeder2.get_temporal_seed(100)
        self.assertEqual(temporal_seed1, temporal_seed2)
    
    def test_no_seed_uniqueness(self):
        """Test that seeders without fixed seeds generate unique values"""
        # Create seeders with small delays to ensure different timestamps
        seeders = []
        for _ in range(3):
            seeders.append(MillisecondSeeder())
            time.sleep(0.001)  # 1ms delay to ensure different timestamps
        
        # Seeds should be different when created at different times
        seeds = [seeder.seed for seeder in seeders]
        self.assertGreater(len(set(seeds)), 1, "Seeds should be unique when created at different times")
        
        # All seeds should be within valid range
        for seed in seeds:
            self.assertGreaterEqual(seed, 0)
            self.assertLess(seed, 2**32)
    
    def test_contextual_seed_consistency(self):
        """Test that contextual seeds are consistent for same context"""
        seeder = MillisecondSeeder(fixed_seed=54321)
        context = "sales_generator"
        
        # Multiple calls with same context should return same seed
        seed1 = seeder.get_contextual_seed(context)
        seed2 = seeder.get_contextual_seed(context)
        seed3 = seeder.get_contextual_seed(context)
        
        self.assertEqual(seed1, seed2)
        self.assertEqual(seed2, seed3)
        
        # Seed should be within valid range
        self.assertGreaterEqual(seed1, 0)
        self.assertLess(seed1, 2**31)
    
    def test_contextual_seed_uniqueness(self):
        """Test that different contexts generate different seeds"""
        seeder = MillisecondSeeder(fixed_seed=98765)
        
        contexts = ["sales", "customers", "products", "orders", "inventory"]
        context_seeds = [seeder.get_contextual_seed(ctx) for ctx in contexts]
        
        # All contextual seeds should be different
        self.assertEqual(len(context_seeds), len(set(context_seeds)))
        
        # All seeds should be within valid range
        for seed in context_seeds:
            self.assertGreaterEqual(seed, 0)
            self.assertLess(seed, 2**31)
    
    def test_temporal_seed_generation(self):
        """Test temporal seed generation with different offsets"""
        seeder = MillisecondSeeder(fixed_seed=11111)
        
        # Test different time offsets
        offsets = [0, 60, 3600, 86400]  # 0s, 1min, 1hour, 1day
        temporal_seeds = [seeder.get_temporal_seed(offset) for offset in offsets]
        
        # All temporal seeds should be different
        self.assertEqual(len(temporal_seeds), len(set(temporal_seeds)))
        
        # All seeds should be within valid range
        for seed in temporal_seeds:
            self.assertGreaterEqual(seed, 0)
            self.assertLess(seed, 2**32)
    
    def test_temporal_seed_reproducibility(self):
        """Test that temporal seeds are reproducible with same base time"""
        fixed_seed = 22222
        seeder1 = MillisecondSeeder(fixed_seed=fixed_seed)
        seeder2 = MillisecondSeeder(fixed_seed=fixed_seed)
        
        # Set same base time for both seeders
        base_time = 1640995200.0  # Fixed timestamp
        seeder1.base_time = base_time
        seeder2.base_time = base_time
        
        # Temporal seeds should be identical for same offset
        offset = 3600  # 1 hour
        temporal_seed1 = seeder1.get_temporal_seed(offset)
        temporal_seed2 = seeder2.get_temporal_seed(offset)
        
        self.assertEqual(temporal_seed1, temporal_seed2)
    
    def test_contextual_seed_algorithm(self):
        """Test the contextual seed generation algorithm"""
        seeder = MillisecondSeeder(fixed_seed=33333)
        context = "test_algorithm"
        
        # Calculate expected seed manually
        expected_hash = hashlib.md5(f"{seeder.seed}_{context}".encode()).hexdigest()
        expected_seed = int(expected_hash[:8], 16) % (2**31)
        
        # Compare with actual result
        actual_seed = seeder.get_contextual_seed(context)
        self.assertEqual(actual_seed, expected_seed)
    
    def test_temporal_seed_algorithm(self):
        """Test the temporal seed generation algorithm"""
        seeder = MillisecondSeeder(fixed_seed=44444)
        offset = 1800  # 30 minutes
        
        # Calculate expected seed manually
        expected_time_offset = int((seeder.base_time + offset) * 1000)
        expected_seed = expected_time_offset % (2**32)
        
        # Compare with actual result
        actual_seed = seeder.get_temporal_seed(offset)
        self.assertEqual(actual_seed, expected_seed)
    
    def test_seed_range_validation(self):
        """Test that all generated seeds are within valid ranges"""
        seeder = MillisecondSeeder()
        
        # Test main seed range
        self.assertGreaterEqual(seeder.seed, 0)
        self.assertLess(seeder.seed, 2**32)
        
        # Test contextual seed range
        contextual_seed = seeder.get_contextual_seed("range_test")
        self.assertGreaterEqual(contextual_seed, 0)
        self.assertLess(contextual_seed, 2**31)
        
        # Test temporal seed range
        temporal_seed = seeder.get_temporal_seed(1000)
        self.assertGreaterEqual(temporal_seed, 0)
        self.assertLess(temporal_seed, 2**32)
    
    def test_context_caching(self):
        """Test that contextual seeds are cached properly"""
        seeder = MillisecondSeeder(fixed_seed=55555)
        context = "cache_test"
        
        # First call should create cache entry
        self.assertNotIn(context, seeder._context_seeds)
        seed1 = seeder.get_contextual_seed(context)
        self.assertIn(context, seeder._context_seeds)
        
        # Second call should use cached value
        seed2 = seeder.get_contextual_seed(context)
        self.assertEqual(seed1, seed2)
        self.assertEqual(seeder._context_seeds[context], seed1)


if __name__ == '__main__':
    unittest.main()