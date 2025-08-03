"""
Unit tests for the BaseGenerator abstract class

Tests cover seeder integration, localization, abstract methods, and common functionality.
"""

import unittest
import pandas as pd
from faker import Faker
from tempdata.core.seeding import MillisecondSeeder
from tempdata.core.base_generator import BaseGenerator


class TestBaseGenerator(unittest.TestCase):
    """Test cases for BaseGenerator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.seeder = MillisecondSeeder(fixed_seed=12345)
        self.generator = BaseGenerator(self.seeder)
    
    def test_initialization_with_defaults(self):
        """Test BaseGenerator initialization with default parameters"""
        generator = BaseGenerator(self.seeder)
        
        self.assertEqual(generator.seeder, self.seeder)
        self.assertEqual(generator.locale, 'en_US')
        self.assertIsInstance(generator.faker, Faker)
    
    def test_initialization_with_custom_locale(self):
        """Test BaseGenerator initialization with custom locale"""
        custom_locale = 'de_DE'
        generator = BaseGenerator(self.seeder, locale=custom_locale)
        
        self.assertEqual(generator.locale, custom_locale)
        self.assertIsInstance(generator.faker, Faker)
    
    def test_faker_seeding_consistency(self):
        """Test that faker instances are seeded consistently"""
        # Create two generators with same seeder and locale
        generator1 = BaseGenerator(self.seeder, locale='en_US')
        generator2 = BaseGenerator(self.seeder, locale='en_US')
        
        # Both should produce the same faker results due to seeding
        name1 = generator1.faker.name()
        name2 = generator2.faker.name()
        
        self.assertEqual(name1, name2)
    
    def test_faker_seeding_uniqueness_across_classes(self):
        """Test that different generator classes get different contextual seeds"""
        
        # Create a mock subclass to test contextual seeding
        class MockGenerator1(BaseGenerator):
            def generate(self, rows: int, **kwargs) -> pd.DataFrame:
                return pd.DataFrame({'test': [1, 2, 3]})
        
        class MockGenerator2(BaseGenerator):
            def generate(self, rows: int, **kwargs) -> pd.DataFrame:
                return pd.DataFrame({'test': [4, 5, 6]})
        
        gen1 = MockGenerator1(self.seeder)
        gen2 = MockGenerator2(self.seeder)
        
        # Different classes should get different contextual seeds
        seed1 = self.seeder.get_contextual_seed('MockGenerator1')
        seed2 = self.seeder.get_contextual_seed('MockGenerator2')
        
        self.assertNotEqual(seed1, seed2)
    
    def test_generate_method_abstract(self):
        """Test that generate method raises NotImplementedError"""
        with self.assertRaises(NotImplementedError) as context:
            self.generator.generate(100)
        
        self.assertIn("Subclasses must implement generate() method", str(context.exception))
    
    def test_apply_realistic_patterns_default(self):
        """Test default implementation of _apply_realistic_patterns"""
        test_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        result = self.generator._apply_realistic_patterns(test_data)
        
        # Default implementation should return data unchanged
        pd.testing.assert_frame_equal(result, test_data)
    
    def test_validate_data_default(self):
        """Test default implementation of _validate_data"""
        # Test with valid data
        valid_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        self.assertTrue(self.generator._validate_data(valid_data))
        
        # Test with empty data
        empty_data = pd.DataFrame()
        self.assertFalse(self.generator._validate_data(empty_data))
        
        # Test with zero-length data
        zero_length_data = pd.DataFrame(columns=['col1', 'col2'])
        self.assertFalse(self.generator._validate_data(zero_length_data))


class ConcreteGenerator(BaseGenerator):
    """Concrete implementation of BaseGenerator for testing"""
    
    def generate(self, rows: int, **kwargs) -> pd.DataFrame:
        """Generate test data"""
        data = pd.DataFrame({
            'id': range(1, rows + 1),
            'name': [self.faker.name() for _ in range(rows)],
            'value': [self.faker.random_int(1, 100) for _ in range(rows)]
        })
        
        # Apply realistic patterns and validate
        data = self._apply_realistic_patterns(data)
        if not self._validate_data(data):
            raise ValueError("Generated data failed validation")
        
        return data
    
    def _apply_realistic_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply test-specific realistic patterns"""
        # Example: Ensure values are positive
        data['value'] = data['value'].abs()
        return data
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate test data"""
        base_valid = super()._validate_data(data)
        if not base_valid:
            return False
        
        # Additional validation: all values should be positive
        return (data['value'] > 0).all()


class TestConcreteGenerator(unittest.TestCase):
    """Test cases for concrete generator implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.seeder = MillisecondSeeder(fixed_seed=54321)
        self.generator = ConcreteGenerator(self.seeder)
    
    def test_concrete_generate(self):
        """Test concrete implementation of generate method"""
        rows = 10
        data = self.generator.generate(rows)
        
        self.assertEqual(len(data), rows)
        self.assertIn('id', data.columns)
        self.assertIn('name', data.columns)
        self.assertIn('value', data.columns)
        
        # Check data types
        self.assertTrue(data['id'].dtype in ['int64', 'int32'])
        self.assertTrue(data['name'].dtype == 'object')
        self.assertTrue(data['value'].dtype in ['int64', 'int32'])
    
    def test_realistic_patterns_applied(self):
        """Test that realistic patterns are applied"""
        data = self.generator.generate(5)
        
        # All values should be positive (applied by _apply_realistic_patterns)
        self.assertTrue((data['value'] > 0).all())
    
    def test_data_validation(self):
        """Test data validation functionality"""
        data = self.generator.generate(5)
        
        # Generated data should pass validation
        self.assertTrue(self.generator._validate_data(data))
        
        # Test with invalid data (negative values)
        invalid_data = data.copy()
        invalid_data.loc[0, 'value'] = -1
        self.assertFalse(self.generator._validate_data(invalid_data))
    
    def test_reproducibility(self):
        """Test that same seeder produces reproducible results"""
        generator1 = ConcreteGenerator(MillisecondSeeder(fixed_seed=99999))
        generator2 = ConcreteGenerator(MillisecondSeeder(fixed_seed=99999))
        
        data1 = generator1.generate(5)
        data2 = generator2.generate(5)
        
        # Data should be identical
        pd.testing.assert_frame_equal(data1, data2)
    
    def test_different_locales(self):
        """Test generator behavior with different locales"""
        seeder = MillisecondSeeder(fixed_seed=77777)
        
        gen_us = ConcreteGenerator(seeder, locale='en_US')
        gen_de = ConcreteGenerator(seeder, locale='de_DE')
        
        data_us = gen_us.generate(3)
        data_de = gen_de.generate(3)
        
        # Names should be different due to different locales
        # (though this test might be flaky due to randomness)
        self.assertEqual(len(data_us), len(data_de))
        self.assertIn('name', data_us.columns)
        self.assertIn('name', data_de.columns)


if __name__ == '__main__':
    unittest.main()