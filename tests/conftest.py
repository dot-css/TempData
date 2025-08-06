"""
Pytest configuration and shared fixtures for TempData tests

This module provides:
- Common test fixtures for all test modules
- Mock objects for consistent testing
- Test data factories
- Configuration for property-based testing
- Performance testing setup
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from unittest.mock import Mock, MagicMock
import tempfile
import shutil
from pathlib import Path
from hypothesis import settings

from tempdata.core.seeding import MillisecondSeeder
from tempdata.core.localization import LocalizationEngine
from tempdata.core.validators import DataValidator
from tempdata.datasets.business.sales import SalesGenerator
from tempdata.datasets.financial.stocks import StockGenerator
from tempdata.datasets.healthcare.patients import PatientGenerator


# Configure Hypothesis for property-based testing
settings.register_profile("default", max_examples=50, deadline=5000)
settings.register_profile("fast", max_examples=10, deadline=2000)
settings.register_profile("thorough", max_examples=200, deadline=10000)
settings.load_profile("default")


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data files"""
    temp_dir = tempfile.mkdtemp(prefix="tempdata_tests_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def fixed_seeder():
    """Provide a fixed seeder for reproducible tests"""
    return MillisecondSeeder(fixed_seed=12345)


@pytest.fixture
def random_seeder():
    """Provide a random seeder for uniqueness tests"""
    return MillisecondSeeder()


@pytest.fixture
def localization_engine():
    """Provide LocalizationEngine instance"""
    return LocalizationEngine()


@pytest.fixture
def data_validator():
    """Provide DataValidator instance"""
    return DataValidator()


@pytest.fixture
def sample_sales_data(fixed_seeder):
    """Generate sample sales data for testing"""
    generator = SalesGenerator(fixed_seeder)
    return generator.generate(100)


@pytest.fixture
def sample_stock_data(fixed_seeder):
    """Generate sample stock data for testing"""
    generator = StockGenerator(fixed_seeder)
    return generator.generate(100)


@pytest.fixture
def sample_patient_data(fixed_seeder):
    """Generate sample patient data for testing"""
    generator = PatientGenerator(fixed_seeder)
    return generator.generate(50)


@pytest.fixture
def mock_country_data():
    """Mock country data for testing"""
    return {
        'united_states': {
            'name': 'United States',
            'code': 'US',
            'currency': 'USD',
            'phone_format': '+1 ({area}) {exchange}-{number}',
            'postal_format': '{zip}',
            'cities': ['New York', 'Los Angeles', 'Chicago'],
            'regions': ['North America']
        },
        'germany': {
            'name': 'Germany',
            'code': 'DE',
            'currency': 'EUR',
            'phone_format': '+49 {area} {number}',
            'postal_format': '{postal}',
            'cities': ['Berlin', 'Munich', 'Hamburg'],
            'regions': ['Europe']
        },
        'pakistan': {
            'name': 'Pakistan',
            'code': 'PK',
            'currency': 'PKR',
            'phone_format': '+92 {area} {number}',
            'postal_format': '{postal}',
            'cities': ['Karachi', 'Lahore', 'Islamabad'],
            'regions': ['Asia Pacific']
        }
    }


@pytest.fixture
def mock_business_data():
    """Mock business data for testing"""
    return {
        'company_names': [
            'TechCorp Inc', 'Global Solutions Ltd', 'Innovation Systems',
            'Digital Dynamics', 'Future Enterprises', 'Smart Solutions'
        ],
        'product_categories': [
            'technology', 'healthcare', 'retail', 'automotive', 
            'food_beverage', 'home_garden'
        ],
        'industries': [
            'Technology', 'Healthcare', 'Finance', 'Manufacturing',
            'Retail', 'Education', 'Government', 'Non-profit'
        ],
        'payment_methods': ['card', 'cash', 'digital', 'bank_transfer', 'check']
    }


@pytest.fixture
def mock_geographical_data():
    """Mock geographical data for testing"""
    return {
        'coordinates': {
            'new_york': (40.7128, -74.0060),
            'los_angeles': (34.0522, -118.2437),
            'chicago': (41.8781, -87.6298),
            'berlin': (52.5200, 13.4050),
            'munich': (48.1351, 11.5820),
            'karachi': (24.8607, 67.0011)
        },
        'addresses': [
            {
                'street': '123 Main St',
                'city': 'New York',
                'state': 'NY',
                'postal_code': '10001',
                'country': 'united_states'
            },
            {
                'street': '456 Oak Ave',
                'city': 'Los Angeles',
                'state': 'CA',
                'postal_code': '90210',
                'country': 'united_states'
            }
        ]
    }


class MockGenerator:
    """Mock generator for testing base functionality"""
    
    def __init__(self, seeder, locale='en_US'):
        self.seeder = seeder
        self.locale = locale
        self.call_count = 0
    
    def generate(self, rows, **kwargs):
        self.call_count += 1
        return pd.DataFrame({
            'id': range(rows),
            'value': np.random.randint(1, 100, rows),
            'timestamp': pd.date_range('2024-01-01', periods=rows, freq='H')
        })


@pytest.fixture
def mock_generator(fixed_seeder):
    """Provide mock generator for testing"""
    return MockGenerator(fixed_seeder)


class MockExporter:
    """Mock exporter for testing export functionality"""
    
    def __init__(self):
        self.export_calls = []
        self.exported_data = None
    
    def export(self, data, filename):
        self.export_calls.append((data, filename))
        self.exported_data = data
        # Simulate file creation
        Path(filename).touch()


@pytest.fixture
def mock_exporter():
    """Provide mock exporter for testing"""
    return MockExporter()


@pytest.fixture
def sample_time_series_data():
    """Generate sample time series data for testing"""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    values = np.cumsum(np.random.randn(100)) + 100  # Random walk starting at 100
    
    return pd.DataFrame({
        'timestamp': dates,
        'value': values,
        'volume': np.random.randint(1000, 10000, 100)
    })


@pytest.fixture
def sample_geographical_data():
    """Generate sample geographical data for testing"""
    return pd.DataFrame({
        'latitude': [40.7128, 34.0522, 41.8781, 29.7604, 39.9526],
        'longitude': [-74.0060, -118.2437, -87.6298, -95.3698, -75.1652],
        'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Philadelphia'],
        'state': ['NY', 'CA', 'IL', 'TX', 'PA'],
        'country': ['united_states'] * 5
    })


@pytest.fixture
def quality_test_data():
    """Generate data with known quality characteristics for testing validators"""
    # High quality data
    high_quality = pd.DataFrame({
        'id': range(100),
        'name': [f'Name_{i}' for i in range(100)],
        'amount': np.random.uniform(10, 1000, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'date': pd.date_range('2024-01-01', periods=100, freq='D')
    })
    
    # Low quality data (with issues)
    low_quality = pd.DataFrame({
        'id': [1, 1, 2, 3, 4] * 20,  # Duplicate IDs
        'name': [''] * 50 + [f'Name_{i}' for i in range(50)],  # Empty names
        'amount': [-10, 0, 5] * 33 + [1000],  # Negative and zero amounts
        'category': ['A'] * 100,  # No variety
        'date': [pd.NaT] * 20 + list(pd.date_range('2024-01-01', periods=80, freq='D'))  # Missing dates
    })
    
    return {'high_quality': high_quality, 'low_quality': low_quality}


@pytest.fixture
def performance_test_sizes():
    """Provide different dataset sizes for performance testing"""
    return [100, 500, 1000, 5000, 10000]


@pytest.fixture
def supported_countries():
    """List of supported countries for testing"""
    return [
        'united_states', 'canada', 'united_kingdom', 'germany', 'france',
        'spain', 'italy', 'netherlands', 'sweden', 'norway', 'denmark',
        'china', 'japan', 'south_korea', 'india', 'pakistan', 'bangladesh',
        'australia', 'new_zealand', 'brazil', 'mexico', 'argentina'
    ]


@pytest.fixture
def supported_locales():
    """List of supported locales for testing"""
    return [
        'en_US', 'en_GB', 'en_CA', 'en_AU',
        'de_DE', 'fr_FR', 'es_ES', 'it_IT', 'nl_NL',
        'sv_SE', 'no_NO', 'da_DK',
        'zh_CN', 'ja_JP', 'ko_KR', 'hi_IN',
        'pt_BR', 'es_MX'
    ]


class DatasetFactory:
    """Factory for creating test datasets with specific characteristics"""
    
    @staticmethod
    def create_sales_data(rows=100, seed=12345, **kwargs):
        """Create sales data with specific characteristics"""
        seeder = MillisecondSeeder(fixed_seed=seed)
        generator = SalesGenerator(seeder)
        return generator.generate(rows, **kwargs)
    
    @staticmethod
    def create_financial_data(rows=100, seed=12345, **kwargs):
        """Create financial data with specific characteristics"""
        seeder = MillisecondSeeder(fixed_seed=seed)
        generator = StockGenerator(seeder)
        return generator.generate(rows, **kwargs)
    
    @staticmethod
    def create_healthcare_data(rows=50, seed=12345, **kwargs):
        """Create healthcare data with specific characteristics"""
        seeder = MillisecondSeeder(fixed_seed=seed)
        generator = PatientGenerator(seeder)
        return generator.generate(rows, **kwargs)
    
    @staticmethod
    def create_time_series_data(rows=100, start_date='2024-01-01', freq='D'):
        """Create time series data for testing"""
        dates = pd.date_range(start_date, periods=rows, freq=freq)
        values = np.cumsum(np.random.randn(rows)) + 100
        
        return pd.DataFrame({
            'timestamp': dates,
            'value': values,
            'trend': np.linspace(0, 10, rows),
            'seasonal': np.sin(np.linspace(0, 4*np.pi, rows)) * 5
        })


@pytest.fixture
def dataset_factory():
    """Provide dataset factory for creating test data"""
    return DatasetFactory


# Pytest markers for different test categories
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance benchmarks")
    config.addinivalue_line("markers", "property: marks tests as property-based tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


# Custom assertions for data quality
def assert_data_quality(data, min_quality_score=0.8):
    """Assert that data meets minimum quality standards"""
    validator = DataValidator()
    quality_score = validator.calculate_quality_score(data)
    assert quality_score >= min_quality_score, f"Data quality score {quality_score} below minimum {min_quality_score}"


def assert_geographical_accuracy(data, min_accuracy=0.95):
    """Assert that geographical data meets accuracy standards"""
    validator = DataValidator()
    accuracy = validator.validate_geographical_accuracy(data)
    assert accuracy >= min_accuracy, f"Geographical accuracy {accuracy} below minimum {min_accuracy}"


def assert_realistic_patterns(data, pattern_threshold=0.7):
    """Assert that data contains realistic patterns"""
    validator = DataValidator()
    patterns = validator.detect_patterns(data)
    
    # Check that at least one pattern is detected above threshold
    pattern_scores = [score for score in patterns.values() if isinstance(score, (int, float))]
    if pattern_scores:
        max_pattern_score = max(pattern_scores)
        assert max_pattern_score >= pattern_threshold, f"No realistic patterns detected above threshold {pattern_threshold}"


# Add custom assertions to pytest namespace
pytest.assert_data_quality = assert_data_quality
pytest.assert_geographical_accuracy = assert_geographical_accuracy
pytest.assert_realistic_patterns = assert_realistic_patterns