"""
Unit tests for main API interface

Tests the primary API functions including create_dataset(), create_batch(),
and geo functions for parameter validation and behavior.
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

from tempdata.api import create_dataset, create_batch, _extract_dataset_type, _validate_time_series_params
from tempdata import geo


class TestCreateDataset:
    """Test cases for create_dataset function"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_dataset_basic_functionality(self):
        """Test basic dataset creation with default parameters"""
        result = create_dataset('sales.csv')
        
        assert isinstance(result, str)
        assert result.endswith('.csv')
        assert os.path.exists(result)
        
        # Verify file has content
        assert os.path.getsize(result) > 0
    
    def test_create_dataset_custom_rows(self):
        """Test dataset creation with custom row count"""
        result = create_dataset('sales.csv', rows=100)
        
        assert os.path.exists(result)
        
        # Read and verify row count
        df = pd.read_csv(result)
        assert len(df) == 100
    
    def test_create_dataset_multiple_formats(self):
        """Test dataset creation with multiple export formats"""
        result = create_dataset('sales.csv', formats=['csv', 'json'])
        
        # Should return comma-separated paths
        paths = result.split(', ')
        assert len(paths) == 2
        
        # Verify both files exist
        for path in paths:
            assert os.path.exists(path)
    
    def test_create_dataset_with_seed(self):
        """Test reproducible dataset generation with fixed seed"""
        result1 = create_dataset('sales.csv', seed=12345)
        result2 = create_dataset('sales2.csv', seed=12345)
        
        # Read both files and compare
        df1 = pd.read_csv(result1)
        df2 = pd.read_csv(result2)
        
        # Should have identical data (excluding any timestamp columns)
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_create_dataset_with_country(self):
        """Test dataset creation with specific country"""
        result = create_dataset('sales.csv', country='pakistan')
        
        assert os.path.exists(result)
        # Country-specific validation would require checking actual data content
    
    def test_create_dataset_time_series(self):
        """Test time series dataset generation"""
        result = create_dataset(
            'sales.csv', 
            time_series=True,
            start_date='2023-01-01',
            end_date='2023-01-31',
            interval='1day'
        )
        
        assert os.path.exists(result)
        
        # Verify time series data
        df = pd.read_csv(result)
        assert 'date' in df.columns or 'timestamp' in df.columns
    
    def test_create_dataset_invalid_filename(self):
        """Test error handling for invalid filename"""
        with pytest.raises(ValueError, match="filename must be a non-empty string"):
            create_dataset('')
        
        with pytest.raises(ValueError, match="filename must be a non-empty string"):
            create_dataset(None)
    
    def test_create_dataset_invalid_rows(self):
        """Test error handling for invalid row count"""
        with pytest.raises(ValueError, match="rows must be a positive integer"):
            create_dataset('sales.csv', rows=0)
        
        with pytest.raises(ValueError, match="rows must be a positive integer"):
            create_dataset('sales.csv', rows=-1)
        
        with pytest.raises(ValueError, match="rows cannot exceed 10,000,000"):
            create_dataset('sales.csv', rows=20_000_000)
    
    def test_create_dataset_unsupported_type(self):
        """Test error handling for unsupported dataset type"""
        with pytest.raises(FileNotFoundError, match="Unsupported dataset type"):
            create_dataset('unknown_type.csv')
    
    def test_create_dataset_invalid_format(self):
        """Test error handling for invalid export format"""
        with pytest.raises(ValueError, match="Unsupported export formats"):
            create_dataset('sales.csv', formats=['invalid_format'])
    
    def test_create_dataset_invalid_country(self):
        """Test error handling for invalid country parameter"""
        with pytest.raises(ValueError, match="country must be a string"):
            create_dataset('sales.csv', country=123)
    
    def test_create_dataset_invalid_time_series_params(self):
        """Test error handling for invalid time series parameters"""
        with pytest.raises(ValueError, match="start_date must be in ISO format"):
            create_dataset('sales.csv', time_series=True, start_date='invalid-date')
        
        with pytest.raises(ValueError, match="interval must be one of"):
            create_dataset('sales.csv', time_series=True, interval='invalid-interval')
        
        with pytest.raises(ValueError, match="start_date must be before end_date"):
            create_dataset(
                'sales.csv', 
                time_series=True,
                start_date='2023-12-31',
                end_date='2023-01-01'
            )


class TestCreateBatch:
    """Test cases for create_batch function"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_batch_basic_functionality(self):
        """Test basic batch dataset creation"""
        datasets = [
            {'filename': 'sales.csv'},
            {'filename': 'customers.csv'}
        ]
        
        results = create_batch(datasets)
        
        assert len(results) == 2
        for result in results:
            assert isinstance(result, str)
            assert os.path.exists(result)
    
    def test_create_batch_with_global_params(self):
        """Test batch creation with global parameters"""
        datasets = [
            {'filename': 'sales.csv'},
            {'filename': 'customers.csv'}
        ]
        
        results = create_batch(datasets, rows=100, country='pakistan', seed=12345)
        
        assert len(results) == 2
        
        # Verify row counts
        for result in results:
            df = pd.read_csv(result)
            assert len(df) == 100
    
    def test_create_batch_with_dataset_specific_params(self):
        """Test batch creation with dataset-specific parameters"""
        datasets = [
            {'filename': 'sales.csv', 'rows': 50},
            {'filename': 'customers.csv', 'rows': 200}
        ]
        
        results = create_batch(datasets, rows=100)  # Global default
        
        assert len(results) == 2
        
        # Verify dataset-specific row counts override global
        df1 = pd.read_csv(results[0])
        df2 = pd.read_csv(results[1])
        assert len(df1) == 50
        assert len(df2) == 200
    
    def test_create_batch_with_multiple_formats(self):
        """Test batch creation with multiple export formats"""
        datasets = [
            {'filename': 'sales.csv', 'formats': ['csv', 'json']},
            {'filename': 'customers.csv'}
        ]
        
        results = create_batch(datasets, formats=['csv'])  # Global default
        
        assert len(results) == 2
        
        # First dataset should have multiple files
        assert ', ' in results[0]  # Multiple paths
        # Second dataset should have single file
        assert ', ' not in results[1]
    
    def test_create_batch_reproducible_with_seed(self):
        """Test batch generation is reproducible with fixed seed"""
        datasets = [
            {'filename': 'sales1.csv'},
            {'filename': 'customers1.csv'}
        ]
        
        results1 = create_batch(datasets, seed=12345)
        
        # Generate again with same seed but different filenames
        datasets2 = [
            {'filename': 'sales2.csv'},
            {'filename': 'customers2.csv'}
        ]
        
        results2 = create_batch(datasets2, seed=12345)
        
        # Compare data content (should be identical)
        df1_sales = pd.read_csv(results1[0])
        df2_sales = pd.read_csv(results2[0])
        df1_customers = pd.read_csv(results1[1])
        df2_customers = pd.read_csv(results2[1])
        
        pd.testing.assert_frame_equal(df1_sales, df2_sales)
        pd.testing.assert_frame_equal(df1_customers, df2_customers)
    
    def test_create_batch_invalid_datasets_param(self):
        """Test error handling for invalid datasets parameter"""
        with pytest.raises(ValueError, match="datasets must be a non-empty list"):
            create_batch([])
        
        with pytest.raises(ValueError, match="datasets must be a non-empty list"):
            create_batch(None)
        
        with pytest.raises(ValueError, match="Cannot process more than 50 datasets"):
            create_batch([{'filename': f'test{i}.csv'} for i in range(51)])
    
    def test_create_batch_invalid_dataset_spec(self):
        """Test error handling for invalid dataset specifications"""
        with pytest.raises(ValueError, match="Dataset 0 must be a dictionary"):
            create_batch(['invalid'])
        
        with pytest.raises(ValueError, match="Dataset 0 must have a 'filename' key"):
            create_batch([{'rows': 100}])
        
        with pytest.raises(ValueError, match="Dataset 0 filename must be a non-empty string"):
            create_batch([{'filename': ''}])


class TestDatasetTypeExtraction:
    """Test cases for dataset type extraction from filenames"""
    
    def test_extract_dataset_type_sales(self):
        """Test sales dataset type extraction"""
        assert _extract_dataset_type('sales.csv') == 'sales'
        assert _extract_dataset_type('transaction_data.csv') == 'sales'
        assert _extract_dataset_type('sales_report.json') == 'sales'
    
    def test_extract_dataset_type_customers(self):
        """Test customers dataset type extraction"""
        assert _extract_dataset_type('customers.csv') == 'customers'
        assert _extract_dataset_type('client_data.csv') == 'customers'
        assert _extract_dataset_type('customer_info.json') == 'customers'
    
    def test_extract_dataset_type_ecommerce(self):
        """Test ecommerce dataset type extraction"""
        assert _extract_dataset_type('ecommerce.csv') == 'ecommerce'
        assert _extract_dataset_type('orders.csv') == 'ecommerce'
        assert _extract_dataset_type('shop_data.csv') == 'ecommerce'
    
    def test_extract_dataset_type_financial(self):
        """Test financial dataset type extraction"""
        assert _extract_dataset_type('stocks.csv') == 'stocks'
        assert _extract_dataset_type('market_data.csv') == 'stocks'
        assert _extract_dataset_type('banking.csv') == 'banking'
        assert _extract_dataset_type('accounts.csv') == 'banking'
    
    def test_extract_dataset_type_healthcare(self):
        """Test healthcare dataset type extraction"""
        assert _extract_dataset_type('patients.csv') == 'patients'
        assert _extract_dataset_type('medical_records.csv') == 'patients'
        assert _extract_dataset_type('appointments.csv') == 'appointments'
        assert _extract_dataset_type('schedule.csv') == 'appointments'
    
    def test_extract_dataset_type_technology(self):
        """Test technology dataset type extraction"""
        assert _extract_dataset_type('web_analytics.csv') == 'web_analytics'
        assert _extract_dataset_type('analytics.csv') == 'web_analytics'
        assert _extract_dataset_type('system_logs.csv') == 'system_logs'
        assert _extract_dataset_type('logs.csv') == 'system_logs'
    
    def test_extract_dataset_type_iot(self):
        """Test IoT dataset type extraction"""
        assert _extract_dataset_type('weather.csv') == 'weather'
        assert _extract_dataset_type('climate_data.csv') == 'weather'
        assert _extract_dataset_type('energy.csv') == 'energy'
        assert _extract_dataset_type('power_consumption.csv') == 'energy'
    
    def test_extract_dataset_type_social(self):
        """Test social dataset type extraction"""
        assert _extract_dataset_type('social_media.csv') == 'social_media'
        assert _extract_dataset_type('posts.csv') == 'social_media'
        assert _extract_dataset_type('user_profiles.csv') == 'user_profiles'
        assert _extract_dataset_type('profiles.csv') == 'user_profiles'
    
    def test_extract_dataset_type_fallback(self):
        """Test fallback behavior for unknown types"""
        assert _extract_dataset_type('unknown_type.csv') == 'unknown_type'
        assert _extract_dataset_type('custom_data.csv') == 'custom_data'


class TestTimeSeriesValidation:
    """Test cases for time series parameter validation"""
    
    def test_validate_time_series_params_valid_dates(self):
        """Test validation with valid date parameters"""
        params = {
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'interval': '1day'
        }
        
        # Should not raise any exception
        _validate_time_series_params(params)
    
    def test_validate_time_series_params_datetime_objects(self):
        """Test validation with datetime objects"""
        from datetime import datetime
        
        params = {
            'start_date': datetime(2023, 1, 1),
            'end_date': datetime(2023, 12, 31),
            'interval': '1hour'
        }
        
        # Should not raise any exception
        _validate_time_series_params(params)
    
    def test_validate_time_series_params_invalid_date_format(self):
        """Test validation with invalid date formats"""
        params = {
            'start_date': 'invalid-date',
            'end_date': '2023-12-31',
            'interval': '1day'
        }
        
        with pytest.raises(ValueError, match="start_date must be in ISO format"):
            _validate_time_series_params(params)
    
    def test_validate_time_series_params_invalid_date_type(self):
        """Test validation with invalid date types"""
        params = {
            'start_date': 123,
            'end_date': '2023-12-31',
            'interval': '1day'
        }
        
        with pytest.raises(ValueError, match="start_date must be a datetime object"):
            _validate_time_series_params(params)
    
    def test_validate_time_series_params_invalid_interval(self):
        """Test validation with invalid interval"""
        params = {
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'interval': 'invalid'
        }
        
        with pytest.raises(ValueError, match="interval must be one of"):
            _validate_time_series_params(params)
    
    def test_validate_time_series_params_invalid_date_range(self):
        """Test validation with invalid date range"""
        params = {
            'start_date': '2023-12-31',
            'end_date': '2023-01-01',
            'interval': '1day'
        }
        
        with pytest.raises(ValueError, match="start_date must be before end_date"):
            _validate_time_series_params(params)


class TestGeoFunctions:
    """Test cases for geo module functions"""
    
    def test_geo_addresses_basic(self):
        """Test basic address generation"""
        addresses = geo.addresses('united_states', count=5)
        
        assert isinstance(addresses, list)
        assert len(addresses) == 5
        
        for address in addresses:
            assert isinstance(address, dict)
            assert 'street' in address
            assert 'city' in address
            assert 'country' in address
    
    def test_geo_addresses_different_countries(self):
        """Test address generation for different countries"""
        us_addresses = geo.addresses('united_states', count=2)
        pk_addresses = geo.addresses('pakistan', count=2)
        
        assert len(us_addresses) == 2
        assert len(pk_addresses) == 2
        
        # Addresses should be different for different countries
        assert us_addresses != pk_addresses
    
    def test_geo_route_basic(self):
        """Test basic route generation"""
        route = geo.route('New York', 'Los Angeles')
        
        assert isinstance(route, dict)
        assert 'start_point' in route or 'distance_km' in route
    
    def test_geo_route_with_waypoints(self):
        """Test route generation with waypoints"""
        route = geo.route('New York', 'Los Angeles', waypoints=2)
        
        assert isinstance(route, dict)
        # Should include waypoint information
        assert 'waypoints' in route or 'distance_km' in route


if __name__ == '__main__':
    pytest.main([__file__])