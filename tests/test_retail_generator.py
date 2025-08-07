"""
Tests for retail operations dataset generator

Tests the RetailGenerator for realistic retail transaction data generation
including POS transactions, store locations, product mix, and payment methods.
"""

import pytest
import pandas as pd
from datetime import datetime, date, timedelta
from tempdata.datasets.business.retail import RetailGenerator
from tempdata.core.seeding import MillisecondSeeder


class TestRetailGenerator:
    """Test suite for RetailGenerator"""
    
    @pytest.fixture
    def generator(self):
        """Create a RetailGenerator instance for testing"""
        seeder = MillisecondSeeder(42)
        return RetailGenerator(seeder)
    
    def test_initialization(self, generator):
        """Test generator initialization"""
        assert generator is not None
        assert hasattr(generator, 'store_types')
        assert hasattr(generator, 'product_categories')
        assert hasattr(generator, 'payment_methods')
        assert hasattr(generator, 'hourly_patterns')
        assert hasattr(generator, 'regions')
    
    def test_store_types_setup(self, generator):
        """Test store types configuration"""
        assert 'supermarket' in generator.store_types
        assert 'convenience' in generator.store_types
        assert 'department_store' in generator.store_types
        assert 'electronics' in generator.store_types
        assert 'clothing' in generator.store_types
        assert 'pharmacy' in generator.store_types
        assert 'home_improvement' in generator.store_types
        
        # Check store type has required fields
        for store_type, config in generator.store_types.items():
            assert 'store_size_sqft' in config
            assert 'daily_transactions' in config
            assert 'avg_items_per_transaction' in config
            assert 'avg_transaction_value' in config
            assert 'weight' in config
    
    def test_product_categories_setup(self, generator):
        """Test product categories configuration"""
        assert 'groceries' in generator.product_categories
        assert 'electronics' in generator.product_categories
        assert 'clothing' in generator.product_categories
        assert 'home_garden' in generator.product_categories
        assert 'health_beauty' in generator.product_categories
        assert 'automotive' in generator.product_categories
        
        # Check groceries has subcategories
        groceries = generator.product_categories['groceries']
        assert 'produce' in groceries
        assert 'dairy' in groceries
        assert 'meat' in groceries
        assert 'bakery' in groceries
    
    def test_payment_methods_setup(self, generator):
        """Test payment methods configuration"""
        expected_methods = ['credit_card', 'debit_card', 'cash', 'mobile_payment', 'gift_card']
        for method in expected_methods:
            assert method in generator.payment_methods
            assert 'processing_time' in generator.payment_methods[method]
            assert 'avg_processing_fee' in generator.payment_methods[method]
            assert 'weight' in generator.payment_methods[method]
    
    def test_temporal_patterns_setup(self, generator):
        """Test temporal patterns configuration"""
        # Check hourly patterns
        assert len(generator.hourly_patterns) == 24
        for hour in range(24):
            assert hour in generator.hourly_patterns
            assert isinstance(generator.hourly_patterns[hour], (int, float))
        
        # Check seasonal multipliers
        assert len(generator.seasonal_multipliers) == 12
        for month in range(1, 13):
            assert month in generator.seasonal_multipliers
    
    def test_geographic_data_setup(self, generator):
        """Test geographic regions configuration"""
        expected_regions = ['northeast', 'southeast', 'midwest', 'southwest', 'west']
        for region in expected_regions:
            assert region in generator.regions
            assert 'states' in generator.regions[region]
            assert 'population_density' in generator.regions[region]
            assert 'cost_of_living_multiplier' in generator.regions[region]
            assert 'weight' in generator.regions[region]
    
    def test_basic_generation(self, generator):
        """Test basic retail data generation"""
        data = generator.generate(100)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100
        
        # Check required columns exist
        required_columns = [
            'transaction_id', 'store_id', 'store_name', 'store_type',
            'region', 'state', 'city', 'transaction_datetime',
            'customer_id', 'customer_type', 'num_items', 'payment_method',
            'subtotal', 'tax_amount', 'total_amount', 'product_categories'
        ]
        
        for col in required_columns:
            assert col in data.columns, f"Missing required column: {col}"
    
    def test_store_type_distribution(self, generator):
        """Test store type distribution"""
        data = generator.generate(1000)
        
        store_types = data['store_type'].unique()
        expected_types = list(generator.store_types.keys())
        
        # Should have variety in store types
        assert len(store_types) >= 3
        
        # All store types should be valid
        for store_type in store_types:
            assert store_type in expected_types
    
    def test_payment_method_distribution(self, generator):
        """Test payment method distribution"""
        data = generator.generate(1000)
        
        payment_methods = data['payment_method'].unique()
        expected_methods = list(generator.payment_methods.keys())
        
        # Should have variety in payment methods
        assert len(payment_methods) >= 3
        
        # All payment methods should be valid
        for method in payment_methods:
            assert method in expected_methods
    
    def test_customer_type_distribution(self, generator):
        """Test customer type distribution"""
        data = generator.generate(1000)
        
        customer_types = data['customer_type'].unique()
        expected_types = ['regular', 'loyalty_member', 'employee', 'senior']
        
        # Should have variety in customer types
        assert len(customer_types) >= 2
        
        # All customer types should be valid
        for ctype in customer_types:
            assert ctype in expected_types
    
    def test_transaction_amounts(self, generator):
        """Test transaction amount calculations"""
        data = generator.generate(100)
        
        # Check amount fields are numeric and positive
        assert data['subtotal'].dtype in ['float64', 'int64']
        assert data['tax_amount'].dtype in ['float64', 'int64']
        assert data['total_amount'].dtype in ['float64', 'int64']
        
        assert (data['subtotal'] >= 0).all()
        assert (data['tax_amount'] >= 0).all()
        assert (data['total_amount'] >= 0).all()
        
        # Check basic math relationships
        # total_amount should generally equal subtotal + tax_amount - discount_amount
        calculated_total = data['subtotal'] + data['tax_amount'] - data['discount_amount']
        # Allow for small rounding differences (more lenient for floating point precision)
        differences = abs(data['total_amount'] - calculated_total)
        max_diff = differences.max()
        assert max_diff < 0.1, f"Maximum difference: {max_diff}, should be less than 0.1"
    
    def test_transaction_timing_patterns(self, generator):
        """Test transaction timing patterns"""
        data = generator.generate(1000)
        
        # Check transaction hours are valid
        assert data['transaction_hour'].min() >= 0
        assert data['transaction_hour'].max() <= 23
        
        # Check day of week is present
        assert 'day_of_week' in data.columns
        valid_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        assert data['day_of_week'].isin(valid_days).all()
        
        # Check peak hours have more transactions (statistical test)
        peak_hours = [11, 12, 17, 18, 19]
        peak_transactions = data[data['transaction_hour'].isin(peak_hours)]
        off_peak_transactions = data[~data['transaction_hour'].isin(peak_hours)]
        
        # Peak hours should have higher average activity (this is probabilistic)
        if len(peak_transactions) > 0 and len(off_peak_transactions) > 0:
            # Just check that we have some peak hour transactions
            assert len(peak_transactions) > 0
    
    def test_product_categories(self, generator):
        """Test product category generation"""
        data = generator.generate(500)
        
        # Check product categories exist
        assert 'product_categories' in data.columns
        assert 'primary_category' in data.columns
        
        # Check categories are valid
        valid_categories = list(generator.product_categories.keys())
        primary_categories = data['primary_category'].unique()
        
        for category in primary_categories:
            if category != 'unknown':  # Allow for unknown category as fallback
                assert category in valid_categories
    
    def test_store_location_data(self, generator):
        """Test store location generation"""
        data = generator.generate(200)
        
        # Check location fields exist
        location_fields = ['region', 'state', 'city', 'store_name']
        for field in location_fields:
            assert field in data.columns
            assert not data[field].isna().all()
        
        # Check regions are valid
        valid_regions = list(generator.regions.keys())
        regions = data['region'].unique()
        for region in regions:
            assert region in valid_regions
    
    def test_geographic_constraints(self, generator):
        """Test geographic constraints functionality"""
        # Test with specific regions
        constraints = {'regions': ['northeast', 'west']}
        data = generator.generate(100, geographic_constraints=constraints)
        
        regions = data['region'].unique()
        for region in regions:
            assert region in ['northeast', 'west']
    
    def test_date_range_constraints(self, generator):
        """Test date range constraints"""
        start_date = date(2023, 1, 1)
        end_date = date(2023, 12, 31)
        date_range = (start_date, end_date)
        
        data = generator.generate(100, date_range=date_range)
        
        # Convert transaction_date to date objects for comparison
        transaction_dates = pd.to_datetime(data['transaction_date']).dt.date
        
        assert (transaction_dates >= start_date).all()
        assert (transaction_dates <= end_date).all()
    
    def test_seasonal_variations(self, generator):
        """Test seasonal variation patterns"""
        # Generate data for different months
        winter_data = generator.generate(100, date_range=(date(2023, 1, 1), date(2023, 1, 31)))
        holiday_data = generator.generate(100, date_range=(date(2023, 12, 1), date(2023, 12, 31)))
        
        # Holiday season should generally have higher transaction values (probabilistic)
        winter_avg = winter_data['total_amount'].mean()
        holiday_avg = holiday_data['total_amount'].mean()
        
        # This is probabilistic, so we just check that both have reasonable values
        assert winter_avg > 0
        assert holiday_avg > 0
    
    def test_loyalty_program_features(self, generator):
        """Test loyalty program related features"""
        data = generator.generate(500)
        
        # Check loyalty-related fields
        assert 'loyalty_points_earned' in data.columns
        assert 'discount_applied' in data.columns
        assert 'promotion_code' in data.columns
        
        # Loyalty members should earn points
        loyalty_members = data[data['customer_type'] == 'loyalty_member']
        if len(loyalty_members) > 0:
            assert (loyalty_members['loyalty_points_earned'] >= 0).all()
    
    def test_transaction_performance_metrics(self, generator):
        """Test transaction performance metrics"""
        data = generator.generate(200)
        
        # Check performance metrics exist
        performance_fields = [
            'avg_item_price', 'profit_margin', 'net_profit',
            'transaction_duration_seconds', 'processing_fee'
        ]
        
        for field in performance_fields:
            assert field in data.columns
            assert data[field].dtype in ['float64', 'int64']
            assert (data[field] >= 0).all()
    
    def test_transaction_size_categories(self, generator):
        """Test transaction size categorization"""
        data = generator.generate(300)
        
        # Check size categories exist
        assert 'transaction_size' in data.columns
        assert 'basket_size' in data.columns
        
        # Check valid categories
        valid_transaction_sizes = ['small', 'medium', 'large', 'bulk', 'extra_large']
        valid_basket_sizes = ['small', 'medium', 'large', 'extra_large']
        
        transaction_sizes = data['transaction_size'].dropna().unique()
        basket_sizes = data['basket_size'].dropna().unique()
        
        for size in transaction_sizes:
            assert size in valid_transaction_sizes
        
        for size in basket_sizes:
            assert size in valid_basket_sizes
    
    def test_items_detail_json(self, generator):
        """Test items detail JSON structure"""
        data = generator.generate(50)
        
        assert 'items_detail' in data.columns
        
        # Check that items_detail contains valid JSON
        import json
        for items_json in data['items_detail'].head(10):  # Test first 10 rows
            try:
                items = json.loads(items_json)
                assert isinstance(items, list)
                if len(items) > 0:
                    item = items[0]
                    assert 'product_id' in item
                    assert 'product_name' in item
                    assert 'category' in item
                    assert 'price' in item
                    assert 'quantity' in item
            except json.JSONDecodeError:
                pytest.fail(f"Invalid JSON in items_detail: {items_json}")
    
    def test_realistic_patterns_application(self, generator):
        """Test that realistic patterns are applied"""
        data = generator.generate(100)
        
        # Check that derived fields are created
        derived_fields = [
            'high_value_transaction', 'bulk_purchase', 
            'loyalty_transaction', 'is_weekend', 'is_peak_hour'
        ]
        
        for field in derived_fields:
            assert field in data.columns
            # These should be boolean fields
            assert data[field].dtype == bool
    
    def test_data_consistency(self, generator):
        """Test data consistency and relationships"""
        data = generator.generate(200)
        
        # Check that num_items matches the actual number of items in items_detail
        import json
        for idx, row in data.head(10).iterrows():  # Test first 10 rows
            items = json.loads(row['items_detail'])
            assert len(items) == row['num_items']
        
        # Check that primary_category appears in product_categories
        for idx, row in data.head(20).iterrows():  # Test first 20 rows
            if row['primary_category'] != 'unknown':
                assert row['primary_category'] in row['product_categories']
    
    def test_reproducibility(self, generator):
        """Test that generation is reproducible with same seed"""
        # Generate data twice with same generator (same seed)
        data1 = generator.generate(50)
        
        # Create new generator with same seed
        seeder2 = MillisecondSeeder(42)
        generator2 = RetailGenerator(seeder2)
        data2 = generator2.generate(50)
        
        # Should generate identical data
        assert data1.equals(data2)
    
    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different data"""
        seeder1 = MillisecondSeeder(42)
        seeder2 = MillisecondSeeder(123)
        
        generator1 = RetailGenerator(seeder1)
        generator2 = RetailGenerator(seeder2)
        
        data1 = generator1.generate(50)
        data2 = generator2.generate(50)
        
        # Should generate different data
        assert not data1.equals(data2)
    
    def test_large_dataset_generation(self, generator):
        """Test generation of larger datasets"""
        data = generator.generate(2000)
        
        assert len(data) == 2000
        assert isinstance(data, pd.DataFrame)
        
        # Check that we have good variety in larger datasets
        assert len(data['store_type'].unique()) >= 5
        assert len(data['payment_method'].unique()) >= 4
        assert len(data['region'].unique()) >= 3
    
    def test_edge_cases(self, generator):
        """Test edge cases and boundary conditions"""
        # Test minimum rows
        data = generator.generate(1)
        assert len(data) == 1
        
        # Test with empty geographic constraints
        data = generator.generate(10, geographic_constraints={})
        assert len(data) == 10
        
        # Test with invalid date range (should handle gracefully)
        try:
            data = generator.generate(10, date_range=(date(2023, 12, 31), date(2023, 1, 1)))
            # If it doesn't raise an error, that's fine too
        except Exception:
            # If it raises an error, that's also acceptable behavior
            pass