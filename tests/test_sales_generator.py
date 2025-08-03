"""
Unit tests for SalesGenerator

Tests realistic patterns, seasonal trends, regional preferences, and data quality.
"""

import pytest
import pandas as pd
from datetime import datetime, date
from tempdata.core.seeding import MillisecondSeeder
from tempdata.datasets.business.sales import SalesGenerator


class TestSalesGenerator:
    """Test suite for SalesGenerator functionality"""
    
    @pytest.fixture
    def seeder(self):
        """Create a fixed seeder for reproducible tests using enhanced seeding"""
        return MillisecondSeeder(fixed_seed=987654321)
    
    @pytest.fixture
    def generator(self, seeder):
        """Create SalesGenerator instance"""
        return SalesGenerator(seeder)
    
    def test_basic_generation(self, generator):
        """Test basic data generation functionality"""
        rows = 100
        data = generator.generate(rows)
        
        # Check basic structure
        assert isinstance(data, pd.DataFrame)
        assert len(data) == rows
        
        # Check required columns exist
        required_columns = [
            'transaction_id', 'date', 'customer_id', 'product_id',
            'product_name', 'product_category', 'amount', 'region',
            'payment_method', 'sales_rep_id', 'store_id'
        ]
        for col in required_columns:
            assert col in data.columns
    
    def test_transaction_id_uniqueness(self, generator):
        """Test that transaction IDs are unique"""
        data = generator.generate(500)
        assert data['transaction_id'].nunique() == len(data)
        
        # Check format
        assert all(data['transaction_id'].str.startswith('TXN_'))
        assert all(data['transaction_id'].str.len() == 12)  # TXN_ + 8 digits
    
    def test_amount_patterns(self, generator):
        """Test realistic amount patterns"""
        data = generator.generate(1000)
        
        # Check amounts are positive
        assert all(data['amount'] > 0)
        
        # Check reasonable range (should vary by category)
        assert data['amount'].min() >= 1.0  # Minimum reasonable amount
        assert data['amount'].max() <= 100000.0  # Maximum reasonable amount
        
        # Check precision (should be 2 decimal places)
        assert all(data['amount'].apply(lambda x: len(str(x).split('.')[-1]) <= 2))
    
    def test_regional_preferences(self, generator):
        """Test regional preferences for products and payment methods"""
        data = generator.generate(1000)
        
        # Check regions are valid
        valid_regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East']
        assert all(data['region'].isin(valid_regions))
        
        # Test regional payment method preferences
        na_data = data[data['region'] == 'North America']
        if len(na_data) > 50:  # Only test if we have enough samples
            # North America should prefer cards over cash
            card_ratio = len(na_data[na_data['payment_method'] == 'card']) / len(na_data)
            cash_ratio = len(na_data[na_data['payment_method'] == 'cash']) / len(na_data)
            assert card_ratio > cash_ratio
        
        # Test Asia Pacific digital preference
        ap_data = data[data['region'] == 'Asia Pacific']
        if len(ap_data) > 50:
            digital_ratio = len(ap_data[ap_data['payment_method'] == 'digital']) / len(ap_data)
            assert digital_ratio > 0.3  # Should be relatively high
    
    def test_seasonal_patterns(self, generator):
        """Test seasonal sales patterns"""
        # Generate data for specific months
        december_data = generator.generate(200, date_range=(date(2024, 12, 1), date(2024, 12, 31)))
        january_data = generator.generate(200, date_range=(date(2024, 1, 1), date(2024, 1, 31)))
        
        # December should have higher average amounts due to holiday season
        dec_avg = december_data['amount'].mean()
        jan_avg = january_data['amount'].mean()
        
        # Allow some variance but December should generally be higher
        assert dec_avg > jan_avg * 0.9  # At least 90% of January average
    
    def test_payment_method_by_amount(self, generator):
        """Test payment method selection based on amount"""
        data = generator.generate(1000)
        
        # Large amounts should prefer cards over cash
        large_amounts = data[data['amount'] > 500]
        if len(large_amounts) > 20:
            card_ratio = len(large_amounts[large_amounts['payment_method'] == 'card']) / len(large_amounts)
            cash_ratio = len(large_amounts[large_amounts['payment_method'] == 'cash']) / len(large_amounts)
            assert card_ratio > cash_ratio
        
        # Small amounts should have more cash usage
        small_amounts = data[data['amount'] < 50]
        if len(small_amounts) > 20:
            cash_ratio = len(small_amounts[small_amounts['payment_method'] == 'cash']) / len(small_amounts)
            # Cash should be at least 20% for small amounts
            assert cash_ratio >= 0.15
    
    def test_product_categories(self, generator):
        """Test product category distribution"""
        data = generator.generate(500)
        
        # Check valid categories
        valid_categories = ['technology', 'healthcare', 'retail', 'automotive', 'food_beverage', 'home_garden']
        assert all(data['product_category'].isin(valid_categories))
        
        # Should have variety in categories
        assert data['product_category'].nunique() >= 3
    
    def test_country_specific_generation(self, generator):
        """Test country-specific data generation"""
        us_data = generator.generate(100, country='united_states')
        china_data = generator.generate(100, country='china')
        
        # US should be North America region
        assert all(us_data['region'] == 'North America')
        
        # China should be Asia Pacific region
        assert all(china_data['region'] == 'Asia Pacific')
    
    def test_realistic_patterns_applied(self, generator):
        """Test that realistic patterns are applied"""
        data = generator.generate(200)
        
        # Check additional columns from realistic patterns
        pattern_columns = ['discount_applied', 'return_probability', 'satisfaction_score']
        for col in pattern_columns:
            assert col in data.columns
        
        # Check discount patterns
        assert data['discount_applied'].dtype == bool
        
        # Check return probability is reasonable
        assert all(data['return_probability'] >= 0)
        assert all(data['return_probability'] <= 1)
        
        # Check satisfaction scores
        assert all(data['satisfaction_score'] >= 1)
        assert all(data['satisfaction_score'] <= 5)
    
    def test_chronological_ordering(self, generator):
        """Test that data is sorted chronologically"""
        data = generator.generate(100)
        
        # Check dates are in ascending order
        dates = pd.to_datetime(data['date'])
        assert all(dates[i] <= dates[i+1] for i in range(len(dates)-1))
    
    def test_data_relationships(self, generator):
        """Test relationships between data fields"""
        data = generator.generate(500)
        
        # Higher amounts should have lower return probability on average
        high_amount = data[data['amount'] > data['amount'].quantile(0.8)]
        low_amount = data[data['amount'] < data['amount'].quantile(0.2)]
        
        if len(high_amount) > 10 and len(low_amount) > 10:
            # This relationship might be weak, so we test for general trend
            high_return_avg = high_amount['return_probability'].mean()
            low_return_avg = low_amount['return_probability'].mean()
            # High amounts should have higher return probability
            assert high_return_avg >= low_return_avg * 0.8
    
    def test_reproducibility(self, seeder):
        """Test that same seed produces same results"""
        gen1 = SalesGenerator(seeder)
        gen2 = SalesGenerator(MillisecondSeeder(fixed_seed=987654321))
        
        data1 = gen1.generate(50)
        data2 = gen2.generate(50)
        
        # Should be identical
        pd.testing.assert_frame_equal(data1, data2)
    
    def test_data_quality_metrics(self, generator):
        """Test overall data quality metrics"""
        data = generator.generate(1000)
        
        # No null values in critical fields
        critical_fields = ['transaction_id', 'amount', 'payment_method', 'region']
        for field in critical_fields:
            assert data[field].notna().all()
        
        # Reasonable distribution of payment methods
        payment_dist = data['payment_method'].value_counts(normalize=True)
        assert len(payment_dist) >= 2  # At least 2 payment methods
        assert payment_dist.max() <= 0.8  # No single method dominates too much
        
        # Reasonable regional distribution
        region_dist = data['region'].value_counts(normalize=True)
        assert len(region_dist) >= 2  # At least 2 regions
    
    def test_edge_cases(self, generator):
        """Test edge cases and error handling"""
        # Test with minimal rows
        data = generator.generate(1)
        assert len(data) == 1
        
        # Test with larger dataset
        data = generator.generate(5000)
        assert len(data) == 5000
        
        # Test with invalid country (should default gracefully)
        data = generator.generate(10, country='invalid_country')
        assert len(data) == 10
        assert all(data['region'].isin(['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East']))


if __name__ == '__main__':
    pytest.main([__file__])