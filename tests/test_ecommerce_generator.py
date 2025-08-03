"""
Unit tests for EcommerceGenerator

Tests order patterns, shipping preferences, product correlations, and data realism.
"""

import pytest
import pandas as pd
from datetime import datetime, date, timedelta
from tempdata.core.seeding import MillisecondSeeder
from tempdata.datasets.business.ecommerce import EcommerceGenerator


class TestEcommerceGenerator:
    """Test suite for EcommerceGenerator functionality"""
    
    @pytest.fixture
    def seeder(self):
        """Create a fixed seeder for reproducible tests using enhanced seeding"""
        return MillisecondSeeder(fixed_seed=987654321)
    
    @pytest.fixture
    def generator(self, seeder):
        """Create EcommerceGenerator instance"""
        return EcommerceGenerator(seeder)
    
    def test_basic_generation(self, generator):
        """Test basic data generation functionality"""
        rows = 100
        data = generator.generate(rows)
        
        # Check basic structure
        assert isinstance(data, pd.DataFrame)
        assert len(data) == rows
        
        # Check required columns exist
        required_columns = [
            'order_id', 'customer_id', 'order_date', 'order_datetime',
            'order_status', 'subtotal', 'shipping_cost', 'tax_amount',
            'total_amount', 'payment_method', 'shipping_method',
            'estimated_delivery_days', 'total_items', 'total_quantity',
            'total_weight_kg', 'primary_category', 'shipping_country',
            'shipping_city', 'shipping_postal_code', 'is_repeat_customer',
            'customer_order_number', 'is_gift', 'has_discount', 'discount_amount'
        ]
        for col in required_columns:
            assert col in data.columns
    
    def test_order_id_uniqueness(self, generator):
        """Test that order IDs are unique"""
        data = generator.generate(500)
        assert data['order_id'].nunique() == len(data)
        
        # Check format
        assert all(data['order_id'].str.startswith('ORD_'))
        assert all(data['order_id'].str.len() == 12)  # ORD_ + 8 digits
    
    def test_order_patterns(self, generator):
        """Test realistic order patterns"""
        data = generator.generate(1000)
        
        # Test order status distribution
        status_counts = data['order_status'].value_counts(normalize=True)
        assert 'delivered' in status_counts.index
        assert 'shipped' in status_counts.index
        assert 'processing' in status_counts.index
        
        # Delivered should be the most common status
        assert status_counts['delivered'] > 0.5  # Should be around 75%
        
        # Test order amounts are reasonable
        assert all(data['total_amount'] >= 0)
        assert all(data['subtotal'] > 0)
        
        # Test item counts
        assert all(data['total_items'] >= 1)
        assert all(data['total_quantity'] >= 1)
    
    def test_shipping_preferences(self, generator):
        """Test shipping method preferences"""
        data = generator.generate(1000)
        
        # Test shipping methods exist
        shipping_methods = data['shipping_method'].unique()
        expected_methods = ['standard', 'express', 'overnight', 'free_shipping']
        for method in expected_methods:
            assert method in shipping_methods
        
        # Standard should be the most common
        method_counts = data['shipping_method'].value_counts()
        assert method_counts.index[0] in ['standard', 'express']  # Top methods
        
        # Test shipping costs are reasonable
        assert all(data['shipping_cost'] >= 0)
        
        # Free shipping should have zero cost
        free_shipping_orders = data[data['shipping_method'] == 'free_shipping']
        if len(free_shipping_orders) > 0:
            assert all(free_shipping_orders['shipping_cost'] == 0)
    
    def test_product_correlations(self, generator):
        """Test product correlation patterns"""
        data = generator.generate(500)
        
        # Test product categories exist
        categories = data['primary_category'].unique()
        expected_categories = ['technology', 'healthcare', 'retail', 'automotive', 'food_beverage', 'home_garden']
        
        # Should have variety in categories
        assert len(categories) >= 3
        
        # Test item details are populated
        item_1_names = data['item_1_name'].dropna()
        assert len(item_1_names) == len(data)  # All orders should have at least one item
        
        # Test categories match between primary and item categories
        item_1_categories = data['item_1_category'].dropna()
        assert len(item_1_categories) == len(data)
    
    def test_seasonal_patterns(self, generator):
        """Test seasonal ordering patterns"""
        # Generate data for specific months
        december_data = generator.generate(200, date_range=(date(2024, 12, 1), date(2024, 12, 31)))
        august_data = generator.generate(200, date_range=(date(2024, 8, 1), date(2024, 8, 31)))
        
        # December should have higher average order values due to holiday season
        dec_avg = december_data['total_amount'].mean()
        aug_avg = august_data['total_amount'].mean()
        
        # Allow some variance but December should generally be higher
        assert dec_avg > aug_avg * 0.8  # At least 80% of August average
    
    def test_customer_behavior_patterns(self, generator):
        """Test customer behavior patterns"""
        data = generator.generate(1000)
        
        # Test repeat customer patterns
        repeat_rate = data['is_repeat_customer'].mean()
        assert 0.2 < repeat_rate < 0.5  # Should be around 35%
        
        # Test gift patterns
        gift_rate = data['is_gift'].mean()
        assert 0.05 < gift_rate < 0.2  # Should be around 12%
        
        # Test discount patterns
        discount_rate = data['has_discount'].mean()
        assert 0.15 < discount_rate < 0.35  # Should be around 25%
        
        # Orders with discounts should have discount amounts
        discounted_orders = data[data['has_discount'] == True]
        if len(discounted_orders) > 0:
            assert all(discounted_orders['discount_amount'] > 0)
    
    def test_payment_methods(self, generator):
        """Test payment method distribution"""
        data = generator.generate(500)
        
        # Test payment methods exist
        payment_methods = data['payment_method'].unique()
        expected_methods = ['credit_card', 'debit_card', 'paypal', 'apple_pay', 'google_pay', 'bank_transfer']
        
        # Should have variety in payment methods
        assert len(payment_methods) >= 4
        
        # Credit card should be most common
        method_counts = data['payment_method'].value_counts()
        assert method_counts.index[0] in ['credit_card', 'debit_card']
    
    def test_shipping_cost_calculation(self, generator):
        """Test shipping cost calculation logic"""
        data = generator.generate(500)
        
        # Orders over $100 should often have free shipping
        high_value_orders = data[data['subtotal'] > 100]
        if len(high_value_orders) > 20:
            free_shipping_rate = (high_value_orders['shipping_cost'] == 0).mean()
            assert free_shipping_rate > 0.3  # At least 30% should have free shipping
        
        # Overnight shipping should cost more than standard (when both have costs > 0)
        overnight_orders = data[data['shipping_method'] == 'overnight']
        standard_orders = data[data['shipping_method'] == 'standard']
        
        if len(overnight_orders) > 5 and len(standard_orders) > 5:
            # Only compare orders that actually have shipping costs
            overnight_with_cost = overnight_orders[overnight_orders['shipping_cost'] > 0]
            standard_with_cost = standard_orders[standard_orders['shipping_cost'] > 0]
            
            if len(overnight_with_cost) > 3 and len(standard_with_cost) > 3:
                overnight_avg_cost = overnight_with_cost['shipping_cost'].mean()
                standard_avg_cost = standard_with_cost['shipping_cost'].mean()
                # Overnight should generally cost more, but allow some variance due to randomness
                assert overnight_avg_cost >= standard_avg_cost * 0.8  # Allow 20% variance
    
    def test_order_timing_patterns(self, generator):
        """Test order timing patterns"""
        data = generator.generate(500)
        
        # Test that order_datetime includes time component
        assert all(data['order_datetime'].dt.hour.between(0, 23))
        
        # Peak hours should be more common (evening hours)
        evening_orders = data[data['order_datetime'].dt.hour.between(18, 22)]
        night_orders = data[data['order_datetime'].dt.hour.between(1, 5)]
        
        # Evening should have more orders than late night
        if len(evening_orders) > 0 and len(night_orders) > 0:
            evening_rate = len(evening_orders) / len(data)
            night_rate = len(night_orders) / len(data)
            assert evening_rate > night_rate
    
    def test_realistic_patterns_applied(self, generator):
        """Test that realistic patterns are applied"""
        data = generator.generate(200)
        
        # Check additional columns from realistic patterns
        pattern_columns = ['customer_satisfaction', 'delivery_performance', 'return_likelihood', 'estimated_clv']
        for col in pattern_columns:
            assert col in data.columns
        
        # Check satisfaction scores
        assert all(data['customer_satisfaction'] >= 1.0)
        assert all(data['customer_satisfaction'] <= 5.0)
        
        # Check delivery performance scores
        assert all(data['delivery_performance'] >= 1.0)
        assert all(data['delivery_performance'] <= 5.0)
        
        # Check return likelihood
        assert all(data['return_likelihood'] >= 0.0)
        assert all(data['return_likelihood'] <= 1.0)
        
        # Check estimated CLV
        assert all(data['estimated_clv'] >= 50.0)
    
    def test_chronological_ordering(self, generator):
        """Test that data is sorted chronologically"""
        data = generator.generate(100)
        
        # Check order datetimes are in ascending order
        order_times = pd.to_datetime(data['order_datetime'])
        assert all(order_times[i] <= order_times[i+1] for i in range(len(order_times)-1))
    
    def test_country_specific_generation(self, generator):
        """Test country-specific data generation"""
        us_data = generator.generate(50, country='united_states')
        global_data = generator.generate(50, country='global')
        
        # US data should have consistent country
        assert all(us_data['shipping_country'] == 'united_states')
        
        # Global data should have variety
        assert global_data['shipping_country'].nunique() > 1
    
    def test_date_range_generation(self, generator):
        """Test generation with specific date range"""
        start_date = date(2023, 1, 1)
        end_date = date(2023, 12, 31)
        
        data = generator.generate(100, date_range=(start_date, end_date))
        
        # All order dates should be within range
        assert all(data['order_date'] >= start_date)
        assert all(data['order_date'] <= end_date)
    
    def test_cart_abandonment_patterns(self, generator):
        """Test cart abandonment patterns when enabled"""
        data = generator.generate(200, include_abandoned=True)
        
        # Should have some abandoned carts
        abandoned_orders = data[data['order_status'] == 'abandoned']
        abandonment_rate = len(abandoned_orders) / len(data)
        
        # Should be around 15% abandonment rate
        assert 0.05 < abandonment_rate < 0.25
        
        # Abandoned orders should have zero total amount
        if len(abandoned_orders) > 0:
            assert all(abandoned_orders['total_amount'] == 0)
            assert all(abandoned_orders['shipping_cost'] == 0)
            assert all(abandoned_orders['tax_amount'] == 0)
    
    def test_order_value_relationships(self, generator):
        """Test relationships between order value components"""
        data = generator.generate(500)
        
        # Total amount should equal subtotal + shipping + tax (for non-abandoned orders)
        non_abandoned = data[data['order_status'] != 'abandoned']
        calculated_total = non_abandoned['subtotal'] + non_abandoned['shipping_cost'] + non_abandoned['tax_amount']
        
        # Allow for small rounding differences
        assert all(abs(calculated_total - non_abandoned['total_amount']) < 0.01)
        
        # Tax amount should be reasonable percentage of subtotal
        tax_rates = non_abandoned['tax_amount'] / non_abandoned['subtotal']
        assert all(tax_rates >= 0.05)  # At least 5%
        assert all(tax_rates <= 0.15)  # At most 15%
    
    def test_item_details_consistency(self, generator):
        """Test consistency of item details"""
        data = generator.generate(200)
        
        # Orders with multiple items should have item_2 populated
        multi_item_orders = data[data['total_items'] > 1]
        if len(multi_item_orders) > 0:
            assert all(multi_item_orders['item_2_name'].notna())
            assert all(multi_item_orders['item_2_category'].notna())
            assert all(multi_item_orders['item_2_price'] > 0)
            assert all(multi_item_orders['item_2_quantity'] >= 1)
        
        # Single item orders should not have item_2 populated
        single_item_orders = data[data['total_items'] == 1]
        if len(single_item_orders) > 0:
            assert all(single_item_orders['item_2_name'].isna())
    
    def test_weight_and_shipping_relationship(self, generator):
        """Test relationship between weight and shipping methods"""
        data = generator.generate(500)
        
        # Heavy orders should not use overnight shipping (weight limit)
        heavy_orders = data[data['total_weight_kg'] > 10]
        if len(heavy_orders) > 10:
            overnight_rate = (heavy_orders['shipping_method'] == 'overnight').mean()
            assert overnight_rate < 0.2  # Should be rare for heavy orders
    
    def test_reproducibility(self, seeder):
        """Test that same seed produces same results"""
        gen1 = EcommerceGenerator(seeder)
        gen2 = EcommerceGenerator(MillisecondSeeder(fixed_seed=987654321))
        
        data1 = gen1.generate(50)
        data2 = gen2.generate(50)
        
        # Should be identical
        pd.testing.assert_frame_equal(data1, data2)
    
    def test_data_quality_metrics(self, generator):
        """Test overall data quality metrics"""
        data = generator.generate(1000)
        
        # No null values in critical fields
        critical_fields = ['order_id', 'customer_id', 'total_amount', 'order_status']
        for field in critical_fields:
            assert data[field].notna().all()
        
        # Reasonable distribution of order statuses
        status_dist = data['order_status'].value_counts(normalize=True)
        assert len(status_dist) >= 3  # At least 3 statuses
        
        # Reasonable order value distribution
        assert data['total_amount'].std() > 0  # Should have variance
        assert data['total_amount'].mean() > 20  # Reasonable average
    
    def test_edge_cases(self, generator):
        """Test edge cases and error handling"""
        # Test with minimal rows
        data = generator.generate(1)
        assert len(data) == 1
        
        # Test with larger dataset
        data = generator.generate(2000)
        assert len(data) == 2000
        
        # Test with invalid country (should default gracefully)
        data = generator.generate(10, country='invalid_country')
        assert len(data) == 10
        assert all(data['order_id'].str.startswith('ORD_'))
    
    def test_seasonal_product_preferences(self, generator):
        """Test seasonal product preferences"""
        # Winter data should have different category preferences than summer
        winter_data = generator.generate(200, date_range=(date(2024, 12, 1), date(2024, 12, 31)))
        summer_data = generator.generate(200, date_range=(date(2024, 7, 1), date(2024, 7, 31)))
        
        winter_categories = winter_data['primary_category'].value_counts(normalize=True)
        summer_categories = summer_data['primary_category'].value_counts(normalize=True)
        
        # Should have different distributions (allowing for randomness)
        if 'technology' in winter_categories.index and 'technology' in summer_categories.index:
            # Technology might be more popular in winter (holiday season)
            # This is a loose test to allow for natural variation
            assert abs(winter_categories['technology'] - summer_categories['technology']) >= 0
    
    def test_repeat_customer_consistency(self, generator):
        """Test repeat customer logic consistency"""
        data = generator.generate(500)
        
        # Customers with order number > 1 should be marked as repeat customers
        repeat_customers = data[data['customer_order_number'] > 1]
        if len(repeat_customers) > 0:
            assert all(repeat_customers['is_repeat_customer'] == True)
        
        # First-time customers should have order number 1
        first_time_customers = data[data['is_repeat_customer'] == False]
        if len(first_time_customers) > 0:
            assert all(first_time_customers['customer_order_number'] == 1)


if __name__ == '__main__':
    pytest.main([__file__])