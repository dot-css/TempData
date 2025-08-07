"""
Unit tests for InventoryGenerator

Tests realistic patterns, stock management data, demand patterns, and inventory relationships.
"""

import pytest
import pandas as pd
from datetime import datetime, date, timedelta
from tempdata.core.seeding import MillisecondSeeder
from tempdata.datasets.business.inventory import InventoryGenerator


class TestInventoryGenerator:
    """Test suite for InventoryGenerator functionality"""
    
    @pytest.fixture
    def seeder(self):
        """Create a fixed seeder for reproducible tests"""
        return MillisecondSeeder(fixed_seed=123456789)
    
    @pytest.fixture
    def generator(self, seeder):
        """Create InventoryGenerator instance"""
        return InventoryGenerator(seeder)
    
    def test_basic_generation(self, generator):
        """Test basic data generation functionality"""
        rows = 100
        data = generator.generate(rows)
        
        # Check basic structure
        assert isinstance(data, pd.DataFrame)
        assert len(data) == rows
        assert not data.empty
    
    def test_required_columns(self, generator):
        """Test that all required columns are present"""
        data = generator.generate(50)
        
        required_columns = [
            'sku', 'product_name', 'category', 'abc_classification',
            'current_stock', 'unit_cost', 'stock_value', 'reorder_point',
            'reorder_quantity', 'safety_stock_quantity', 'warehouse_id',
            'warehouse_type', 'warehouse_location', 'bin_location',
            'storage_requirements', 'supplier_id', 'supplier_type',
            'supplier_reliability', 'lead_time_days', 'avg_daily_demand',
            'annual_turnover', 'days_of_supply', 'last_received_date',
            'last_sold_date', 'expiration_date', 'expired', 'stock_status',
            'demand_volatility', 'seasonal_factor', 'holding_cost_annual',
            'ordering_cost', 'service_level_target', 'created_date',
            'last_updated'
        ]
        
        for column in required_columns:
            assert column in data.columns, f"Missing required column: {column}"
    
    def test_sku_uniqueness_and_format(self, generator):
        """Test that SKUs are unique and properly formatted"""
        data = generator.generate(500)
        assert data['sku'].nunique() == len(data)
        
        # Check format (category code + hyphen + 6 digits)
        assert all(data['sku'].str.contains(r'^[A-Z]{3}-\d{6}$'))
        
        # Check that SKUs start with valid category codes
        valid_prefixes = ['ELC', 'CLT', 'FDB', 'HGD', 'AUT', 'HBT', 'SPT', 'BKM', 'OFC']
        sku_prefixes = data['sku'].str[:3].unique()
        assert all(prefix in valid_prefixes for prefix in sku_prefixes)
    
    def test_product_categories_valid(self, generator):
        """Test that product categories are from valid set"""
        data = generator.generate(200)
        
        expected_categories = {
            'electronics', 'clothing', 'food_beverage', 'home_garden',
            'automotive', 'health_beauty', 'sports_outdoors', 'books_media',
            'office_supplies'
        }
        
        actual_categories = set(data['category'].unique())
        assert actual_categories.issubset(expected_categories)
    
    def test_abc_classification_valid(self, generator):
        """Test that ABC classifications are valid and distributed correctly"""
        data = generator.generate(1000)
        
        expected_classes = {'A', 'B', 'C'}
        actual_classes = set(data['abc_classification'].unique())
        assert actual_classes.issubset(expected_classes)
        
        # Check approximate distribution (A: 20%, B: 30%, C: 50%)
        class_counts = data['abc_classification'].value_counts(normalize=True)
        
        # Allow for some variance due to randomness
        if 'A' in class_counts:
            assert 0.15 <= class_counts['A'] <= 0.25
        if 'B' in class_counts:
            assert 0.25 <= class_counts['B'] <= 0.35
        if 'C' in class_counts:
            assert 0.45 <= class_counts['C'] <= 0.55
    
    def test_stock_levels_realistic(self, generator):
        """Test that stock levels are realistic and non-negative"""
        data = generator.generate(300)
        
        # All stock quantities should be non-negative
        assert (data['current_stock'] >= 0).all()
        assert (data['reorder_point'] >= 0).all()
        assert (data['reorder_quantity'] > 0).all()
        assert (data['safety_stock_quantity'] >= 0).all()
        
        # Reorder point should generally be greater than safety stock
        assert (data['reorder_point'] >= data['safety_stock_quantity']).mean() > 0.8
    
    def test_cost_and_value_calculations(self, generator):
        """Test that cost and value calculations are correct"""
        data = generator.generate(200)
        
        # Unit costs should be positive
        assert (data['unit_cost'] > 0).all()
        
        # Stock value should equal current stock * unit cost
        calculated_value = data['current_stock'] * data['unit_cost']
        assert abs(data['stock_value'] - calculated_value).max() < 0.01
        
        # Holding cost should be reasonable percentage of stock value
        # Only check for items with non-zero stock value
        non_zero_stock = data['stock_value'] > 0
        if non_zero_stock.any():
            holding_rate = data[non_zero_stock]['holding_cost_annual'] / data[non_zero_stock]['stock_value']
            # Should be between 10% and 30% annually
            assert (holding_rate >= 0.10).all()
            assert (holding_rate <= 0.35).all()
    
    def test_warehouse_types_valid(self, generator):
        """Test that warehouse types are from valid set"""
        data = generator.generate(200)
        
        expected_types = {
            'main_distribution', 'regional_hub', 'local_store',
            'specialty_storage', 'overflow_storage'
        }
        
        actual_types = set(data['warehouse_type'].unique())
        assert actual_types.issubset(expected_types)
    
    def test_warehouse_id_format(self, generator):
        """Test that warehouse IDs are properly formatted"""
        data = generator.generate(100)
        
        # Warehouse IDs should follow pattern WH_XXX_NN
        assert all(data['warehouse_id'].str.match(r'^WH_[A-Z]{3}_\d{2}$'))
    
    def test_bin_location_format(self, generator):
        """Test that bin locations are properly formatted"""
        data = generator.generate(150)
        
        # Bin locations should be non-empty strings
        assert data['bin_location'].notna().all()
        assert (data['bin_location'].str.len() > 0).all()
        
        # Should contain alphanumeric characters and possibly hyphens
        assert all(data['bin_location'].str.match(r'^[A-Z0-9\-]+$'))
    
    def test_supplier_information_valid(self, generator):
        """Test that supplier information is valid"""
        data = generator.generate(200)
        
        # Supplier IDs should follow format
        assert all(data['supplier_id'].str.match(r'^SUP_\d{3}$'))
        
        # Supplier types should be valid
        expected_supplier_types = {
            'domestic_premium', 'domestic_standard', 'international_premium',
            'international_standard', 'local_supplier'
        }
        actual_supplier_types = set(data['supplier_type'].unique())
        assert actual_supplier_types.issubset(expected_supplier_types)
        
        # Supplier reliability should be between 0 and 100
        assert (data['supplier_reliability'] >= 0).all()
        assert (data['supplier_reliability'] <= 100).all()
        
        # Lead times should be positive
        assert (data['lead_time_days'] > 0).all()
    
    def test_demand_patterns_realistic(self, generator):
        """Test that demand patterns are realistic"""
        data = generator.generate(400)
        
        # Daily demand should be non-negative
        assert (data['avg_daily_demand'] >= 0).all()
        
        # Annual turnover should be positive
        assert (data['annual_turnover'] > 0).all()
        
        # Days of supply should be calculated correctly
        # Use the same calculation as the generator: max(avg_daily_demand, 0.1)
        calculated_days = data['current_stock'] / data['avg_daily_demand'].apply(lambda x: max(x, 0.1))
        calculated_days = calculated_days.round(1)
        assert abs(data['days_of_supply'] - calculated_days).max() < 0.1
        
        # Demand volatility should be between 0 and 1
        assert (data['demand_volatility'] >= 0).all()
        assert (data['demand_volatility'] <= 1).all()
    
    def test_abc_classification_patterns(self, generator):
        """Test that ABC classification follows expected patterns"""
        data = generator.generate(600)
        
        # A items should generally have higher turnover than C items
        a_items = data[data['abc_classification'] == 'A']
        c_items = data[data['abc_classification'] == 'C']
        
        if not a_items.empty and not c_items.empty:
            a_turnover = a_items['annual_turnover'].mean()
            c_turnover = c_items['annual_turnover'].mean()
            assert a_turnover > c_turnover
        
        # A items should have higher service level targets
        if not a_items.empty and not c_items.empty:
            a_service = a_items['service_level_target'].mean()
            c_service = c_items['service_level_target'].mean()
            assert a_service > c_service
    
    def test_stock_status_logic(self, generator):
        """Test that stock status follows logical rules"""
        data = generator.generate(300)
        
        expected_statuses = {'out_of_stock', 'critical', 'reorder_needed', 'normal', 'overstock'}
        actual_statuses = set(data['stock_status'].unique())
        assert actual_statuses.issubset(expected_statuses)
        
        # Out of stock items should have zero current stock
        out_of_stock = data[data['stock_status'] == 'out_of_stock']
        if not out_of_stock.empty:
            assert (out_of_stock['current_stock'] == 0).all()
        
        # Critical items should have stock <= safety stock
        critical = data[data['stock_status'] == 'critical']
        if not critical.empty:
            assert (critical['current_stock'] <= critical['safety_stock_quantity']).all()
        
        # Reorder needed items should have stock <= reorder point
        reorder_needed = data[data['stock_status'] == 'reorder_needed']
        if not reorder_needed.empty:
            assert (reorder_needed['current_stock'] <= reorder_needed['reorder_point']).all()
    
    def test_storage_requirements_valid(self, generator):
        """Test that storage requirements are valid"""
        data = generator.generate(200)
        
        expected_requirements = {'standard', 'climate_controlled', 'temperature_controlled'}
        actual_requirements = set(data['storage_requirements'].unique())
        assert actual_requirements.issubset(expected_requirements)
    
    def test_expiration_date_logic(self, generator):
        """Test that expiration dates follow logical rules"""
        data = generator.generate(400)
        
        # Items with expiration dates should have expired flag set correctly
        has_expiration = data['expiration_date'].notna()
        if has_expiration.any():
            today = datetime.now().date()
            expired_items = data[has_expiration]
            
            for idx, row in expired_items.iterrows():
                exp_date = pd.to_datetime(row['expiration_date']).date()
                expected_expired = exp_date < today
                assert row['expired'] == expected_expired
        
        # Items without expiration dates should not be marked as expired
        no_expiration = data['expiration_date'].isna()
        if no_expiration.any():
            assert not data[no_expiration]['expired'].any()
    
    def test_date_relationships(self, generator):
        """Test that dates have logical relationships"""
        data = generator.generate(200)
        
        today = datetime.now().date()
        
        # Created date should be before last updated
        created_dates = pd.to_datetime(data['created_date']).dt.date
        updated_dates = pd.to_datetime(data['last_updated']).dt.date
        assert (updated_dates >= created_dates).all()
        
        # Last received and sold dates should not be in the future
        received_dates = pd.to_datetime(data['last_received_date']).dt.date
        sold_dates = pd.to_datetime(data['last_sold_date']).dt.date
        assert (received_dates <= today).all()
        assert (sold_dates <= today).all()
        
        # Last updated should not be in the future
        assert (updated_dates <= today).all()
    
    def test_derived_metrics_calculation(self, generator):
        """Test that derived metrics are calculated correctly"""
        data = generator.generate(300)
        
        # Check that derived columns exist
        derived_columns = [
            'stock_turn_ratio', 'reorder_urgency', 'days_since_received',
            'inventory_health_score', 'movement_frequency', 'daily_holding_cost',
            'stockout_risk'
        ]
        
        for column in derived_columns:
            assert column in data.columns, f"Missing derived column: {column}"
        
        # Stock turn ratio should be calculated correctly
        expected_turn_ratio = data['annual_turnover'] / (data['current_stock'] + 1)
        assert abs(data['stock_turn_ratio'] - expected_turn_ratio).max() < 0.01
        
        # Daily holding cost should be annual cost / 365
        expected_daily_cost = data['holding_cost_annual'] / 365
        assert abs(data['daily_holding_cost'] - expected_daily_cost).max() < 0.01
        
        # Health scores should be between 0 and 100
        assert (data['inventory_health_score'] >= 0).all()
        assert (data['inventory_health_score'] <= 100).all()
        
        # Stockout risk should be between 0 and 100
        assert (data['stockout_risk'] >= 0).all()
        assert (data['stockout_risk'] <= 100).all()
    
    def test_seasonal_factors_realistic(self, generator):
        """Test that seasonal factors are realistic"""
        data = generator.generate(200)
        
        # Seasonal factors should be positive
        assert (data['seasonal_factor'] > 0).all()
        
        # Should be reasonable multipliers (between 0.5 and 2.0)
        assert (data['seasonal_factor'] >= 0.5).all()
        assert (data['seasonal_factor'] <= 2.0).all()
    
    def test_time_series_generation(self, generator):
        """Test time series data generation"""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        data = generator.generate(
            100,
            time_series=True,
            start_date=start_date,
            end_date=end_date,
            interval='1week'
        )
        
        # Should have time series specific columns
        time_series_columns = [
            'timestamp', 'seasonal_multiplier', 'weekly_multiplier',
            'activity_intensity', 'demand_forecast'
        ]
        
        for column in time_series_columns:
            assert column in data.columns, f"Missing time series column: {column}"
        
        # Timestamps should be within specified range
        timestamps = pd.to_datetime(data['timestamp'])
        assert (timestamps >= start_date).all()
        assert (timestamps <= end_date).all()
        
        # Multipliers should be positive
        assert (data['seasonal_multiplier'] > 0).all()
        assert (data['weekly_multiplier'] > 0).all()
        assert (data['activity_intensity'] > 0).all()
    
    def test_warehouse_location_format(self, generator):
        """Test that warehouse locations are properly formatted"""
        data = generator.generate(100)
        
        # Warehouse locations should contain city and region
        assert data['warehouse_location'].notna().all()
        assert all(', ' in location for location in data['warehouse_location'])
        
        # Should end with region names
        valid_regions = ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West']
        locations = data['warehouse_location'].tolist()
        regions_found = [loc.split(', ')[-1] for loc in locations]
        assert all(region in valid_regions for region in regions_found)
    
    def test_reorder_logic_consistency(self, generator):
        """Test that reorder logic is consistent"""
        data = generator.generate(400)
        
        # Reorder quantity should be reasonable relative to demand
        # Should cover at least a few days of demand
        min_coverage_days = data['reorder_quantity'] / data['avg_daily_demand'].replace(0, 1)
        assert (min_coverage_days >= 1).mean() > 0.9  # 90% should cover at least 1 day
        
        # Safety stock should be reasonable
        safety_coverage = data['safety_stock_quantity'] / data['avg_daily_demand'].replace(0, 1)
        assert (safety_coverage >= 1).mean() > 0.8  # 80% should cover at least 1 day
    
    def test_category_cost_patterns(self, generator):
        """Test that different categories have appropriate cost patterns"""
        data = generator.generate(800)
        
        # Electronics should generally be more expensive than office supplies
        electronics = data[data['category'] == 'electronics']
        office = data[data['category'] == 'office_supplies']
        
        if not electronics.empty and not office.empty:
            electronics_avg_cost = electronics['unit_cost'].mean()
            office_avg_cost = office['unit_cost'].mean()
            # Allow some variance but expect general trend
            assert electronics_avg_cost >= office_avg_cost * 0.8
    
    def test_supplier_reliability_correlation(self, generator):
        """Test that supplier reliability correlates with supplier type"""
        data = generator.generate(500)
        
        # Premium suppliers should have higher reliability than standard
        premium_domestic = data[data['supplier_type'] == 'domestic_premium']
        standard_domestic = data[data['supplier_type'] == 'domestic_standard']
        
        if not premium_domestic.empty and not standard_domestic.empty:
            premium_reliability = premium_domestic['supplier_reliability'].mean()
            standard_reliability = standard_domestic['supplier_reliability'].mean()
            assert premium_reliability > standard_reliability
    
    def test_lead_time_patterns(self, generator):
        """Test that lead times follow expected patterns by supplier type"""
        data = generator.generate(600)
        
        # International suppliers should have longer lead times than domestic
        international = data[data['supplier_type'].str.contains('international')]
        domestic = data[data['supplier_type'].str.contains('domestic')]
        
        if not international.empty and not domestic.empty:
            international_lead_time = international['lead_time_days'].mean()
            domestic_lead_time = domestic['lead_time_days'].mean()
            assert international_lead_time > domestic_lead_time
        
        # Local suppliers should have shortest lead times
        local = data[data['supplier_type'] == 'local_supplier']
        if not local.empty and not domestic.empty:
            local_lead_time = local['lead_time_days'].mean()
            domestic_lead_time = domestic['lead_time_days'].mean()
            assert local_lead_time <= domestic_lead_time
    
    def test_movement_frequency_calculation(self, generator):
        """Test that movement frequency is calculated correctly"""
        data = generator.generate(300)
        
        # Movement frequency should be non-negative
        assert (data['movement_frequency'] >= 0).all()
        
        # Should correlate with annual turnover
        high_turnover = data[data['annual_turnover'] > 10]
        low_turnover = data[data['annual_turnover'] < 2]
        
        if not high_turnover.empty and not low_turnover.empty:
            high_movement = high_turnover['movement_frequency'].mean()
            low_movement = low_turnover['movement_frequency'].mean()
            assert high_movement > low_movement
    
    def test_stockout_risk_logic(self, generator):
        """Test that stockout risk follows logical patterns"""
        data = generator.generate(400)
        
        # Out of stock items should have highest risk
        out_of_stock = data[data['stock_status'] == 'out_of_stock']
        normal_stock = data[data['stock_status'] == 'normal']
        
        if not out_of_stock.empty and not normal_stock.empty:
            out_of_stock_risk = out_of_stock['stockout_risk'].mean()
            normal_stock_risk = normal_stock['stockout_risk'].mean()
            assert out_of_stock_risk > normal_stock_risk
        
        # Items with unreliable suppliers should have higher risk
        unreliable = data[data['supplier_reliability'] < 80]
        reliable = data[data['supplier_reliability'] > 95]
        
        if not unreliable.empty and not reliable.empty:
            unreliable_risk = unreliable['stockout_risk'].mean()
            reliable_risk = reliable['stockout_risk'].mean()
            assert unreliable_risk > reliable_risk
    
    def test_data_quality_validation(self, generator):
        """Test overall data quality and consistency"""
        data = generator.generate(300)
        
        # No null values in required fields
        critical_fields = [
            'sku', 'product_name', 'category', 'abc_classification',
            'current_stock', 'unit_cost', 'reorder_point', 'warehouse_id'
        ]
        
        for field in critical_fields:
            assert data[field].notna().all(), f"Null values found in critical field: {field}"
        
        # Numeric fields should be numeric
        numeric_fields = [
            'current_stock', 'unit_cost', 'stock_value', 'reorder_point',
            'reorder_quantity', 'safety_stock_quantity', 'supplier_reliability',
            'lead_time_days', 'avg_daily_demand', 'annual_turnover'
        ]
        
        for field in numeric_fields:
            assert pd.api.types.is_numeric_dtype(data[field]), f"Non-numeric data in field: {field}"
    
    def test_seeding_reproducibility(self):
        """Test that seeding produces reproducible results"""
        seeder1 = MillisecondSeeder(fixed_seed=999)
        seeder2 = MillisecondSeeder(fixed_seed=999)
        
        generator1 = InventoryGenerator(seeder1)
        generator2 = InventoryGenerator(seeder2)
        
        data1 = generator1.generate(100)
        data2 = generator2.generate(100)
        
        # Should produce identical results with same seed
        pd.testing.assert_frame_equal(data1, data2)
    
    def test_realistic_patterns_applied(self, generator):
        """Test that realistic patterns are applied to generated data"""
        data = generator.generate(500)
        
        # Should have realistic distribution of stock statuses
        status_counts = data['stock_status'].value_counts(normalize=True)
        
        # Normal should be most common status
        assert status_counts.get('normal', 0) > 0.3
        
        # Out of stock should be relatively rare
        assert status_counts.get('out_of_stock', 0) < 0.1
        
        # Should have variety in ABC classifications
        abc_counts = data['abc_classification'].nunique()
        assert abc_counts >= 2  # Should have at least 2 different classifications
        
        # Should have variety in categories
        category_counts = data['category'].nunique()
        assert category_counts >= 3  # Should have at least 3 different categories