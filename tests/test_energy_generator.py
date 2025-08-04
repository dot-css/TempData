"""
Unit tests for EnergyGenerator

Tests realistic patterns, consumption patterns, peak usage times, seasonal variations,
and power factor calculations.
"""

import pytest
import pandas as pd
import math
from datetime import datetime, date
from tempdata.core.seeding import MillisecondSeeder
from tempdata.datasets.iot_sensors.energy import EnergyGenerator


class TestEnergyGenerator:
    """Test suite for EnergyGenerator functionality"""
    
    @pytest.fixture
    def seeder(self):
        """Create a fixed seeder for reproducible tests"""
        return MillisecondSeeder(fixed_seed=987654321)
    
    @pytest.fixture
    def generator(self, seeder):
        """Create EnergyGenerator instance"""
        return EnergyGenerator(seeder)
    
    def test_basic_generation(self, generator):
        """Test basic data generation functionality"""
        rows = 100
        data = generator.generate(rows)
        
        # Check basic structure
        assert isinstance(data, pd.DataFrame)
        assert len(data) == rows
        
        # Check required columns exist
        required_columns = [
            'meter_id', 'timestamp', 'building_type', 'region',
            'consumption_kwh', 'cumulative_kwh', 'voltage_v', 'current_a',
            'power_factor', 'frequency_hz', 'real_power_w', 'apparent_power_va',
            'reactive_power_var', 'demand_kw'
        ]
        for col in required_columns:
            assert col in data.columns
    
    def test_meter_id_format(self, generator):
        """Test meter ID format and uniqueness"""
        data = generator.generate(500)
        
        # Check format
        assert all(data['meter_id'].str.startswith('MTR_'))
        assert all(data['meter_id'].str.len() == 12)  # MTR_ + 8 digits
        
        # Check uniqueness
        assert data['meter_id'].nunique() == len(data)
    
    def test_consumption_ranges(self, generator):
        """Test energy consumption ranges are realistic"""
        data = generator.generate(1000)
        
        # Consumption should be positive
        assert all(data['consumption_kwh'] > 0)
        
        # Check reasonable range for hourly consumption
        assert data['consumption_kwh'].min() >= 0.001  # Minimum consumption
        assert data['consumption_kwh'].max() <= 200.0  # Maximum reasonable hourly consumption
        
        # Check precision (3 decimal places)
        assert all(data['consumption_kwh'].apply(lambda x: len(str(x).split('.')[-1]) <= 3))
    
    def test_voltage_ranges(self, generator):
        """Test voltage ranges are realistic"""
        data = generator.generate(1000)
        
        # Voltage should be in reasonable range
        assert all(data['voltage_v'] >= 100)  # Minimum voltage
        assert all(data['voltage_v'] <= 450)  # Maximum voltage
        
        # Check precision (1 decimal place)
        assert all(data['voltage_v'].apply(lambda x: len(str(x).split('.')[-1]) <= 1))
    
    def test_power_factor_ranges(self, generator):
        """Test power factor ranges are realistic"""
        data = generator.generate(1000)
        
        # Power factor should be between 0 and 1
        assert all(data['power_factor'] >= 0.7)  # Minimum reasonable power factor
        assert all(data['power_factor'] <= 1.0)  # Maximum power factor
        
        # Check precision (3 decimal places)
        assert all(data['power_factor'].apply(lambda x: len(str(x).split('.')[-1]) <= 3))
    
    def test_current_calculations(self, generator):
        """Test current calculations are realistic"""
        data = generator.generate(500)
        
        # Current should be positive
        assert all(data['current_a'] > 0)
        
        # Current should be reasonable for the power levels
        assert all(data['current_a'] <= 1000)  # Maximum reasonable current
    
    def test_frequency_standards(self, generator):
        """Test frequency standards by region"""
        # Test North America (60 Hz)
        na_data = generator.generate(100, country='united_states')
        assert all(na_data['frequency_hz'] == 60)
        
        # Test Europe (50 Hz)
        eu_data = generator.generate(100, country='germany')
        assert all(eu_data['frequency_hz'] == 50)
    
    def test_building_type_distribution(self, generator):
        """Test building type distribution"""
        data = generator.generate(1000)
        
        # Check valid building types
        valid_types = ['residential', 'commercial', 'industrial', 'retail', 'hospital']
        assert all(data['building_type'].isin(valid_types))
        
        # Residential should be most common
        type_dist = data['building_type'].value_counts(normalize=True)
        assert type_dist['residential'] > 0.4  # At least 40% residential
    
    def test_building_type_consumption_patterns(self, generator):
        """Test consumption patterns by building type"""
        data = generator.generate(1000)
        
        # Industrial should have higher average consumption than residential
        industrial_data = data[data['building_type'] == 'industrial']
        residential_data = data[data['building_type'] == 'residential']
        
        if len(industrial_data) > 10 and len(residential_data) > 10:
            industrial_avg = industrial_data['consumption_kwh'].mean()
            residential_avg = residential_data['consumption_kwh'].mean()
            assert industrial_avg > residential_avg
    
    def test_seasonal_patterns(self, generator):
        """Test seasonal energy consumption patterns"""
        # Generate data for summer and winter months
        summer_data = generator.generate(
            100,
            time_series=True,
            date_range=(date(2024, 7, 1), date(2024, 7, 31)),
            country='united_states'
        )
        
        winter_data = generator.generate(
            100,
            time_series=True,
            date_range=(date(2024, 1, 1), date(2024, 1, 31)),
            country='united_states'
        )
        
        # Winter should generally have higher consumption due to heating
        summer_avg = summer_data['consumption_kwh'].mean()
        winter_avg = winter_data['consumption_kwh'].mean()
        
        # Allow some variance but winter should generally be higher
        assert winter_avg > summer_avg * 0.9  # At least 90% of summer average
    
    def test_daily_patterns(self, generator):
        """Test daily consumption patterns"""
        # Generate hourly data for one day
        data = generator.generate(
            24,
            time_series=True,
            date_range=(datetime(2024, 6, 15, 0, 0), datetime(2024, 6, 15, 23, 0)),
            interval='1hour',
            building_type='residential'
        )
        
        if len(data) >= 20:  # Need sufficient data points
            # Extract hour from timestamp
            data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
            
            # Evening peak hours should have higher consumption than early morning
            evening_consumption = data[data['hour'].isin([18, 19, 20])]['consumption_kwh']
            morning_consumption = data[data['hour'].isin([2, 3, 4])]['consumption_kwh']
            
            if len(evening_consumption) > 0 and len(morning_consumption) > 0:
                assert evening_consumption.mean() > morning_consumption.mean()
    
    def test_weekend_patterns(self, generator):
        """Test weekend vs weekday consumption patterns"""
        # Generate data for a week
        data = generator.generate(
            168,  # 24 hours * 7 days
            time_series=True,
            date_range=(datetime(2024, 6, 10, 0, 0), datetime(2024, 6, 16, 23, 0)),
            interval='1hour',
            building_type='residential'
        )
        
        if len(data) >= 100:
            data['weekday'] = pd.to_datetime(data['timestamp']).dt.weekday
            
            # Separate weekday and weekend data
            weekday_data = data[data['weekday'] < 5]  # Monday-Friday
            weekend_data = data[data['weekday'] >= 5]  # Saturday-Sunday
            
            if len(weekday_data) > 20 and len(weekend_data) > 10:
                # For residential, weekend consumption might be higher
                weekend_avg = weekend_data['consumption_kwh'].mean()
                weekday_avg = weekday_data['consumption_kwh'].mean()
                
                # Weekend should be at least 90% of weekday (allowing for variation)
                assert weekend_avg >= weekday_avg * 0.9
    
    def test_power_calculations(self, generator):
        """Test power calculations are consistent"""
        data = generator.generate(200)
        
        # Real power should be less than or equal to apparent power
        assert all(data['real_power_w'] <= data['apparent_power_va'] + 1)  # Allow small rounding
        
        # Demand (kW) should equal real power (W) / 1000
        calculated_demand = data['real_power_w'] / 1000
        assert all(abs(data['demand_kw'] - calculated_demand) < 0.01)  # Allow small rounding
        
        # Power factor should equal real power / apparent power
        calculated_pf = data['real_power_w'] / data['apparent_power_va']
        assert all(abs(data['power_factor'] - calculated_pf) < 0.01)  # Allow small rounding
    
    def test_cumulative_readings(self, generator):
        """Test cumulative meter readings"""
        data = generator.generate(100)
        
        # Cumulative readings should be positive and increasing
        assert all(data['cumulative_kwh'] > 0)
        
        # Cumulative should be greater than individual consumption
        assert all(data['cumulative_kwh'] >= data['consumption_kwh'])
    
    def test_regional_patterns(self, generator):
        """Test regional energy patterns"""
        # Test different regions
        us_data = generator.generate(100, country='united_states')
        eu_data = generator.generate(100, country='germany')
        
        # Check regions are assigned correctly
        assert all(us_data['region'] == 'north_america')
        assert all(eu_data['region'] == 'europe')
        
        # Check frequency standards
        assert all(us_data['frequency_hz'] == 60)
        assert all(eu_data['frequency_hz'] == 50)
    
    def test_time_series_generation(self, generator):
        """Test time series data generation"""
        data = generator.generate(
            100,
            time_series=True,
            date_range=(date(2024, 1, 1), date(2024, 1, 10)),
            interval='1hour'
        )
        
        # Check timestamps are in chronological order
        timestamps = pd.to_datetime(data['timestamp'])
        assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
        
        # Check interval consistency
        time_diffs = timestamps.diff().dropna()
        hour_diffs = time_diffs.dt.total_seconds() / 3600
        
        # Most intervals should be close to 1 hour
        close_to_hour = hour_diffs[(hour_diffs >= 0.5) & (hour_diffs <= 1.5)]
        assert len(close_to_hour) / len(hour_diffs) > 0.7  # 70% within reasonable range
    
    def test_cost_calculations(self, generator):
        """Test energy cost calculations"""
        data = generator.generate(200)
        
        # Cost per kWh should be positive
        assert all(data['cost_per_kwh'] > 0)
        
        # Total cost should equal consumption * cost per kWh
        calculated_cost = data['consumption_kwh'] * data['cost_per_kwh']
        assert all(abs(data['total_cost'] - calculated_cost) < 0.01)  # Allow rounding
        
        # Cost should vary by region
        if data['region'].nunique() > 1:
            region_costs = data.groupby('region')['cost_per_kwh'].mean()
            assert region_costs.std() > 0.01  # Some variation between regions
    
    def test_efficiency_ratings(self, generator):
        """Test efficiency rating calculations"""
        data = generator.generate(500)
        
        # Check valid efficiency ratings
        valid_ratings = ['excellent', 'good', 'fair', 'poor', 'very_poor']
        assert all(data['efficiency_rating'].isin(valid_ratings))
        
        # Higher power factor should correlate with better ratings
        excellent_data = data[data['efficiency_rating'] == 'excellent']
        poor_data = data[data['efficiency_rating'] == 'poor']
        
        if len(excellent_data) > 5 and len(poor_data) > 5:
            excellent_pf = excellent_data['power_factor'].mean()
            poor_pf = poor_data['power_factor'].mean()
            assert excellent_pf > poor_pf
    
    def test_rate_period_classification(self, generator):
        """Test rate period classification"""
        data = generator.generate(200)
        
        # Check valid rate periods
        valid_periods = ['peak', 'off_peak', 'standard', 'weekend']
        assert all(data['rate_period'].isin(valid_periods))
        
        # Peak hours should have higher costs
        peak_data = data[data['rate_period'] == 'peak']
        standard_data = data[data['rate_period'] == 'standard']
        
        if len(peak_data) > 5 and len(standard_data) > 5:
            peak_cost = peak_data['cost_per_kwh'].mean()
            standard_cost = standard_data['cost_per_kwh'].mean()
            assert peak_cost > standard_cost
    
    def test_meter_status_distribution(self, generator):
        """Test meter status distribution"""
        data = generator.generate(1000)
        
        # Check valid status values
        valid_statuses = ['normal', 'maintenance_required', 'calibration_needed']
        assert all(data['meter_status'].isin(valid_statuses))
        
        # Most meters should be normal
        normal_ratio = len(data[data['meter_status'] == 'normal']) / len(data)
        assert normal_ratio > 0.9  # At least 90% should be normal
    
    def test_carbon_footprint_calculation(self, generator):
        """Test carbon footprint calculations"""
        data = generator.generate(200)
        
        # Carbon footprint should be positive
        assert all(data['carbon_footprint_kg'] > 0)
        
        # Should correlate with consumption
        correlation = data['consumption_kwh'].corr(data['carbon_footprint_kg'])
        assert correlation > 0.9  # Strong positive correlation
        
        # Different regions should have different carbon intensities
        if data['region'].nunique() > 1:
            region_carbon = data.groupby('region')['carbon_footprint_kg'].mean() / data.groupby('region')['consumption_kwh'].mean()
            assert region_carbon.std() > 0.01  # Some variation between regions
    
    def test_building_type_specific_generation(self, generator):
        """Test building type specific generation"""
        # Test specific building type
        industrial_data = generator.generate(100, building_type='industrial')
        residential_data = generator.generate(100, building_type='residential')
        
        # All should be the specified type
        assert all(industrial_data['building_type'] == 'industrial')
        assert all(residential_data['building_type'] == 'residential')
        
        # Industrial should have higher consumption on average
        industrial_avg = industrial_data['consumption_kwh'].mean()
        residential_avg = residential_data['consumption_kwh'].mean()
        assert industrial_avg > residential_avg
    
    def test_realistic_patterns_applied(self, generator):
        """Test that realistic patterns are applied"""
        data = generator.generate(200)
        
        # Check additional columns from realistic patterns
        pattern_columns = ['cost_per_kwh', 'total_cost', 'efficiency_rating', 
                          'rate_period', 'meter_status', 'carbon_footprint_kg']
        for col in pattern_columns:
            assert col in data.columns
        
        # Check data is sorted by timestamp
        timestamps = pd.to_datetime(data['timestamp'])
        assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
    
    def test_reproducibility(self, seeder):
        """Test that same seed produces same results"""
        gen1 = EnergyGenerator(seeder)
        gen2 = EnergyGenerator(MillisecondSeeder(fixed_seed=987654321))
        
        data1 = gen1.generate(50)
        data2 = gen2.generate(50)
        
        # Should be identical
        pd.testing.assert_frame_equal(data1, data2)
    
    def test_data_quality_metrics(self, generator):
        """Test overall data quality metrics"""
        data = generator.generate(1000)
        
        # No null values in critical fields
        critical_fields = ['meter_id', 'consumption_kwh', 'voltage_v', 'power_factor']
        for field in critical_fields:
            assert data[field].notna().all()
        
        # Reasonable distribution of building types
        building_dist = data['building_type'].value_counts(normalize=True)
        assert len(building_dist) >= 3  # At least 3 different building types
        
        # Reasonable distribution of regions
        region_dist = data['region'].value_counts(normalize=True)
        assert len(region_dist) >= 2  # At least 2 regions
    
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
        
        # Test with extreme date ranges
        data = generator.generate(
            10,
            time_series=True,
            date_range=(date(2020, 1, 1), date(2020, 1, 2)),
            interval='1hour'
        )
        assert len(data) == 10
    
    def test_electrical_relationships(self, generator):
        """Test electrical relationships and calculations"""
        data = generator.generate(500)
        
        # Test Ohm's law relationships (approximately)
        # P = V * I * PF, so I = P / (V * PF)
        calculated_current = (data['real_power_w']) / (data['voltage_v'] * data['power_factor'])
        
        # Allow for some variance due to rounding and approximations
        current_diff = abs(data['current_a'] - calculated_current)
        assert (current_diff < 1.0).mean() > 0.8  # 80% should be close
        
        # Reactive power should be calculated correctly
        # Q = sqrt(S^2 - P^2)
        calculated_reactive = (data['apparent_power_va']**2 - data['real_power_w']**2)**0.5
        reactive_diff = abs(data['reactive_power_var'] - calculated_reactive)
        assert (reactive_diff < 10.0).mean() > 0.8  # 80% should be close


if __name__ == '__main__':
    pytest.main([__file__])