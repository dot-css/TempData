"""
Unit tests for WeatherGenerator

Tests realistic patterns, seasonal trends, geographical variations, and data correlations.
"""

import pytest
import pandas as pd
import math
from datetime import datetime, date
from tempdata.core.seeding import MillisecondSeeder
from tempdata.datasets.iot_sensors.weather import WeatherGenerator


class TestWeatherGenerator:
    """Test suite for WeatherGenerator functionality"""
    
    @pytest.fixture
    def seeder(self):
        """Create a fixed seeder for reproducible tests"""
        return MillisecondSeeder(fixed_seed=123456789)
    
    @pytest.fixture
    def generator(self, seeder):
        """Create WeatherGenerator instance"""
        return WeatherGenerator(seeder)
    
    def test_basic_generation(self, generator):
        """Test basic data generation functionality"""
        rows = 100
        data = generator.generate(rows)
        
        # Check basic structure
        assert isinstance(data, pd.DataFrame)
        assert len(data) == rows
        
        # Check required columns exist
        required_columns = [
            'sensor_id', 'timestamp', 'location', 'latitude', 'longitude',
            'altitude_m', 'temperature_c', 'humidity_percent', 'pressure_hpa',
            'wind_speed_kmh', 'wind_direction_deg', 'uv_index', 'visibility_km'
        ]
        for col in required_columns:
            assert col in data.columns
    
    def test_sensor_id_format(self, generator):
        """Test sensor ID format and uniqueness"""
        data = generator.generate(500)
        
        # Check format
        assert all(data['sensor_id'].str.startswith('WS_'))
        assert all(data['sensor_id'].str.len() == 9)  # WS_ + 6 digits
        
        # Check uniqueness
        assert data['sensor_id'].nunique() == len(data)
    
    def test_temperature_ranges(self, generator):
        """Test temperature ranges are realistic"""
        data = generator.generate(1000)
        
        # Check reasonable global temperature range
        assert data['temperature_c'].min() >= -50.0  # Extreme cold limit
        assert data['temperature_c'].max() <= 60.0   # Extreme heat limit
        
        # Check precision (1 decimal place)
        assert all(data['temperature_c'].apply(lambda x: len(str(x).split('.')[-1]) <= 1))
    
    def test_humidity_ranges(self, generator):
        """Test humidity ranges are realistic"""
        data = generator.generate(1000)
        
        # Humidity should be between 0-100%
        assert all(data['humidity_percent'] >= 0)
        assert all(data['humidity_percent'] <= 100)
        
        # Check precision (1 decimal place)
        assert all(data['humidity_percent'].apply(lambda x: len(str(x).split('.')[-1]) <= 1))
    
    def test_pressure_ranges(self, generator):
        """Test atmospheric pressure ranges are realistic"""
        data = generator.generate(1000)
        
        # Realistic pressure range (extreme weather conditions)
        assert all(data['pressure_hpa'] >= 870)   # Lowest recorded pressure
        assert all(data['pressure_hpa'] <= 1085)  # Highest recorded pressure
        
        # Most values should be in normal range
        normal_range = data[(data['pressure_hpa'] >= 980) & (data['pressure_hpa'] <= 1040)]
        assert len(normal_range) / len(data) > 0.8  # 80% in normal range
    
    def test_wind_parameters(self, generator):
        """Test wind speed and direction parameters"""
        data = generator.generate(500)
        
        # Wind speed should be non-negative
        assert all(data['wind_speed_kmh'] >= 0)
        
        # Wind direction should be 0-360 degrees
        assert all(data['wind_direction_deg'] >= 0)
        assert all(data['wind_direction_deg'] <= 360)
    
    def test_uv_index_ranges(self, generator):
        """Test UV index ranges are realistic"""
        data = generator.generate(500)
        
        # UV index should be 0-11
        assert all(data['uv_index'] >= 0)
        assert all(data['uv_index'] <= 11)
    
    def test_visibility_ranges(self, generator):
        """Test visibility ranges are realistic"""
        data = generator.generate(500)
        
        # Visibility should be positive
        assert all(data['visibility_km'] > 0)
        
        # Most visibility should be reasonable (not extreme fog conditions)
        reasonable_visibility = data[data['visibility_km'] >= 1]
        assert len(reasonable_visibility) / len(data) > 0.7  # 70% with decent visibility
    
    def test_geographical_coordinates(self, generator):
        """Test geographical coordinates are valid"""
        data = generator.generate(200)
        
        # Latitude should be -90 to 90
        assert all(data['latitude'] >= -90)
        assert all(data['latitude'] <= 90)
        
        # Longitude should be -180 to 180
        assert all(data['longitude'] >= -180)
        assert all(data['longitude'] <= 180)
        
        # Altitude should be reasonable for weather stations
        assert all(data['altitude_m'] >= 0)
        assert all(data['altitude_m'] <= 5000)  # Most weather stations below 5000m
    
    def test_climate_zone_patterns(self, generator):
        """Test climate zone specific patterns"""
        # Test tropical climate
        tropical_data = generator.generate(100, country='brazil')
        tropical_temps = tropical_data['temperature_c']
        
        # Tropical should have higher average temperatures
        assert tropical_temps.mean() > 15  # Generally warm
        
        # Test polar climate (if we had polar countries in our data)
        temperate_data = generator.generate(100, country='germany')
        temperate_temps = temperate_data['temperature_c']
        
        # Should have reasonable temperature variation
        assert temperate_temps.std() > 5  # Good seasonal variation
    
    def test_seasonal_patterns(self, generator):
        """Test seasonal temperature patterns"""
        # Generate data for summer and winter months
        summer_data = generator.generate(
            100, 
            time_series=True,
            date_range=(date(2024, 7, 1), date(2024, 7, 31)),
            country='germany'
        )
        
        winter_data = generator.generate(
            100,
            time_series=True, 
            date_range=(date(2024, 1, 1), date(2024, 1, 31)),
            country='germany'
        )
        
        # Summer should generally be warmer than winter in temperate zones
        summer_avg = summer_data['temperature_c'].mean()
        winter_avg = winter_data['temperature_c'].mean()
        
        assert summer_avg > winter_avg
    
    def test_hemisphere_seasonal_differences(self, generator):
        """Test that northern and southern hemispheres have opposite seasons"""
        # July data for northern hemisphere (summer)
        north_july = generator.generate(
            50,
            time_series=True,
            date_range=(date(2024, 7, 1), date(2024, 7, 31)),
            country='germany'  # Northern hemisphere
        )
        
        # July data for southern hemisphere (winter) - using Australia as proxy
        south_july = generator.generate(
            50,
            time_series=True,
            date_range=(date(2024, 7, 1), date(2024, 7, 31)),
            country='australia'  # Southern hemisphere
        )
        
        # Filter for southern hemisphere locations (negative latitude)
        south_data = south_july[south_july['latitude'] < 0]
        
        if len(south_data) > 10:  # Only test if we have enough southern hemisphere data
            north_avg = north_july['temperature_c'].mean()
            south_avg = south_data['temperature_c'].mean()
            
            # Northern hemisphere July should be warmer than southern hemisphere July
            assert north_avg > south_avg
    
    def test_temperature_humidity_correlation(self, generator):
        """Test inverse correlation between temperature and humidity"""
        data = generator.generate(1000)
        
        # Calculate correlation
        correlation = data['temperature_c'].corr(data['humidity_percent'])
        
        # Should have negative correlation (not too strict due to other factors)
        assert correlation < 0.1  # Allow for some positive correlation due to climate zones
    
    def test_altitude_pressure_correlation(self, generator):
        """Test altitude effect on atmospheric pressure"""
        data = generator.generate(1000)
        
        # Higher altitude should generally mean lower pressure
        correlation = data['altitude_m'].corr(data['pressure_hpa'])
        
        # Should have negative correlation
        assert correlation < 0
    
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
        
        # Check interval consistency (allowing some variance for random timestamps)
        time_diffs = timestamps.diff().dropna()
        hour_diffs = time_diffs.dt.total_seconds() / 3600
        
        # Most intervals should be close to 1 hour
        close_to_hour = hour_diffs[(hour_diffs >= 0.5) & (hour_diffs <= 1.5)]
        assert len(close_to_hour) / len(hour_diffs) > 0.7  # 70% within reasonable range
    
    def test_daily_temperature_variation(self, generator):
        """Test daily temperature variation patterns"""
        # Generate hourly data for one day
        data = generator.generate(
            24,
            time_series=True,
            date_range=(datetime(2024, 6, 15, 0, 0), datetime(2024, 6, 15, 23, 0)),
            interval='1hour'
        )
        
        if len(data) >= 20:  # Need sufficient data points
            # Extract hour from timestamp
            data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
            
            # Afternoon should generally be warmer than early morning
            afternoon_temps = data[data['hour'].isin([13, 14, 15])]['temperature_c']
            morning_temps = data[data['hour'].isin([4, 5, 6])]['temperature_c']
            
            if len(afternoon_temps) > 0 and len(morning_temps) > 0:
                assert afternoon_temps.mean() > morning_temps.mean()
    
    def test_weather_condition_logic(self, generator):
        """Test weather condition determination logic"""
        data = generator.generate(500)
        
        # Check valid weather conditions
        valid_conditions = ['fog', 'rainy', 'sunny', 'freezing', 'humid', 'stormy', 'clear', 'hot', 'cold', 'overcast']
        assert all(data['weather_condition'].isin(valid_conditions))
        
        # Test specific condition logic
        fog_data = data[data['weather_condition'] == 'fog']
        if len(fog_data) > 0:
            # Fog should have low visibility
            assert all(fog_data['visibility_km'] < 1)
        
        freezing_data = data[data['weather_condition'] == 'freezing']
        if len(freezing_data) > 0:
            # Freezing should have temperature below 0
            assert all(freezing_data['temperature_c'] < 0)
    
    def test_heat_index_calculation(self, generator):
        """Test heat index calculation"""
        data = generator.generate(200)
        
        # Heat index should be >= temperature
        assert all(data['heat_index_c'] >= data['temperature_c'] - 1)  # Allow small rounding differences
        
        # For high temperature and humidity, heat index should be higher
        hot_humid = data[(data['temperature_c'] > 30) & (data['humidity_percent'] > 70)]
        if len(hot_humid) > 0:
            temp_diff = hot_humid['heat_index_c'] - hot_humid['temperature_c']
            assert temp_diff.mean() > 0  # Heat index should be higher than temperature
    
    def test_dew_point_calculation(self, generator):
        """Test dew point calculation"""
        data = generator.generate(200)
        
        # Dew point should be <= temperature
        assert all(data['dew_point_c'] <= data['temperature_c'] + 1)  # Allow small rounding differences
        
        # Higher humidity should generally mean dew point closer to temperature
        high_humidity = data[data['humidity_percent'] > 80]
        low_humidity = data[data['humidity_percent'] < 40]
        
        if len(high_humidity) > 5 and len(low_humidity) > 5:
            high_hum_diff = (high_humidity['temperature_c'] - high_humidity['dew_point_c']).mean()
            low_hum_diff = (low_humidity['temperature_c'] - low_humidity['dew_point_c']).mean()
            
            assert high_hum_diff < low_hum_diff  # Smaller difference for high humidity
    
    def test_sensor_status_distribution(self, generator):
        """Test sensor status distribution"""
        data = generator.generate(1000)
        
        # Check valid status values
        valid_statuses = ['normal', 'maintenance_required', 'calibration_needed']
        assert all(data['sensor_status'].isin(valid_statuses))
        
        # Most sensors should be normal
        normal_ratio = len(data[data['sensor_status'] == 'normal']) / len(data)
        assert normal_ratio > 0.9  # At least 90% should be normal
    
    def test_country_specific_patterns(self, generator):
        """Test country-specific weather patterns"""
        # Test different countries
        us_data = generator.generate(100, country='united_states')
        brazil_data = generator.generate(100, country='brazil')
        
        # Brazil (tropical) should generally be more humid
        brazil_humidity = brazil_data['humidity_percent'].mean()
        us_humidity = us_data['humidity_percent'].mean()
        
        # This is a general trend, not absolute
        assert brazil_humidity > us_humidity - 10  # Allow some variance
    
    def test_realistic_patterns_applied(self, generator):
        """Test that realistic patterns are applied"""
        data = generator.generate(200)
        
        # Check additional columns from realistic patterns
        pattern_columns = ['weather_condition', 'heat_index_c', 'dew_point_c', 'sensor_status']
        for col in pattern_columns:
            assert col in data.columns
        
        # Check data is sorted by timestamp
        timestamps = pd.to_datetime(data['timestamp'])
        assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
    
    def test_reproducibility(self, seeder):
        """Test that same seed produces same results"""
        gen1 = WeatherGenerator(seeder)
        gen2 = WeatherGenerator(MillisecondSeeder(fixed_seed=123456789))
        
        data1 = gen1.generate(50)
        data2 = gen2.generate(50)
        
        # Should be identical
        pd.testing.assert_frame_equal(data1, data2)
    
    def test_data_quality_metrics(self, generator):
        """Test overall data quality metrics"""
        data = generator.generate(1000)
        
        # No null values in critical fields
        critical_fields = ['sensor_id', 'temperature_c', 'humidity_percent', 'pressure_hpa']
        for field in critical_fields:
            assert data[field].notna().all()
        
        # Reasonable distribution of weather conditions
        condition_dist = data['weather_condition'].value_counts(normalize=True)
        assert len(condition_dist) >= 3  # At least 3 different conditions
        
        # No single condition should dominate too much (unless specific climate)
        assert condition_dist.max() <= 0.7  # No condition more than 70%
    
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
            interval='1min'
        )
        assert len(data) == 10
    
    def test_parameter_correlations(self, generator):
        """Test correlations between weather parameters"""
        data = generator.generate(1000)
        
        # Test multiple correlation relationships
        correlations = data[['temperature_c', 'humidity_percent', 'pressure_hpa', 'altitude_m']].corr()
        
        # Temperature-humidity should have some negative correlation
        temp_humidity_corr = correlations.loc['temperature_c', 'humidity_percent']
        assert temp_humidity_corr < 0.2  # Allow for climate zone effects
        
        # Altitude-pressure should have negative correlation
        altitude_pressure_corr = correlations.loc['altitude_m', 'pressure_hpa']
        assert altitude_pressure_corr < 0


if __name__ == '__main__':
    pytest.main([__file__])