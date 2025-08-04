"""
Unit tests for time series generation system

Tests TimeSeriesConfig, TimeSeriesGenerator, and seasonal pattern functionality.
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tempdata.core.time_series import (
    TimeSeriesConfig, 
    TimeSeriesGenerator, 
    SeasonalPatternGenerator,
    create_time_series_config
)
from tempdata.core.seeding import MillisecondSeeder


class TestTimeSeriesConfig:
    """Test TimeSeriesConfig dataclass functionality"""
    
    def test_basic_config_creation(self):
        """Test basic TimeSeriesConfig creation"""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        config = TimeSeriesConfig(
            start_date=start_date,
            end_date=end_date,
            interval='1day'
        )
        
        assert config.start_date == start_date
        assert config.end_date == end_date
        assert config.interval == '1day'
        assert config.seasonal_patterns is True
        assert config.trend_direction == 'random'
        assert config.volatility_level == 0.1
    
    def test_config_validation_invalid_dates(self):
        """Test validation of invalid date ranges"""
        start_date = datetime(2024, 1, 31)
        end_date = datetime(2024, 1, 1)  # End before start
        
        with pytest.raises(ValueError, match="start_date must be before end_date"):
            TimeSeriesConfig(start_date=start_date, end_date=end_date)
    
    def test_config_validation_invalid_volatility(self):
        """Test validation of invalid volatility levels"""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        with pytest.raises(ValueError, match="volatility_level must be between 0.0 and 1.0"):
            TimeSeriesConfig(
                start_date=start_date,
                end_date=end_date,
                volatility_level=1.5
            )
    
    def test_config_validation_invalid_trend_direction(self):
        """Test validation of invalid trend directions"""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        with pytest.raises(ValueError, match="trend_direction must be"):
            TimeSeriesConfig(
                start_date=start_date,
                end_date=end_date,
                trend_direction='invalid'
            )
    
    def test_config_validation_excessive_seasonal_weights(self):
        """Test validation of excessive seasonal weights"""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        with pytest.raises(ValueError, match="Sum of seasonal weights should not exceed 2.0"):
            TimeSeriesConfig(
                start_date=start_date,
                end_date=end_date,
                seasonal_weights={
                    'daily': 1.0,
                    'weekly': 1.0,
                    'monthly': 1.0,
                    'yearly': 1.0
                }
            )
    
    def test_get_timestamps_daily(self):
        """Test timestamp generation for daily intervals"""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 5)
        
        config = TimeSeriesConfig(
            start_date=start_date,
            end_date=end_date,
            interval='1day'
        )
        
        timestamps = config.get_timestamps()
        assert len(timestamps) == 5  # 1st, 2nd, 3rd, 4th, 5th
        assert timestamps[0] == start_date
        assert timestamps[-1] == end_date
        assert timestamps[1] == start_date + timedelta(days=1)
    
    def test_get_timestamps_hourly(self):
        """Test timestamp generation for hourly intervals"""
        start_date = datetime(2024, 1, 1, 0, 0)
        end_date = datetime(2024, 1, 1, 3, 0)
        
        config = TimeSeriesConfig(
            start_date=start_date,
            end_date=end_date,
            interval='1hour'
        )
        
        timestamps = config.get_timestamps()
        assert len(timestamps) == 4  # 0:00, 1:00, 2:00, 3:00
        assert timestamps[1] == start_date + timedelta(hours=1)
    
    def test_get_timestamps_minutes(self):
        """Test timestamp generation for minute intervals"""
        start_date = datetime(2024, 1, 1, 0, 0)
        end_date = datetime(2024, 1, 1, 0, 5)
        
        config = TimeSeriesConfig(
            start_date=start_date,
            end_date=end_date,
            interval='1min'
        )
        
        timestamps = config.get_timestamps()
        assert len(timestamps) == 6  # 0, 1, 2, 3, 4, 5 minutes
        assert timestamps[1] == start_date + timedelta(minutes=1)
    
    def test_get_total_periods(self):
        """Test total periods calculation"""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)
        
        config = TimeSeriesConfig(
            start_date=start_date,
            end_date=end_date,
            interval='1day'
        )
        
        total_periods = config.get_total_periods()
        assert total_periods == 10
    
    def test_unsupported_interval(self):
        """Test handling of unsupported intervals"""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        config = TimeSeriesConfig(
            start_date=start_date,
            end_date=end_date,
            interval='invalid_interval'
        )
        
        with pytest.raises(ValueError, match="Unsupported interval"):
            config.get_timestamps()


class TestTimeSeriesGenerator:
    """Test TimeSeriesGenerator functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.seeder = MillisecondSeeder(fixed_seed=12345)
        self.generator = TimeSeriesGenerator(self.seeder)
        
        self.config = TimeSeriesConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            interval='1day',
            volatility_level=0.1,
            trend_direction='up'
        )
    
    def test_generator_initialization(self):
        """Test TimeSeriesGenerator initialization"""
        assert self.generator.seeder == self.seeder
        assert hasattr(self.generator, 'ts_random')
        assert isinstance(self.generator.ts_random, np.random.RandomState)
    
    def test_generate_time_series_base_structure(self):
        """Test basic time series generation structure"""
        result = self.generator.generate_time_series_base(self.config)
        
        assert isinstance(result, pd.DataFrame)
        assert 'timestamp' in result.columns
        assert 'value' in result.columns
        assert len(result) == self.config.get_total_periods()
        assert all(result['value'] > 0)  # Should be positive values
    
    def test_generate_time_series_base_reproducibility(self):
        """Test reproducibility with fixed seed"""
        result1 = self.generator.generate_time_series_base(self.config)
        
        # Create new generator with same seed
        generator2 = TimeSeriesGenerator(MillisecondSeeder(fixed_seed=12345))
        result2 = generator2.generate_time_series_base(self.config)
        
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_generate_time_series_base_value_range(self):
        """Test value range clamping"""
        value_range = (50.0, 150.0)
        result = self.generator.generate_time_series_base(
            self.config, 
            base_value=100.0,
            value_range=value_range
        )
        
        assert all(result['value'] >= value_range[0])
        assert all(result['value'] <= value_range[1])
    
    def test_generate_time_series_trend_directions(self):
        """Test different trend directions"""
        # Test upward trend
        config_up = TimeSeriesConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            interval='1day',
            trend_direction='up',
            trend_strength=0.05,
            volatility_level=0.01  # Low volatility to see trend clearly
        )
        
        result_up = self.generator.generate_time_series_base(config_up)
        
        # Check that there's generally an upward trend
        first_half_avg = result_up['value'][:15].mean()
        second_half_avg = result_up['value'][15:].mean()
        assert second_half_avg > first_half_avg * 0.95  # Allow some variance
        
        # Test stable trend
        config_stable = TimeSeriesConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            interval='1day',
            trend_direction='stable',
            volatility_level=0.01
        )
        
        result_stable = self.generator.generate_time_series_base(config_stable)
        
        # Values should be relatively stable
        value_std = result_stable['value'].std()
        value_mean = result_stable['value'].mean()
        coefficient_of_variation = value_std / value_mean
        assert coefficient_of_variation < 0.1  # Low variation for stable trend
    
    def test_add_correlation(self):
        """Test adding correlated series"""
        base_series = self.generator.generate_time_series_base(self.config)
        correlated_series = self.generator.add_correlation(base_series, self.config)
        
        assert 'correlated_value' in correlated_series.columns
        assert len(correlated_series) == len(base_series)
        
        # Check correlation exists (should be positive with default settings)
        correlation = np.corrcoef(
            correlated_series['value'], 
            correlated_series['correlated_value']
        )[0, 1]
        
        # Should have some positive correlation (allowing for randomness)
        assert correlation > 0.1
    
    def test_add_correlation_strength(self):
        """Test correlation strength parameter"""
        base_series = self.generator.generate_time_series_base(self.config)
        
        # High correlation
        high_corr = self.generator.add_correlation(
            base_series, self.config, correlation_strength=0.9
        )
        
        # Low correlation
        low_corr = self.generator.add_correlation(
            base_series, self.config, correlation_strength=0.1
        )
        
        high_correlation = np.corrcoef(
            high_corr['value'], 
            high_corr['correlated_value']
        )[0, 1]
        
        low_correlation = np.corrcoef(
            low_corr['value'], 
            low_corr['correlated_value']
        )[0, 1]
        
        # High correlation should be stronger than low correlation
        assert high_correlation > low_correlation


class TestSeasonalPatternGenerator:
    """Test SeasonalPatternGenerator functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = TimeSeriesConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            interval='1day',
            seasonal_patterns=True,
            seasonal_weights={
                'daily': 0.2,
                'weekly': 0.3,
                'monthly': 0.3,
                'yearly': 0.2
            }
        )
        
        self.timestamps = self.config.get_timestamps()
        self.random_state = np.random.RandomState(12345)
        self.generator = SeasonalPatternGenerator(
            self.config, self.timestamps, self.random_state
        )
    
    def test_seasonal_generator_initialization(self):
        """Test SeasonalPatternGenerator initialization"""
        assert self.generator.config == self.config
        assert self.generator.timestamps == self.timestamps
        assert self.generator.random_state == self.random_state
        assert hasattr(self.generator, 'daily_phase')
        assert hasattr(self.generator, 'weekly_phase')
        assert hasattr(self.generator, 'monthly_phase')
        assert hasattr(self.generator, 'yearly_phase')
    
    def test_get_seasonal_factor_range(self):
        """Test seasonal factor is within expected range"""
        timestamp = datetime(2024, 6, 15, 12, 0)  # Mid-year, mid-day
        factor = self.generator.get_seasonal_factor(timestamp, 165)
        
        # Factor should be reasonable (not extreme)
        assert -1.0 <= factor <= 1.0
    
    def test_get_seasonal_factor_disabled(self):
        """Test seasonal factor when both seasonal and cyclical patterns are disabled"""
        config_no_patterns = TimeSeriesConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            interval='1day',
            seasonal_patterns=False,
            cyclical_patterns=False
        )
        
        generator = SeasonalPatternGenerator(
            config_no_patterns, self.timestamps, self.random_state
        )
        
        timestamp = datetime(2024, 6, 15, 12, 0)
        factor = generator.get_seasonal_factor(timestamp, 165)
        
        assert factor == 0.0
    
    def test_seasonal_factor_consistency(self):
        """Test seasonal factor consistency for same timestamp"""
        timestamp = datetime(2024, 6, 15, 12, 0)
        
        factor1 = self.generator.get_seasonal_factor(timestamp, 165)
        factor2 = self.generator.get_seasonal_factor(timestamp, 165)
        
        assert factor1 == factor2  # Should be deterministic
    
    def test_seasonal_patterns_vary_by_time(self):
        """Test that seasonal patterns vary across different times"""
        factors = []
        
        # Test different times of day
        for hour in [0, 6, 12, 18]:
            timestamp = datetime(2024, 6, 15, hour, 0)
            factor = self.generator.get_seasonal_factor(timestamp, 165)
            factors.append(factor)
        
        # Factors should vary (not all the same)
        assert len(set(factors)) > 1
    
    def test_cyclical_patterns(self):
        """Test cyclical patterns when enabled"""
        config_cyclical = TimeSeriesConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            interval='1day',
            cyclical_patterns=True,
            seasonal_patterns=False  # Disable seasonal to isolate cyclical
        )
        
        generator = SeasonalPatternGenerator(
            config_cyclical, self.timestamps, self.random_state
        )
        
        # Test different indices
        factor1 = generator.get_seasonal_factor(datetime(2024, 1, 1), 0)
        factor2 = generator.get_seasonal_factor(datetime(2024, 6, 1), 150)
        
        # Should have some variation due to cyclical patterns
        assert factor1 != factor2


class TestCreateTimeSeriesConfig:
    """Test create_time_series_config convenience function"""
    
    def test_create_config_with_string_dates(self):
        """Test creating config with string dates"""
        config = create_time_series_config(
            start_date='2024-01-01',
            end_date='2024-01-31',
            interval='1day'
        )
        
        assert config.start_date == datetime(2024, 1, 1)
        assert config.end_date == datetime(2024, 1, 31)
        assert config.interval == '1day'
    
    def test_create_config_with_datetime_objects(self):
        """Test creating config with datetime objects"""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        config = create_time_series_config(
            start_date=start_date,
            end_date=end_date,
            interval='1hour'
        )
        
        assert config.start_date == start_date
        assert config.end_date == end_date
        assert config.interval == '1hour'
    
    def test_create_config_with_additional_params(self):
        """Test creating config with additional parameters"""
        config = create_time_series_config(
            start_date='2024-01-01',
            end_date='2024-01-31',
            interval='1day',
            trend_direction='up',
            volatility_level=0.2,
            seasonal_patterns=False
        )
        
        assert config.trend_direction == 'up'
        assert config.volatility_level == 0.2
        assert config.seasonal_patterns is False


class TestTimeSeriesIntegration:
    """Integration tests for time series system"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.seeder = MillisecondSeeder(fixed_seed=12345)
        self.generator = TimeSeriesGenerator(self.seeder)
    
    def test_full_time_series_workflow(self):
        """Test complete time series generation workflow"""
        # Create configuration
        config = create_time_series_config(
            start_date='2024-01-01',
            end_date='2024-01-31',
            interval='1day',
            trend_direction='up',
            seasonal_patterns=True,
            volatility_level=0.1
        )
        
        # Generate base series
        base_series = self.generator.generate_time_series_base(config, base_value=100.0)
        
        # Add correlation
        correlated_series = self.generator.add_correlation(base_series, config)
        
        # Verify structure
        assert len(correlated_series) == 31  # January has 31 days
        assert 'timestamp' in correlated_series.columns
        assert 'value' in correlated_series.columns
        assert 'correlated_value' in correlated_series.columns
        
        # Verify data quality
        assert all(correlated_series['value'] > 0)
        assert all(correlated_series['correlated_value'] > 0)
        assert correlated_series['timestamp'].is_monotonic_increasing
    
    def test_different_intervals_consistency(self):
        """Test consistency across different time intervals"""
        intervals = ['1min', '1hour', '1day']
        
        for interval in intervals:
            if interval == '1min':
                start_date = '2024-01-01T00:00:00'
                end_date = '2024-01-01T00:10:00'
            elif interval == '1hour':
                start_date = '2024-01-01T00:00:00'
                end_date = '2024-01-01T10:00:00'
            else:  # 1day
                start_date = '2024-01-01'
                end_date = '2024-01-10'
            
            config = create_time_series_config(
                start_date=start_date,
                end_date=end_date,
                interval=interval
            )
            
            result = self.generator.generate_time_series_base(config)
            
            # Basic structure checks
            assert len(result) > 0
            assert 'timestamp' in result.columns
            assert 'value' in result.columns
            assert result['timestamp'].is_monotonic_increasing
    
    def test_anomaly_injection(self):
        """Test anomaly injection in time series"""
        config = TimeSeriesConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            interval='1day',
            anomaly_probability=0.5,  # High probability for testing
            anomaly_magnitude=3.0,
            volatility_level=0.01  # Low volatility to see anomalies clearly
        )
        
        result = self.generator.generate_time_series_base(config, base_value=100.0)
        
        # With high anomaly probability, we should see some extreme values
        values = result['value'].values
        mean_value = np.mean(values)
        std_value = np.std(values)
        
        # Check for outliers (values more than 2 standard deviations from mean)
        outliers = np.abs(values - mean_value) > 2 * std_value
        assert np.any(outliers)  # Should have some outliers due to anomalies
