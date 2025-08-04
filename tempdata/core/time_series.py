"""
Time series generation system for TempData

Provides configuration and base functionality for generating time series datasets
with realistic temporal patterns, seasonal variations, and trend directions.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple, Union
import pandas as pd
import numpy as np
from .seeding import MillisecondSeeder
# BaseGenerator import moved to avoid circular dependency


@dataclass
class TimeSeriesConfig:
    """
    Configuration for time series data generation
    
    Defines temporal parameters, patterns, and behaviors for generating
    realistic time series datasets with seasonal variations and trends.
    """
    
    # Time range configuration
    start_date: datetime
    end_date: datetime
    interval: str = '1day'  # '1min', '5min', '1hour', '1day', '1week', '1month'
    
    # Pattern configuration
    seasonal_patterns: bool = True
    trend_direction: str = 'random'  # 'up', 'down', 'random', 'stable'
    volatility_level: float = 0.1  # 0.0 to 1.0
    
    # Advanced pattern options
    cyclical_patterns: bool = True
    noise_level: float = 0.05  # Random noise as percentage of value
    correlation_strength: float = 0.7  # For multi-variate time series
    
    # Seasonal pattern weights (0.0 to 1.0)
    seasonal_weights: Dict[str, float] = field(default_factory=lambda: {
        'daily': 0.3,      # Daily patterns (hourly variations)
        'weekly': 0.2,     # Weekly patterns (weekday/weekend)
        'monthly': 0.3,    # Monthly patterns (month-to-month variations)
        'yearly': 0.2      # Yearly patterns (seasonal variations)
    })
    
    # Trend configuration
    trend_strength: float = 0.1  # How strong the trend is
    trend_change_probability: float = 0.05  # Probability of trend reversal per period
    
    # Anomaly injection
    anomaly_probability: float = 0.02  # Probability of anomalies per data point
    anomaly_magnitude: float = 2.0  # Multiplier for anomaly values
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        
        if not 0.0 <= self.volatility_level <= 1.0:
            raise ValueError("volatility_level must be between 0.0 and 1.0")
        
        if not 0.0 <= self.noise_level <= 1.0:
            raise ValueError("noise_level must be between 0.0 and 1.0")
        
        if not 0.0 <= self.correlation_strength <= 1.0:
            raise ValueError("correlation_strength must be between 0.0 and 1.0")
        
        if self.trend_direction not in ['up', 'down', 'random', 'stable']:
            raise ValueError("trend_direction must be 'up', 'down', 'random', or 'stable'")
        
        # Validate seasonal weights sum to reasonable range
        total_weight = sum(self.seasonal_weights.values())
        if total_weight > 2.0:
            raise ValueError("Sum of seasonal weights should not exceed 2.0")
    
    def get_total_periods(self) -> int:
        """Calculate total number of periods in the time series"""
        return len(self.get_timestamps())
    
    def get_timestamps(self) -> List[datetime]:
        """Generate list of timestamps based on interval"""
        timestamps = []
        current = self.start_date
        
        # Parse interval
        interval_map = {
            '1min': timedelta(minutes=1),
            '5min': timedelta(minutes=5),
            '15min': timedelta(minutes=15),
            '30min': timedelta(minutes=30),
            '1hour': timedelta(hours=1),
            '2hour': timedelta(hours=2),
            '4hour': timedelta(hours=4),
            '6hour': timedelta(hours=6),
            '12hour': timedelta(hours=12),
            '1day': timedelta(days=1),
            '1week': timedelta(weeks=1),
            '1month': timedelta(days=30),  # Approximate
            '1quarter': timedelta(days=90),  # Approximate
            '1year': timedelta(days=365)  # Approximate
        }
        
        if self.interval not in interval_map:
            raise ValueError(f"Unsupported interval: {self.interval}")
        
        delta = interval_map[self.interval]
        
        while current <= self.end_date:
            timestamps.append(current)
            current += delta
        
        return timestamps


class TimeSeriesGenerator:
    """
    Base class for time series data generation
    
    Provides common functionality for generating realistic time series data
    with temporal patterns, seasonal variations, and configurable trends.
    """
    
    def __init__(self, seeder: MillisecondSeeder, locale: str = 'en_US'):
        """
        Initialize time series generator
        
        Args:
            seeder: MillisecondSeeder instance for reproducible randomness
            locale: Locale string for localization
        """
        self.seeder = seeder
        self.locale = locale
        self.ts_random = np.random.RandomState(seeder.get_contextual_seed('time_series'))
    
    def generate_time_series_base(self, config: TimeSeriesConfig, 
                                 base_value: float = 100.0,
                                 value_range: Tuple[float, float] = (0.0, float('inf'))) -> pd.DataFrame:
        """
        Generate base time series with temporal patterns
        
        Args:
            config: TimeSeriesConfig with generation parameters
            base_value: Starting/baseline value for the series
            value_range: (min, max) allowed values for clamping
            
        Returns:
            pd.DataFrame: Base time series with timestamp and value columns
        """
        timestamps = config.get_timestamps()
        values = []
        
        # Initialize trend and pattern generators
        trend_generator = self._create_trend_generator(config, len(timestamps))
        seasonal_generator = self._create_seasonal_generator(config, timestamps)
        
        current_value = base_value
        trend_direction = self._initialize_trend_direction(config.trend_direction)
        
        for i, timestamp in enumerate(timestamps):
            # Apply trend
            trend_factor = next(trend_generator)
            if config.trend_direction != 'stable':
                # Check for trend reversal
                if self.ts_random.random() < config.trend_change_probability:
                    trend_direction *= -1
                
                trend_change = trend_direction * config.trend_strength * trend_factor
                current_value *= (1 + trend_change)
            
            # Apply seasonal patterns
            seasonal_factor = seasonal_generator.get_seasonal_factor(timestamp, i)
            seasonal_value = current_value * (1 + seasonal_factor)
            
            # Apply volatility and noise
            volatility_factor = 1 + self.ts_random.normal(0, config.volatility_level)
            noise_factor = 1 + self.ts_random.normal(0, config.noise_level)
            
            final_value = seasonal_value * volatility_factor * noise_factor
            
            # Apply anomalies
            if self.ts_random.random() < config.anomaly_probability:
                anomaly_direction = 1 if self.ts_random.random() > 0.5 else -1
                final_value *= (1 + anomaly_direction * config.anomaly_magnitude * config.volatility_level)
            
            # Clamp to valid range
            final_value = max(value_range[0], min(value_range[1], final_value))
            values.append(final_value)
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'value': values
        })
    
    def _create_trend_generator(self, config: TimeSeriesConfig, length: int):
        """Create generator for trend values"""
        if config.trend_direction == 'stable':
            while True:
                yield 0.0
        else:
            # Generate smooth trend using sine wave with random phase
            phase = self.ts_random.random() * 2 * np.pi
            frequency = 1.0 / max(length // 10, 1)  # Complete cycle over 10% of data
            
            for i in range(length):
                trend_value = np.sin(2 * np.pi * frequency * i + phase) * 0.5
                yield trend_value
    
    def _create_seasonal_generator(self, config: TimeSeriesConfig, timestamps: List[datetime]):
        """Create generator for seasonal patterns"""
        return SeasonalPatternGenerator(config, timestamps, self.ts_random)
    
    def _initialize_trend_direction(self, trend_direction: str) -> int:
        """Initialize trend direction multiplier"""
        if trend_direction == 'up':
            return 1
        elif trend_direction == 'down':
            return -1
        elif trend_direction == 'random':
            return 1 if self.ts_random.random() > 0.5 else -1
        else:  # stable
            return 0
    
    def add_correlation(self, base_series: pd.DataFrame, 
                       config: TimeSeriesConfig,
                       correlation_strength: float = None) -> pd.DataFrame:
        """
        Add correlated series to base time series
        
        Args:
            base_series: Base time series DataFrame
            config: TimeSeriesConfig for correlation parameters
            correlation_strength: Override correlation strength
            
        Returns:
            pd.DataFrame: DataFrame with additional correlated column
        """
        if correlation_strength is None:
            correlation_strength = config.correlation_strength
        
        base_values = base_series['value'].values
        correlated_values = []
        
        for i, base_val in enumerate(base_values):
            # Generate correlated value
            if i == 0:
                correlated_val = base_val * (1 + self.ts_random.normal(0, 0.1))
            else:
                # Use previous value and correlation with base series change
                prev_corr = correlated_values[i-1]
                base_change = (base_val - base_values[i-1]) / base_values[i-1] if base_values[i-1] != 0 else 0
                
                # Apply correlation
                correlated_change = base_change * correlation_strength
                correlated_change += self.ts_random.normal(0, config.volatility_level * (1 - correlation_strength))
                
                correlated_val = prev_corr * (1 + correlated_change)
            
            correlated_values.append(correlated_val)
        
        result = base_series.copy()
        result['correlated_value'] = correlated_values
        return result


class SeasonalPatternGenerator:
    """
    Generator for seasonal patterns in time series data
    
    Handles daily, weekly, monthly, and yearly seasonal variations
    with configurable weights and realistic patterns.
    """
    
    def __init__(self, config: TimeSeriesConfig, timestamps: List[datetime], random_state: np.random.RandomState):
        """
        Initialize seasonal pattern generator
        
        Args:
            config: TimeSeriesConfig with seasonal parameters
            timestamps: List of timestamps for the series
            random_state: Random state for reproducible patterns
        """
        self.config = config
        self.timestamps = timestamps
        self.random_state = random_state
        
        # Pre-calculate pattern phases for consistency
        self.daily_phase = random_state.random() * 2 * np.pi
        self.weekly_phase = random_state.random() * 2 * np.pi
        self.monthly_phase = random_state.random() * 2 * np.pi
        self.yearly_phase = random_state.random() * 2 * np.pi
    
    def get_seasonal_factor(self, timestamp: datetime, index: int) -> float:
        """
        Calculate seasonal factor for given timestamp
        
        Args:
            timestamp: Current timestamp
            index: Index in the time series
            
        Returns:
            float: Seasonal adjustment factor (-1.0 to 1.0)
        """
        total_factor = 0.0
        
        # Apply seasonal patterns if enabled
        if self.config.seasonal_patterns:
            # Daily patterns (hour of day)
            if self.config.seasonal_weights.get('daily', 0) > 0:
                hour_factor = np.sin(2 * np.pi * timestamp.hour / 24 + self.daily_phase)
                total_factor += hour_factor * self.config.seasonal_weights['daily']
            
            # Weekly patterns (day of week)
            if self.config.seasonal_weights.get('weekly', 0) > 0:
                weekday_factor = np.sin(2 * np.pi * timestamp.weekday() / 7 + self.weekly_phase)
                total_factor += weekday_factor * self.config.seasonal_weights['weekly']
            
            # Monthly patterns (day of month)
            if self.config.seasonal_weights.get('monthly', 0) > 0:
                day_factor = np.sin(2 * np.pi * timestamp.day / 31 + self.monthly_phase)
                total_factor += day_factor * self.config.seasonal_weights['monthly']
            
            # Yearly patterns (day of year)
            if self.config.seasonal_weights.get('yearly', 0) > 0:
                day_of_year = timestamp.timetuple().tm_yday
                year_factor = np.sin(2 * np.pi * day_of_year / 365 + self.yearly_phase)
                total_factor += year_factor * self.config.seasonal_weights['yearly']
        
        # Add cyclical patterns if enabled (independent of seasonal patterns)
        if self.config.cyclical_patterns:
            # Add longer-term cycles
            cycle_length = len(self.timestamps) // 4  # Quarter of the series
            if cycle_length > 0:
                cycle_factor = np.sin(2 * np.pi * index / cycle_length)
                total_factor += cycle_factor * 0.1  # Small cyclical component
        
        return total_factor * self.config.volatility_level


def create_time_series_config(start_date: Union[str, datetime],
                            end_date: Union[str, datetime],
                            interval: str = '1day',
                            **kwargs) -> TimeSeriesConfig:
    """
    Convenience function to create TimeSeriesConfig with string dates
    
    Args:
        start_date: Start date as string (YYYY-MM-DD) or datetime
        end_date: End date as string (YYYY-MM-DD) or datetime
        interval: Time interval between data points
        **kwargs: Additional configuration parameters
        
    Returns:
        TimeSeriesConfig: Configured time series parameters
    """
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = datetime.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = datetime.fromisoformat(end_date)
    
    return TimeSeriesConfig(
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        **kwargs
    )