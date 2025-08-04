"""
Base generator class for all dataset generators

Provides common functionality including seeding, localization, validation, and time series support.
"""

import pandas as pd
from faker import Faker
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
from .seeding import MillisecondSeeder
from .time_series import TimeSeriesConfig, TimeSeriesGenerator, create_time_series_config


class BaseGenerator:
    """
    Abstract base class for all dataset generators
    
    Provides common functionality including seeder integration, localization,
    and data validation that all specific generators inherit.
    """
    
    def __init__(self, seeder: MillisecondSeeder, locale: str = 'en_US'):
        """
        Initialize base generator with seeder and locale
        
        Args:
            seeder: MillisecondSeeder instance for reproducible randomness
            locale: Locale string for localization (default: 'en_US')
        """
        self.seeder = seeder
        self.locale = locale
        self.faker = Faker(locale)
        self.faker.seed_instance(seeder.get_contextual_seed(self.__class__.__name__))
        
        # Initialize time series generator for time-based data
        self.time_series_generator = TimeSeriesGenerator(seeder, locale)
    
    def generate(self, rows: int, **kwargs) -> pd.DataFrame:
        """
        Abstract method for generating dataset
        
        Args:
            rows: Number of rows to generate
            **kwargs: Additional generation parameters
            
        Returns:
            pd.DataFrame: Generated dataset
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement generate() method")
    
    def _apply_realistic_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply domain-specific realistic patterns to generated data
        
        Args:
            data: Generated data to apply patterns to
            
        Returns:
            pd.DataFrame: Data with realistic patterns applied
        """
        # Base implementation - subclasses should override for specific patterns
        return data
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate generated data meets quality standards
        
        Args:
            data: Generated data to validate
            
        Returns:
            bool: True if data passes validation
        """
        # Basic validation - subclasses can extend
        return not data.empty and len(data) > 0
    
    def _create_time_series_config(self, **kwargs) -> Optional[TimeSeriesConfig]:
        """
        Create time series configuration from kwargs
        
        Args:
            **kwargs: Parameters that may include time series configuration
            
        Returns:
            TimeSeriesConfig or None if not time series generation
        """
        if not kwargs.get('time_series', False):
            return None
        
        # Extract time series parameters
        start_date = kwargs.get('start_date', datetime.now() - timedelta(days=30))
        end_date = kwargs.get('end_date', datetime.now())
        interval = kwargs.get('interval', '1day')
        
        # Convert string dates if needed
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)
        
        return create_time_series_config(
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            trend_direction=kwargs.get('trend_direction', 'random'),
            seasonal_patterns=kwargs.get('seasonal_patterns', True),
            volatility_level=kwargs.get('volatility_level', 0.1),
            cyclical_patterns=kwargs.get('cyclical_patterns', True),
            noise_level=kwargs.get('noise_level', 0.05),
            correlation_strength=kwargs.get('correlation_strength', 0.7)
        )
    
    def _generate_time_series_timestamps(self, config: TimeSeriesConfig, rows: int) -> List[datetime]:
        """
        Generate timestamps for time series data
        
        Args:
            config: Time series configuration
            rows: Maximum number of rows requested
            
        Returns:
            List of timestamps
        """
        timestamps = config.get_timestamps()
        
        # Limit to requested rows if needed
        if len(timestamps) > rows:
            timestamps = timestamps[:rows]
        
        return timestamps
    
    def _apply_time_series_correlation(self, data: pd.DataFrame, 
                                     config: TimeSeriesConfig,
                                     value_column: str) -> pd.DataFrame:
        """
        Apply time series correlation to a value column
        
        Args:
            data: DataFrame with time series data
            config: Time series configuration
            value_column: Name of the column to correlate
            
        Returns:
            DataFrame with correlated time series
        """
        if value_column not in data.columns:
            return data
        
        # Create base time series from the value column
        base_series = pd.DataFrame({
            'timestamp': data['timestamp'] if 'timestamp' in data.columns else data.index,
            'value': data[value_column]
        })
        
        # Add correlation using time series generator
        correlated_series = self.time_series_generator.add_correlation(base_series, config)
        
        # Add correlated values back to original data
        data[f'{value_column}_correlated'] = correlated_series['correlated_value']
        
        return data
    
    def _maintain_cross_dataset_relationships(self, data: pd.DataFrame, 
                                            related_datasets: Dict[str, pd.DataFrame],
                                            relationship_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Maintain relationships between different dataset types in time series generation
        
        Args:
            data: Current dataset being generated
            related_datasets: Dictionary of related datasets {dataset_name: dataframe}
            relationship_config: Configuration for maintaining relationships
            
        Returns:
            DataFrame with maintained cross-dataset relationships
        """
        if 'timestamp' not in data.columns:
            return data
        
        # Sort data by timestamp for proper relationship maintenance
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        for dataset_name, related_df in related_datasets.items():
            if 'timestamp' not in related_df.columns:
                continue
                
            # Get relationship configuration for this dataset pair
            rel_config = relationship_config.get(dataset_name, {})
            
            if not rel_config:
                continue
            
            # Merge on timestamp to align data points
            merged = pd.merge_asof(
                data.sort_values('timestamp'),
                related_df.sort_values('timestamp'),
                on='timestamp',
                direction='nearest',
                suffixes=('', f'_{dataset_name}')
            )
            
            # Apply relationship rules
            for rule in rel_config.get('rules', []):
                data = self._apply_relationship_rule(data, merged, rule, dataset_name)
        
        return data
    
    def _apply_relationship_rule(self, data: pd.DataFrame, merged: pd.DataFrame, 
                               rule: Dict[str, Any], dataset_name: str) -> pd.DataFrame:
        """Apply a specific relationship rule between datasets"""
        rule_type = rule.get('type')
        source_column = rule.get('source_column')
        target_column = rule.get('target_column')
        correlation_strength = rule.get('correlation_strength', 0.7)
        
        if rule_type == 'positive_correlation':
            # Apply positive correlation between columns
            if f'{source_column}_{dataset_name}' in merged.columns and target_column in data.columns:
                source_values = merged[f'{source_column}_{dataset_name}'].fillna(0)
                source_normalized = (source_values - source_values.mean()) / (source_values.std() + 1e-8)
                
                # Apply correlation to target column
                target_adjustment = source_normalized * correlation_strength * data[target_column].std()
                data[target_column] += target_adjustment
                
        elif rule_type == 'negative_correlation':
            # Apply negative correlation between columns
            if f'{source_column}_{dataset_name}' in merged.columns and target_column in data.columns:
                source_values = merged[f'{source_column}_{dataset_name}'].fillna(0)
                source_normalized = (source_values - source_values.mean()) / (source_values.std() + 1e-8)
                
                # Apply negative correlation to target column
                target_adjustment = -source_normalized * correlation_strength * data[target_column].std()
                data[target_column] += target_adjustment
                
        elif rule_type == 'threshold_dependency':
            # Apply threshold-based dependency
            threshold = rule.get('threshold', 0)
            multiplier = rule.get('multiplier', 1.0)
            
            if f'{source_column}_{dataset_name}' in merged.columns and target_column in data.columns:
                source_values = merged[f'{source_column}_{dataset_name}'].fillna(0)
                threshold_mask = source_values > threshold
                data.loc[threshold_mask, target_column] *= multiplier
                
        elif rule_type == 'id_reference':
            # Maintain ID references between datasets
            id_column = rule.get('id_column')
            reference_column = rule.get('reference_column')
            
            if f'{reference_column}_{dataset_name}' in merged.columns and id_column in data.columns:
                # Use referenced IDs where available
                valid_refs = merged[f'{reference_column}_{dataset_name}'].notna()
                data.loc[valid_refs, id_column] = merged.loc[valid_refs, f'{reference_column}_{dataset_name}']
        
        return data
    
    def _add_temporal_relationships(self, data: pd.DataFrame, 
                                  config: TimeSeriesConfig) -> pd.DataFrame:
        """
        Add temporal relationships and patterns to time series data
        
        Args:
            data: DataFrame with time series data
            config: Time series configuration
            
        Returns:
            DataFrame with temporal relationships added
        """
        if 'timestamp' not in data.columns:
            return data
        
        # Sort by timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Add time-based features
        data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
        data['day_of_week'] = pd.to_datetime(data['timestamp']).dt.dayofweek
        data['month'] = pd.to_datetime(data['timestamp']).dt.month
        data['quarter'] = pd.to_datetime(data['timestamp']).dt.quarter
        
        # Add lag features for numeric columns
        numeric_columns = data.select_dtypes(include=['number']).columns
        for col in numeric_columns:
            if col not in ['hour', 'day_of_week', 'month', 'quarter']:
                # Add previous value (lag-1)
                data[f'{col}_prev'] = data[col].shift(1)
                
                # Add change from previous value
                data[f'{col}_change'] = data[col] - data[f'{col}_prev']
                
                # Add percentage change
                data[f'{col}_pct_change'] = (data[f'{col}_change'] / data[f'{col}_prev'] * 100).round(2)
        
        return data