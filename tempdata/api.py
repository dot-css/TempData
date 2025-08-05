"""
Main API interface for TempData library

Provides the primary functions for dataset generation including
create_dataset() and create_batch() functions.
"""

from typing import List, Dict, Any, Optional, Union
import os
import re
from pathlib import Path

from .core.seeding import MillisecondSeeder
from .exporters.export_manager import ExportManager

# Import all available generators
from .datasets.business import SalesGenerator, CustomerGenerator, EcommerceGenerator
from .datasets.financial import StockGenerator, BankingGenerator
from .datasets.healthcare import PatientGenerator, AppointmentGenerator
from .datasets.technology import WebAnalyticsGenerator, SystemLogsGenerator
from .datasets.iot_sensors import WeatherGenerator, EnergyGenerator
from .datasets.social import SocialMediaGenerator, UserProfilesGenerator


# Dataset type mapping
DATASET_GENERATORS = {
    # Business datasets
    'sales': SalesGenerator,
    'customers': CustomerGenerator,
    'ecommerce': EcommerceGenerator,
    
    # Financial datasets
    'stocks': StockGenerator,
    'banking': BankingGenerator,
    
    # Healthcare datasets
    'patients': PatientGenerator,
    'appointments': AppointmentGenerator,
    
    # Technology datasets
    'web_analytics': WebAnalyticsGenerator,
    'system_logs': SystemLogsGenerator,
    
    # IoT sensor datasets
    'weather': WeatherGenerator,
    'energy': EnergyGenerator,
    
    # Social datasets
    'social_media': SocialMediaGenerator,
    'user_profiles': UserProfilesGenerator
}


def create_dataset(filename: str, rows: int = 500, **kwargs) -> str:
    """
    Generate single dataset with specified parameters
    
    Args:
        filename: Output filename (determines dataset type from name)
        rows: Number of rows to generate (default: 500)
        **kwargs: Additional parameters including:
            - country: Country code for geographical data (default: 'global')
            - seed: Fixed seed for reproducible results (default: None)
            - formats: List of export formats (default: ['csv'])
            - time_series: Enable time series generation (default: False)
            - start_date: Start date for time series (default: None)
            - end_date: End date for time series (default: None)
            - interval: Time series interval (default: '1day')
    
    Returns:
        str: Path to generated file(s) or comma-separated paths for multiple formats
        
    Raises:
        ValueError: If parameters are invalid
        FileNotFoundError: If dataset type is not supported
        IOError: If file generation fails
    """
    # Parameter validation
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename must be a non-empty string")
    
    if not isinstance(rows, int) or rows <= 0:
        raise ValueError("rows must be a positive integer")
    
    if rows > 10_000_000:  # 10 million row limit
        raise ValueError("rows cannot exceed 10,000,000 for performance reasons")
    
    # Extract dataset type from filename
    dataset_type = _extract_dataset_type(filename)
    
    if dataset_type not in DATASET_GENERATORS:
        available_types = ', '.join(sorted(DATASET_GENERATORS.keys()))
        raise FileNotFoundError(
            f"Unsupported dataset type '{dataset_type}'. "
            f"Available types: {available_types}"
        )
    
    # Extract and validate parameters
    country = kwargs.get('country', 'global')
    seed = kwargs.get('seed', None)
    formats = kwargs.get('formats', ['csv'])
    time_series = kwargs.get('time_series', False)
    
    # Validate formats
    if not isinstance(formats, list):
        formats = [formats] if isinstance(formats, str) else ['csv']
    
    export_manager = ExportManager()
    supported_formats = export_manager.get_supported_formats()
    invalid_formats = [f for f in formats if f not in supported_formats]
    if invalid_formats:
        raise ValueError(
            f"Unsupported export formats: {invalid_formats}. "
            f"Supported formats: {supported_formats}"
        )
    
    # Validate country parameter
    if not isinstance(country, str):
        raise ValueError("country must be a string")
    
    # Handle 'global' country by using a default locale
    if country == 'global':
        country = 'united_states'  # Default to US for global datasets
    
    # Validate time series parameters
    if time_series:
        _validate_time_series_params(kwargs)
    
    # Initialize seeder
    seeder = MillisecondSeeder(fixed_seed=seed)
    
    # Get proper locale from country using localization engine
    from .core.localization import LocalizationEngine
    localization = LocalizationEngine()
    locale = localization.get_locale(country)
    
    # Get generator class and create instance
    generator_class = DATASET_GENERATORS[dataset_type]
    generator = generator_class(seeder, locale)
    
    # Generate data
    try:
        if time_series:
            # Pass time series parameters to generator
            time_series_params = {
                'time_series': True,
                'start_date': kwargs.get('start_date'),
                'end_date': kwargs.get('end_date'),
                'interval': kwargs.get('interval', '1day')
            }
            data = generator.generate(rows, **time_series_params)
        else:
            # Filter out API-level parameters before passing to generator
            generator_kwargs = {k: v for k, v in kwargs.items() 
                              if k not in ['country', 'seed', 'formats', 'time_series']}
            data = generator.generate(rows, **generator_kwargs)
    except Exception as e:
        raise IOError(f"Failed to generate {dataset_type} data: {str(e)}")
    
    # Export data
    try:
        # Remove extension from filename for export manager
        base_filename = _remove_extension(filename)
        
        if len(formats) == 1:
            # Single format export
            result_path = export_manager.export_single(data, base_filename, formats[0])
            return result_path
        else:
            # Multiple format export
            result_paths = export_manager.export_multiple(data, base_filename, formats)
            return ', '.join(result_paths.values())
            
    except Exception as e:
        raise IOError(f"Failed to export data: {str(e)}")


def _extract_dataset_type(filename: str) -> str:
    """
    Extract dataset type from filename
    
    Args:
        filename: Input filename
        
    Returns:
        str: Dataset type
    """
    # Remove path and extension
    base_name = Path(filename).stem.lower()
    
    # Handle common patterns
    if 'sales' in base_name or 'transaction' in base_name:
        return 'sales'
    elif 'customer' in base_name or 'client' in base_name:
        return 'customers'
    elif 'ecommerce' in base_name or 'order' in base_name or 'shop' in base_name:
        return 'ecommerce'
    elif 'stock' in base_name or 'market' in base_name:
        return 'stocks'
    elif 'bank' in base_name or 'account' in base_name:
        return 'banking'
    elif 'patient' in base_name or 'medical' in base_name:
        return 'patients'
    elif 'appointment' in base_name or 'schedule' in base_name:
        return 'appointments'
    elif 'web' in base_name or 'analytics' in base_name:
        return 'web_analytics'
    elif 'log' in base_name or 'system' in base_name:
        return 'system_logs'
    elif 'weather' in base_name or 'climate' in base_name:
        return 'weather'
    elif 'energy' in base_name or 'power' in base_name:
        return 'energy'
    elif 'social' in base_name or 'post' in base_name:
        return 'social_media'
    elif 'user' in base_name or 'profile' in base_name:
        return 'user_profiles'
    else:
        # Default to the base name if no pattern matches
        return base_name


def _remove_extension(filename: str) -> str:
    """Remove file extension from filename"""
    return str(Path(filename).with_suffix(''))


def _validate_time_series_params(kwargs: Dict[str, Any]) -> None:
    """
    Validate time series parameters
    
    Args:
        kwargs: Parameters dictionary
        
    Raises:
        ValueError: If time series parameters are invalid
    """
    from datetime import datetime
    
    start_date = kwargs.get('start_date')
    end_date = kwargs.get('end_date')
    interval = kwargs.get('interval', '1day')
    
    # Validate dates if provided
    if start_date is not None:
        if isinstance(start_date, str):
            try:
                datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError("start_date must be in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
        elif not isinstance(start_date, datetime):
            raise ValueError("start_date must be a datetime object or ISO format string")
    
    if end_date is not None:
        if isinstance(end_date, str):
            try:
                datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError("end_date must be in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
        elif not isinstance(end_date, datetime):
            raise ValueError("end_date must be a datetime object or ISO format string")
    
    # Validate interval
    valid_intervals = ['1min', '5min', '15min', '30min', '1hour', '1day', '1week', '1month']
    if interval not in valid_intervals:
        raise ValueError(f"interval must be one of: {valid_intervals}")
    
    # Validate date range if both provided
    if start_date and end_date:
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        if start_date >= end_date:
            raise ValueError("start_date must be before end_date")


def create_batch(datasets: List[Dict[str, Any]], **kwargs) -> List[str]:
    """
    Generate multiple related datasets with maintained relationships
    
    Args:
        datasets: List of dataset specifications, each containing:
            - filename: Output filename
            - rows: Number of rows (optional, uses global default)
            - relationships: List of dataset names this depends on (optional)
            - Additional dataset-specific parameters
        **kwargs: Global parameters applied to all datasets including:
            - country: Country code for geographical data
            - seed: Base seed for reproducible results
            - formats: List of export formats
            - relationships: List of relationship specifications (optional)
            - Other parameters passed to individual generators
    
    Returns:
        List[str]: Paths to generated files
        
    Raises:
        ValueError: If datasets parameter is invalid
        IOError: If batch generation fails
    """
    from .core.batch_generator import BatchGenerator, DatasetSpec, RelationshipSpec
    
    # Parameter validation
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("datasets must be a non-empty list")
    
    if len(datasets) > 50:  # Reasonable limit for batch processing
        raise ValueError("Cannot process more than 50 datasets in a single batch")
    
    # Validate each dataset specification
    for i, dataset_spec in enumerate(datasets):
        if not isinstance(dataset_spec, dict):
            raise ValueError(f"Dataset {i} must be a dictionary")
        
        if 'filename' not in dataset_spec:
            raise ValueError(f"Dataset {i} must have a 'filename' key")
        
        if not isinstance(dataset_spec['filename'], str) or not dataset_spec['filename'].strip():
            raise ValueError(f"Dataset {i} filename must be a non-empty string")
    
    # Extract global parameters
    global_seed = kwargs.get('seed', None)
    global_rows = kwargs.get('rows', 500)
    
    # Initialize batch generator
    base_seeder = MillisecondSeeder(fixed_seed=global_seed)
    batch_generator = BatchGenerator(base_seeder)
    
    try:
        # Add datasets to batch generator
        for i, dataset_spec in enumerate(datasets):
            filename = dataset_spec['filename']
            rows = dataset_spec.get('rows', global_rows)
            dataset_type = _extract_dataset_type(filename)
            relationships = dataset_spec.get('relationships', [])
            
            # Extract custom parameters (excluding standard ones)
            custom_params = {k: v for k, v in dataset_spec.items() 
                           if k not in ['filename', 'rows', 'relationships']}
            
            # Create dataset specification
            spec = DatasetSpec(
                name=f"dataset_{i}_{dataset_type}",
                filename=filename,
                rows=rows,
                dataset_type=dataset_type,
                relationships=relationships,
                custom_params=custom_params
            )
            
            batch_generator.add_dataset(spec)
        
        # Add explicit relationships if provided
        explicit_relationships = kwargs.get('relationships', [])
        for rel_spec in explicit_relationships:
            if isinstance(rel_spec, dict):
                relationship = RelationshipSpec(
                    source_dataset=rel_spec['source_dataset'],
                    target_dataset=rel_spec['target_dataset'],
                    source_column=rel_spec['source_column'],
                    target_column=rel_spec['target_column'],
                    relationship_type=rel_spec.get('relationship_type', 'one_to_many'),
                    cascade_delete=rel_spec.get('cascade_delete', False)
                )
                batch_generator.add_relationship(relationship)
        
        # Generate batch with global parameters
        global_params = {k: v for k, v in kwargs.items() 
                        if k not in ['relationships']}
        
        result_dict = batch_generator.generate_batch(**global_params)
        
        # Convert result dictionary to list for backward compatibility
        return list(result_dict.values())
        
    except Exception as e:
        raise IOError(f"Batch generation failed: {str(e)}")


def _cleanup_batch_files(file_paths: List[str]) -> None:
    """
    Clean up files from failed batch generation
    
    Args:
        file_paths: List of file paths to clean up
    """
    for path_str in file_paths:
        # Handle comma-separated paths (multiple formats)
        paths = path_str.split(', ') if ', ' in path_str else [path_str]
        
        for path in paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass  # Ignore cleanup errors