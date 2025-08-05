"""
Command-line interface commands for TempData

Provides CLI commands for dataset generation, batch operations, and configuration.
"""

import click
import sys
import os
import json
from typing import List, Optional
from pathlib import Path

# Import API functions
from ..api import create_dataset, DATASET_GENERATORS


@click.group()
@click.version_option(version='0.1.0')
def cli():
    """
    TempData - Realistic fake data generation library
    
    Generate realistic fake datasets for testing, development, and prototyping
    with worldwide geographical capabilities and time-based dynamic seeding.
    
    Examples:
    
        # Generate 1000 sales records
        tempdata generate sales.csv --rows 1000
        
        # Generate customer data for Pakistan in multiple formats
        tempdata generate customers.json --country pakistan --formats csv,json,excel
        
        # Generate reproducible stock data with fixed seed
        tempdata generate stocks.parquet --rows 5000 --seed 12345
        
        # Generate time series weather data
        tempdata generate weather.csv --time-series --interval 1hour --start-date 2024-01-01
    """
    pass


@cli.command()
@click.argument('filename')
@click.option('--rows', '-r', default=500, type=int, 
              help='Number of rows to generate (default: 500, max: 10,000,000)')
@click.option('--country', '-c', default='global', 
              help='Country for geographical data (default: global)')
@click.option('--formats', '-f', default='csv', 
              help='Export formats: csv,json,parquet,excel,geojson (comma-separated)')
@click.option('--seed', '-s', type=int, 
              help='Fixed seed for reproducible results')
@click.option('--time-series', is_flag=True, 
              help='Generate time series data with temporal patterns')
@click.option('--start-date', 
              help='Start date for time series (ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)')
@click.option('--end-date', 
              help='End date for time series (ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)')
@click.option('--interval', default='1day', 
              help='Time series interval: 1min,5min,15min,30min,1hour,1day,1week,1month')
@click.option('--verbose', '-v', is_flag=True, 
              help='Enable verbose output')
@click.option('--quiet', '-q', is_flag=True, 
              help='Suppress all output except errors')
def generate(filename: str, rows: int, country: str, formats: str, seed: Optional[int],
             time_series: bool, start_date: Optional[str], end_date: Optional[str], 
             interval: str, verbose: bool, quiet: bool):
    """
    Generate a dataset with specified parameters
    
    FILENAME: Output filename (determines dataset type from name)
    
    The dataset type is automatically detected from the filename:
    - sales.csv -> sales dataset
    - customers.json -> customers dataset  
    - stocks.parquet -> stocks dataset
    - etc.
    
    Examples:
    
        tempdata generate sales.csv --rows 1000
        tempdata generate customers.json --country pakistan --formats csv,json
        tempdata generate weather.csv --time-series --interval 1hour
    """
    # Handle quiet/verbose flags
    if quiet and verbose:
        click.echo("Error: Cannot use both --quiet and --verbose flags", err=True)
        sys.exit(1)
    
    # Validate parameters
    try:
        _validate_generate_params(filename, rows, country, formats, interval, 
                                time_series, start_date, end_date)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    
    # Parse formats
    format_list = [f.strip().lower() for f in formats.split(',')]
    
    # Build parameters dictionary
    params = {
        'country': country,
        'seed': seed,
        'formats': format_list,
        'time_series': time_series,
        'interval': interval
    }
    
    # Add time series parameters if provided
    if start_date:
        params['start_date'] = start_date
    if end_date:
        params['end_date'] = end_date
    
    # Show generation info unless quiet
    if not quiet:
        dataset_type = _extract_dataset_type_from_filename(filename)
        click.echo(f"Generating {dataset_type} dataset...")
        if verbose:
            click.echo(f"  Filename: {filename}")
            click.echo(f"  Rows: {rows:,}")
            click.echo(f"  Country: {country}")
            click.echo(f"  Formats: {', '.join(format_list)}")
            if seed is not None:
                click.echo(f"  Seed: {seed}")
            if time_series:
                click.echo(f"  Time series: enabled")
                click.echo(f"  Interval: {interval}")
                if start_date:
                    click.echo(f"  Start date: {start_date}")
                if end_date:
                    click.echo(f"  End date: {end_date}")
    
    # Generate dataset
    try:
        result_paths = create_dataset(filename, rows, **params)
        
        if not quiet:
            if ',' in result_paths:
                # Multiple files generated
                paths = result_paths.split(', ')
                click.echo(f"✓ Successfully generated {len(paths)} files:")
                for path in paths:
                    file_size = _get_file_size_str(path)
                    click.echo(f"  - {path} ({file_size})")
            else:
                # Single file generated
                file_size = _get_file_size_str(result_paths)
                click.echo(f"✓ Successfully generated: {result_paths} ({file_size})")
                
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('config_file')
@click.option('--output-dir', '-o', default='.', 
              help='Output directory for generated files (default: current directory)')
@click.option('--parallel', '-p', is_flag=True, 
              help='Enable parallel generation for better performance')
@click.option('--progress/--no-progress', default=True, 
              help='Show/hide progress indicators (default: show)')
@click.option('--verbose', '-v', is_flag=True, 
              help='Enable verbose output')
@click.option('--quiet', '-q', is_flag=True, 
              help='Suppress all output except errors')
@click.option('--dry-run', is_flag=True, 
              help='Show what would be generated without actually creating files')
def batch(config_file: str, output_dir: str, parallel: bool, progress: bool, 
          verbose: bool, quiet: bool, dry_run: bool):
    """
    Generate multiple related datasets from configuration file
    
    CONFIG_FILE: Path to batch configuration file (JSON or YAML)
    
    The configuration file should contain a list of dataset specifications
    with optional relationships between datasets.
    
    Example JSON configuration:
    
    \b
    {
      "global": {
        "country": "pakistan",
        "seed": 12345,
        "formats": ["csv", "json"]
      },
      "datasets": [
        {
          "filename": "customers.csv",
          "rows": 1000,
          "type": "customers"
        },
        {
          "filename": "sales.csv", 
          "rows": 5000,
          "type": "sales",
          "relationships": ["customers"]
        }
      ],
      "relationships": [
        {
          "source_dataset": "customers",
          "target_dataset": "sales",
          "source_column": "customer_id",
          "target_column": "customer_id"
        }
      ]
    }
    
    Examples:
    
        tempdata batch config.json
        tempdata batch config.yaml --output-dir ./data --parallel
        tempdata batch config.json --dry-run --verbose
    """
    # Handle quiet/verbose flags
    if quiet and verbose:
        click.echo("Error: Cannot use both --quiet and --verbose flags", err=True)
        sys.exit(1)
    
    # Validate configuration file
    try:
        config = _load_batch_config(config_file)
    except Exception as e:
        click.echo(f"Error loading configuration file: {e}", err=True)
        sys.exit(1)
    
    # Validate output directory
    output_path = Path(output_dir)
    if not output_path.exists():
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            click.echo(f"Error creating output directory: {e}", err=True)
            sys.exit(1)
    
    # Extract configuration
    global_config = config.get('global', {})
    datasets = config.get('datasets', [])
    relationships = config.get('relationships', [])
    
    if not datasets:
        click.echo("Error: No datasets specified in configuration file", err=True)
        sys.exit(1)
    
    # Show configuration summary unless quiet
    if not quiet:
        click.echo(f"Loading batch configuration from: {config_file}")
        if verbose or dry_run:
            click.echo(f"Output directory: {output_path.absolute()}")
            click.echo(f"Global settings: {global_config}")
            click.echo(f"Datasets to generate: {len(datasets)}")
            if relationships:
                click.echo(f"Relationships: {len(relationships)}")
            if parallel:
                click.echo("Parallel generation: enabled")
    
    # Dry run mode - show what would be generated
    if dry_run:
        _show_dry_run_summary(datasets, relationships, global_config, output_path)
        return
    
    # Prepare dataset specifications for API
    try:
        dataset_specs = []
        for i, dataset_config in enumerate(datasets):
            # Merge global and dataset-specific configuration
            merged_config = {**global_config, **dataset_config}
            
            # Ensure filename includes output directory
            filename = merged_config['filename']
            if not os.path.isabs(filename):
                filename = str(output_path / filename)
                merged_config['filename'] = filename
            
            dataset_specs.append(merged_config)
        
        # Add relationships to global config
        batch_params = {**global_config}
        if relationships:
            batch_params['relationships'] = relationships
        
        # Generate datasets with progress tracking
        if progress and not quiet:
            _generate_batch_with_progress(dataset_specs, batch_params, parallel, verbose)
        else:
            _generate_batch_simple(dataset_specs, batch_params, parallel, quiet, verbose)
            
    except Exception as e:
        click.echo(f"Error during batch generation: {e}", err=True)
        sys.exit(1)


@cli.command()
def list_types():
    """
    List all available dataset types
    """
    click.echo("Available dataset types:")
    click.echo()
    
    # Group by category
    categories = {
        'Business': ['sales', 'customers', 'ecommerce'],
        'Financial': ['stocks', 'banking'],
        'Healthcare': ['patients', 'appointments'],
        'Technology': ['web_analytics', 'system_logs'],
        'IoT Sensors': ['weather', 'energy'],
        'Social': ['social_media', 'user_profiles']
    }
    
    for category, types in categories.items():
        click.echo(f"{category}:")
        for dataset_type in types:
            if dataset_type in DATASET_GENERATORS:
                click.echo(f"  - {dataset_type}")
        click.echo()


@cli.command()
@click.argument('dataset_type')
def info(dataset_type: str):
    """
    Show information about a specific dataset type
    
    DATASET_TYPE: Name of the dataset type (e.g., 'sales', 'customers')
    """
    if dataset_type not in DATASET_GENERATORS:
        available_types = ', '.join(sorted(DATASET_GENERATORS.keys()))
        click.echo(f"Error: Unknown dataset type '{dataset_type}'", err=True)
        click.echo(f"Available types: {available_types}", err=True)
        sys.exit(1)
    
    generator_class = DATASET_GENERATORS[dataset_type]
    
    click.echo(f"Dataset Type: {dataset_type}")
    click.echo(f"Generator: {generator_class.__name__}")
    
    # Show docstring if available
    if generator_class.__doc__:
        click.echo(f"Description: {generator_class.__doc__.strip()}")
    
    click.echo()
    click.echo("Example usage:")
    click.echo(f"  tempdata generate {dataset_type}.csv --rows 1000")
    click.echo(f"  tempdata generate {dataset_type}.json --country pakistan")


@cli.command()
def examples():
    """
    Show example batch configuration files
    """
    click.echo("Example batch configuration files:")
    click.echo()
    
    examples_dir = Path(__file__).parent / 'examples'
    
    if examples_dir.exists():
        example_files = list(examples_dir.glob('*.json')) + list(examples_dir.glob('*.yaml'))
        
        for example_file in sorted(example_files):
            click.echo(f"• {example_file.name}")
            
            # Show brief description based on filename
            if 'simple' in example_file.name:
                click.echo("  Basic batch generation with multiple datasets")
            elif 'related' in example_file.name:
                click.echo("  Datasets with relationships and referential integrity")
            elif 'time_series' in example_file.name:
                click.echo("  Time series data generation with temporal patterns")
            elif 'large' in example_file.name:
                click.echo("  Large dataset generation with performance optimization")
            
            click.echo(f"  Location: {example_file}")
            click.echo()
        
        click.echo("Usage:")
        click.echo("  tempdata batch path/to/config.json")
        click.echo("  tempdata batch path/to/config.yaml --output-dir ./data")
    else:
        click.echo("No example files found. Example configurations should be in:")
        click.echo(f"  {examples_dir}")
        click.echo()
        click.echo("Basic JSON configuration structure:")
        click.echo("""
{
  "global": {
    "country": "global",
    "seed": 12345,
    "formats": ["csv"]
  },
  "datasets": [
    {
      "filename": "customers.csv",
      "rows": 1000,
      "type": "customers"
    }
  ]
}
        """.strip())


def _validate_generate_params(filename: str, rows: int, country: str, formats: str,
                            interval: str, time_series: bool, start_date: Optional[str], 
                            end_date: Optional[str]) -> None:
    """
    Validate generate command parameters
    
    Args:
        filename: Output filename
        rows: Number of rows
        country: Country code
        formats: Format string
        interval: Time series interval
        time_series: Time series flag
        start_date: Start date string
        end_date: End date string
        
    Raises:
        ValueError: If any parameter is invalid
    """
    # Validate filename
    if not filename or not filename.strip():
        raise ValueError("filename cannot be empty")
    
    # Validate rows
    if rows <= 0:
        raise ValueError("rows must be a positive integer")
    if rows > 10_000_000:
        raise ValueError("rows cannot exceed 10,000,000")
    
    # Validate dataset type
    dataset_type = _extract_dataset_type_from_filename(filename)
    if dataset_type not in DATASET_GENERATORS:
        available_types = ', '.join(sorted(DATASET_GENERATORS.keys()))
        raise ValueError(f"Unsupported dataset type '{dataset_type}'. Available: {available_types}")
    
    # Validate formats
    format_list = [f.strip().lower() for f in formats.split(',')]
    valid_formats = ['csv', 'json', 'parquet', 'excel', 'geojson']
    invalid_formats = [f for f in format_list if f not in valid_formats]
    if invalid_formats:
        raise ValueError(f"Invalid formats: {invalid_formats}. Valid: {valid_formats}")
    
    # Validate interval
    valid_intervals = ['1min', '5min', '15min', '30min', '1hour', '1day', '1week', '1month']
    if interval not in valid_intervals:
        raise ValueError(f"Invalid interval '{interval}'. Valid: {valid_intervals}")
    
    # Validate time series parameters
    if time_series:
        if start_date and end_date:
            from datetime import datetime
            try:
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                if start_dt >= end_dt:
                    raise ValueError("start_date must be before end_date")
            except ValueError as e:
                if "start_date must be before end_date" in str(e):
                    raise e
                raise ValueError("Invalid date format. Use ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS")


def _extract_dataset_type_from_filename(filename: str) -> str:
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


def _get_file_size_str(file_path: str) -> str:
    """
    Get human-readable file size string
    
    Args:
        file_path: Path to file
        
    Returns:
        str: Human-readable file size
    """
    try:
        size_bytes = os.path.getsize(file_path)
        return _format_bytes(size_bytes)
    except OSError:
        return "unknown size"


def _format_bytes(size_bytes: int) -> str:
    """
    Format bytes into human-readable string
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        str: Human-readable size string
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def _load_batch_config(config_file: str) -> dict:
    """
    Load batch configuration from JSON or YAML file
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        dict: Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file format is invalid
    """
    config_path = Path(config_file)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                try:
                    import yaml
                    config = yaml.safe_load(f)
                except ImportError:
                    raise ValueError("PyYAML is required for YAML configuration files. Install with: pip install PyYAML")
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                # Try to parse as JSON first, then YAML
                content = f.read()
                try:
                    config = json.loads(content)
                except json.JSONDecodeError:
                    try:
                        import yaml
                        config = yaml.safe_load(content)
                    except ImportError:
                        raise ValueError("Could not parse configuration file. Ensure it's valid JSON or install PyYAML for YAML support.")
                    except yaml.YAMLError:
                        raise ValueError("Configuration file must be valid JSON or YAML")
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        raise ValueError(f"Error reading configuration file: {e}")
    
    # Validate configuration structure
    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a JSON object or YAML mapping")
    
    if 'datasets' not in config:
        raise ValueError("Configuration file must contain a 'datasets' key")
    
    if not isinstance(config['datasets'], list):
        raise ValueError("'datasets' must be a list")
    
    # Validate each dataset specification
    for i, dataset in enumerate(config['datasets']):
        if not isinstance(dataset, dict):
            raise ValueError(f"Dataset {i} must be an object/mapping")
        
        if 'filename' not in dataset:
            raise ValueError(f"Dataset {i} must have a 'filename' field")
        
        # Set default values
        dataset.setdefault('rows', 500)
        
        # Validate dataset type if specified
        if 'type' in dataset:
            dataset_type = dataset['type']
            if dataset_type not in DATASET_GENERATORS:
                available_types = ', '.join(sorted(DATASET_GENERATORS.keys()))
                raise ValueError(f"Dataset {i} has invalid type '{dataset_type}'. Available: {available_types}")
    
    return config


def _show_dry_run_summary(datasets: List[dict], relationships: List[dict], 
                         global_config: dict, output_path: Path) -> None:
    """
    Show what would be generated in dry run mode
    
    Args:
        datasets: List of dataset configurations
        relationships: List of relationship configurations
        global_config: Global configuration
        output_path: Output directory path
    """
    click.echo()
    click.echo("=== DRY RUN - No files will be generated ===")
    click.echo()
    
    click.echo("Global Configuration:")
    for key, value in global_config.items():
        click.echo(f"  {key}: {value}")
    click.echo()
    
    click.echo("Datasets to generate:")
    total_rows = 0
    for i, dataset in enumerate(datasets, 1):
        filename = dataset['filename']
        rows = dataset.get('rows', 500)
        total_rows += rows
        
        # Make filename relative to output path for display
        if filename.startswith(str(output_path)):
            display_filename = os.path.relpath(filename, output_path)
        else:
            display_filename = filename
            
        click.echo(f"  {i}. {display_filename}")
        click.echo(f"     Rows: {rows:,}")
        
        dataset_type = dataset.get('type')
        if not dataset_type:
            dataset_type = _extract_dataset_type_from_filename(filename)
        click.echo(f"     Type: {dataset_type}")
        
        if 'relationships' in dataset:
            click.echo(f"     Depends on: {', '.join(dataset['relationships'])}")
        
        # Show merged configuration
        merged_config = {**global_config, **dataset}
        config_items = []
        for key, value in merged_config.items():
            if key not in ['filename', 'rows', 'type', 'relationships']:
                config_items.append(f"{key}={value}")
        if config_items:
            click.echo(f"     Config: {', '.join(config_items)}")
        click.echo()
    
    if relationships:
        click.echo("Relationships:")
        for i, rel in enumerate(relationships, 1):
            click.echo(f"  {i}. {rel['source_dataset']}.{rel['source_column']} -> {rel['target_dataset']}.{rel['target_column']}")
        click.echo()
    
    click.echo(f"Total rows to generate: {total_rows:,}")
    click.echo(f"Output directory: {output_path.absolute()}")


def _generate_batch_with_progress(dataset_specs: List[dict], batch_params: dict, 
                                parallel: bool, verbose: bool) -> None:
    """
    Generate batch datasets with progress indicators
    
    Args:
        dataset_specs: List of dataset specifications
        batch_params: Global batch parameters
        parallel: Whether to use parallel generation
        verbose: Whether to show verbose output
    """
    from ..api import create_batch
    import time
    
    # Import tqdm for progress bars
    try:
        from tqdm import tqdm
    except ImportError:
        click.echo("Warning: tqdm not available, falling back to simple progress", err=True)
        _generate_batch_simple(dataset_specs, batch_params, parallel, False, verbose)
        return
    
    click.echo("Starting batch generation...")
    
    if parallel:
        click.echo("Note: Parallel generation not yet implemented, using sequential generation")
    
    # Calculate total rows for more detailed progress
    total_rows = sum(spec.get('rows', 500) for spec in dataset_specs)
    
    # Show dataset summary
    if verbose:
        click.echo(f"Generating {len(dataset_specs)} datasets with {total_rows:,} total rows")
        for i, spec in enumerate(dataset_specs, 1):
            rows = spec.get('rows', 500)
            click.echo(f"  {i}. {spec['filename']} - {rows:,} rows")
        click.echo()
    
    # Show overall progress with estimated time
    with tqdm(total=len(dataset_specs), desc="Generating datasets", unit="dataset") as pbar:
        start_time = time.time()
        
        # Add nested progress for large datasets
        large_dataset_threshold = 100000
        has_large_datasets = any(spec.get('rows', 500) > large_dataset_threshold for spec in dataset_specs)
        
        if has_large_datasets:
            click.echo("Large datasets detected - this may take several minutes...")
        
        try:
            # Use the API's create_batch function
            result_paths = create_batch(dataset_specs, **batch_params)
            
            # Update progress bar
            pbar.update(len(dataset_specs))
            
            end_time = time.time()
            duration = end_time - start_time
            
            click.echo()
            click.echo(f"✓ Successfully generated {len(result_paths)} datasets in {duration:.2f}s")
            
            # Show performance stats
            if total_rows > 0 and duration > 0:
                rows_per_second = total_rows / duration
                click.echo(f"  Performance: {rows_per_second:,.0f} rows/second")
            
            if verbose:
                click.echo("Generated files:")
                total_size = 0
                for path in result_paths:
                    if isinstance(path, str) and ', ' in path:
                        # Multiple formats
                        files = path.split(', ')
                        for file_path in files:
                            file_size = _get_file_size_str(file_path)
                            click.echo(f"  - {file_path} ({file_size})")
                            try:
                                total_size += os.path.getsize(file_path)
                            except OSError:
                                pass
                    else:
                        file_size = _get_file_size_str(path)
                        click.echo(f"  - {path} ({file_size})")
                        try:
                            total_size += os.path.getsize(path)
                        except OSError:
                            pass
                
                if total_size > 0:
                    total_size_str = _format_bytes(total_size)
                    click.echo(f"  Total size: {total_size_str}")
                        
        except Exception as e:
            pbar.close()
            raise e


def _generate_batch_simple(dataset_specs: List[dict], batch_params: dict, 
                          parallel: bool, quiet: bool, verbose: bool) -> None:
    """
    Generate batch datasets with simple progress reporting
    
    Args:
        dataset_specs: List of dataset specifications
        batch_params: Global batch parameters
        parallel: Whether to use parallel generation
        quiet: Whether to suppress output
        verbose: Whether to show verbose output
    """
    from ..api import create_batch
    import time
    
    if not quiet:
        click.echo("Starting batch generation...")
    
    if parallel:
        click.echo("Note: Parallel generation not yet implemented, using sequential generation")
    
    start_time = time.time()
    
    try:
        # Use the API's create_batch function
        result_paths = create_batch(dataset_specs, **batch_params)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if not quiet:
            click.echo(f"✓ Successfully generated {len(result_paths)} datasets in {duration:.2f}s")
            
            if verbose:
                click.echo("Generated files:")
                for path in result_paths:
                    if isinstance(path, str) and ', ' in path:
                        # Multiple formats
                        files = path.split(', ')
                        for file_path in files:
                            file_size = _get_file_size_str(file_path)
                            click.echo(f"  - {file_path} ({file_size})")
                    else:
                        file_size = _get_file_size_str(path)
                        click.echo(f"  - {path} ({file_size})")
                        
    except Exception as e:
        raise e


if __name__ == '__main__':
    cli()