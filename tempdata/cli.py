"""
Command-line interface for TempData
"""

import click
from . import create_dataset, create_batch

@click.group()
@click.version_option()
def cli():
    """TempData - Realistic fake data generation library"""
    pass

@cli.command()
@click.argument('filename')
@click.option('--rows', default=500, help='Number of rows to generate')
@click.option('--country', default='united_states', help='Country for geographical data')
@click.option('--seed', type=int, help='Random seed for reproducible results')
@click.option('--formats', help='Comma-separated list of export formats')
@click.option('--time-series', is_flag=True, help='Generate time series data')
@click.option('--start-date', help='Start date for time series')
@click.option('--end-date', help='End date for time series')
@click.option('--interval', help='Time interval for time series')
def generate(filename, rows, country, seed, formats, time_series, start_date, end_date, interval):
    """Generate a dataset and save to file"""
    try:
        kwargs = {}
        if time_series:
            kwargs.update({
                'time_series': True,
                'start_date': start_date,
                'end_date': end_date,
                'interval': interval
            })
        
        if formats:
            kwargs['formats'] = formats.split(',')
        
        result = create_dataset(filename, rows=rows, country=country, seed=seed, **kwargs)
        click.echo(f"Generated: {result}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()

@cli.command()
@click.argument('config_file')
@click.option('--country', default='united_states', help='Country for geographical data')
@click.option('--seed', type=int, help='Random seed for reproducible results')
def batch(config_file, country, seed):
    """Generate multiple related datasets from configuration file"""
    try:
        import json
        
        with open(config_file, 'r') as f:
            datasets = json.load(f)
        
        results = create_batch(datasets, country=country, seed=seed)
        click.echo("Generated files:")
        for result in results:
            click.echo(f"  - {result}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()

if __name__ == '__main__':
    cli()