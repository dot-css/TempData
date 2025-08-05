"""
Unit tests for CLI command parsing and execution

Tests the command-line interface functionality including parameter validation,
command parsing, and integration with the API layer.
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from tempdata.cli.commands import cli, _validate_generate_params, _extract_dataset_type_from_filename, _get_file_size_str


class TestCLICommands:
    """Test CLI command functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cli_help(self):
        """Test CLI help command"""
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'TempData - Realistic fake data generation library' in result.output
        assert 'generate' in result.output
        assert 'batch' in result.output
    
    def test_cli_version(self):
        """Test CLI version command"""
        result = self.runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert '0.1.0' in result.output
    
    def test_generate_help(self):
        """Test generate command help"""
        result = self.runner.invoke(cli, ['generate', '--help'])
        assert result.exit_code == 0
        assert 'Generate a dataset with specified parameters' in result.output
        assert '--rows' in result.output
        assert '--country' in result.output
        assert '--formats' in result.output
        assert '--seed' in result.output
    
    @patch('tempdata.cli.commands.create_dataset')
    def test_generate_basic(self, mock_create_dataset):
        """Test basic generate command"""
        mock_create_dataset.return_value = 'sales.csv'
        
        result = self.runner.invoke(cli, ['generate', 'sales.csv'])
        
        assert result.exit_code == 0
        assert 'Generating sales dataset...' in result.output
        assert 'Successfully generated: sales.csv' in result.output
        
        mock_create_dataset.assert_called_once_with(
            'sales.csv', 500,
            country='global',
            seed=None,
            formats=['csv'],
            time_series=False,
            interval='1day'
        )
    
    @patch('tempdata.cli.commands.create_dataset')
    def test_generate_with_options(self, mock_create_dataset):
        """Test generate command with various options"""
        mock_create_dataset.return_value = 'customers.json'
        
        result = self.runner.invoke(cli, [
            'generate', 'customers.json',
            '--rows', '1000',
            '--country', 'pakistan',
            '--formats', 'csv,json',
            '--seed', '12345',
            '--verbose'
        ])
        
        assert result.exit_code == 0
        assert 'Generating customers dataset...' in result.output
        assert 'Rows: 1,000' in result.output
        assert 'Country: pakistan' in result.output
        assert 'Formats: csv, json' in result.output
        assert 'Seed: 12345' in result.output
        
        mock_create_dataset.assert_called_once_with(
            'customers.json', 1000,
            country='pakistan',
            seed=12345,
            formats=['csv', 'json'],
            time_series=False,
            interval='1day'
        )
    
    @patch('tempdata.cli.commands.create_dataset')
    def test_generate_time_series(self, mock_create_dataset):
        """Test generate command with time series options"""
        mock_create_dataset.return_value = 'weather.csv'
        
        result = self.runner.invoke(cli, [
            'generate', 'weather.csv',
            '--time-series',
            '--interval', '1hour',
            '--start-date', '2024-01-01',
            '--end-date', '2024-01-02'
        ])
        
        assert result.exit_code == 0
        
        mock_create_dataset.assert_called_once_with(
            'weather.csv', 500,
            country='global',
            seed=None,
            formats=['csv'],
            time_series=True,
            interval='1hour',
            start_date='2024-01-01',
            end_date='2024-01-02'
        )
    
    @patch('tempdata.cli.commands.create_dataset')
    def test_generate_multiple_formats(self, mock_create_dataset):
        """Test generate command with multiple output formats"""
        mock_create_dataset.return_value = 'stocks.csv, stocks.json, stocks.parquet'
        
        result = self.runner.invoke(cli, [
            'generate', 'stocks.csv',
            '--formats', 'csv,json,parquet'
        ])
        
        assert result.exit_code == 0
        assert 'Successfully generated 3 files:' in result.output
        assert 'stocks.csv' in result.output
        assert 'stocks.json' in result.output
        assert 'stocks.parquet' in result.output
    
    def test_generate_quiet_mode(self):
        """Test generate command in quiet mode"""
        with patch('tempdata.cli.commands.create_dataset') as mock_create_dataset:
            mock_create_dataset.return_value = 'sales.csv'
            
            result = self.runner.invoke(cli, ['generate', 'sales.csv', '--quiet'])
            
            assert result.exit_code == 0
            assert result.output.strip() == ''  # No output in quiet mode
    
    def test_generate_quiet_and_verbose_error(self):
        """Test error when both quiet and verbose flags are used"""
        result = self.runner.invoke(cli, ['generate', 'sales.csv', '--quiet', '--verbose'])
        
        assert result.exit_code == 1
        assert 'Cannot use both --quiet and --verbose flags' in result.output
    
    def test_generate_invalid_rows(self):
        """Test generate command with invalid rows parameter"""
        result = self.runner.invoke(cli, ['generate', 'sales.csv', '--rows', '0'])
        
        assert result.exit_code == 1
        assert 'rows must be a positive integer' in result.output
    
    def test_generate_too_many_rows(self):
        """Test generate command with too many rows"""
        result = self.runner.invoke(cli, ['generate', 'sales.csv', '--rows', '20000000'])
        
        assert result.exit_code == 1
        assert 'rows cannot exceed 10,000,000' in result.output
    
    def test_generate_invalid_dataset_type(self):
        """Test generate command with invalid dataset type"""
        result = self.runner.invoke(cli, ['generate', 'invalid_type.csv'])
        
        assert result.exit_code == 1
        assert 'Unsupported dataset type' in result.output
    
    def test_generate_invalid_formats(self):
        """Test generate command with invalid formats"""
        result = self.runner.invoke(cli, ['generate', 'sales.csv', '--formats', 'csv,invalid'])
        
        assert result.exit_code == 1
        assert 'Invalid formats: [\'invalid\']' in result.output
    
    def test_generate_invalid_interval(self):
        """Test generate command with invalid time series interval"""
        result = self.runner.invoke(cli, [
            'generate', 'weather.csv',
            '--time-series',
            '--interval', 'invalid'
        ])
        
        assert result.exit_code == 1
        assert 'Invalid interval \'invalid\'' in result.output
    
    def test_generate_invalid_date_range(self):
        """Test generate command with invalid date range"""
        result = self.runner.invoke(cli, [
            'generate', 'weather.csv',
            '--time-series',
            '--start-date', '2024-01-02',
            '--end-date', '2024-01-01'
        ])
        
        assert result.exit_code == 1
        assert 'start_date must be before end_date' in result.output
    
    def test_generate_api_error(self):
        """Test generate command when API raises an error"""
        with patch('tempdata.cli.commands.create_dataset') as mock_create_dataset:
            mock_create_dataset.side_effect = ValueError("Test error")
            
            result = self.runner.invoke(cli, ['generate', 'sales.csv'])
            
            assert result.exit_code == 1
            assert 'Error: Test error' in result.output
    
    def test_list_types_command(self):
        """Test list-types command"""
        result = self.runner.invoke(cli, ['list-types'])
        
        assert result.exit_code == 0
        assert 'Available dataset types:' in result.output
        assert 'Business:' in result.output
        assert 'Financial:' in result.output
        assert 'Healthcare:' in result.output
        assert 'sales' in result.output
        assert 'customers' in result.output
    
    def test_info_command_valid_type(self):
        """Test info command with valid dataset type"""
        result = self.runner.invoke(cli, ['info', 'sales'])
        
        assert result.exit_code == 0
        assert 'Dataset Type: sales' in result.output
        assert 'Example usage:' in result.output
        assert 'tempdata generate sales.csv' in result.output
    
    def test_info_command_invalid_type(self):
        """Test info command with invalid dataset type"""
        result = self.runner.invoke(cli, ['info', 'invalid_type'])
        
        assert result.exit_code == 1
        assert 'Unknown dataset type \'invalid_type\'' in result.output
        assert 'Available types:' in result.output
    
    def test_batch_command_placeholder(self):
        """Test batch command (placeholder implementation)"""
        result = self.runner.invoke(cli, ['batch', 'config.json'])
        
        assert result.exit_code == 1  # NotImplementedError causes exit code 1
        assert 'Generating batch datasets from config.json...' in result.output


class TestCLIHelperFunctions:
    """Test CLI helper functions"""
    
    def test_validate_generate_params_valid(self):
        """Test parameter validation with valid parameters"""
        # Should not raise any exception
        _validate_generate_params(
            filename='sales.csv',
            rows=1000,
            country='global',
            formats='csv,json',
            interval='1day',
            time_series=False,
            start_date=None,
            end_date=None
        )
    
    def test_validate_generate_params_empty_filename(self):
        """Test parameter validation with empty filename"""
        with pytest.raises(ValueError, match="filename cannot be empty"):
            _validate_generate_params(
                filename='',
                rows=1000,
                country='global',
                formats='csv',
                interval='1day',
                time_series=False,
                start_date=None,
                end_date=None
            )
    
    def test_validate_generate_params_invalid_rows(self):
        """Test parameter validation with invalid rows"""
        with pytest.raises(ValueError, match="rows must be a positive integer"):
            _validate_generate_params(
                filename='sales.csv',
                rows=0,
                country='global',
                formats='csv',
                interval='1day',
                time_series=False,
                start_date=None,
                end_date=None
            )
    
    def test_validate_generate_params_too_many_rows(self):
        """Test parameter validation with too many rows"""
        with pytest.raises(ValueError, match="rows cannot exceed 10,000,000"):
            _validate_generate_params(
                filename='sales.csv',
                rows=20_000_000,
                country='global',
                formats='csv',
                interval='1day',
                time_series=False,
                start_date=None,
                end_date=None
            )
    
    def test_validate_generate_params_invalid_dataset_type(self):
        """Test parameter validation with invalid dataset type"""
        with pytest.raises(ValueError, match="Unsupported dataset type"):
            _validate_generate_params(
                filename='invalid.csv',
                rows=1000,
                country='global',
                formats='csv',
                interval='1day',
                time_series=False,
                start_date=None,
                end_date=None
            )
    
    def test_validate_generate_params_invalid_formats(self):
        """Test parameter validation with invalid formats"""
        with pytest.raises(ValueError, match="Invalid formats"):
            _validate_generate_params(
                filename='sales.csv',
                rows=1000,
                country='global',
                formats='csv,invalid',
                interval='1day',
                time_series=False,
                start_date=None,
                end_date=None
            )
    
    def test_validate_generate_params_invalid_interval(self):
        """Test parameter validation with invalid interval"""
        with pytest.raises(ValueError, match="Invalid interval"):
            _validate_generate_params(
                filename='sales.csv',
                rows=1000,
                country='global',
                formats='csv',
                interval='invalid',
                time_series=False,
                start_date=None,
                end_date=None
            )
    
    def test_validate_generate_params_time_series_invalid_dates(self):
        """Test parameter validation with invalid time series dates"""
        with pytest.raises(ValueError, match="start_date must be before end_date"):
            _validate_generate_params(
                filename='weather.csv',
                rows=1000,
                country='global',
                formats='csv',
                interval='1day',
                time_series=True,
                start_date='2024-01-02',
                end_date='2024-01-01'
            )
    
    def test_extract_dataset_type_from_filename(self):
        """Test dataset type extraction from filename"""
        test_cases = [
            ('sales.csv', 'sales'),
            ('transactions.json', 'sales'),
            ('customers.parquet', 'customers'),
            ('clients.xlsx', 'customers'),
            ('ecommerce_orders.csv', 'ecommerce'),
            ('shop_data.json', 'ecommerce'),
            ('stock_prices.csv', 'stocks'),
            ('market_data.parquet', 'stocks'),
            ('bank_accounts.csv', 'banking'),
            ('patient_records.json', 'patients'),
            ('medical_data.csv', 'patients'),
            ('appointment_schedule.csv', 'appointments'),
            ('web_analytics.json', 'web_analytics'),
            ('system_logs.csv', 'system_logs'),
            ('weather_data.csv', 'weather'),
            ('climate_info.json', 'weather'),
            ('energy_consumption.csv', 'energy'),
            ('power_usage.parquet', 'energy'),
            ('social_posts.csv', 'social_media'),
            ('user_profiles.json', 'user_profiles'),
            ('unknown_type.csv', 'unknown_type')
        ]
        
        for filename, expected_type in test_cases:
            assert _extract_dataset_type_from_filename(filename) == expected_type
    
    def test_get_file_size_str(self):
        """Test file size string formatting"""
        # Create temporary files with known sizes
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Test small file (bytes)
            small_file = os.path.join(temp_dir, 'small.txt')
            with open(small_file, 'w') as f:
                f.write('x' * 100)  # 100 bytes
            
            size_str = _get_file_size_str(small_file)
            assert '100 B' == size_str
            
            # Test medium file (KB)
            medium_file = os.path.join(temp_dir, 'medium.txt')
            with open(medium_file, 'w') as f:
                f.write('x' * 2048)  # 2KB
            
            size_str = _get_file_size_str(medium_file)
            assert '2.0 KB' == size_str
            
            # Test non-existent file
            size_str = _get_file_size_str('/non/existent/file.txt')
            assert size_str == 'unknown size'
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestCLIBatchCommands:
    """Test CLI batch command functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_batch_help(self):
        """Test batch command help"""
        result = self.runner.invoke(cli, ['batch', '--help'])
        assert result.exit_code == 0
        assert 'Generate multiple related datasets from configuration file' in result.output
        assert '--output-dir' in result.output
        assert '--parallel' in result.output
        assert '--progress' in result.output
        assert '--dry-run' in result.output
    
    def test_batch_missing_config_file(self):
        """Test batch command with missing configuration file"""
        result = self.runner.invoke(cli, ['batch', 'nonexistent.json'])
        
        assert result.exit_code == 1
        assert 'Configuration file not found' in result.output
    
    def test_batch_invalid_json_config(self):
        """Test batch command with invalid JSON configuration"""
        config_file = os.path.join(self.temp_dir, 'invalid.json')
        with open(config_file, 'w') as f:
            f.write('{ invalid json }')
        
        result = self.runner.invoke(cli, ['batch', config_file])
        
        assert result.exit_code == 1
        assert 'Error loading configuration file' in result.output
    
    def test_batch_missing_datasets_key(self):
        """Test batch command with configuration missing datasets key"""
        config_file = os.path.join(self.temp_dir, 'no_datasets.json')
        config = {'global': {'country': 'pakistan'}}
        
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        result = self.runner.invoke(cli, ['batch', config_file])
        
        assert result.exit_code == 1
        assert "must contain a 'datasets' key" in result.output
    
    def test_batch_empty_datasets(self):
        """Test batch command with empty datasets list"""
        config_file = os.path.join(self.temp_dir, 'empty_datasets.json')
        config = {'datasets': []}
        
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        result = self.runner.invoke(cli, ['batch', config_file])
        
        assert result.exit_code == 1
        assert 'No datasets specified in configuration file' in result.output
    
    def test_batch_invalid_dataset_type(self):
        """Test batch command with invalid dataset type"""
        config_file = os.path.join(self.temp_dir, 'invalid_type.json')
        config = {
            'datasets': [
                {'filename': 'test.csv', 'type': 'invalid_type', 'rows': 100}
            ]
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        result = self.runner.invoke(cli, ['batch', config_file])
        
        assert result.exit_code == 1
        assert 'invalid type' in result.output
    
    def test_batch_dry_run(self):
        """Test batch command in dry run mode"""
        config_file = os.path.join(self.temp_dir, 'dry_run.json')
        config = {
            'global': {'country': 'pakistan', 'seed': 12345},
            'datasets': [
                {'filename': 'customers.csv', 'rows': 1000},
                {'filename': 'sales.csv', 'rows': 5000, 'relationships': ['customers']}
            ],
            'relationships': [
                {
                    'source_dataset': 'customers',
                    'target_dataset': 'sales',
                    'source_column': 'customer_id',
                    'target_column': 'customer_id'
                }
            ]
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        result = self.runner.invoke(cli, ['batch', config_file, '--dry-run'])
        
        assert result.exit_code == 0
        assert '=== DRY RUN - No files will be generated ===' in result.output
        assert 'Global Configuration:' in result.output
        assert 'Datasets to generate:' in result.output
        assert 'customers.csv' in result.output
        assert 'sales.csv' in result.output
        assert 'Relationships:' in result.output
        assert 'Total rows to generate: 6,000' in result.output
    
    @patch('tempdata.api.create_batch')
    def test_batch_successful_generation(self, mock_create_batch):
        """Test successful batch generation"""
        config_file = os.path.join(self.temp_dir, 'success.json')
        config = {
            'global': {'country': 'global', 'formats': ['csv']},
            'datasets': [
                {'filename': 'customers.csv', 'rows': 100},
                {'filename': 'sales.csv', 'rows': 200}
            ]
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        mock_create_batch.return_value = ['customers.csv', 'sales.csv']
        
        result = self.runner.invoke(cli, ['batch', config_file, '--no-progress'])
        
        assert result.exit_code == 0
        assert 'Starting batch generation...' in result.output
        assert 'Successfully generated 2 datasets' in result.output
        
        # Verify create_batch was called with correct parameters
        mock_create_batch.assert_called_once()
        call_args = mock_create_batch.call_args
        datasets = call_args[0][0]  # First positional argument
        
        assert len(datasets) == 2
        assert datasets[0]['filename'].endswith('customers.csv')
        assert datasets[1]['filename'].endswith('sales.csv')
    
    @patch('tempdata.api.create_batch')
    def test_batch_with_progress(self, mock_create_batch):
        """Test batch generation with progress indicators"""
        config_file = os.path.join(self.temp_dir, 'progress.json')
        config = {
            'datasets': [
                {'filename': 'customers.csv', 'rows': 100},
                {'filename': 'sales.csv', 'rows': 200}
            ]
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        mock_create_batch.return_value = ['customers.csv', 'sales.csv']
        
        # Test with tqdm available
        with patch('tqdm.tqdm') as mock_tqdm:
            mock_progress_bar = MagicMock()
            mock_tqdm.return_value.__enter__.return_value = mock_progress_bar
            
            result = self.runner.invoke(cli, ['batch', config_file, '--progress'])
            
            assert result.exit_code == 0
            mock_tqdm.assert_called_once()
            mock_progress_bar.update.assert_called_once_with(2)
    
    @patch('tempdata.api.create_batch')
    def test_batch_verbose_output(self, mock_create_batch):
        """Test batch generation with verbose output"""
        config_file = os.path.join(self.temp_dir, 'verbose.json')
        config = {
            'global': {'country': 'pakistan'},
            'datasets': [
                {'filename': 'customers.csv', 'rows': 100}
            ]
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        mock_create_batch.return_value = ['customers.csv']
        
        result = self.runner.invoke(cli, ['batch', config_file, '--verbose', '--no-progress'])
        
        assert result.exit_code == 0
        assert 'Output directory:' in result.output
        assert 'Global settings:' in result.output
        assert 'Datasets to generate: 1' in result.output
        assert 'Generated files:' in result.output
    
    def test_batch_quiet_and_verbose_error(self):
        """Test error when both quiet and verbose flags are used"""
        result = self.runner.invoke(cli, ['batch', 'config.json', '--quiet', '--verbose'])
        
        assert result.exit_code == 1
        assert 'Cannot use both --quiet and --verbose flags' in result.output
    
    @patch('tempdata.api.create_batch')
    def test_batch_quiet_mode(self, mock_create_batch):
        """Test batch generation in quiet mode"""
        config_file = os.path.join(self.temp_dir, 'quiet.json')
        config = {
            'datasets': [
                {'filename': 'customers.csv', 'rows': 100}
            ]
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        mock_create_batch.return_value = ['customers.csv']
        
        result = self.runner.invoke(cli, ['batch', config_file, '--quiet'])
        
        assert result.exit_code == 0
        # Should have minimal output in quiet mode
        assert 'Loading batch configuration' not in result.output
    
    def test_batch_output_directory_creation(self):
        """Test batch command creates output directory if it doesn't exist"""
        config_file = os.path.join(self.temp_dir, 'output_dir.json')
        output_dir = os.path.join(self.temp_dir, 'new_output_dir')
        
        config = {
            'datasets': [
                {'filename': 'customers.csv', 'rows': 100}
            ]
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        with patch('tempdata.api.create_batch') as mock_create_batch:
            mock_create_batch.return_value = [os.path.join(output_dir, 'customers.csv')]
            
            result = self.runner.invoke(cli, ['batch', config_file, '--output-dir', output_dir, '--no-progress'])
            
            assert result.exit_code == 0
            assert os.path.exists(output_dir)
    
    @patch('tempdata.api.create_batch')
    def test_batch_api_error(self, mock_create_batch):
        """Test batch command when API raises an error"""
        config_file = os.path.join(self.temp_dir, 'error.json')
        config = {
            'datasets': [
                {'filename': 'customers.csv', 'rows': 100}
            ]
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        mock_create_batch.side_effect = ValueError("Test batch error")
        
        result = self.runner.invoke(cli, ['batch', config_file])
        
        assert result.exit_code == 1
        assert 'Error during batch generation: Test batch error' in result.output
    
    def test_batch_yaml_config_support(self):
        """Test batch command with YAML configuration file"""
        config_file = os.path.join(self.temp_dir, 'config.yaml')
        yaml_content = """
global:
  country: pakistan
  seed: 12345
datasets:
  - filename: customers.csv
    rows: 1000
  - filename: sales.csv
    rows: 5000
    relationships: [customers]
relationships:
  - source_dataset: customers
    target_dataset: sales
    source_column: customer_id
    target_column: customer_id
"""
        
        with open(config_file, 'w') as f:
            f.write(yaml_content)
        
        # Test the YAML loading function directly
        from tempdata.cli.commands import _load_batch_config
        
        # Mock yaml import to raise ImportError
        import sys
        original_modules = sys.modules.copy()
        
        try:
            # Remove yaml from modules if it exists
            if 'yaml' in sys.modules:
                del sys.modules['yaml']
            
            # Mock the import to fail
            def mock_import(name, *args, **kwargs):
                if name == 'yaml':
                    raise ImportError("No module named 'yaml'")
                return __import__(name, *args, **kwargs)
            
            with patch('builtins.__import__', side_effect=mock_import):
                try:
                    _load_batch_config(config_file)
                    assert False, "Should have raised ValueError"
                except ValueError as e:
                    assert 'PyYAML is required for YAML configuration files' in str(e)
        finally:
            # Restore original modules
            sys.modules.clear()
            sys.modules.update(original_modules)
    
    def test_batch_parallel_option_note(self):
        """Test that parallel option shows appropriate note"""
        config_file = os.path.join(self.temp_dir, 'parallel.json')
        config = {
            'datasets': [
                {'filename': 'customers.csv', 'rows': 100}
            ]
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        with patch('tempdata.api.create_batch') as mock_create_batch:
            mock_create_batch.return_value = ['customers.csv']
            
            result = self.runner.invoke(cli, ['batch', config_file, '--parallel', '--no-progress'])
            
            assert result.exit_code == 0
            assert 'Parallel generation not yet implemented' in result.output
    
    @patch('tempdata.api.create_batch')
    def test_batch_large_dataset_progress(self, mock_create_batch):
        """Test batch generation with large datasets shows appropriate messages"""
        config_file = os.path.join(self.temp_dir, 'large.json')
        config = {
            'datasets': [
                {'filename': 'large_customers.csv', 'rows': 200000},  # Large dataset
                {'filename': 'small_sales.csv', 'rows': 100}
            ]
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        mock_create_batch.return_value = ['large_customers.csv', 'small_sales.csv']
        
        # Test with tqdm available and simulate some time passing
        with patch('tqdm.tqdm') as mock_tqdm:
            mock_progress_bar = MagicMock()
            mock_tqdm.return_value.__enter__.return_value = mock_progress_bar
            
            # Mock time.time to simulate duration
            with patch('time.time', side_effect=[0, 1.5]):  # 1.5 second duration
                result = self.runner.invoke(cli, ['batch', config_file, '--progress', '--verbose'])
                
                assert result.exit_code == 0
                assert 'Large datasets detected' in result.output
                assert 'Generating 2 datasets with 200,100 total rows' in result.output
                assert 'Performance:' in result.output
    
    def test_batch_examples_command(self):
        """Test the examples command"""
        result = self.runner.invoke(cli, ['examples'])
        
        assert result.exit_code == 0
        assert 'Example batch configuration files:' in result.output
        assert 'Usage:' in result.output
    
    @patch('tempdata.api.create_batch')
    def test_batch_performance_stats(self, mock_create_batch):
        """Test that batch generation shows performance statistics"""
        config_file = os.path.join(self.temp_dir, 'perf.json')
        config = {
            'datasets': [
                {'filename': 'customers.csv', 'rows': 1000},
                {'filename': 'sales.csv', 'rows': 2000}
            ]
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        mock_create_batch.return_value = ['customers.csv', 'sales.csv']
        
        # Mock file sizes
        def mock_getsize(path):
            if 'customers' in path:
                return 50000  # 50KB
            elif 'sales' in path:
                return 100000  # 100KB
            return 0
        
        with patch('os.path.getsize', side_effect=mock_getsize):
            with patch('tqdm.tqdm') as mock_tqdm:
                mock_progress_bar = MagicMock()
                mock_tqdm.return_value.__enter__.return_value = mock_progress_bar
                
                # Mock time.time to simulate duration > 0
                with patch('time.time', side_effect=[0, 2.0]):  # 2 second duration
                    result = self.runner.invoke(cli, ['batch', config_file, '--progress', '--verbose'])
                    
                    assert result.exit_code == 0
                    assert 'Performance:' in result.output
                    assert 'rows/second' in result.output
                    assert 'Total size:' in result.output


if __name__ == '__main__':
    pytest.main([__file__])