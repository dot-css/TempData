"""
Unit tests for batch generator functionality

Tests the BatchGenerator class for related dataset generation with
referential integrity maintenance across datasets.
"""

import pytest
import os
import tempfile
import shutil
import pandas as pd
from pathlib import Path

from tempdata.core.batch_generator import BatchGenerator, DatasetSpec, RelationshipSpec
from tempdata.core.seeding import MillisecondSeeder


class TestBatchGenerator:
    """Test cases for BatchGenerator class"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Initialize batch generator
        self.seeder = MillisecondSeeder(fixed_seed=12345)
        self.batch_generator = BatchGenerator(self.seeder)
    
    def teardown_method(self):
        """Clean up test environment"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_add_dataset_basic(self):
        """Test adding a basic dataset specification"""
        spec = DatasetSpec(
            name='sales_data',
            filename='sales.csv',
            rows=100,
            dataset_type='sales'
        )
        
        self.batch_generator.add_dataset(spec)
        
        assert 'sales_data' in self.batch_generator.datasets
        assert self.batch_generator.datasets['sales_data'] == spec
    
    def test_add_dataset_duplicate_name(self):
        """Test error handling for duplicate dataset names"""
        spec1 = DatasetSpec(
            name='sales_data',
            filename='sales1.csv',
            rows=100,
            dataset_type='sales'
        )
        
        spec2 = DatasetSpec(
            name='sales_data',
            filename='sales2.csv',
            rows=100,
            dataset_type='sales'
        )
        
        self.batch_generator.add_dataset(spec1)
        
        with pytest.raises(ValueError, match="Dataset 'sales_data' already exists"):
            self.batch_generator.add_dataset(spec2)
    
    def test_add_relationship_basic(self):
        """Test adding a basic relationship specification"""
        # Add datasets first
        customer_spec = DatasetSpec(
            name='customers',
            filename='customers.csv',
            rows=50,
            dataset_type='customers'
        )
        
        sales_spec = DatasetSpec(
            name='sales',
            filename='sales.csv',
            rows=200,
            dataset_type='sales',
            relationships=['customers']
        )
        
        self.batch_generator.add_dataset(customer_spec)
        self.batch_generator.add_dataset(sales_spec)
        
        # Add relationship
        relationship = RelationshipSpec(
            source_dataset='customers',
            target_dataset='sales',
            source_column='customer_id',
            target_column='customer_id',
            relationship_type='one_to_many'
        )
        
        self.batch_generator.add_relationship(relationship)
        
        assert len(self.batch_generator.relationships) == 1
        assert self.batch_generator.relationships[0] == relationship
    
    def test_add_relationship_invalid_dataset(self):
        """Test error handling for relationships with invalid datasets"""
        relationship = RelationshipSpec(
            source_dataset='nonexistent',
            target_dataset='sales',
            source_column='id',
            target_column='customer_id',
            relationship_type='one_to_many'
        )
        
        with pytest.raises(ValueError, match="Source dataset 'nonexistent' not found"):
            self.batch_generator.add_relationship(relationship)
    
    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies"""
        # Create circular dependency: A -> B -> C -> A
        spec_a = DatasetSpec(
            name='dataset_a',
            filename='a.csv',
            rows=10,
            dataset_type='sales',
            relationships=['dataset_c']  # A depends on C
        )
        
        spec_b = DatasetSpec(
            name='dataset_b',
            filename='b.csv',
            rows=10,
            dataset_type='customers',
            relationships=['dataset_a']  # B depends on A
        )
        
        spec_c = DatasetSpec(
            name='dataset_c',
            filename='c.csv',
            rows=10,
            dataset_type='ecommerce',
            relationships=['dataset_b']  # C depends on B
        )
        
        self.batch_generator.add_dataset(spec_a)
        self.batch_generator.add_dataset(spec_b)
        self.batch_generator.add_dataset(spec_c)
        
        with pytest.raises(ValueError, match="Circular dependencies detected"):
            self.batch_generator.generate_batch()
    
    def test_generation_order_calculation(self):
        """Test calculation of correct generation order"""
        # Create dependency chain: customers -> sales -> orders
        customers_spec = DatasetSpec(
            name='customers',
            filename='customers.csv',
            rows=50,
            dataset_type='customers'
        )
        
        sales_spec = DatasetSpec(
            name='sales',
            filename='sales.csv',
            rows=100,
            dataset_type='sales',
            relationships=['customers']
        )
        
        orders_spec = DatasetSpec(
            name='orders',
            filename='orders.csv',
            rows=200,
            dataset_type='ecommerce',
            relationships=['sales']
        )
        
        self.batch_generator.add_dataset(customers_spec)
        self.batch_generator.add_dataset(sales_spec)
        self.batch_generator.add_dataset(orders_spec)
        
        # Calculate generation order
        order = self.batch_generator._calculate_generation_order()
        
        # Customers should come first, then sales, then orders
        assert order.index('customers') < order.index('sales')
        assert order.index('sales') < order.index('orders')
    
    def test_simple_batch_generation(self):
        """Test generation of a simple batch without relationships"""
        sales_spec = DatasetSpec(
            name='sales',
            filename='sales.csv',
            rows=50,
            dataset_type='sales'
        )
        
        customers_spec = DatasetSpec(
            name='customers',
            filename='customers.csv',
            rows=30,
            dataset_type='customers'
        )
        
        self.batch_generator.add_dataset(sales_spec)
        self.batch_generator.add_dataset(customers_spec)
        
        # Generate batch
        results = self.batch_generator.generate_batch()
        
        assert len(results) == 2
        assert 'sales' in results
        assert 'customers' in results
        
        # Verify files exist
        for file_path in results.values():
            assert os.path.exists(file_path)
        
        # Verify row counts
        sales_df = pd.read_csv(results['sales'])
        customers_df = pd.read_csv(results['customers'])
        
        assert len(sales_df) == 50
        assert len(customers_df) == 30
    
    def test_batch_generation_with_relationships(self):
        """Test batch generation with maintained relationships"""
        # Create customers and sales with relationship
        customers_spec = DatasetSpec(
            name='customers',
            filename='customers.csv',
            rows=20,
            dataset_type='customers'
        )
        
        sales_spec = DatasetSpec(
            name='sales',
            filename='sales.csv',
            rows=100,
            dataset_type='sales',
            relationships=['customers']
        )
        
        self.batch_generator.add_dataset(customers_spec)
        self.batch_generator.add_dataset(sales_spec)
        
        # Add relationship specification
        relationship = RelationshipSpec(
            source_dataset='customers',
            target_dataset='sales',
            source_column='customer_id',
            target_column='customer_id',
            relationship_type='one_to_many'
        )
        
        self.batch_generator.add_relationship(relationship)
        
        # Generate batch
        results = self.batch_generator.generate_batch()
        
        assert len(results) == 2
        
        # Load generated data
        customers_df = pd.read_csv(results['customers'])
        sales_df = pd.read_csv(results['sales'])
        
        # Verify relationship integrity (if columns exist)
        if 'customer_id' in customers_df.columns and 'customer_id' in sales_df.columns:
            customer_ids = set(customers_df['customer_id'].unique())
            sales_customer_ids = set(sales_df['customer_id'].unique())
            
            # All sales customer_ids should exist in customers
            orphaned_ids = sales_customer_ids - customer_ids
            assert len(orphaned_ids) == 0, f"Orphaned customer IDs found: {orphaned_ids}"
    
    def test_batch_generation_with_custom_params(self):
        """Test batch generation with custom parameters"""
        sales_spec = DatasetSpec(
            name='sales',
            filename='sales.csv',
            rows=50,
            dataset_type='sales',
            custom_params={'region': 'north_america'}
        )
        
        self.batch_generator.add_dataset(sales_spec)
        
        # Generate with global parameters
        results = self.batch_generator.generate_batch(
            country='united_states',
            formats=['csv', 'json']
        )
        
        assert len(results) == 1
        
        # Should generate multiple formats
        result_path = results['sales']
        assert ', ' in result_path  # Multiple file paths
        
        # Verify both files exist
        paths = result_path.split(', ')
        for path in paths:
            assert os.path.exists(path)
    
    def test_relationship_validation(self):
        """Test relationship validation after generation"""
        customers_spec = DatasetSpec(
            name='customers',
            filename='customers.csv',
            rows=10,
            dataset_type='customers'
        )
        
        sales_spec = DatasetSpec(
            name='sales',
            filename='sales.csv',
            rows=50,
            dataset_type='sales'
        )
        
        self.batch_generator.add_dataset(customers_spec)
        self.batch_generator.add_dataset(sales_spec)
        
        # Generate without relationships first
        self.batch_generator.generate_batch()
        
        # Validate batch integrity
        integrity_result = self.batch_generator.validate_batch_integrity()
        
        assert integrity_result['status'] == 'valid'
        assert integrity_result['datasets_generated'] == 2
    
    def test_relationship_summary(self):
        """Test relationship summary generation"""
        customers_spec = DatasetSpec(
            name='customers',
            filename='customers.csv',
            rows=20,
            dataset_type='customers'
        )
        
        sales_spec = DatasetSpec(
            name='sales',
            filename='sales.csv',
            rows=100,
            dataset_type='sales',
            relationships=['customers']
        )
        
        self.batch_generator.add_dataset(customers_spec)
        self.batch_generator.add_dataset(sales_spec)
        
        relationship = RelationshipSpec(
            source_dataset='customers',
            target_dataset='sales',
            source_column='customer_id',
            target_column='customer_id',
            relationship_type='one_to_many'
        )
        
        self.batch_generator.add_relationship(relationship)
        
        # Get relationship summary
        summary = self.batch_generator.get_relationship_summary()
        
        assert summary['total_datasets'] == 2
        assert summary['total_relationships'] == 1
        assert len(summary['relationships']) == 1
        assert summary['relationships'][0]['source'] == 'customers'
        assert summary['relationships'][0]['target'] == 'sales'
        assert summary['relationships'][0]['type'] == 'one_to_many'
    
    def test_invalid_relationship_type(self):
        """Test error handling for invalid relationship types"""
        customers_spec = DatasetSpec(
            name='customers',
            filename='customers.csv',
            rows=20,
            dataset_type='customers'
        )
        
        sales_spec = DatasetSpec(
            name='sales',
            filename='sales.csv',
            rows=100,
            dataset_type='sales'
        )
        
        self.batch_generator.add_dataset(customers_spec)
        self.batch_generator.add_dataset(sales_spec)
        
        # Add invalid relationship
        relationship = RelationshipSpec(
            source_dataset='customers',
            target_dataset='sales',
            source_column='customer_id',
            target_column='customer_id',
            relationship_type='invalid_type'
        )
        
        self.batch_generator.add_relationship(relationship)
        
        with pytest.raises(ValueError, match="Invalid relationship type"):
            self.batch_generator.generate_batch()
    
    def test_cleanup_on_failure(self):
        """Test cleanup of generated files when batch generation fails"""
        # Create a spec that will cause failure
        invalid_spec = DatasetSpec(
            name='invalid',
            filename='invalid.csv',
            rows=10,
            dataset_type='nonexistent_type'  # This should cause failure
        )
        
        self.batch_generator.add_dataset(invalid_spec)
        
        with pytest.raises(IOError, match="Batch generation failed"):
            self.batch_generator.generate_batch()
        
        # Verify no files were left behind
        csv_files = list(Path('.').glob('*.csv'))
        json_files = list(Path('.').glob('*.json'))
        
        assert len(csv_files) == 0
        assert len(json_files) == 0


if __name__ == '__main__':
    pytest.main([__file__])