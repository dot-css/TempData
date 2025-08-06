#!/usr/bin/env python3
"""
Batch Generation Examples

This example demonstrates how to create related datasets using TempData's
batch generation capabilities, showing how to maintain referential integrity
and relationships between different dataset types.

This tutorial covers:
- Requirement 8.1: Batch generation with maintained referential integrity
- Requirement 2.1: Business datasets with realistic relationships
- Requirement 5.1: Time series generation across related datasets

The example creates comprehensive ecosystems suitable for:
- Testing complex database schemas
- ETL pipeline validation with referential integrity
- Data warehouse star/snowflake schema testing
- Multi-table analytics and reporting
- Relationship-aware machine learning datasets
"""

import tempdata
import pandas as pd
import os
from datetime import datetime

def create_ecommerce_ecosystem():
    """
    Create a complete e-commerce ecosystem with related datasets
    """
    print("Creating E-commerce Ecosystem...")
    print("=" * 35)
    
    output_dir = "ecommerce_ecosystem"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the complete e-commerce dataset relationships
    ecommerce_datasets = [
        # Foundation datasets (no dependencies)
        {
            'filename': f'{output_dir}/product_categories.csv',
            'rows': 50,
            'description': 'Product category master data'
        },
        {
            'filename': f'{output_dir}/regions.csv',
            'rows': 25,
            'description': 'Sales regions and territories'
        },
        {
            'filename': f'{output_dir}/suppliers.csv',
            'rows': 100,
            'description': 'Product suppliers and vendors'
        },
        
        # Second level - depends on foundation
        {
            'filename': f'{output_dir}/products.csv',
            'rows': 2000,
            'relationships': ['product_categories', 'suppliers'],
            'description': 'Product catalog with categories and suppliers'
        },
        {
            'filename': f'{output_dir}/customers.csv',
            'rows': 5000,
            'relationships': ['regions'],
            'description': 'Customer database with regional assignments'
        },
        {
            'filename': f'{output_dir}/warehouses.csv',
            'rows': 20,
            'relationships': ['regions'],
            'description': 'Warehouse locations by region'
        },
        
        # Third level - depends on second level
        {
            'filename': f'{output_dir}/inventory.csv',
            'rows': 10000,
            'relationships': ['products', 'warehouses'],
            'description': 'Inventory levels by product and warehouse'
        },
        {
            'filename': f'{output_dir}/customer_segments.csv',
            'rows': 5000,
            'relationships': ['customers'],
            'description': 'Customer segmentation and preferences'
        },
        
        # Fourth level - transactional data
        {
            'filename': f'{output_dir}/orders.csv',
            'rows': 25000,
            'relationships': ['customers', 'regions'],
            'time_series': True,
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'description': 'Customer orders with time series'
        },
        
        # Fifth level - order details
        {
            'filename': f'{output_dir}/order_items.csv',
            'rows': 75000,  # Multiple items per order
            'relationships': ['orders', 'products'],
            'description': 'Individual items within orders'
        },
        {
            'filename': f'{output_dir}/shipments.csv',
            'rows': 25000,
            'relationships': ['orders', 'warehouses'],
            'time_series': True,
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'description': 'Order shipment tracking'
        },
        
        # Analytics datasets
        {
            'filename': f'{output_dir}/customer_reviews.csv',
            'rows': 15000,
            'relationships': ['customers', 'products'],
            'time_series': True,
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'description': 'Product reviews and ratings'
        }
    ]
    
    print(f"\nGenerating {len(ecommerce_datasets)} related datasets...")
    
    # Generate all datasets with relationships
    batch_paths = tempdata.create_batch(
        ecommerce_datasets,
        country='united_states',
        formats=['csv', 'parquet'],  # Multiple formats
        seed=12345
    )
    
    # Display results
    print("\n✓ E-commerce ecosystem generated:")
    for i, dataset in enumerate(ecommerce_datasets):
        print(f"   {i+1:2d}. {dataset['filename'].split('/')[-1]:<25} ({dataset['rows']:,} rows)")
        print(f"       {dataset['description']}")
        if 'relationships' in dataset:
            deps = ', '.join(dataset['relationships'])
            print(f"       Dependencies: {deps}")
        print()
    
    print(f"All files saved in: {output_dir}/")
    return output_dir, ecommerce_datasets

def create_healthcare_system():
    """
    Create a healthcare system with patient, doctor, and appointment relationships
    """
    print("\nCreating Healthcare System...")
    print("=" * 30)
    
    healthcare_dir = "healthcare_system"
    os.makedirs(healthcare_dir, exist_ok=True)
    
    healthcare_datasets = [
        # Medical infrastructure
        {
            'filename': f'{healthcare_dir}/hospitals.csv',
            'rows': 25,
            'description': 'Hospital and clinic information'
        },
        {
            'filename': f'{healthcare_dir}/departments.csv',
            'rows': 100,
            'relationships': ['hospitals'],
            'description': 'Medical departments by hospital'
        },
        {
            'filename': f'{healthcare_dir}/doctors.csv',
            'rows': 200,
            'relationships': ['departments'],
            'description': 'Doctor profiles and specializations'
        },
        
        # Patient data
        {
            'filename': f'{healthcare_dir}/patients.csv',
            'rows': 10000,
            'description': 'Patient demographic and contact information'
        },
        {
            'filename': f'{healthcare_dir}/patient_medical_history.csv',
            'rows': 10000,
            'relationships': ['patients'],
            'description': 'Patient medical history and conditions'
        },
        
        # Appointments and visits
        {
            'filename': f'{healthcare_dir}/appointments.csv',
            'rows': 50000,
            'relationships': ['patients', 'doctors'],
            'time_series': True,
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'description': 'Scheduled appointments'
        },
        {
            'filename': f'{healthcare_dir}/visit_records.csv',
            'rows': 45000,  # Some appointments result in visits
            'relationships': ['appointments'],
            'description': 'Actual visit records and outcomes'
        },
        
        # Medical procedures and billing
        {
            'filename': f'{healthcare_dir}/procedures.csv',
            'rows': 30000,
            'relationships': ['visit_records', 'doctors'],
            'description': 'Medical procedures performed'
        },
        {
            'filename': f'{healthcare_dir}/prescriptions.csv',
            'rows': 40000,
            'relationships': ['visit_records', 'patients'],
            'time_series': True,
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'description': 'Prescription medications'
        },
        {
            'filename': f'{healthcare_dir}/billing_records.csv',
            'rows': 45000,
            'relationships': ['visit_records', 'procedures'],
            'description': 'Medical billing and insurance claims'
        }
    ]
    
    print(f"Generating {len(healthcare_datasets)} healthcare datasets...")
    
    batch_paths = tempdata.create_batch(
        healthcare_datasets,
        country='canada',  # Canadian healthcare system
        seed=12345
    )
    
    print("\n✓ Healthcare system generated:")
    for dataset in healthcare_datasets:
        filename = dataset['filename'].split('/')[-1]
        print(f"   • {filename:<30} ({dataset['rows']:,} rows)")
    
    print(f"\nHealthcare system saved in: {healthcare_dir}/")
    return healthcare_dir

def create_financial_institution():
    """
    Create a complete financial institution dataset ecosystem
    """
    print("\nCreating Financial Institution...")
    print("=" * 32)
    
    finance_dir = "financial_institution"
    os.makedirs(finance_dir, exist_ok=True)
    
    financial_datasets = [
        # Bank structure
        {
            'filename': f'{finance_dir}/branches.csv',
            'rows': 50,
            'description': 'Bank branch locations'
        },
        {
            'filename': f'{finance_dir}/employees.csv',
            'rows': 500,
            'relationships': ['branches'],
            'description': 'Bank employees by branch'
        },
        
        # Customer base
        {
            'filename': f'{finance_dir}/customers.csv',
            'rows': 20000,
            'description': 'Bank customer profiles'
        },
        {
            'filename': f'{finance_dir}/customer_kyc.csv',
            'rows': 20000,
            'relationships': ['customers'],
            'description': 'Know Your Customer compliance data'
        },
        
        # Account structure
        {
            'filename': f'{finance_dir}/account_types.csv',
            'rows': 20,
            'description': 'Types of bank accounts offered'
        },
        {
            'filename': f'{finance_dir}/accounts.csv',
            'rows': 35000,  # Customers can have multiple accounts
            'relationships': ['customers', 'account_types', 'branches'],
            'description': 'Customer bank accounts'
        },
        
        # Products and services
        {
            'filename': f'{finance_dir}/loan_products.csv',
            'rows': 15,
            'description': 'Available loan products'
        },
        {
            'filename': f'{finance_dir}/loans.csv',
            'rows': 8000,
            'relationships': ['customers', 'loan_products', 'employees'],
            'time_series': True,
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'description': 'Customer loans'
        },
        
        # Transactional data
        {
            'filename': f'{finance_dir}/transactions.csv',
            'rows': 500000,
            'relationships': ['accounts'],
            'time_series': True,
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'interval': '1hour',
            'description': 'All account transactions'
        },
        {
            'filename': f'{finance_dir}/atm_transactions.csv',
            'rows': 100000,
            'relationships': ['accounts', 'branches'],
            'time_series': True,
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'description': 'ATM transactions'
        },
        
        # Risk and compliance
        {
            'filename': f'{finance_dir}/fraud_alerts.csv',
            'rows': 2000,
            'relationships': ['transactions', 'accounts'],
            'time_series': True,
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'description': 'Fraud detection alerts'
        },
        {
            'filename': f'{finance_dir}/compliance_reports.csv',
            'rows': 1000,
            'relationships': ['customers', 'transactions'],
            'time_series': True,
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'description': 'Regulatory compliance reports'
        }
    ]
    
    print(f"Generating {len(financial_datasets)} financial datasets...")
    
    # Use streaming for large datasets
    batch_paths = tempdata.create_batch(
        financial_datasets,
        country='united_states',
        formats=['parquet', 'csv'],  # Parquet for large datasets
        seed=12345
    )
    
    print("\n✓ Financial institution generated:")
    for dataset in financial_datasets:
        filename = dataset['filename'].split('/')[-1]
        print(f"   • {filename:<25} ({dataset['rows']:,} rows)")
    
    print(f"\nFinancial institution saved in: {finance_dir}/")
    return finance_dir

def create_manufacturing_system():
    """
    Create a manufacturing system with supply chain relationships
    """
    print("\nCreating Manufacturing System...")
    print("=" * 30)
    
    manufacturing_dir = "manufacturing_system"
    os.makedirs(manufacturing_dir, exist_ok=True)
    
    manufacturing_datasets = [
        # Facilities and equipment
        {
            'filename': f'{manufacturing_dir}/factories.csv',
            'rows': 10,
            'description': 'Manufacturing facilities'
        },
        {
            'filename': f'{manufacturing_dir}/production_lines.csv',
            'rows': 50,
            'relationships': ['factories'],
            'description': 'Production lines by factory'
        },
        {
            'filename': f'{manufacturing_dir}/machines.csv',
            'rows': 200,
            'relationships': ['production_lines'],
            'description': 'Manufacturing equipment'
        },
        
        # Materials and products
        {
            'filename': f'{manufacturing_dir}/raw_materials.csv',
            'rows': 500,
            'description': 'Raw materials inventory'
        },
        {
            'filename': f'{manufacturing_dir}/products.csv',
            'rows': 1000,
            'description': 'Manufactured products'
        },
        {
            'filename': f'{manufacturing_dir}/bill_of_materials.csv',
            'rows': 5000,
            'relationships': ['products', 'raw_materials'],
            'description': 'Product recipes and material requirements'
        },
        
        # Production data
        {
            'filename': f'{manufacturing_dir}/production_orders.csv',
            'rows': 10000,
            'relationships': ['products', 'production_lines'],
            'time_series': True,
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'description': 'Production orders and schedules'
        },
        {
            'filename': f'{manufacturing_dir}/production_runs.csv',
            'rows': 15000,
            'relationships': ['production_orders', 'machines'],
            'time_series': True,
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'description': 'Actual production runs'
        },
        
        # Quality and maintenance
        {
            'filename': f'{manufacturing_dir}/quality_inspections.csv',
            'rows': 20000,
            'relationships': ['production_runs'],
            'time_series': True,
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'description': 'Quality control inspections'
        },
        {
            'filename': f'{manufacturing_dir}/machine_maintenance.csv',
            'rows': 3000,
            'relationships': ['machines'],
            'time_series': True,
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'description': 'Equipment maintenance records'
        }
    ]
    
    print(f"Generating {len(manufacturing_datasets)} manufacturing datasets...")
    
    batch_paths = tempdata.create_batch(
        manufacturing_datasets,
        country='germany',  # German manufacturing excellence
        seed=12345
    )
    
    print("\n✓ Manufacturing system generated:")
    for dataset in manufacturing_datasets:
        filename = dataset['filename'].split('/')[-1]
        print(f"   • {filename:<30} ({dataset['rows']:,} rows)")
    
    print(f"\nManufacturing system saved in: {manufacturing_dir}/")
    return manufacturing_dir

def analyze_batch_relationships(data_dir, datasets_config):
    """
    Analyze the relationships between generated batch datasets
    """
    print(f"\nAnalyzing Batch Relationships in {data_dir}...")
    print("-" * 50)
    
    # Load datasets and analyze relationships
    loaded_datasets = {}
    
    for dataset_config in datasets_config:
        filename = dataset_config['filename']
        csv_filename = filename.replace('.csv', '.csv')  # Ensure .csv extension
        
        if os.path.exists(csv_filename):
            try:
                df = pd.read_csv(csv_filename)
                dataset_name = os.path.basename(filename).replace('.csv', '')
                loaded_datasets[dataset_name] = df
                
                print(f"\n{dataset_name}:")
                print(f"  Rows: {len(df):,}")
                print(f"  Columns: {len(df.columns)}")
                
                # Check for ID columns that might be foreign keys
                id_columns = [col for col in df.columns if col.endswith('_id')]
                if id_columns:
                    print(f"  ID columns: {', '.join(id_columns)}")
                
                # Check for time series data
                if 'timestamp' in df.columns or 'date' in df.columns:
                    time_col = 'timestamp' if 'timestamp' in df.columns else 'date'
                    print(f"  Time range: {df[time_col].min()} to {df[time_col].max()}")
                
                # Show relationships if specified
                if 'relationships' in dataset_config:
                    print(f"  Depends on: {', '.join(dataset_config['relationships'])}")
                
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
    
    # Analyze cross-dataset relationships
    print(f"\nRelationship Analysis:")
    print("-" * 20)
    
    relationship_count = 0
    for dataset_config in datasets_config:
        if 'relationships' in dataset_config:
            dataset_name = os.path.basename(dataset_config['filename']).replace('.csv', '')
            relationships = dataset_config['relationships']
            
            print(f"\n{dataset_name} relationships:")
            for related_dataset in relationships:
                if dataset_name in loaded_datasets and related_dataset in loaded_datasets:
                    parent_df = loaded_datasets[related_dataset]
                    child_df = loaded_datasets[dataset_name]
                    
                    # Look for potential foreign key relationships
                    parent_id_col = f"{related_dataset.rstrip('s')}_id"  # Remove 's' and add '_id'
                    if parent_id_col in child_df.columns:
                        unique_parent_ids = child_df[parent_id_col].nunique()
                        total_parent_records = len(parent_df)
                        
                        print(f"  → {related_dataset}: {unique_parent_ids}/{total_parent_records} referenced")
                        relationship_count += 1
    
    print(f"\nTotal relationships analyzed: {relationship_count}")

def create_custom_relationship_example():
    """
    Create a custom example showing explicit relationship configuration
    """
    print("\nCreating Custom Relationship Example...")
    print("=" * 40)
    
    custom_dir = "custom_relationships"
    os.makedirs(custom_dir, exist_ok=True)
    
    # Define datasets with explicit relationship specifications
    custom_datasets = [
        {
            'filename': f'{custom_dir}/authors.csv',
            'rows': 100
        },
        {
            'filename': f'{custom_dir}/publishers.csv',
            'rows': 20
        },
        {
            'filename': f'{custom_dir}/books.csv',
            'rows': 1000,
            'relationships': ['authors', 'publishers']
        },
        {
            'filename': f'{custom_dir}/book_sales.csv',
            'rows': 10000,
            'relationships': ['books'],
            'time_series': True,
            'start_date': '2024-01-01',
            'end_date': '2024-12-31'
        }
    ]
    
    # Define explicit relationships
    explicit_relationships = [
        {
            'source_dataset': 'authors',
            'target_dataset': 'books',
            'source_column': 'author_id',
            'target_column': 'author_id',
            'relationship_type': 'one_to_many'
        },
        {
            'source_dataset': 'publishers',
            'target_dataset': 'books',
            'source_column': 'publisher_id',
            'target_column': 'publisher_id',
            'relationship_type': 'one_to_many'
        },
        {
            'source_dataset': 'books',
            'target_dataset': 'book_sales',
            'source_column': 'book_id',
            'target_column': 'book_id',
            'relationship_type': 'one_to_many'
        }
    ]
    
    print("Generating datasets with explicit relationships...")
    
    batch_paths = tempdata.create_batch(
        custom_datasets,
        relationships=explicit_relationships,
        country='united_kingdom',
        seed=12345
    )
    
    print("\n✓ Custom relationship example generated:")
    for path in batch_paths:
        filename = os.path.basename(path)
        print(f"   • {filename}")
    
    print(f"\nCustom relationships saved in: {custom_dir}/")
    return custom_dir, custom_datasets

def main():
    """
    Main function to run all batch generation examples
    """
    print("TempData Batch Generation Examples")
    print("=" * 40)
    
    # Create different batch generation examples
    print("Creating comprehensive batch generation examples...")
    
    # E-commerce ecosystem
    ecommerce_dir, ecommerce_config = create_ecommerce_ecosystem()
    
    # Healthcare system
    healthcare_dir = create_healthcare_system()
    
    # Financial institution
    finance_dir = create_financial_institution()
    
    # Manufacturing system
    manufacturing_dir = create_manufacturing_system()
    
    # Custom relationships
    custom_dir, custom_config = create_custom_relationship_example()
    
    # Analyze relationships in the e-commerce example
    analyze_batch_relationships(ecommerce_dir, ecommerce_config)
    
    print("\n" + "=" * 40)
    print("All Batch Generation Examples Complete!")
    print("\nGenerated directories:")
    print(f"  - E-commerce Ecosystem: {ecommerce_dir}/")
    print(f"  - Healthcare System: {healthcare_dir}/")
    print(f"  - Financial Institution: {finance_dir}/")
    print(f"  - Manufacturing System: {manufacturing_dir}/")
    print(f"  - Custom Relationships: {custom_dir}/")
    
    print("\nBatch generation benefits:")
    print("  1. Referential integrity maintained automatically")
    print("  2. Consistent data relationships across datasets")
    print("  3. Realistic foreign key distributions")
    print("  4. Time series alignment for related data")
    print("  5. Scalable to complex multi-table systems")
    print("  6. Support for custom relationship specifications")
    
    print("\nNext steps:")
    print("  1. Load related datasets into your database")
    print("  2. Verify foreign key relationships")
    print("  3. Test complex queries across related tables")
    print("  4. Use for ETL pipeline testing")
    print("  5. Create data warehouse star/snowflake schemas")

if __name__ == "__main__":
    main()