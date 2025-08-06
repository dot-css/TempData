#!/usr/bin/env python3
"""
IoT Data Pipeline Example

This example demonstrates how to create comprehensive IoT sensor datasets
using TempData, including weather sensors, energy monitoring, and smart home
devices with realistic temporal patterns and correlations.
"""

import tempdata
import pandas as pd
from datetime import datetime, timedelta
import os
import json

def create_smart_city_pipeline():
    """
    Create a smart city IoT data pipeline with multiple sensor types
    """
    print("Creating Smart City IoT Pipeline...")
    print("=" * 40)
    
    # Create output directory
    output_dir = "smart_city_iot"
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Weather monitoring network
    print("\n1. Creating weather monitoring network...")
    
    # Multiple weather stations across the city
    weather_stations = ['downtown', 'airport', 'industrial', 'residential', 'coastal']
    
    for station in weather_stations:
        print(f"   - Generating weather data for {station} station...")
        weather_path = tempdata.create_dataset(
            f'{output_dir}/weather_{station}.csv',
            rows=8760,  # One year of hourly data
            time_series=True,
            start_date='2024-01-01',
            end_date='2024-12-31',
            interval='1hour',
            country='netherlands',  # Temperate climate
            seed=12345 + hash(station) % 1000  # Unique but reproducible seed per station
        )
        print(f"     ✓ Weather station {station}: {weather_path}")
    
    # Step 2: Energy monitoring system
    print("\n2. Creating energy monitoring system...")
    
    # City-wide energy consumption
    print("   - Generating city-wide energy consumption...")
    city_energy_path = tempdata.create_dataset(
        f'{output_dir}/city_energy_consumption.csv',
        rows=8760,  # Hourly for one year
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='1hour',
        country='germany',  # European energy patterns
        seed=12345
    )
    print(f"   ✓ City energy consumption: {city_energy_path}")
    
    # Residential energy monitoring
    print("   - Generating residential energy data...")
    residential_energy_path = tempdata.create_dataset(
        f'{output_dir}/residential_energy.csv',
        rows=35040,  # 15-minute intervals for one year (4 * 24 * 365)
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='15min',
        country='sweden',  # Nordic energy patterns
        seed=12345
    )
    print(f"   ✓ Residential energy: {residential_energy_path}")
    
    # Step 3: Traffic monitoring system
    print("\n3. Creating traffic monitoring system...")
    
    # Traffic sensors at major intersections
    traffic_locations = ['main_st_1st_ave', 'highway_101_exit', 'downtown_plaza', 'university_district']
    
    for location in traffic_locations:
        print(f"   - Generating traffic data for {location}...")
        traffic_path = tempdata.create_dataset(
            f'{output_dir}/traffic_{location}.csv',
            rows=17520,  # 30-minute intervals for one year
            time_series=True,
            start_date='2024-01-01',
            end_date='2024-12-31',
            interval='30min',
            seed=12345 + hash(location) % 1000
        )
        print(f"     ✓ Traffic sensor {location}: {traffic_path}")
    
    # Step 4: Environmental monitoring
    print("\n4. Creating environmental monitoring network...")
    
    # Air quality sensors
    print("   - Generating air quality data...")
    air_quality_path = tempdata.create_dataset(
        f'{output_dir}/air_quality_sensors.csv',
        rows=8760,  # Hourly measurements
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='1hour',
        country='germany',  # European environmental standards
        seed=12345
    )
    print(f"   ✓ Air quality sensors: {air_quality_path}")
    
    # Noise level monitoring
    print("   - Generating noise level data...")
    noise_path = tempdata.create_dataset(
        f'{output_dir}/noise_level_sensors.csv',
        rows=35040,  # 15-minute intervals
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='15min',
        seed=12345
    )
    print(f"   ✓ Noise level sensors: {noise_path}")
    
    print(f"\nSmart City IoT Pipeline complete in: {output_dir}/")
    return output_dir

def create_industrial_iot_pipeline():
    """
    Create an industrial IoT monitoring pipeline
    """
    print("\nCreating Industrial IoT Pipeline...")
    print("=" * 35)
    
    industrial_dir = "industrial_iot"
    os.makedirs(industrial_dir, exist_ok=True)
    
    # Step 1: Manufacturing equipment monitoring
    print("\n1. Creating manufacturing equipment monitoring...")
    
    # High-frequency machine data
    print("   - Generating machine sensor data (1-minute intervals)...")
    machine_sensors_path = tempdata.create_dataset(
        f'{industrial_dir}/machine_sensors.csv',
        rows=525600,  # One year of minute-by-minute data
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='1min',
        use_streaming=True,  # Large dataset requires streaming
        formats=['parquet'],  # Efficient format for large data
        seed=12345
    )
    print(f"   ✓ Machine sensors: {machine_sensors_path}")
    
    # Equipment maintenance data
    print("   - Generating equipment maintenance logs...")
    maintenance_path = tempdata.create_dataset(
        f'{industrial_dir}/equipment_maintenance.csv',
        rows=2000,  # Maintenance events over the year
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        seed=12345
    )
    print(f"   ✓ Equipment maintenance: {maintenance_path}")
    
    # Step 2: Quality control sensors
    print("\n2. Creating quality control monitoring...")
    
    # Product quality measurements
    print("   - Generating quality control data...")
    quality_path = tempdata.create_dataset(
        f'{industrial_dir}/quality_control.csv',
        rows=50000,  # Quality checks throughout the year
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='5min',
        seed=12345
    )
    print(f"   ✓ Quality control: {quality_path}")
    
    # Step 3: Environmental monitoring in factory
    print("\n3. Creating factory environmental monitoring...")
    
    # Temperature and humidity in production areas
    print("   - Generating factory environmental data...")
    factory_env_path = tempdata.create_dataset(
        f'{industrial_dir}/factory_environment.csv',
        rows=105120,  # 5-minute intervals for one year
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='5min',
        use_streaming=True,
        formats=['parquet'],
        seed=12345
    )
    print(f"   ✓ Factory environment: {factory_env_path}")
    
    print(f"\nIndustrial IoT Pipeline complete in: {industrial_dir}/")
    return industrial_dir

def create_smart_home_pipeline():
    """
    Create a smart home IoT data pipeline
    """
    print("\nCreating Smart Home IoT Pipeline...")
    print("=" * 35)
    
    smart_home_dir = "smart_home_iot"
    os.makedirs(smart_home_dir, exist_ok=True)
    
    # Step 1: Home automation sensors
    print("\n1. Creating home automation sensors...")
    
    # Smart thermostat data
    print("   - Generating smart thermostat data...")
    thermostat_path = tempdata.create_dataset(
        f'{smart_home_dir}/smart_thermostat.csv',
        rows=17520,  # 30-minute intervals
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='30min',
        country='canada',  # Cold climate patterns
        seed=12345
    )
    print(f"   ✓ Smart thermostat: {thermostat_path}")
    
    # Smart lighting system
    print("   - Generating smart lighting data...")
    lighting_path = tempdata.create_dataset(
        f'{smart_home_dir}/smart_lighting.csv',
        rows=35040,  # 15-minute intervals
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='15min',
        seed=12345
    )
    print(f"   ✓ Smart lighting: {lighting_path}")
    
    # Step 2: Security system
    print("\n2. Creating security system data...")
    
    # Motion sensors
    print("   - Generating motion sensor data...")
    motion_path = tempdata.create_dataset(
        f'{smart_home_dir}/motion_sensors.csv',
        rows=52560,  # 10-minute intervals
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='10min',
        seed=12345
    )
    print(f"   ✓ Motion sensors: {motion_path}")
    
    # Door/window sensors
    print("   - Generating door/window sensor data...")
    door_window_path = tempdata.create_dataset(
        f'{smart_home_dir}/door_window_sensors.csv',
        rows=8760,  # Hourly status checks
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='1hour',
        seed=12345
    )
    print(f"   ✓ Door/window sensors: {door_window_path}")
    
    # Step 3: Appliance monitoring
    print("\n3. Creating appliance monitoring...")
    
    # Smart appliances energy usage
    appliances = ['refrigerator', 'washing_machine', 'dishwasher', 'oven', 'water_heater']
    
    for appliance in appliances:
        print(f"   - Generating {appliance} data...")
        appliance_path = tempdata.create_dataset(
            f'{smart_home_dir}/{appliance}_usage.csv',
            rows=8760,  # Hourly monitoring
            time_series=True,
            start_date='2024-01-01',
            end_date='2024-12-31',
            interval='1hour',
            seed=12345 + hash(appliance) % 1000
        )
        print(f"     ✓ {appliance}: {appliance_path}")
    
    print(f"\nSmart Home IoT Pipeline complete in: {smart_home_dir}/")
    return smart_home_dir

def create_multi_format_iot_exports():
    """
    Create IoT datasets in multiple formats for different analytics tools
    """
    print("\nCreating Multi-Format IoT Exports...")
    print("=" * 35)
    
    export_dir = "iot_multi_format"
    os.makedirs(export_dir, exist_ok=True)
    
    # Real-time analytics format (JSON for streaming)
    print("1. Creating real-time analytics data...")
    realtime_paths = tempdata.create_dataset(
        f'{export_dir}/realtime_sensors',
        rows=1440,  # One day of minute data
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-01-01 23:59:00',
        interval='1min',
        formats=['json', 'csv'],
        seed=12345
    )
    print(f"   ✓ Real-time data: {realtime_paths}")
    
    # Historical analytics format (Parquet for data science)
    print("2. Creating historical analytics data...")
    historical_paths = tempdata.create_dataset(
        f'{export_dir}/historical_sensors',
        rows=8760,  # One year hourly
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='1hour',
        formats=['parquet', 'csv'],
        seed=12345
    )
    print(f"   ✓ Historical data: {historical_paths}")
    
    # Executive reporting format (Excel for business users)
    print("3. Creating executive reporting data...")
    executive_paths = tempdata.create_dataset(
        f'{export_dir}/executive_iot_summary',
        rows=52,  # Weekly summaries
        time_series=True,
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='1week',
        formats=['excel', 'json'],
        seed=12345
    )
    print(f"   ✓ Executive reports: {executive_paths}")
    
    # GIS mapping format (GeoJSON for location-based sensors)
    print("4. Creating GIS mapping data...")
    
    # Generate sensor locations
    sensor_locations = tempdata.geo.addresses('united_states', count=50)
    
    # Create GeoJSON-compatible sensor data
    gis_paths = tempdata.create_dataset(
        f'{export_dir}/sensor_locations',
        rows=50,
        formats=['geojson', 'csv'],
        seed=12345
    )
    print(f"   ✓ GIS mapping data: {gis_paths}")
    
    print(f"\nMulti-Format IoT Exports complete in: {export_dir}/")
    return export_dir

def analyze_iot_data(data_dirs):
    """
    Perform basic analysis on generated IoT data
    """
    print("\nPerforming IoT Data Analysis...")
    print("-" * 30)
    
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            continue
            
        print(f"\nAnalyzing {data_dir}:")
        
        # Find CSV files in the directory
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        for csv_file in csv_files[:3]:  # Analyze first 3 files
            try:
                file_path = os.path.join(data_dir, csv_file)
                df = pd.read_csv(file_path)
                
                print(f"  {csv_file}:")
                print(f"    Records: {len(df):,}")
                print(f"    Columns: {len(df.columns)}")
                
                # Check for time series data
                if 'timestamp' in df.columns:
                    print(f"    Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                elif 'date' in df.columns:
                    print(f"    Date range: {df['date'].min()} to {df['date'].max()}")
                
                # Show numeric column statistics
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    sample_col = numeric_cols[0]
                    print(f"    {sample_col} range: {df[sample_col].min():.2f} to {df[sample_col].max():.2f}")
                
            except Exception as e:
                print(f"    Error analyzing {csv_file}: {e}")

def create_iot_batch_pipeline():
    """
    Create related IoT datasets using batch generation
    """
    print("\nCreating IoT Batch Pipeline...")
    print("=" * 30)
    
    batch_dir = "iot_batch_pipeline"
    os.makedirs(batch_dir, exist_ok=True)
    
    # Define related IoT datasets
    iot_batch_datasets = [
        {
            'filename': f'{batch_dir}/sensor_registry.csv',
            'rows': 100  # Sensor device registry
        },
        {
            'filename': f'{batch_dir}/sensor_locations.csv',
            'rows': 100,
            'relationships': ['sensor_registry']  # Locations for each sensor
        },
        {
            'filename': f'{batch_dir}/sensor_readings.csv',
            'rows': 50000,
            'relationships': ['sensor_registry'],
            'time_series': True,
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'interval': '1hour'
        },
        {
            'filename': f'{batch_dir}/sensor_alerts.csv',
            'rows': 2000,
            'relationships': ['sensor_readings'],
            'time_series': True,
            'start_date': '2024-01-01',
            'end_date': '2024-12-31'
        }
    ]
    
    # Generate batch with relationships
    batch_paths = tempdata.create_batch(
        iot_batch_datasets,
        country='united_states',
        seed=12345
    )
    
    for path in batch_paths:
        print(f"   ✓ {path}")
    
    print(f"\nIoT Batch Pipeline complete in: {batch_dir}/")
    return batch_dir

def main():
    """
    Main function to run all IoT pipeline examples
    """
    print("TempData IoT Data Pipeline Examples")
    print("=" * 40)
    
    # Create different IoT pipelines
    smart_city_dir = create_smart_city_pipeline()
    industrial_dir = create_industrial_iot_pipeline()
    smart_home_dir = create_smart_home_pipeline()
    multi_format_dir = create_multi_format_iot_exports()
    batch_dir = create_iot_batch_pipeline()
    
    # Analyze generated data
    all_dirs = [smart_city_dir, industrial_dir, smart_home_dir, multi_format_dir, batch_dir]
    analyze_iot_data(all_dirs)
    
    print("\n" + "=" * 40)
    print("All IoT Pipeline Examples Complete!")
    print("\nGenerated directories:")
    print(f"  - Smart City IoT: {smart_city_dir}/")
    print(f"  - Industrial IoT: {industrial_dir}/")
    print(f"  - Smart Home IoT: {smart_home_dir}/")
    print(f"  - Multi-Format Exports: {multi_format_dir}/")
    print(f"  - Batch Pipeline: {batch_dir}/")
    
    print("\nUse cases for generated data:")
    print("  1. Time series analysis and forecasting")
    print("  2. Anomaly detection in sensor readings")
    print("  3. IoT dashboard development and testing")
    print("  4. Machine learning model training")
    print("  5. Real-time streaming analytics testing")
    print("  6. Edge computing simulation")
    print("  7. Predictive maintenance algorithms")

if __name__ == "__main__":
    main()