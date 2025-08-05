"""
Unit tests for GeoJSON exporter functionality
"""

import pytest
import pandas as pd
import json
import os
import tempfile
from datetime import datetime
from tempdata.exporters.geojson_exporter import GeoJSONExporter


class TestGeoJSONExporter:
    """Test cases for GeoJSON exporter"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.exporter = GeoJSONExporter()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample geographical data
        self.sample_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Location A', 'Location B', 'Location C', 'Location D', 'Location E'],
            'latitude': [40.7128, 34.0522, 41.8781, 29.7604, 39.9526],
            'longitude': [-74.0060, -118.2437, -87.6298, -95.3698, -75.1652],
            'category': ['restaurant', 'hotel', 'museum', 'park', 'office'],
            'rating': [4.5, 3.8, 4.2, 4.0, 3.9]
        })
        
        # Create route data for LineString testing
        self.route_data = pd.DataFrame({
            'route_id': [1, 1, 1, 2, 2, 2],
            'sequence': [1, 2, 3, 1, 2, 3],
            'latitude': [40.7128, 40.7589, 40.7831, 34.0522, 34.0689, 34.0851],
            'longitude': [-74.0060, -73.9851, -73.9712, -118.2437, -118.2468, -118.2501],
            'waypoint_name': ['Start', 'Middle', 'End', 'Begin', 'Center', 'Finish']
        })
    
    def teardown_method(self):
        """Clean up test fixtures"""
        # Clean up temporary files
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_basic_point_geojson_export(self):
        """Test basic Point GeoJSON export functionality"""
        filename = os.path.join(self.temp_dir, 'test_points.geojson')
        
        result_path = self.exporter.export(self.sample_data, filename)
        
        assert os.path.exists(result_path)
        assert result_path.endswith('.geojson')
        
        # Verify GeoJSON structure
        with open(result_path, 'r') as f:
            geojson_data = json.load(f)
        
        assert geojson_data['type'] == 'FeatureCollection'
        assert len(geojson_data['features']) == len(self.sample_data)
        
        # Check first feature
        first_feature = geojson_data['features'][0]
        assert first_feature['type'] == 'Feature'
        assert first_feature['geometry']['type'] == 'Point'
        assert len(first_feature['geometry']['coordinates']) == 2
        assert 'properties' in first_feature
    
    def test_linestring_geojson_export(self):
        """Test LineString GeoJSON export functionality"""
        filename = os.path.join(self.temp_dir, 'test_lines.geojson')
        
        result_path = self.exporter.export(
            self.route_data, 
            filename,
            geometry_type='LineString',
            group_by='route_id',
            order_by='sequence'
        )
        
        assert os.path.exists(result_path)
        
        # Verify GeoJSON structure
        with open(result_path, 'r') as f:
            geojson_data = json.load(f)
        
        assert geojson_data['type'] == 'FeatureCollection'
        assert len(geojson_data['features']) == 2  # Two routes
        
        # Check first feature
        first_feature = geojson_data['features'][0]
        assert first_feature['type'] == 'Feature'
        assert first_feature['geometry']['type'] == 'LineString'
        assert len(first_feature['geometry']['coordinates']) == 3  # Three points per route
    
    def test_polygon_geojson_export(self):
        """Test Polygon GeoJSON export functionality"""
        # Create polygon data (square around NYC)
        polygon_data = pd.DataFrame({
            'polygon_id': [1, 1, 1, 1, 1],  # Same polygon
            'sequence': [1, 2, 3, 4, 5],
            'latitude': [40.7, 40.8, 40.8, 40.7, 40.7],  # Square coordinates
            'longitude': [-74.1, -74.1, -73.9, -73.9, -74.1],
            'area_name': ['Central Park Area'] * 5
        })
        
        filename = os.path.join(self.temp_dir, 'test_polygon.geojson')
        
        result_path = self.exporter.export(
            polygon_data,
            filename,
            geometry_type='Polygon',
            order_by='sequence'
        )
        
        assert os.path.exists(result_path)
        
        # Verify GeoJSON structure
        with open(result_path, 'r') as f:
            geojson_data = json.load(f)
        
        assert geojson_data['type'] == 'FeatureCollection'
        assert len(geojson_data['features']) == 1
        
        # Check polygon feature
        polygon_feature = geojson_data['features'][0]
        assert polygon_feature['type'] == 'Feature'
        assert polygon_feature['geometry']['type'] == 'Polygon'
        
        # Polygon coordinates should be closed (first == last)
        coords = polygon_feature['geometry']['coordinates'][0]
        assert coords[0] == coords[-1]
    
    def test_custom_coordinate_columns(self):
        """Test GeoJSON export with custom coordinate column names"""
        custom_data = self.sample_data.rename(columns={
            'latitude': 'lat',
            'longitude': 'lng'
        })
        
        filename = os.path.join(self.temp_dir, 'test_custom_coords.geojson')
        
        result_path = self.exporter.export(
            custom_data,
            filename,
            lat_col='lat',
            lon_col='lng'
        )
        
        assert os.path.exists(result_path)
        
        # Verify coordinates are correct
        with open(result_path, 'r') as f:
            geojson_data = json.load(f)
        
        first_coords = geojson_data['features'][0]['geometry']['coordinates']
        assert first_coords[0] == custom_data.iloc[0]['lng']  # longitude first in GeoJSON
        assert first_coords[1] == custom_data.iloc[0]['lat']  # latitude second
    
    def test_coordinate_precision(self):
        """Test coordinate precision handling"""
        filename = os.path.join(self.temp_dir, 'test_precision.geojson')
        
        result_path = self.exporter.export(
            self.sample_data,
            filename,
            precision=2  # Only 2 decimal places
        )
        
        assert os.path.exists(result_path)
        
        # Verify precision
        with open(result_path, 'r') as f:
            geojson_data = json.load(f)
        
        first_coords = geojson_data['features'][0]['geometry']['coordinates']
        # Check that coordinates are rounded to 2 decimal places
        assert len(str(first_coords[0]).split('.')[-1]) <= 2
        assert len(str(first_coords[1]).split('.')[-1]) <= 2
    
    def test_properties_filtering(self):
        """Test filtering which columns are included as properties"""
        filename = os.path.join(self.temp_dir, 'test_properties.geojson')
        
        result_path = self.exporter.export(
            self.sample_data,
            filename,
            properties_cols=['name', 'category']  # Only include specific columns
        )
        
        assert os.path.exists(result_path)
        
        # Verify properties
        with open(result_path, 'r') as f:
            geojson_data = json.load(f)
        
        first_properties = geojson_data['features'][0]['properties']
        assert set(first_properties.keys()) == {'name', 'category'}
        assert 'id' not in first_properties
        assert 'rating' not in first_properties
    
    def test_crs_specification(self):
        """Test CRS (Coordinate Reference System) specification"""
        filename = os.path.join(self.temp_dir, 'test_crs.geojson')
        
        result_path = self.exporter.export(
            self.sample_data,
            filename,
            crs='EPSG:3857'  # Web Mercator
        )
        
        assert os.path.exists(result_path)
        
        # Verify CRS
        with open(result_path, 'r') as f:
            geojson_data = json.load(f)
        
        assert 'crs' in geojson_data
        assert geojson_data['crs']['properties']['name'] == 'EPSG:3857'
    
    def test_coordinate_validation(self):
        """Test coordinate validation"""
        # Create data with invalid coordinates
        invalid_data = pd.DataFrame({
            'id': [1, 2, 3],
            'latitude': [91.0, 40.7128, -91.0],  # Invalid latitudes
            'longitude': [-74.0060, 181.0, -118.2437]  # Invalid longitude
        })
        
        filename = os.path.join(self.temp_dir, 'test_invalid.geojson')
        
        with pytest.raises(ValueError, match="Latitude values must be between -90 and 90"):
            self.exporter.export(invalid_data, filename)
    
    def test_missing_coordinate_columns(self):
        """Test error handling for missing coordinate columns"""
        data_no_coords = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C']
        })
        
        filename = os.path.join(self.temp_dir, 'test_no_coords.geojson')
        
        with pytest.raises(ValueError, match="Latitude column 'latitude' not found"):
            self.exporter.export(data_no_coords, filename)
    
    def test_nan_coordinates_error(self):
        """Test error handling for NaN coordinates"""
        nan_data = pd.DataFrame({
            'id': [1, 2, 3],
            'latitude': [40.7128, float('nan'), 41.8781],
            'longitude': [-74.0060, -118.2437, -87.6298]
        })
        
        filename = os.path.join(self.temp_dir, 'test_nan_coords.geojson')
        
        with pytest.raises(ValueError, match="Coordinate columns cannot contain NaN values"):
            self.exporter.export(nan_data, filename)
    
    def test_non_numeric_coordinates_error(self):
        """Test error handling for non-numeric coordinates"""
        text_coords_data = pd.DataFrame({
            'id': [1, 2, 3],
            'latitude': ['40.7128', 'invalid', '41.8781'],
            'longitude': [-74.0060, -118.2437, -87.6298]
        })
        
        filename = os.path.join(self.temp_dir, 'test_text_coords.geojson')
        
        with pytest.raises(ValueError, match="contains non-numeric values"):
            self.exporter.export(text_coords_data, filename)
    
    def test_datetime_properties_handling(self):
        """Test handling of datetime properties"""
        datetime_data = self.sample_data.copy()
        datetime_data['timestamp'] = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
        
        filename = os.path.join(self.temp_dir, 'test_datetime.geojson')
        
        result_path = self.exporter.export(datetime_data, filename)
        
        assert os.path.exists(result_path)
        
        # Verify datetime is converted to ISO format
        with open(result_path, 'r') as f:
            geojson_data = json.load(f)
        
        first_timestamp = geojson_data['features'][0]['properties']['timestamp']
        assert isinstance(first_timestamp, str)
        assert 'T' in first_timestamp  # ISO format indicator
    
    def test_filename_validation(self):
        """Test filename validation and extension handling"""
        # Test without extension
        filename_no_ext = os.path.join(self.temp_dir, 'test_no_ext')
        result_path = self.exporter.export(self.sample_data, filename_no_ext)
        assert result_path.endswith('.geojson')
        
        # Test with .json extension
        filename_json = os.path.join(self.temp_dir, 'test.json')
        result_path = self.exporter.export(self.sample_data, filename_json)
        assert result_path.endswith('.json')
    
    def test_empty_data_error(self):
        """Test GeoJSON export with empty data raises error"""
        empty_data = pd.DataFrame()
        filename = os.path.join(self.temp_dir, 'test_empty.geojson')
        
        with pytest.raises(ValueError, match="Data cannot be empty"):
            self.exporter.export(empty_data, filename)
    
    def test_invalid_data_error(self):
        """Test GeoJSON export with invalid data raises error"""
        filename = os.path.join(self.temp_dir, 'test_invalid.geojson')
        
        with pytest.raises(ValueError, match="Data must be a pandas DataFrame"):
            self.exporter.export("not a dataframe", filename)
        
        with pytest.raises(ValueError, match="Data cannot be None"):
            self.exporter.export(None, filename)
    
    def test_invalid_geometry_type_error(self):
        """Test error handling for invalid geometry type"""
        filename = os.path.join(self.temp_dir, 'test_invalid_geom.geojson')
        
        with pytest.raises(ValueError, match="Unsupported geometry type"):
            self.exporter.export(self.sample_data, filename, geometry_type='InvalidType')
    
    def test_directory_creation(self):
        """Test GeoJSON export creates directories if they don't exist"""
        nested_dir = os.path.join(self.temp_dir, 'nested', 'directory')
        filename = os.path.join(nested_dir, 'test.geojson')
        
        result_path = self.exporter.export(self.sample_data, filename)
        
        assert os.path.exists(result_path)
        assert os.path.exists(nested_dir)
    
    def test_get_export_info(self):
        """Test getting export information"""
        info = self.exporter.get_export_info(self.sample_data)
        
        assert info['rows'] == len(self.sample_data)
        assert info['columns'] == len(self.sample_data.columns)
        assert info['column_names'] == list(self.sample_data.columns)
        assert info['has_coordinates'] == True
        assert info['coordinate_validation'] == 'passed'
        assert 'coordinate_bounds' in info
        assert 'estimated_geojson_size_mb' in info
    
    def test_get_export_info_missing_coordinates(self):
        """Test export info with missing coordinates"""
        no_coords_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C']
        })
        
        info = self.exporter.get_export_info(no_coords_data)
        
        assert info['has_coordinates'] == False
        assert 'failed: missing coordinate columns' in info['coordinate_validation']
    
    def test_get_export_info_invalid_coordinates(self):
        """Test export info with invalid coordinates"""
        invalid_coords_data = pd.DataFrame({
            'id': [1, 2, 3],
            'latitude': [91.0, 40.7128, -91.0],  # Invalid latitudes
            'longitude': [-74.0060, 181.0, -118.2437]
        })
        
        info = self.exporter.get_export_info(invalid_coords_data)
        
        assert 'failed:' in info['coordinate_validation']
    
    def test_coordinate_bounds_calculation(self):
        """Test coordinate bounds calculation in export info"""
        info = self.exporter.get_export_info(self.sample_data)
        
        bounds = info['coordinate_bounds']
        assert bounds['min_lat'] == self.sample_data['latitude'].min()
        assert bounds['max_lat'] == self.sample_data['latitude'].max()
        assert bounds['min_lon'] == self.sample_data['longitude'].min()
        assert bounds['max_lon'] == self.sample_data['longitude'].max()
    
    def test_json_formatting_options(self):
        """Test JSON formatting options"""
        filename = os.path.join(self.temp_dir, 'test_formatting.geojson')
        
        result_path = self.exporter.export(
            self.sample_data,
            filename,
            indent=4  # Custom indentation
        )
        
        assert os.path.exists(result_path)
        
        # Verify formatting by checking file content
        with open(result_path, 'r') as f:
            content = f.read()
            # Should have proper indentation
            assert '    ' in content  # 4-space indentation
    
    def test_disable_coordinate_validation(self):
        """Test disabling coordinate validation"""
        # Create data with slightly invalid coordinates
        edge_case_data = pd.DataFrame({
            'id': [1, 2],
            'latitude': [90.1, -90.1],  # Slightly outside valid range
            'longitude': [180.1, -180.1]
        })
        
        filename = os.path.join(self.temp_dir, 'test_no_validation.geojson')
        
        # Should work when validation is disabled
        result_path = self.exporter.export(
            edge_case_data,
            filename,
            validate_coordinates=False
        )
        
        assert os.path.exists(result_path)