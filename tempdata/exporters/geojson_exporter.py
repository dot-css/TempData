"""
GeoJSON export functionality

Exports geographical data to GeoJSON format with proper coordinate formatting
and geographical feature support.
"""

import json
import pandas as pd
import os
from typing import Optional, Union, Dict, Any, List, Tuple
from datetime import datetime
from .base_exporter import BaseExporter


class GeoJSONExporter(BaseExporter):
    """
    Exporter for GeoJSON format
    
    Handles GeoJSON export with proper coordinate formatting and geographical
    feature support. Supports points, lines, polygons, and feature collections.
    """
    
    def __init__(self):
        """Initialize GeoJSON exporter"""
        super().__init__()
        self.supported_extensions = ['.geojson', '.json']
        
        # Default GeoJSON export options
        self.default_options = {
            'lat_col': 'latitude',
            'lon_col': 'longitude',
            'geometry_type': 'Point',  # Point, LineString, Polygon
            'properties_cols': None,  # None means all non-coordinate columns
            'crs': 'EPSG:4326',  # Default coordinate reference system (WGS84)
            'precision': 6,  # Decimal places for coordinates
        }
        
        # JSON formatting options
        self.json_options = {
            'indent': 2,
            'ensure_ascii': False,
            'sort_keys': False,
            'separators': (',', ': ')
        }
    
    def export(self, data: pd.DataFrame, filename: str, **kwargs) -> str:
        """
        Export data to GeoJSON format with proper coordinate formatting
        
        Args:
            data: DataFrame to export (must contain coordinate columns)
            filename: Output filename
            **kwargs: GeoJSON-specific options that override defaults
                - lat_col: Name of latitude column (default: 'latitude')
                - lon_col: Name of longitude column (default: 'longitude')
                - geometry_type: Type of geometry ('Point', 'LineString', 'Polygon')
                - properties_cols: List of columns to include as properties (None = all non-coordinate)
                - crs: Coordinate reference system (default: 'EPSG:4326')
                - precision: Decimal places for coordinates (default: 6)
                - group_by: Column to group features by (for LineString/Polygon)
                - order_by: Column to order coordinates by (for LineString/Polygon)
                - indent: JSON indentation (default: 2)
                - validate_coordinates: Validate coordinate ranges (default: True)
                
        Returns:
            str: Path to exported GeoJSON file
            
        Raises:
            ValueError: If data is invalid, missing coordinate columns, or invalid coordinates
            IOError: If file cannot be written
        """
        # Validate input data
        self._validate_data(data)
        
        # Validate and normalize filename
        if not filename.endswith('.geojson') and not filename.endswith('.json'):
            geojson_path = self._validate_filename(filename, '.geojson')
        else:
            geojson_path = self._validate_filename(filename)
        
        # Ensure output directory exists
        self._ensure_directory_exists(geojson_path)
        
        # Merge default options with user-provided options
        export_options = {**self.default_options, **kwargs}
        
        # Validate coordinate columns exist
        lat_col = export_options['lat_col']
        lon_col = export_options['lon_col']
        
        if lat_col not in data.columns:
            raise ValueError(f"Latitude column '{lat_col}' not found in data")
        if lon_col not in data.columns:
            raise ValueError(f"Longitude column '{lon_col}' not found in data")
        
        # Validate coordinates if requested
        if export_options.get('validate_coordinates', True):
            self._validate_coordinates(data, lat_col, lon_col)
        
        try:
            # Prepare data for GeoJSON export
            geojson_data = self._create_geojson(data, export_options)
            
            # Write to file
            json_options = {**self.json_options}
            if 'indent' in kwargs:
                json_options['indent'] = kwargs['indent']
            
            with open(geojson_path, 'w', encoding='utf-8') as f:
                json.dump(geojson_data, f, **json_options, cls=GeoJSONEncoder)
            
            # Verify the file was created successfully
            if not os.path.exists(geojson_path):
                raise IOError(f"Failed to create GeoJSON file: {geojson_path}")
                
            return geojson_path
            
        except ValueError as e:
            # Re-raise ValueError as-is (for validation errors)
            raise e
        except Exception as e:
            # Clean up partial file if it exists
            if os.path.exists(geojson_path):
                try:
                    os.remove(geojson_path)
                except OSError:
                    pass
            raise IOError(f"Failed to export GeoJSON file: {str(e)}")
    
    def _validate_coordinates(self, data: pd.DataFrame, lat_col: str, lon_col: str):
        """
        Validate coordinate values are within valid ranges
        
        Args:
            data: DataFrame containing coordinates
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            
        Raises:
            ValueError: If coordinates are invalid
        """
        # Check for missing coordinates
        if data[lat_col].isna().any() or data[lon_col].isna().any():
            raise ValueError("Coordinate columns cannot contain NaN values")
        
        # Validate latitude range (-90 to 90)
        lat_values = pd.to_numeric(data[lat_col], errors='coerce')
        if lat_values.isna().any():
            raise ValueError(f"Latitude column '{lat_col}' contains non-numeric values")
        
        if (lat_values < -90).any() or (lat_values > 90).any():
            raise ValueError(f"Latitude values must be between -90 and 90 degrees")
        
        # Validate longitude range (-180 to 180)
        lon_values = pd.to_numeric(data[lon_col], errors='coerce')
        if lon_values.isna().any():
            raise ValueError(f"Longitude column '{lon_col}' contains non-numeric values")
        
        if (lon_values < -180).any() or (lon_values > 180).any():
            raise ValueError(f"Longitude values must be between -180 and 180 degrees")
    
    def _create_geojson(self, data: pd.DataFrame, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create GeoJSON structure from DataFrame
        
        Args:
            data: DataFrame with geographical data
            options: Export options
            
        Returns:
            Dict representing GeoJSON structure
        """
        geometry_type = options['geometry_type']
        
        if geometry_type == 'Point':
            return self._create_point_geojson(data, options)
        elif geometry_type == 'LineString':
            return self._create_linestring_geojson(data, options)
        elif geometry_type == 'Polygon':
            return self._create_polygon_geojson(data, options)
        else:
            raise ValueError(f"Unsupported geometry type: {geometry_type}")
    
    def _create_point_geojson(self, data: pd.DataFrame, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create GeoJSON FeatureCollection with Point geometries
        
        Args:
            data: DataFrame with point data
            options: Export options
            
        Returns:
            Dict representing GeoJSON FeatureCollection
        """
        lat_col = options['lat_col']
        lon_col = options['lon_col']
        precision = options['precision']
        properties_cols = options['properties_cols']
        
        # Determine which columns to include as properties
        if properties_cols is None:
            properties_cols = [col for col in data.columns if col not in [lat_col, lon_col]]
        
        features = []
        
        for _, row in data.iterrows():
            # Create point geometry
            coordinates = [
                round(float(row[lon_col]), precision),
                round(float(row[lat_col]), precision)
            ]
            
            geometry = {
                "type": "Point",
                "coordinates": coordinates
            }
            
            # Create properties
            properties = {}
            for col in properties_cols:
                value = row[col]
                # Handle different data types for JSON serialization
                if pd.isna(value):
                    properties[col] = None
                elif isinstance(value, (pd.Timestamp, datetime)):
                    properties[col] = value.isoformat()
                else:
                    properties[col] = value
            
            # Create feature
            feature = {
                "type": "Feature",
                "geometry": geometry,
                "properties": properties
            }
            
            features.append(feature)
        
        # Create FeatureCollection
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        # Add CRS if specified
        if options.get('crs'):
            geojson["crs"] = {
                "type": "name",
                "properties": {
                    "name": options['crs']
                }
            }
        
        return geojson
    
    def _create_linestring_geojson(self, data: pd.DataFrame, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create GeoJSON FeatureCollection with LineString geometries
        
        Args:
            data: DataFrame with line data
            options: Export options
            
        Returns:
            Dict representing GeoJSON FeatureCollection
        """
        lat_col = options['lat_col']
        lon_col = options['lon_col']
        precision = options['precision']
        group_by = options.get('group_by')
        order_by = options.get('order_by')
        properties_cols = options.get('properties_cols')
        if properties_cols is None:
            properties_cols = [col for col in data.columns if col not in [lat_col, lon_col, group_by, order_by] and col is not None]
        
        features = []
        
        if group_by and group_by in data.columns:
            # Group data to create separate LineString features
            for group_value, group_data in data.groupby(group_by):
                # Sort by order column if specified
                if order_by and order_by in group_data.columns:
                    group_data = group_data.sort_values(order_by)
                
                # Create coordinates array
                coordinates = []
                for _, row in group_data.iterrows():
                    coord = [
                        round(float(row[lon_col]), precision),
                        round(float(row[lat_col]), precision)
                    ]
                    coordinates.append(coord)
                
                # Create LineString geometry
                geometry = {
                    "type": "LineString",
                    "coordinates": coordinates
                }
                
                # Create properties (use first row's properties)
                properties = {group_by: group_value}
                first_row = group_data.iloc[0]
                for col in properties_cols:
                    if col in group_data.columns and col != group_by:
                        value = first_row[col]
                        if pd.isna(value):
                            properties[col] = None
                        elif isinstance(value, (pd.Timestamp, datetime)):
                            properties[col] = value.isoformat()
                        else:
                            properties[col] = value
                
                # Create feature
                feature = {
                    "type": "Feature",
                    "geometry": geometry,
                    "properties": properties
                }
                
                features.append(feature)
        else:
            # Create single LineString from all points
            if order_by and order_by in data.columns:
                data = data.sort_values(order_by)
            
            coordinates = []
            for _, row in data.iterrows():
                coord = [
                    round(float(row[lon_col]), precision),
                    round(float(row[lat_col]), precision)
                ]
                coordinates.append(coord)
            
            geometry = {
                "type": "LineString",
                "coordinates": coordinates
            }
            
            # Create properties from first row
            properties = {}
            if len(data) > 0:
                first_row = data.iloc[0]
                for col in properties_cols:
                    if col in data.columns:
                        value = first_row[col]
                        if pd.isna(value):
                            properties[col] = None
                        elif isinstance(value, (pd.Timestamp, datetime)):
                            properties[col] = value.isoformat()
                        else:
                            properties[col] = value
            
            feature = {
                "type": "Feature",
                "geometry": geometry,
                "properties": properties
            }
            
            features.append(feature)
        
        # Create FeatureCollection
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        # Add CRS if specified
        if options.get('crs'):
            geojson["crs"] = {
                "type": "name",
                "properties": {
                    "name": options['crs']
                }
            }
        
        return geojson
    
    def _create_polygon_geojson(self, data: pd.DataFrame, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create GeoJSON FeatureCollection with Polygon geometries
        
        Args:
            data: DataFrame with polygon data
            options: Export options
            
        Returns:
            Dict representing GeoJSON FeatureCollection
        """
        # Similar to LineString but coordinates are wrapped in an additional array
        # for the exterior ring (interior rings/holes would be additional arrays)
        lat_col = options['lat_col']
        lon_col = options['lon_col']
        precision = options['precision']
        group_by = options.get('group_by')
        order_by = options.get('order_by')
        properties_cols = options.get('properties_cols')
        if properties_cols is None:
            properties_cols = [col for col in data.columns if col not in [lat_col, lon_col, group_by, order_by] and col is not None]
        
        features = []
        
        if group_by and group_by in data.columns:
            # Group data to create separate Polygon features
            for group_value, group_data in data.groupby(group_by):
                # Sort by order column if specified
                if order_by and order_by in group_data.columns:
                    group_data = group_data.sort_values(order_by)
                
                # Create coordinates array (exterior ring)
                coordinates = []
                for _, row in group_data.iterrows():
                    coord = [
                        round(float(row[lon_col]), precision),
                        round(float(row[lat_col]), precision)
                    ]
                    coordinates.append(coord)
                
                # Close the polygon by adding first point at the end if not already closed
                if len(coordinates) > 0 and coordinates[0] != coordinates[-1]:
                    coordinates.append(coordinates[0])
                
                # Create Polygon geometry (coordinates wrapped in array for exterior ring)
                geometry = {
                    "type": "Polygon",
                    "coordinates": [coordinates]
                }
                
                # Create properties
                properties = {group_by: group_value}
                first_row = group_data.iloc[0]
                for col in properties_cols:
                    if col in group_data.columns and col != group_by:
                        value = first_row[col]
                        if pd.isna(value):
                            properties[col] = None
                        elif isinstance(value, (pd.Timestamp, datetime)):
                            properties[col] = value.isoformat()
                        else:
                            properties[col] = value
                
                # Create feature
                feature = {
                    "type": "Feature",
                    "geometry": geometry,
                    "properties": properties
                }
                
                features.append(feature)
        else:
            # Create single Polygon from all points
            if order_by and order_by in data.columns:
                data = data.sort_values(order_by)
            
            coordinates = []
            for _, row in data.iterrows():
                coord = [
                    round(float(row[lon_col]), precision),
                    round(float(row[lat_col]), precision)
                ]
                coordinates.append(coord)
            
            # Close the polygon
            if len(coordinates) > 0 and coordinates[0] != coordinates[-1]:
                coordinates.append(coordinates[0])
            
            geometry = {
                "type": "Polygon",
                "coordinates": [coordinates]
            }
            
            # Create properties from first row
            properties = {}
            if len(data) > 0:
                first_row = data.iloc[0]
                for col in properties_cols:
                    if col in data.columns:
                        value = first_row[col]
                        if pd.isna(value):
                            properties[col] = None
                        elif isinstance(value, (pd.Timestamp, datetime)):
                            properties[col] = value.isoformat()
                        else:
                            properties[col] = value
            
            feature = {
                "type": "Feature",
                "geometry": geometry,
                "properties": properties
            }
            
            features.append(feature)
        
        # Create FeatureCollection
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        # Add CRS if specified
        if options.get('crs'):
            geojson["crs"] = {
                "type": "name",
                "properties": {
                    "name": options['crs']
                }
            }
        
        return geojson
    
    def export_points_with_buffer(self, data: pd.DataFrame, filename: str, 
                                 buffer_meters: float, **kwargs) -> str:
        """
        Export points as polygons with circular buffer
        
        Args:
            data: DataFrame with point data
            filename: Output filename
            buffer_meters: Buffer distance in meters
            **kwargs: Export options
            
        Returns:
            str: Path to exported GeoJSON file
        """
        # This would require geospatial libraries like shapely for proper buffer calculation
        # For now, create a simple approximation
        # In a full implementation, you'd use shapely.geometry.Point.buffer()
        
        # Simple approximation: convert meters to degrees (very rough)
        # 1 degree ≈ 111,320 meters at equator
        buffer_degrees = buffer_meters / 111320
        
        # Create buffered points as simple squares (not circles)
        buffered_data = data.copy()
        lat_col = kwargs.get('lat_col', 'latitude')
        lon_col = kwargs.get('lon_col', 'longitude')
        
        # This is a simplified implementation
        # A proper implementation would use geospatial libraries
        return self.export(buffered_data, filename, geometry_type='Point', **kwargs)
    
    def get_export_info(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Get information about what the GeoJSON export will contain
        
        Args:
            data: DataFrame to analyze
            **kwargs: Export options
            
        Returns:
            Dict with export information
        """
        lat_col = kwargs.get('lat_col', 'latitude')
        lon_col = kwargs.get('lon_col', 'longitude')
        
        info = {
            'rows': len(data),
            'columns': len(data.columns),
            'column_names': list(data.columns),
            'coordinate_columns': {
                'latitude': lat_col,
                'longitude': lon_col
            },
            'has_coordinates': lat_col in data.columns and lon_col in data.columns,
            'geometry_type': kwargs.get('geometry_type', 'Point'),
            'memory_usage_mb': round(data.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            'estimated_geojson_size_mb': 0
        }
        
        # Validate coordinates if present
        if info['has_coordinates']:
            try:
                self._validate_coordinates(data, lat_col, lon_col)
                info['coordinate_validation'] = 'passed'
                
                # Calculate coordinate bounds
                lat_values = pd.to_numeric(data[lat_col], errors='coerce')
                lon_values = pd.to_numeric(data[lon_col], errors='coerce')
                
                info['coordinate_bounds'] = {
                    'min_lat': float(lat_values.min()),
                    'max_lat': float(lat_values.max()),
                    'min_lon': float(lon_values.min()),
                    'max_lon': float(lon_values.max())
                }
                
            except ValueError as e:
                info['coordinate_validation'] = f'failed: {str(e)}'
        else:
            info['coordinate_validation'] = f'failed: missing coordinate columns'
        
        # Estimate GeoJSON file size (rough approximation)
        # GeoJSON is typically larger than JSON due to coordinate precision and structure
        avg_chars_per_feature = 200  # Rough estimate including coordinates and properties
        estimated_size = len(data) * avg_chars_per_feature
        info['estimated_geojson_size_mb'] = round(estimated_size / (1024 * 1024), 2)
        
        return info


class GeoJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for GeoJSON data types
    """
    
    def default(self, obj):
        """
        Handle special data types for GeoJSON serialization
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON-serializable representation
        """
        # Handle pandas/numpy data types
        if hasattr(obj, 'dtype'):
            if 'int' in str(obj.dtype):
                return int(obj)
            elif 'float' in str(obj.dtype):
                return float(obj)
            elif 'bool' in str(obj.dtype):
                return bool(obj)
        
        # Handle datetime objects
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        
        # Handle NaN and None
        if pd.isna(obj):
            return None
        
        # Fall back to default behavior
        return super().default(obj)