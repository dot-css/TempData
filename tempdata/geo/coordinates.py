"""
Coordinate generation with city boundaries

Generates accurate latitude/longitude coordinates within specified city boundaries.
"""

from typing import Tuple, Optional, Dict, Any
import json
import random
import math
from pathlib import Path
from ..core.base_generator import BaseGenerator


class CoordinateGenerator(BaseGenerator):
    """
    Generator for accurate geographical coordinates
    
    Generates latitude/longitude pairs that respect city boundaries
    and geographical constraints.
    """
    
    def __init__(self, seeder, locale: str = 'en_US'):
        """Initialize coordinate generator"""
        super().__init__(seeder, locale)
        self._city_data = None
        self._load_city_boundaries()
    
    def _load_city_boundaries(self) -> None:
        """Load city boundary data from JSON file"""
        try:
            # Try comprehensive data first, fallback to original
            comprehensive_file = Path(__file__).parent.parent / "data" / "countries" / "comprehensive_city_boundaries.json"
            city_boundaries_file = Path(__file__).parent.parent / "data" / "countries" / "city_boundaries.json"
            
            if comprehensive_file.exists():
                with open(comprehensive_file, 'r', encoding='utf-8') as f:
                    self._city_data = json.load(f)
            elif city_boundaries_file.exists():
                with open(city_boundaries_file, 'r', encoding='utf-8') as f:
                    self._city_data = json.load(f)
            else:
                self._city_data = {}
        except (FileNotFoundError, json.JSONDecodeError):
            self._city_data = {}
    
    def generate_coordinates(self, city: str, country: str) -> Tuple[float, float]:
        """
        Generate coordinates within city boundaries
        
        Args:
            city: Target city name
            country: Country containing the city
            
        Returns:
            Tuple[float, float]: (latitude, longitude) coordinates
        """
        city_key = city.lower().replace(' ', '_')
        country_key = country.lower()
        
        # Try to find city in boundary data
        if (self._city_data and 
            country_key in self._city_data and 
            city_key in self._city_data[country_key]):
            
            city_info = self._city_data[country_key][city_key]
            bounds = city_info['bounds']
            
            # Generate random coordinates within city bounds using seeder
            coord_seed = self.seeder.get_contextual_seed(f'coordinates_{city}_{country}')
            random.seed(coord_seed)
            lat = random.uniform(bounds['south'], bounds['north'])
            lon = random.uniform(bounds['west'], bounds['east'])
            
            return (lat, lon)
        
        # Fallback with country-specific coordinate ranges
        country_bounds = {
            'united_states': {'lat': (24.0, 49.0), 'lon': (-125.0, -66.0)},
            'pakistan': {'lat': (23.0, 37.0), 'lon': (60.0, 77.0)},
            'germany': {'lat': (47.0, 55.0), 'lon': (5.0, 15.0)},
            'united_kingdom': {'lat': (49.0, 61.0), 'lon': (-8.0, 2.0)},
            'france': {'lat': (41.0, 51.0), 'lon': (-5.0, 10.0)},
            'japan': {'lat': (24.0, 46.0), 'lon': (123.0, 146.0)},
            'china': {'lat': (18.0, 54.0), 'lon': (73.0, 135.0)},
            'india': {'lat': (6.0, 37.0), 'lon': (68.0, 97.0)},
            'brazil': {'lat': (-34.0, 5.0), 'lon': (-74.0, -32.0)},
            'canada': {'lat': (41.0, 84.0), 'lon': (-141.0, -52.0)},
            'australia': {'lat': (-44.0, -10.0), 'lon': (113.0, 154.0)},
            'russia': {'lat': (41.0, 82.0), 'lon': (19.0, 170.0)}
        }
        
        if country_key in country_bounds:
            bounds = country_bounds[country_key]
            # Use seeder for reproducible coordinate generation
            coord_seed = self.seeder.get_contextual_seed(f'coordinates_{city}_{country}')
            random.seed(coord_seed)
            lat = random.uniform(bounds['lat'][0], bounds['lat'][1])
            lon = random.uniform(bounds['lon'][0], bounds['lon'][1])
            return (lat, lon)
        
        # Final fallback to faker
        return (float(self.faker.latitude()), float(self.faker.longitude()))
    
    def generate_coordinates_near(self, 
                                 center_lat: float, 
                                 center_lon: float, 
                                 radius_km: float = 10.0) -> Tuple[float, float]:
        """
        Generate coordinates near a center point
        
        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            radius_km: Radius in kilometers
            
        Returns:
            Tuple[float, float]: (latitude, longitude) coordinates
        """
        # Convert radius from km to degrees (approximate)
        # 1 degree latitude ≈ 111 km
        # 1 degree longitude ≈ 111 km * cos(latitude)
        lat_radius = radius_km / 111.0
        lon_radius = radius_km / (111.0 * math.cos(math.radians(center_lat)))
        
        # Generate random offset within radius
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(0, 1) * radius_km
        
        # Convert back to coordinate offset
        lat_offset = (distance / 111.0) * math.cos(angle)
        lon_offset = (distance / (111.0 * math.cos(math.radians(center_lat)))) * math.sin(angle)
        
        new_lat = center_lat + lat_offset
        new_lon = center_lon + lon_offset
        
        return (new_lat, new_lon)
    
    def get_city_center(self, city: str, country: str) -> Optional[Tuple[float, float]]:
        """
        Get the center coordinates of a city
        
        Args:
            city: City name
            country: Country name
            
        Returns:
            Optional[Tuple[float, float]]: Center coordinates or None if not found
        """
        city_key = city.lower().replace(' ', '_')
        country_key = country.lower()
        
        if (self._city_data and 
            country_key in self._city_data and 
            city_key in self._city_data[country_key]):
            
            center = self._city_data[country_key][city_key]['center']
            return (center[0], center[1])
        
        return None
    
    def get_city_bounds(self, city: str, country: str) -> Optional[Dict[str, float]]:
        """
        Get the boundary coordinates of a city
        
        Args:
            city: City name
            country: Country name
            
        Returns:
            Optional[Dict[str, float]]: Boundary coordinates or None if not found
        """
        city_key = city.lower().replace(' ', '_')
        country_key = country.lower()
        
        if (self._city_data and 
            country_key in self._city_data and 
            city_key in self._city_data[country_key]):
            
            return self._city_data[country_key][city_key]['bounds']
        
        return None
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two coordinates using Haversine formula
        
        Args:
            lat1: First point latitude
            lon1: First point longitude
            lat2: Second point latitude
            lon2: Second point longitude
            
        Returns:
            float: Distance in kilometers
        """
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of earth in kilometers
        r = 6371
        
        return c * r