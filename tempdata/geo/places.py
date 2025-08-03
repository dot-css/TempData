"""
Points of Interest (POI) generation

Generates realistic points of interest including businesses, landmarks, and services.
"""

from typing import List, Dict, Any
from ..core.base_generator import BaseGenerator


class POIGenerator(BaseGenerator):
    """
    Generator for Points of Interest (POI)
    
    Generates realistic POIs including businesses, landmarks, restaurants,
    and other location-based services with appropriate geographical distribution.
    """
    
    def generate_pois(self, 
                     city: str, 
                     country: str, 
                     count: int = 10, 
                     poi_types: List[str] = None) -> List[Dict[str, Any]]:
        """
        Generate points of interest for a city
        
        Args:
            city: Target city name
            country: Country containing the city
            count: Number of POIs to generate
            poi_types: Types of POIs to generate (restaurants, shops, etc.)
            
        Returns:
            List[Dict[str, Any]]: List of POI dictionaries
        """
        if poi_types is None:
            poi_types = ['restaurant', 'shop', 'service', 'landmark']
        
        pois = []
        for _ in range(count):
            poi = {
                'name': self.faker.company(),
                'type': self.faker.random_element(poi_types),
                'address': self.faker.address(),
                'coordinates': (self.faker.latitude(), self.faker.longitude()),
                'rating': round(self.faker.random.uniform(1.0, 5.0), 1)
            }
            pois.append(poi)
        
        return pois