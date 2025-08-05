"""
Geographical data generation module

Provides worldwide geographical data generation including addresses,
coordinates, routes, and points of interest with country-specific accuracy.
"""

from .address import AddressGenerator, Address
from .coordinates import CoordinateGenerator
from .routes import RouteSimulator, Route
from .places import POIGenerator

# Convenience functions for the geo API
def addresses(country: str, count: int = 1, **kwargs):
    """Generate realistic addresses for specified country"""
    from ..core.seeding import MillisecondSeeder
    seeder = MillisecondSeeder()
    generator = AddressGenerator(seeder, country)
    return generator.generate_multiple(count, **kwargs)

def route(start_city: str, end_city: str, waypoints: int = 0, **kwargs):
    """Generate route with waypoints between cities"""
    from ..core.seeding import MillisecondSeeder
    seeder = MillisecondSeeder()
    simulator = RouteSimulator(seeder)
    route_obj = simulator.generate_route(start_city, end_city, waypoints, **kwargs)
    
    # Convert Route object to dictionary for API consistency
    return {
        'start_point': {
            'street': route_obj.start_point.street,
            'city': route_obj.start_point.city,
            'state_province': route_obj.start_point.state_province,
            'postal_code': route_obj.start_point.postal_code,
            'country': route_obj.start_point.country,
            'coordinates': route_obj.start_point.coordinates
        },
        'end_point': {
            'street': route_obj.end_point.street,
            'city': route_obj.end_point.city,
            'state_province': route_obj.end_point.state_province,
            'postal_code': route_obj.end_point.postal_code,
            'country': route_obj.end_point.country,
            'coordinates': route_obj.end_point.coordinates
        },
        'waypoints': [
            {
                'street': wp.street,
                'city': wp.city,
                'state_province': wp.state_province,
                'postal_code': wp.postal_code,
                'country': wp.country,
                'coordinates': wp.coordinates
            } for wp in route_obj.waypoints
        ],
        'distance_km': route_obj.distance_km,
        'estimated_time_minutes': route_obj.estimated_time_minutes,
        'transportation_mode': route_obj.transportation_mode,
        'route_type': route_obj.route_type
    }

__all__ = [
    "AddressGenerator",
    "Address", 
    "CoordinateGenerator",
    "RouteSimulator",
    "Route",
    "POIGenerator",
    "addresses",
    "route"
]