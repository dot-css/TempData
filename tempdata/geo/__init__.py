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
    return simulator.generate_route(start_city, end_city, waypoints, **kwargs)

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