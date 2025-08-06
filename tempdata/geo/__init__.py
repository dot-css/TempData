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
    """
    Generate realistic addresses for specified country
    
    Creates geographically accurate addresses with proper formatting for the
    specified country, including realistic street names, postal codes, and
    coordinate pairs within city boundaries.
    
    Args:
        country (str): Country code for address generation. Supported countries include:
            'united_states', 'canada', 'united_kingdom', 'germany', 'france', 
            'spain', 'italy', 'netherlands', 'sweden', 'norway', 'denmark',
            'finland', 'poland', 'czech_republic', 'austria', 'switzerland',
            'australia', 'new_zealand', 'japan', 'south_korea', 'india', 
            'pakistan', 'brazil', 'mexico', 'argentina', 'chile', 'south_africa',
            'egypt', 'nigeria', 'kenya'
        count (int, optional): Number of addresses to generate. Defaults to 1.
        **kwargs: Additional parameters:
            - city (str): Specific city to generate addresses for
            - state_province (str): Specific state/province to limit addresses to
            - urban_bias (float): Bias towards urban areas (0.0-1.0, default: 0.7)
            - include_coordinates (bool): Include lat/lng coordinates (default: True)
            - address_type (str): Type of address ('residential', 'commercial', 'mixed')
    
    Returns:
        List[Dict]: List of address dictionaries, each containing:
            - street (str): Street address with number and name
            - city (str): City name
            - state_province (str): State, province, or region
            - postal_code (str): Postal/ZIP code in country format
            - country (str): Country name
            - coordinates (Tuple[float, float]): (latitude, longitude) if enabled
            - address_type (str): Type of address
    
    Examples:
        >>> import tempdata
        
        # Generate single address for Germany
        >>> address = tempdata.geo.addresses('germany')[0]
        >>> print(address['street'])
        'Hauptstraße 42'
        >>> print(address['postal_code'])
        '10115'
        
        # Generate multiple addresses for Pakistan
        >>> addresses = tempdata.geo.addresses('pakistan', count=5)
        >>> len(addresses)
        5
        
        # Generate addresses for specific city
        >>> tokyo_addresses = tempdata.geo.addresses('japan', count=3, city='Tokyo')
        >>> all(addr['city'] == 'Tokyo' for addr in tokyo_addresses)
        True
        
        # Generate commercial addresses only
        >>> commercial = tempdata.geo.addresses('united_states', count=10, 
        ...                                    address_type='commercial')
        >>> all(addr['address_type'] == 'commercial' for addr in commercial)
        True
    
    Raises:
        ValueError: If country is not supported or parameters are invalid
        IOError: If geographical data cannot be loaded
    
    Note:
        Addresses are generated with realistic patterns for each country including:
        - Proper street naming conventions
        - Accurate postal code formats
        - Realistic city and state distributions
        - Coordinates within actual city boundaries
        - Cultural and linguistic appropriateness
    """
    from ..core.seeding import MillisecondSeeder
    seeder = MillisecondSeeder()
    generator = AddressGenerator(seeder, country)
    return generator.generate_multiple(count, **kwargs)

def route(start_city: str, end_city: str, waypoints: int = 0, **kwargs):
    """
    Generate realistic route with waypoints between cities
    
    Creates a travel route between two cities with optional waypoints, including
    realistic distance calculations, travel time estimates, and geographical
    accuracy. Supports multiple transportation modes and route types.
    
    Args:
        start_city (str): Starting city name (e.g., 'Berlin', 'New York', 'Tokyo')
        end_city (str): Destination city name
        waypoints (int, optional): Number of intermediate waypoints. Defaults to 0.
        **kwargs: Additional parameters:
            - country (str): Country to limit route to (optional)
            - transportation_mode (str): Mode of transport ('driving', 'walking', 
              'cycling', 'public_transit'). Defaults to 'driving'
            - route_type (str): Type of route ('fastest', 'shortest', 'scenic').
              Defaults to 'fastest'
            - avoid_highways (bool): Avoid major highways (default: False)
            - include_traffic (bool): Include traffic considerations (default: True)
            - max_waypoint_deviation (float): Max deviation for waypoints in km (default: 50)
    
    Returns:
        Dict: Route information containing:
            - start_point (Dict): Starting address with full details
            - end_point (Dict): Destination address with full details  
            - waypoints (List[Dict]): List of waypoint addresses
            - distance_km (float): Total route distance in kilometers
            - estimated_time_minutes (int): Estimated travel time in minutes
            - transportation_mode (str): Mode of transportation used
            - route_type (str): Type of route generated
            - elevation_gain (float): Total elevation gain in meters (if applicable)
            - traffic_factor (float): Traffic impact factor (1.0 = no traffic)
    
    Examples:
        >>> import tempdata
        
        # Simple route between two cities
        >>> route = tempdata.geo.route('Berlin', 'Munich')
        >>> route['distance_km']
        584.2
        >>> route['estimated_time_minutes']
        351
        
        # Route with waypoints
        >>> scenic_route = tempdata.geo.route('Paris', 'Rome', waypoints=3, 
        ...                                  route_type='scenic')
        >>> len(scenic_route['waypoints'])
        3
        
        # Walking route in same country
        >>> walk = tempdata.geo.route('London', 'Oxford', 
        ...                          transportation_mode='walking',
        ...                          country='united_kingdom')
        >>> walk['transportation_mode']
        'walking'
        
        # Route avoiding highways
        >>> local_route = tempdata.geo.route('Los Angeles', 'San Francisco',
        ...                                 avoid_highways=True)
        >>> local_route['route_type']
        'scenic'
    
    Raises:
        ValueError: If cities are not found or parameters are invalid
        IOError: If geographical data cannot be accessed
        
    Note:
        Routes are generated with realistic characteristics:
        - Accurate distances based on geographical coordinates
        - Realistic travel times considering transportation mode
        - Waypoints positioned along logical route paths
        - Traffic considerations for driving routes
        - Elevation data for walking/cycling routes
        - Cultural route preferences by region
    """
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