"""
Route simulation with realistic travel patterns

Generates realistic travel routes with waypoints and distance calculations.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import random
import math
from .address import Address, AddressGenerator
from .coordinates import CoordinateGenerator
from ..core.base_generator import BaseGenerator


@dataclass
class Route:
    """Data class representing a travel route"""
    start_point: Address
    end_point: Address
    waypoints: List[Address]
    distance_km: float
    estimated_time_minutes: int
    transportation_mode: str = 'car'
    route_type: str = 'direct'  # direct, scenic, fastest, shortest


class RouteSimulator(BaseGenerator):
    """
    Simulator for realistic travel routes
    
    Generates routes with realistic waypoints, distances, and travel times
    based on geographical constraints and transportation patterns.
    """
    
    def __init__(self, seeder, locale: str = 'en_US'):
        """Initialize route simulator"""
        super().__init__(seeder, locale)
        self.coordinate_gen = CoordinateGenerator(seeder, locale)
        
        # Transportation speed averages (km/h)
        self.transport_speeds = {
            'car': 60,
            'truck': 50,
            'motorcycle': 70,
            'bicycle': 20,
            'walking': 5,
            'bus': 40,
            'train': 80
        }
    
    def generate_route(self, 
                      start_city: str, 
                      end_city: str, 
                      waypoints: int = 0, 
                      **kwargs) -> Route:
        """
        Generate route between cities with waypoints
        
        Args:
            start_city: Starting city name
            end_city: Destination city name
            waypoints: Number of waypoints to include
            **kwargs: Additional route parameters (country, transportation_mode, route_type)
            
        Returns:
            Route: Generated route object
        """
        country = kwargs.get('country', 'united_states')
        transportation_mode = kwargs.get('transportation_mode', 'car')
        route_type = kwargs.get('route_type', 'direct')
        
        # Generate start and end addresses with realistic coordinates
        addr_gen = AddressGenerator(self.seeder, country)
        
        # Get realistic coordinates for the cities
        start_coords = self.coordinate_gen.generate_coordinates(start_city, country)
        end_coords = self.coordinate_gen.generate_coordinates(end_city, country)
        
        # Generate addresses with these coordinates
        start_addr = addr_gen.generate_address_near(start_coords[0], start_coords[1], radius_km=5)
        end_addr = addr_gen.generate_address_near(end_coords[0], end_coords[1], radius_km=5)
        
        # Override the city names to match the requested cities
        start_addr.city = start_city
        end_addr.city = end_city
        
        # Generate waypoints along the route
        waypoint_addrs = self._generate_waypoints(
            start_addr, end_addr, waypoints, country, addr_gen
        )
        
        # Calculate total distance and time
        total_distance = self._calculate_route_distance(
            start_addr, waypoint_addrs, end_addr
        )
        
        estimated_time = self._calculate_travel_time(
            total_distance, transportation_mode, route_type
        )
        
        return Route(
            start_point=start_addr,
            end_point=end_addr,
            waypoints=waypoint_addrs,
            distance_km=round(total_distance, 2),
            estimated_time_minutes=int(estimated_time),
            transportation_mode=transportation_mode,
            route_type=route_type
        )
    
    def _generate_waypoints(self, 
                           start_addr: Address, 
                           end_addr: Address, 
                           count: int,
                           country: str,
                           addr_gen: AddressGenerator) -> List[Address]:
        """Generate waypoints between start and end addresses"""
        if count == 0:
            return []
        
        waypoints = []
        start_lat, start_lon = start_addr.coordinates
        end_lat, end_lon = end_addr.coordinates
        
        for i in range(count):
            # Calculate intermediate position
            progress = (i + 1) / (count + 1)
            
            # Linear interpolation with some random deviation
            waypoint_lat = start_lat + (end_lat - start_lat) * progress
            waypoint_lon = start_lon + (end_lon - start_lon) * progress
            
            # Add some random deviation to make route more realistic
            deviation_km = random.uniform(5, 20)  # 5-20 km deviation
            deviation_lat = (deviation_km / 111.0) * random.uniform(-1, 1)
            deviation_lon = (deviation_km / (111.0 * math.cos(math.radians(waypoint_lat)))) * random.uniform(-1, 1)
            
            waypoint_lat += deviation_lat
            waypoint_lon += deviation_lon
            
            # Generate address near these coordinates
            waypoint_addr = addr_gen.generate_address_near(
                waypoint_lat, waypoint_lon, radius_km=10
            )
            waypoints.append(waypoint_addr)
        
        return waypoints
    
    def _calculate_route_distance(self, 
                                 start_addr: Address, 
                                 waypoints: List[Address], 
                                 end_addr: Address) -> float:
        """Calculate total route distance including waypoints"""
        total_distance = 0.0
        current_addr = start_addr
        
        # Distance through waypoints
        for waypoint in waypoints:
            distance = self.coordinate_gen.calculate_distance(
                current_addr.coordinates[0], current_addr.coordinates[1],
                waypoint.coordinates[0], waypoint.coordinates[1]
            )
            total_distance += distance
            current_addr = waypoint
        
        # Distance from last waypoint (or start) to end
        final_distance = self.coordinate_gen.calculate_distance(
            current_addr.coordinates[0], current_addr.coordinates[1],
            end_addr.coordinates[0], end_addr.coordinates[1]
        )
        total_distance += final_distance
        
        return total_distance
    
    def _calculate_travel_time(self, 
                              distance_km: float, 
                              transportation_mode: str, 
                              route_type: str) -> float:
        """Calculate estimated travel time in minutes"""
        base_speed = self.transport_speeds.get(transportation_mode, 60)
        
        # Adjust speed based on route type
        speed_multipliers = {
            'fastest': 1.2,
            'direct': 1.0,
            'scenic': 0.8,
            'shortest': 0.9
        }
        
        adjusted_speed = base_speed * speed_multipliers.get(route_type, 1.0)
        
        # Add some realistic delays (traffic, stops, etc.)
        delay_factor = random.uniform(1.1, 1.4)  # 10-40% delay
        
        travel_time_hours = (distance_km / adjusted_speed) * delay_factor
        return travel_time_hours * 60  # Convert to minutes
    
    def generate_route_with_stops(self, 
                                 start_city: str, 
                                 end_city: str, 
                                 stop_types: List[str] = None,
                                 **kwargs) -> Dict[str, Any]:
        """
        Generate route with specific types of stops (gas stations, restaurants, etc.)
        
        Args:
            start_city: Starting city name
            end_city: Destination city name
            stop_types: Types of stops to include
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Route with detailed stop information
        """
        if stop_types is None:
            stop_types = ['gas_station', 'restaurant', 'rest_area']
        
        # Generate base route
        base_route = self.generate_route(start_city, end_city, **kwargs)
        
        # Add stops along the route
        stops = []
        for i, waypoint in enumerate(base_route.waypoints):
            stop_type = random.choice(stop_types)
            stop = {
                'address': waypoint,
                'type': stop_type,
                'name': self._generate_stop_name(stop_type),
                'duration_minutes': self._get_stop_duration(stop_type)
            }
            stops.append(stop)
        
        return {
            'route': base_route,
            'stops': stops,
            'total_time_with_stops': base_route.estimated_time_minutes + sum(s['duration_minutes'] for s in stops)
        }
    
    def _generate_stop_name(self, stop_type: str) -> str:
        """Generate realistic name for a stop"""
        stop_names = {
            'gas_station': ['Shell', 'BP', 'Exxon', 'Chevron', 'Mobil', 'Texaco'],
            'restaurant': ['McDonald\'s', 'Subway', 'KFC', 'Burger King', 'Taco Bell', 'Pizza Hut'],
            'rest_area': ['Highway Rest Stop', 'Travel Plaza', 'Service Area', 'Roadside Rest'],
            'hotel': ['Holiday Inn', 'Best Western', 'Marriott', 'Hampton Inn', 'Comfort Inn'],
            'shopping': ['Walmart', 'Target', 'Mall', 'Shopping Center', 'Outlet Store']
        }
        
        names = stop_names.get(stop_type, ['Generic Stop'])
        return random.choice(names)
    
    def _get_stop_duration(self, stop_type: str) -> int:
        """Get typical stop duration in minutes"""
        durations = {
            'gas_station': random.randint(5, 15),
            'restaurant': random.randint(20, 60),
            'rest_area': random.randint(10, 30),
            'hotel': 480,  # 8 hours
            'shopping': random.randint(30, 120)
        }
        
        return durations.get(stop_type, 15)
    
    def calculate_fuel_consumption(self, 
                                  route: Route, 
                                  vehicle_mpg: float = 25.0) -> Dict[str, float]:
        """
        Calculate fuel consumption for a route
        
        Args:
            route: Route object
            vehicle_mpg: Vehicle fuel efficiency in miles per gallon
            
        Returns:
            Dict[str, float]: Fuel consumption details
        """
        distance_miles = route.distance_km * 0.621371  # Convert km to miles
        fuel_gallons = distance_miles / vehicle_mpg
        
        # Estimate fuel cost (average US gas price)
        fuel_cost = fuel_gallons * 3.50  # $3.50 per gallon
        
        return {
            'distance_miles': round(distance_miles, 2),
            'fuel_gallons': round(fuel_gallons, 2),
            'estimated_cost_usd': round(fuel_cost, 2)
        }