"""
Unit tests for route simulation and POI generation

Tests RouteSimulator and POIGenerator for route realism and POI accuracy.
"""

import pytest
import math
from tempdata.core.seeding import MillisecondSeeder
from tempdata.geo.routes import RouteSimulator, Route
from tempdata.geo.places import POIGenerator, POI
from tempdata.geo.address import Address


class TestRoute:
    """Test cases for Route dataclass"""
    
    def test_route_creation(self):
        """Test Route dataclass creation with all fields"""
        start_addr = Address("123 Start St", "StartCity", "State1", "12345", "US", (40.0, -74.0))
        end_addr = Address("456 End St", "EndCity", "State2", "67890", "US", (41.0, -75.0))
        waypoint = Address("789 Way St", "WayCity", "State3", "11111", "US", (40.5, -74.5))
        
        route = Route(
            start_point=start_addr,
            end_point=end_addr,
            waypoints=[waypoint],
            distance_km=150.5,
            estimated_time_minutes=180,
            transportation_mode='car',
            route_type='direct'
        )
        
        assert route.start_point == start_addr
        assert route.end_point == end_addr
        assert len(route.waypoints) == 1
        assert route.waypoints[0] == waypoint
        assert route.distance_km == 150.5
        assert route.estimated_time_minutes == 180
        assert route.transportation_mode == 'car'
        assert route.route_type == 'direct'


class TestPOI:
    """Test cases for POI dataclass"""
    
    def test_poi_creation(self):
        """Test POI dataclass creation"""
        address = Address("123 Business St", "City", "State", "12345", "US", (40.0, -74.0))
        hours = {'monday': '9:00 AM - 5:00 PM', 'tuesday': '9:00 AM - 5:00 PM'}
        
        poi = POI(
            name="Test Business",
            type="restaurant",
            category="Food & Dining",
            address=address,
            coordinates=(40.0, -74.0),
            rating=4.2,
            price_level=2,
            hours=hours,
            phone="+1-555-123-4567",
            website="www.testbusiness.com",
            description="A great place to eat"
        )
        
        assert poi.name == "Test Business"
        assert poi.type == "restaurant"
        assert poi.category == "Food & Dining"
        assert poi.address == address
        assert poi.coordinates == (40.0, -74.0)
        assert poi.rating == 4.2
        assert poi.price_level == 2
        assert poi.hours == hours
        assert poi.phone == "+1-555-123-4567"
        assert poi.website == "www.testbusiness.com"
        assert poi.description == "A great place to eat"


class TestRouteSimulator:
    """Test cases for RouteSimulator functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.seeder = MillisecondSeeder(fixed_seed=12345)
        self.route_sim = RouteSimulator(self.seeder)
    
    def test_initialization(self):
        """Test RouteSimulator initializes correctly"""
        assert self.route_sim is not None
        assert hasattr(self.route_sim, 'transport_speeds')
        assert 'car' in self.route_sim.transport_speeds
        assert 'walking' in self.route_sim.transport_speeds
    
    def test_generate_basic_route(self):
        """Test basic route generation between cities"""
        route = self.route_sim.generate_route('New York', 'Boston')
        
        assert isinstance(route, Route)
        assert route.start_point.city == 'New York'
        assert route.end_point.city == 'Boston'
        assert route.distance_km > 0
        assert route.estimated_time_minutes > 0
        assert route.transportation_mode == 'car'  # default
        assert route.route_type == 'direct'  # default
        assert len(route.waypoints) == 0  # no waypoints by default
    
    def test_generate_route_with_waypoints(self):
        """Test route generation with waypoints"""
        route = self.route_sim.generate_route('New York', 'Boston', waypoints=2)
        
        assert isinstance(route, Route)
        assert len(route.waypoints) == 2
        assert route.distance_km > 0
        assert route.estimated_time_minutes > 0
        
        # Verify waypoints are Address objects
        for waypoint in route.waypoints:
            assert isinstance(waypoint, Address)
            assert waypoint.coordinates
            assert len(waypoint.coordinates) == 2
    
    def test_generate_route_with_parameters(self):
        """Test route generation with custom parameters"""
        route = self.route_sim.generate_route(
            'Chicago', 'Detroit',
            waypoints=1,
            country='united_states',
            transportation_mode='truck',
            route_type='scenic'
        )
        
        assert route.transportation_mode == 'truck'
        assert route.route_type == 'scenic'
        assert len(route.waypoints) == 1
    
    def test_transportation_modes(self):
        """Test different transportation modes affect travel time"""
        car_route = self.route_sim.generate_route(
            'New York', 'Philadelphia', transportation_mode='car'
        )
        walking_route = self.route_sim.generate_route(
            'New York', 'Philadelphia', transportation_mode='walking'
        )
        
        # Walking should take much longer than driving
        assert walking_route.estimated_time_minutes > car_route.estimated_time_minutes
    
    def test_route_types(self):
        """Test different route types"""
        fastest_route = self.route_sim.generate_route(
            'Los Angeles', 'San Francisco', route_type='fastest'
        )
        scenic_route = self.route_sim.generate_route(
            'Los Angeles', 'San Francisco', route_type='scenic'
        )
        
        # Both should be valid routes
        assert fastest_route.route_type == 'fastest'
        assert scenic_route.route_type == 'scenic'
        assert fastest_route.distance_km > 0
        assert scenic_route.distance_km > 0
    
    def test_distance_calculation_accuracy(self):
        """Test that distance calculations are reasonable"""
        # Test short distance route
        short_route = self.route_sim.generate_route('New York', 'Newark')
        assert 10 <= short_route.distance_km <= 50  # Reasonable for nearby cities
        
        # Test longer distance route
        long_route = self.route_sim.generate_route('New York', 'Los Angeles')
        assert 3000 <= long_route.distance_km <= 5000  # Cross-country distance
    
    def test_waypoint_positioning(self):
        """Test that waypoints are positioned logically between start and end"""
        route = self.route_sim.generate_route('Miami', 'Atlanta', waypoints=2)
        
        start_lat, start_lon = route.start_point.coordinates
        end_lat, end_lon = route.end_point.coordinates
        
        for waypoint in route.waypoints:
            wp_lat, wp_lon = waypoint.coordinates
            
            # Waypoints should be roughly between start and end
            # (allowing for some deviation for realism)
            lat_range = abs(end_lat - start_lat)
            lon_range = abs(end_lon - start_lon)
            
            # Waypoint should be within expanded bounds
            assert min(start_lat, end_lat) - lat_range <= wp_lat <= max(start_lat, end_lat) + lat_range
            assert min(start_lon, end_lon) - lon_range <= wp_lon <= max(start_lon, end_lon) + lon_range
    
    def test_generate_route_with_stops(self):
        """Test route generation with specific stop types"""
        route_with_stops = self.route_sim.generate_route_with_stops(
            'Denver', 'Salt Lake City',
            stop_types=['gas_station', 'restaurant']
        )
        
        assert 'route' in route_with_stops
        assert 'stops' in route_with_stops
        assert 'total_time_with_stops' in route_with_stops
        
        route = route_with_stops['route']
        stops = route_with_stops['stops']
        
        assert isinstance(route, Route)
        assert len(stops) == len(route.waypoints)
        
        for stop in stops:
            assert 'type' in stop
            assert 'name' in stop
            assert 'duration_minutes' in stop
            assert stop['type'] in ['gas_station', 'restaurant']
    
    def test_fuel_consumption_calculation(self):
        """Test fuel consumption calculation"""
        route = self.route_sim.generate_route('Houston', 'Dallas')
        fuel_info = self.route_sim.calculate_fuel_consumption(route, vehicle_mpg=30.0)
        
        assert 'distance_miles' in fuel_info
        assert 'fuel_gallons' in fuel_info
        assert 'estimated_cost_usd' in fuel_info
        
        assert fuel_info['distance_miles'] > 0
        assert fuel_info['fuel_gallons'] > 0
        assert fuel_info['estimated_cost_usd'] > 0
        
        # Verify calculation accuracy
        expected_gallons = fuel_info['distance_miles'] / 30.0
        assert abs(fuel_info['fuel_gallons'] - expected_gallons) < 0.01
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same routes"""
        seeder1 = MillisecondSeeder(fixed_seed=12345)
        seeder2 = MillisecondSeeder(fixed_seed=12345)
        
        sim1 = RouteSimulator(seeder1)
        sim2 = RouteSimulator(seeder2)
        
        route1 = sim1.generate_route('Seattle', 'Portland', waypoints=1)
        route2 = sim2.generate_route('Seattle', 'Portland', waypoints=1)
        
        # Should generate identical routes with same seed
        assert route1.distance_km == route2.distance_km
        assert route1.estimated_time_minutes == route2.estimated_time_minutes
        assert len(route1.waypoints) == len(route2.waypoints)


class TestPOIGenerator:
    """Test cases for POIGenerator functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.seeder = MillisecondSeeder(fixed_seed=12345)
        self.poi_gen = POIGenerator(self.seeder)
    
    def test_initialization(self):
        """Test POIGenerator initializes correctly"""
        assert self.poi_gen is not None
        assert hasattr(self.poi_gen, 'poi_definitions')
        assert 'restaurant' in self.poi_gen.poi_definitions
        assert 'shop' in self.poi_gen.poi_definitions
    
    def test_generate_basic_pois(self):
        """Test basic POI generation"""
        pois = self.poi_gen.generate_pois('New York', 'united_states', count=5)
        
        assert len(pois) == 5
        
        for poi in pois:
            assert 'name' in poi
            assert 'type' in poi
            assert 'category' in poi
            assert 'address' in poi
            assert 'coordinates' in poi
            assert 'rating' in poi
            assert 'price_level' in poi
            assert 'hours' in poi
            assert 'phone' in poi
            assert 'website' in poi
            assert 'description' in poi
            
            # Verify data types and ranges
            assert isinstance(poi['name'], str)
            assert poi['type'] in self.poi_gen.poi_definitions
            assert isinstance(poi['rating'], float)
            assert 1 <= poi['price_level'] <= 4
            assert len(poi['coordinates']) == 2
    
    def test_generate_pois_with_specific_types(self):
        """Test POI generation with specific types"""
        poi_types = ['restaurant', 'gas_station']
        pois = self.poi_gen.generate_pois(
            'Chicago', 'united_states', 
            count=6, 
            poi_types=poi_types
        )
        
        assert len(pois) == 6
        
        # All POIs should be of specified types
        for poi in pois:
            assert poi['type'] in poi_types
    
    def test_poi_type_distribution(self):
        """Test that POI types are distributed according to weights"""
        pois = self.poi_gen.generate_pois(
            'Los Angeles', 'united_states', 
            count=20,
            poi_types=['restaurant', 'shop', 'landmark']
        )
        
        type_counts = {}
        for poi in pois:
            poi_type = poi['type']
            type_counts[poi_type] = type_counts.get(poi_type, 0) + 1
        
        # Restaurants and shops should be more common than landmarks
        # (based on the weighting system)
        assert type_counts.get('restaurant', 0) >= type_counts.get('landmark', 0)
        assert type_counts.get('shop', 0) >= type_counts.get('landmark', 0)
    
    def test_poi_categories(self):
        """Test that POIs have correct categories"""
        pois = self.poi_gen.generate_pois('Miami', 'united_states', count=10)
        
        for poi in pois:
            poi_type = poi['type']
            expected_category = self.poi_gen.poi_definitions[poi_type]['category']
            assert poi['category'] == expected_category
    
    def test_poi_ratings_and_prices(self):
        """Test that POI ratings and prices are within expected ranges"""
        pois = self.poi_gen.generate_pois('Boston', 'united_states', count=15)
        
        for poi in pois:
            poi_type = poi['type']
            poi_def = self.poi_gen.poi_definitions[poi_type]
            
            # Check rating range
            min_rating, max_rating = poi_def['rating_range']
            assert min_rating <= poi['rating'] <= max_rating
            
            # Check price level range
            min_price, max_price = poi_def['price_range']
            assert min_price <= poi['price_level'] <= max_price
    
    def test_poi_hours_generation(self):
        """Test that POI hours are generated correctly"""
        pois = self.poi_gen.generate_pois(
            'San Francisco', 'united_states', 
            count=5,
            poi_types=['restaurant', 'gas_station', 'service']
        )
        
        for poi in pois:
            hours = poi['hours']
            assert isinstance(hours, dict)
            assert 'monday' in hours
            assert 'sunday' in hours
            
            # Gas stations should be 24 hours
            if poi['type'] == 'gas_station':
                assert hours['monday'] == '24 hours'
            
            # Service businesses should be closed on Sunday
            if poi['type'] == 'service':
                assert hours['sunday'] == 'Closed'
    
    def test_poi_contact_info(self):
        """Test that POI contact information is generated"""
        pois = self.poi_gen.generate_pois('Seattle', 'united_states', count=5)
        
        for poi in pois:
            # Phone should be formatted
            assert poi['phone'].startswith('+1-')
            
            # Website should be valid format
            assert poi['website'].startswith('www.')
            assert any(domain in poi['website'] for domain in ['.com', '.net', '.org', '.biz'])
            
            # Description should mention the POI name
            assert poi['name'] in poi['description']
    
    def test_generate_pois_near_coordinates(self):
        """Test POI generation near specific coordinates"""
        center_lat, center_lon = 40.7128, -74.0060  # NYC coordinates
        
        pois = self.poi_gen.generate_pois(
            'New York', 'united_states',
            count=5,
            center_lat=center_lat,
            center_lon=center_lon,
            radius_km=2.0
        )
        
        assert len(pois) == 5
        
        # All POIs should be within reasonable distance of center
        for poi in pois:
            poi_lat, poi_lon = poi['coordinates']
            
            # Calculate approximate distance (rough check)
            lat_diff = abs(poi_lat - center_lat)
            lon_diff = abs(poi_lon - center_lon)
            
            # Should be within a reasonable range (allowing for some margin)
            assert lat_diff <= 0.1  # Roughly 11km in latitude
            assert lon_diff <= 0.1  # Roughly 11km in longitude
    
    def test_generate_pois_near_route(self):
        """Test POI generation near a route"""
        route_coordinates = [
            (40.7128, -74.0060),  # NYC
            (40.6892, -74.0445),  # Jersey City
            (40.6782, -74.0776)   # Newark
        ]
        
        pois = self.poi_gen.generate_pois_near_route(
            route_coordinates,
            poi_types=['gas_station', 'restaurant'],
            radius_km=1.0,
            pois_per_point=2
        )
        
        # Should have 2 POIs per route point
        assert len(pois) == 6  # 3 points * 2 POIs each
        
        for poi in pois:
            assert 'route_point_index' in poi
            assert 'distance_from_route' in poi
            assert poi['type'] in ['gas_station', 'restaurant']
            assert 0 <= poi['route_point_index'] <= 2
            assert 0.1 <= poi['distance_from_route'] <= 1.0
    
    def test_poi_recommendations(self):
        """Test POI recommendation system"""
        # Test basic recommendations
        restaurant_recs = self.poi_gen.get_poi_recommendations('restaurant')
        assert len(restaurant_recs) <= 4
        assert all(isinstance(rec, str) for rec in restaurant_recs)
        
        # Test recommendations with preferences
        high_rating_prefs = {'rating_min': 4.5}
        high_rating_recs = self.poi_gen.get_poi_recommendations('restaurant', high_rating_prefs)
        assert 'Highly rated by customers' in high_rating_recs
        
        budget_prefs = {'price_max': 2}
        budget_recs = self.poi_gen.get_poi_recommendations('shop', budget_prefs)
        assert 'Budget-friendly prices' in budget_recs
    
    def test_poi_name_generation_variety(self):
        """Test that POI names have variety and realistic patterns"""
        restaurant_pois = self.poi_gen.generate_pois(
            'Portland', 'united_states',
            count=10,
            poi_types=['restaurant']
        )
        
        names = [poi['name'] for poi in restaurant_pois]
        
        # Should have variety in names
        assert len(set(names)) >= 8  # Most names should be unique
        
        # Some names should have restaurant-specific suffixes
        suffixes = ['Restaurant', 'Bistro', 'Cafe', 'Grill', 'Kitchen']
        has_suffix = any(any(suffix in name for suffix in suffixes) for name in names)
        assert has_suffix
    
    def test_invalid_poi_types_handling(self):
        """Test handling of invalid POI types"""
        pois = self.poi_gen.generate_pois(
            'Denver', 'united_states',
            count=5,
            poi_types=['invalid_type', 'restaurant', 'nonexistent']
        )
        
        # Should still generate POIs, filtering out invalid types
        assert len(pois) == 5
        for poi in pois:
            assert poi['type'] in self.poi_gen.poi_definitions
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same POIs"""
        seeder1 = MillisecondSeeder(fixed_seed=12345)
        seeder2 = MillisecondSeeder(fixed_seed=12345)
        
        gen1 = POIGenerator(seeder1)
        gen2 = POIGenerator(seeder2)
        
        pois1 = gen1.generate_pois('Austin', 'united_states', count=3)
        pois2 = gen2.generate_pois('Austin', 'united_states', count=3)
        
        # Should generate identical POIs with same seed
        assert len(pois1) == len(pois2)
        for poi1, poi2 in zip(pois1, pois2):
            assert poi1['name'] == poi2['name']
            assert poi1['type'] == poi2['type']
            assert poi1['rating'] == poi2['rating']
    
    def test_poi_geographical_accuracy(self):
        """Test that POIs are geographically accurate for the country"""
        # Test US POIs
        us_pois = self.poi_gen.generate_pois('Phoenix', 'united_states', count=5)
        
        for poi in us_pois:
            lat, lon = poi['coordinates']
            # Should be within US bounds
            assert 24 <= lat <= 49
            assert -125 <= lon <= -66
            
            # Phone should be US format
            assert poi['phone'].startswith('+1-')
    
    def test_poi_data_completeness(self):
        """Test that all POI data fields are properly populated"""
        pois = self.poi_gen.generate_pois('Nashville', 'united_states', count=3)
        
        required_fields = [
            'name', 'type', 'category', 'address', 'coordinates',
            'rating', 'price_level', 'hours', 'phone', 'website', 'description'
        ]
        
        for poi in pois:
            for field in required_fields:
                assert field in poi
                assert poi[field] is not None
                assert poi[field] != ""
            
            # Address should be a dictionary with required fields
            address = poi['address']
            assert isinstance(address, dict)
            assert 'street' in address
            assert 'city' in address
            assert 'country' in address