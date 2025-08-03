"""
Unit tests for geographical data structures

Tests Address and Route dataclasses, AddressGenerator, and CoordinateGenerator
for geographical data accuracy and country-specific patterns.
"""

import pytest
import math
from tempdata.core.seeding import MillisecondSeeder
from tempdata.geo.address import AddressGenerator, Address
from tempdata.geo.coordinates import CoordinateGenerator
from tempdata.geo.routes import RouteSimulator, Route


class TestAddress:
    """Test cases for Address dataclass"""
    
    def test_address_creation(self):
        """Test Address dataclass creation"""
        address = Address(
            street="123 Main St",
            city="New York",
            state_province="NY",
            postal_code="10001",
            country="united_states",
            coordinates=(40.7128, -74.0060)
        )
        
        assert address.street == "123 Main St"
        assert address.city == "New York"
        assert address.state_province == "NY"
        assert address.postal_code == "10001"
        assert address.country == "united_states"
        assert address.coordinates == (40.7128, -74.0060)
        assert address.formatted_address is None  # Optional field


class TestRoute:
    """Test cases for Route dataclass"""
    
    def test_route_creation(self):
        """Test Route dataclass creation"""
        start_addr = Address("123 Start St", "StartCity", "State1", "12345", "US", (40.0, -74.0))
        end_addr = Address("456 End St", "EndCity", "State2", "67890", "US", (41.0, -75.0))
        waypoint = Address("789 Way St", "WayCity", "State3", "11111", "US", (40.5, -74.5))
        
        route = Route(
            start_point=start_addr,
            end_point=end_addr,
            waypoints=[waypoint],
            distance_km=150.5,
            estimated_time_minutes=180
        )
        
        assert route.start_point == start_addr
        assert route.end_point == end_addr
        assert len(route.waypoints) == 1
        assert route.waypoints[0] == waypoint
        assert route.distance_km == 150.5
        assert route.estimated_time_minutes == 180


class TestCoordinateGenerator:
    """Test cases for CoordinateGenerator functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.seeder = MillisecondSeeder(fixed_seed=12345)
        self.coord_gen = CoordinateGenerator(self.seeder)
    
    def test_initialization(self):
        """Test CoordinateGenerator initializes correctly"""
        assert self.coord_gen is not None
        assert hasattr(self.coord_gen, '_city_data')
    
    def test_generate_coordinates_with_city_boundaries(self):
        """Test coordinate generation within city boundaries"""
        # Test New York coordinates
        lat, lon = self.coord_gen.generate_coordinates('new_york', 'united_states')
        
        # Should be within reasonable bounds for New York
        assert 40.4 <= lat <= 41.0  # Approximate NYC latitude range
        assert -74.3 <= lon <= -73.7  # Approximate NYC longitude range
    
    def test_generate_coordinates_fallback(self):
        """Test coordinate generation fallback for unknown cities"""
        lat, lon = self.coord_gen.generate_coordinates('unknown_city', 'unknown_country')
        
        # Should still generate valid coordinates
        assert -90 <= lat <= 90
        assert -180 <= lon <= 180
    
    def test_generate_coordinates_near(self):
        """Test coordinate generation near a center point"""
        center_lat, center_lon = 40.7128, -74.0060  # NYC
        radius_km = 10.0
        
        lat, lon = self.coord_gen.generate_coordinates_near(center_lat, center_lon, radius_km)
        
        # Calculate distance to verify it's within radius
        distance = self.coord_gen.calculate_distance(center_lat, center_lon, lat, lon)
        assert distance <= radius_km
    
    def test_get_city_center(self):
        """Test getting city center coordinates"""
        center = self.coord_gen.get_city_center('new_york', 'united_states')
        
        if center:  # If city data is available
            lat, lon = center
            assert isinstance(lat, float)
            assert isinstance(lon, float)
            assert -90 <= lat <= 90
            assert -180 <= lon <= 180
    
    def test_get_city_bounds(self):
        """Test getting city boundary coordinates"""
        bounds = self.coord_gen.get_city_bounds('new_york', 'united_states')
        
        if bounds:  # If city data is available
            assert 'north' in bounds
            assert 'south' in bounds
            assert 'east' in bounds
            assert 'west' in bounds
            assert bounds['north'] > bounds['south']
            assert bounds['east'] > bounds['west']
    
    def test_calculate_distance(self):
        """Test distance calculation between coordinates"""
        # Test known distance (NYC to LA approximately 3944 km)
        nyc_lat, nyc_lon = 40.7128, -74.0060
        la_lat, la_lon = 34.0522, -118.2437
        
        distance = self.coord_gen.calculate_distance(nyc_lat, nyc_lon, la_lat, la_lon)
        
        # Should be approximately 3944 km (allow some tolerance)
        assert 3800 <= distance <= 4100
    
    def test_distance_calculation_same_point(self):
        """Test distance calculation for same point"""
        lat, lon = 40.7128, -74.0060
        distance = self.coord_gen.calculate_distance(lat, lon, lat, lon)
        assert distance == 0.0


class TestAddressGenerator:
    """Test cases for AddressGenerator functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.seeder = MillisecondSeeder(fixed_seed=12345)
    
    def test_initialization_us(self):
        """Test AddressGenerator initialization for US"""
        addr_gen = AddressGenerator(self.seeder, 'united_states')
        assert addr_gen.country == 'united_states'
        assert len(addr_gen._city_data) > 0
        assert 'New York City' in addr_gen._city_data
    
    def test_initialization_pakistan(self):
        """Test AddressGenerator initialization for Pakistan"""
        addr_gen = AddressGenerator(self.seeder, 'pakistan')
        assert addr_gen.country == 'pakistan'
        assert 'Karachi' in addr_gen._city_data
        assert 'Lahore' in addr_gen._city_data
    
    def test_initialization_germany(self):
        """Test AddressGenerator initialization for Germany"""
        addr_gen = AddressGenerator(self.seeder, 'germany')
        assert addr_gen.country == 'germany'
        assert 'Berlin' in addr_gen._city_data
        assert 'Munich' in addr_gen._city_data
    
    def test_generate_single_address_us(self):
        """Test single address generation for US"""
        addr_gen = AddressGenerator(self.seeder, 'united_states')
        address = addr_gen._generate_single_address()
        
        assert isinstance(address, Address)
        assert address.country == 'united_states'
        assert address.street
        assert address.city
        assert address.state_province
        assert address.postal_code
        assert len(address.coordinates) == 2
        assert isinstance(address.formatted_address, str)
    
    def test_generate_single_address_pakistan(self):
        """Test single address generation for Pakistan"""
        addr_gen = AddressGenerator(self.seeder, 'pakistan')
        address = addr_gen._generate_single_address()
        
        assert isinstance(address, Address)
        assert address.country == 'pakistan'
        assert address.city in addr_gen._city_data
        assert address.state_province in [
            'Punjab', 'Sindh', 'Khyber Pakhtunkhwa', 'Balochistan',
            'Islamabad Capital Territory', 'Gilgit-Baltistan', 'Azad Kashmir'
        ]
    
    def test_generate_multiple_addresses(self):
        """Test generating multiple addresses"""
        addr_gen = AddressGenerator(self.seeder, 'united_states')
        addresses = addr_gen.generate_multiple(5)
        
        assert len(addresses) == 5
        for addr_dict in addresses:
            assert 'street' in addr_dict
            assert 'city' in addr_dict
            assert 'state_province' in addr_dict
            assert 'postal_code' in addr_dict
            assert 'country' in addr_dict
            assert 'coordinates' in addr_dict
            assert addr_dict['country'] == 'united_states'
    
    def test_country_specific_street_patterns(self):
        """Test country-specific street address patterns"""
        # Test Pakistani street patterns
        addr_gen_pk = AddressGenerator(self.seeder, 'pakistan')
        pk_address = addr_gen_pk._generate_single_address()
        assert 'House' in pk_address.street or any(area in pk_address.street 
                                                  for area in ['Model Town', 'DHA', 'Gulberg'])
        
        # Test German street patterns
        addr_gen_de = AddressGenerator(self.seeder, 'germany')
        de_address = addr_gen_de._generate_single_address()
        assert any(suffix in de_address.street.lower() 
                  for suffix in ['straße', 'weg', 'platz', 'allee', 'gasse'])
        
        # Test Japanese street patterns
        addr_gen_jp = AddressGenerator(self.seeder, 'japan')
        jp_address = addr_gen_jp._generate_single_address()
        assert '-' in jp_address.street  # Japanese address format
    
    def test_generate_address_near(self):
        """Test generating address near specific coordinates"""
        addr_gen = AddressGenerator(self.seeder, 'united_states')
        center_lat, center_lon = 40.7128, -74.0060  # NYC
        radius_km = 5.0
        
        address = addr_gen.generate_address_near(center_lat, center_lon, radius_km)
        
        assert isinstance(address, Address)
        
        # Verify coordinates are within radius
        addr_lat, addr_lon = address.coordinates
        distance = addr_gen.coordinate_gen.calculate_distance(
            center_lat, center_lon, addr_lat, addr_lon
        )
        assert distance <= radius_km
    
    def test_address_validation(self):
        """Test address validation"""
        addr_gen = AddressGenerator(self.seeder, 'united_states')
        
        # Valid address
        valid_address = Address(
            street="123 Main St",
            city="New York",
            state_province="NY",
            postal_code="10001",
            country="united_states",
            coordinates=(40.7128, -74.0060)
        )
        assert addr_gen.validate_address(valid_address)
        
        # Invalid coordinates
        invalid_coords = Address(
            street="123 Main St",
            city="New York",
            state_province="NY",
            postal_code="10001",
            country="united_states",
            coordinates=(200.0, -74.0060)  # Invalid latitude
        )
        assert not addr_gen.validate_address(invalid_coords)
        
        # Empty required field
        empty_street = Address(
            street="",
            city="New York",
            state_province="NY",
            postal_code="10001",
            country="united_states",
            coordinates=(40.7128, -74.0060)
        )
        assert not addr_gen.validate_address(empty_street)
    
    def test_postal_code_generation(self):
        """Test postal code generation for different countries"""
        # US postal codes
        addr_gen_us = AddressGenerator(self.seeder, 'united_states')
        us_address = addr_gen_us._generate_single_address()
        assert len(us_address.postal_code) == 5
        assert us_address.postal_code.isdigit()
        
        # UK postal codes (if supported)
        addr_gen_uk = AddressGenerator(self.seeder, 'united_kingdom')
        uk_address = addr_gen_uk._generate_single_address()
        assert len(uk_address.postal_code) >= 5  # UK postcodes vary in length
    
    def test_formatted_address_generation(self):
        """Test that formatted addresses are generated"""
        addr_gen = AddressGenerator(self.seeder, 'united_states')
        address = addr_gen._generate_single_address()
        
        assert address.formatted_address is not None
        assert isinstance(address.formatted_address, str)
        assert len(address.formatted_address) > 0
        
        # Should contain address components
        assert address.street in address.formatted_address
        assert address.city in address.formatted_address
    
    def test_city_selection_with_parameter(self):
        """Test city selection with specific city parameter"""
        addr_gen = AddressGenerator(self.seeder, 'united_states')
        address = addr_gen._generate_single_address(city='Chicago')
        
        assert address.city == 'Chicago'
    
    def test_geographical_accuracy(self):
        """Test geographical accuracy of generated addresses"""
        addr_gen = AddressGenerator(self.seeder, 'united_states')
        
        # Generate multiple addresses and check coordinate ranges
        addresses = addr_gen.generate_multiple(10)
        
        for addr_dict in addresses:
            lat, lon = addr_dict['coordinates']
            
            # US coordinates should be within reasonable bounds
            assert 24 <= lat <= 49  # Continental US latitude range
            assert -125 <= lon <= -66  # Continental US longitude range
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same addresses"""
        seeder1 = MillisecondSeeder(fixed_seed=12345)
        seeder2 = MillisecondSeeder(fixed_seed=12345)
        
        addr_gen1 = AddressGenerator(seeder1, 'united_states')
        addr_gen2 = AddressGenerator(seeder2, 'united_states')
        
        address1 = addr_gen1._generate_single_address()
        address2 = addr_gen2._generate_single_address()
        
        # Should generate identical addresses with same seed
        assert address1.street == address2.street
        assert address1.city == address2.city
        assert address1.state_province == address2.state_province
        assert address1.postal_code == address2.postal_code
    
    def test_multiple_countries_support(self):
        """Test that multiple countries are supported"""
        countries = ['united_states', 'pakistan', 'germany', 'united_kingdom', 'france', 'japan']
        
        for country in countries:
            addr_gen = AddressGenerator(self.seeder, country)
            address = addr_gen._generate_single_address()
            
            assert address.country == country
            assert len(addr_gen._city_data) > 0
            assert address.city in addr_gen._city_data