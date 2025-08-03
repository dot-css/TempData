"""
Address generation with country-specific patterns

Generates realistic addresses following country-specific formatting and patterns.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import random
from ..core.base_generator import BaseGenerator
from ..core.seeding import MillisecondSeeder
from ..core.localization import LocalizationEngine
from .coordinates import CoordinateGenerator


@dataclass
class Address:
    """Data class representing a geographical address"""
    street: str
    city: str
    state_province: str
    postal_code: str
    country: str
    coordinates: Tuple[float, float]
    formatted_address: Optional[str] = None


class AddressGenerator(BaseGenerator):
    """
    Generator for realistic addresses with country-specific patterns
    
    Generates addresses that follow local formatting conventions and
    geographical accuracy for specified countries.
    """
    
    def __init__(self, seeder: MillisecondSeeder, country: str = 'united_states'):
        """
        Initialize address generator for specific country
        
        Args:
            seeder: MillisecondSeeder instance
            country: Target country for address generation
        """
        # Use localization engine to get proper locale
        self.localization = LocalizationEngine()
        locale = self.localization.get_locale(country)
        
        # Fallback to supported locales if the requested one isn't available
        supported_locales = {
            'united_states': 'en_US',
            'pakistan': 'en_US',  # Fallback to English for Pakistan
            'germany': 'de_DE',
            'united_kingdom': 'en_GB',
            'france': 'fr_FR',
            'japan': 'ja_JP',
            'china': 'zh_CN',
            'india': 'en_IN',
            'brazil': 'pt_BR',
            'canada': 'en_CA',
            'australia': 'en_AU',
            'russia': 'ru_RU',
            'south_korea': 'ko_KR',
            'mexico': 'es_MX',
            'italy': 'it_IT',
            'spain': 'es_ES',
            'netherlands': 'nl_NL',
            'sweden': 'sv_SE',
            'norway': 'no_NO',
            'switzerland': 'de_CH',
            'turkey': 'tr_TR',
            'south_africa': 'en_ZA'
        }
        
        # Use supported locale or fallback to en_US
        faker_locale = supported_locales.get(country, 'en_US')
        
        super().__init__(seeder, faker_locale)
        self.country = country
        self.coordinate_gen = CoordinateGenerator(seeder, faker_locale)
        
        # Country-specific city lists for more realistic generation
        self._city_data = self._get_country_cities()
    
    def _get_country_cities(self) -> List[str]:
        """Get list of cities for the country"""
        city_lists = {
            'united_states': [
                'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
                'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
                'Austin', 'Jacksonville', 'Fort Worth', 'Columbus', 'Charlotte',
                'San Francisco', 'Indianapolis', 'Seattle', 'Denver', 'Washington'
            ],
            'pakistan': [
                'Karachi', 'Lahore', 'Islamabad', 'Rawalpindi', 'Faisalabad',
                'Multan', 'Peshawar', 'Quetta', 'Sialkot', 'Gujranwala',
                'Hyderabad', 'Bahawalpur', 'Sargodha', 'Sukkur', 'Larkana'
            ],
            'germany': [
                'Berlin', 'Hamburg', 'Munich', 'Cologne', 'Frankfurt',
                'Stuttgart', 'Düsseldorf', 'Dortmund', 'Essen', 'Leipzig',
                'Bremen', 'Dresden', 'Hanover', 'Nuremberg', 'Duisburg'
            ],
            'united_kingdom': [
                'London', 'Birmingham', 'Manchester', 'Glasgow', 'Liverpool',
                'Leeds', 'Sheffield', 'Edinburgh', 'Bristol', 'Cardiff',
                'Leicester', 'Coventry', 'Bradford', 'Belfast', 'Nottingham'
            ],
            'france': [
                'Paris', 'Marseille', 'Lyon', 'Toulouse', 'Nice',
                'Nantes', 'Strasbourg', 'Montpellier', 'Bordeaux', 'Lille',
                'Rennes', 'Reims', 'Le Havre', 'Saint-Étienne', 'Toulon'
            ],
            'japan': [
                'Tokyo', 'Yokohama', 'Osaka', 'Nagoya', 'Sapporo',
                'Fukuoka', 'Kobe', 'Kyoto', 'Kawasaki', 'Saitama',
                'Hiroshima', 'Sendai', 'Kitakyushu', 'Chiba', 'Sakai'
            ]
        }
        return city_lists.get(self.country, ['Default City'])
    
    def generate_multiple(self, count: int, **kwargs) -> List[Dict[str, Any]]:
        """
        Generate multiple addresses
        
        Args:
            count: Number of addresses to generate
            **kwargs: Additional parameters (city, state_filter, etc.)
            
        Returns:
            List[Dict[str, Any]]: List of address dictionaries
        """
        addresses = []
        for _ in range(count):
            address = self._generate_single_address(**kwargs)
            addresses.append(address.__dict__)
        return addresses
    
    def _generate_single_address(self, **kwargs) -> Address:
        """
        Generate a single realistic address
        
        Args:
            **kwargs: Optional parameters (city, state_filter, etc.)
            
        Returns:
            Address: Generated address object
        """
        # Use provided city or select random from country cities using seeder
        if 'city' in kwargs:
            city = kwargs['city']
        else:
            # Use seeder for reproducible city selection
            city_seed = self.seeder.get_contextual_seed('city_selection')
            random.seed(city_seed)
            city = random.choice(self._city_data)
        
        # Generate country-specific address components
        street = self._generate_street_address()
        state_province = self._generate_state_province()
        postal_code = self._generate_postal_code()
        
        # Generate coordinates for the city
        coordinates = self.coordinate_gen.generate_coordinates(city, self.country)
        
        # Create address object
        address = Address(
            street=street,
            city=city,
            state_province=state_province,
            postal_code=postal_code,
            country=self.country,
            coordinates=coordinates
        )
        
        # Add formatted address using localization
        address.formatted_address = self._format_address(address)
        
        return address
    
    def _generate_street_address(self) -> str:
        """Generate country-specific street address"""
        if self.country == 'pakistan':
            # Pakistani address patterns
            street_types = ['Road', 'Street', 'Lane', 'Block', 'Sector']
            areas = ['Model Town', 'DHA', 'Gulberg', 'Johar Town', 'Clifton']
            number = random.randint(1, 999)
            area = random.choice(areas)
            street_type = random.choice(street_types)
            return f"House {number}, {area} {street_type}"
        
        elif self.country == 'germany':
            # German address patterns
            street_suffixes = ['straße', 'weg', 'platz', 'allee', 'gasse']
            street_names = ['Haupt', 'Kirch', 'Schul', 'Markt', 'Berg']
            number = random.randint(1, 200)
            name = random.choice(street_names)
            suffix = random.choice(street_suffixes)
            return f"{name}{suffix} {number}"
        
        elif self.country == 'japan':
            # Japanese address patterns
            number = random.randint(1, 50)
            chome = random.randint(1, 10)
            ban = random.randint(1, 20)
            return f"{chome}-{ban}-{number}"
        
        else:
            # Default to faker for other countries
            return self.faker.street_address()
    
    def _generate_state_province(self) -> str:
        """Generate country-specific state/province"""
        state_data = {
            'united_states': [
                'California', 'Texas', 'Florida', 'New York', 'Pennsylvania',
                'Illinois', 'Ohio', 'Georgia', 'North Carolina', 'Michigan'
            ],
            'pakistan': [
                'Punjab', 'Sindh', 'Khyber Pakhtunkhwa', 'Balochistan',
                'Islamabad Capital Territory', 'Gilgit-Baltistan', 'Azad Kashmir'
            ],
            'germany': [
                'Bavaria', 'Baden-Württemberg', 'North Rhine-Westphalia',
                'Hesse', 'Saxony', 'Lower Saxony', 'Rhineland-Palatinate',
                'Schleswig-Holstein', 'Brandenburg', 'Saxony-Anhalt'
            ],
            'canada': [
                'Ontario', 'Quebec', 'British Columbia', 'Alberta',
                'Manitoba', 'Saskatchewan', 'Nova Scotia', 'New Brunswick'
            ],
            'united_kingdom': [
                'England', 'Scotland', 'Wales', 'Northern Ireland'
            ],
            'france': [
                'Île-de-France', 'Provence-Alpes-Côte d\'Azur', 'Auvergne-Rhône-Alpes',
                'Nouvelle-Aquitaine', 'Occitanie', 'Hauts-de-France'
            ],
            'japan': [
                'Tokyo', 'Osaka', 'Kanagawa', 'Aichi', 'Saitama', 'Chiba'
            ]
        }
        
        if self.country in state_data:
            # Use seeder for reproducible state selection
            state_seed = self.seeder.get_contextual_seed('state_selection')
            random.seed(state_seed)
            return random.choice(state_data[self.country])
        else:
            # Fallback for countries without state data
            try:
                return self.faker.state()
            except AttributeError:
                # If faker doesn't have state for this locale, use a generic region
                return "Region"
    
    def _generate_postal_code(self) -> str:
        """Generate country-specific postal code"""
        return self.localization.format_postal_code(self.country)
    
    def _format_address(self, address: Address) -> str:
        """Format address according to country standards"""
        address_parts = {
            'street': address.street,
            'city': address.city,
            'state': address.state_province,
            'postal_code': address.postal_code,
            'country': address.country
        }
        
        return self.localization.format_address(self.country, address_parts)
    
    def generate_address_near(self, 
                            center_lat: float, 
                            center_lon: float, 
                            radius_km: float = 5.0) -> Address:
        """
        Generate address near specific coordinates
        
        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            radius_km: Radius in kilometers
            
        Returns:
            Address: Generated address near the center point
        """
        # Generate coordinates near the center
        coordinates = self.coordinate_gen.generate_coordinates_near(
            center_lat, center_lon, radius_km
        )
        
        # Generate other address components
        street = self._generate_street_address()
        city = random.choice(self._city_data)
        state_province = self._generate_state_province()
        postal_code = self._generate_postal_code()
        
        address = Address(
            street=street,
            city=city,
            state_province=state_province,
            postal_code=postal_code,
            country=self.country,
            coordinates=coordinates
        )
        
        address.formatted_address = self._format_address(address)
        return address
    
    def validate_address(self, address: Address) -> bool:
        """
        Validate address components
        
        Args:
            address: Address to validate
            
        Returns:
            bool: True if address is valid
        """
        # Validate postal code format
        if not self.localization.validate_postal_code(self.country, address.postal_code):
            return False
        
        # Validate coordinates are reasonable
        lat, lon = address.coordinates
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return False
        
        # Validate required fields are present
        required_fields = [address.street, address.city, address.country]
        if not all(field.strip() for field in required_fields):
            return False
        
        return True