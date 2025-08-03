"""
Points of Interest (POI) generation

Generates realistic points of interest including businesses, landmarks, and services.
"""

from typing import List, Dict, Any, Optional, Tuple
import random
from dataclasses import dataclass
from .address import AddressGenerator, Address
from .coordinates import CoordinateGenerator
from ..core.base_generator import BaseGenerator


@dataclass
class POI:
    """Data class representing a Point of Interest"""
    name: str
    type: str
    category: str
    address: Address
    coordinates: Tuple[float, float]
    rating: float
    price_level: int  # 1-4 scale
    hours: Dict[str, str]
    phone: Optional[str] = None
    website: Optional[str] = None
    description: Optional[str] = None


class POIGenerator(BaseGenerator):
    """
    Generator for Points of Interest (POI)
    
    Generates realistic POIs including businesses, landmarks, restaurants,
    and other location-based services with appropriate geographical distribution.
    """
    
    def __init__(self, seeder, locale: str = 'en_US'):
        """Initialize POI generator"""
        super().__init__(seeder, locale)
        self.coordinate_gen = CoordinateGenerator(seeder, locale)
        
        # POI type definitions with realistic names and categories
        self.poi_definitions = {
            'restaurant': {
                'category': 'Food & Dining',
                'names': [
                    'The Golden Spoon', 'Mama\'s Kitchen', 'Urban Bistro', 'Sunset Grill',
                    'The Corner Cafe', 'Riverside Restaurant', 'Chef\'s Table', 'Garden Terrace',
                    'The Local Eatery', 'Fusion Kitchen', 'Harvest Moon', 'The Daily Bread'
                ],
                'price_range': (1, 4),
                'rating_range': (2.5, 4.8)
            },
            'shop': {
                'category': 'Shopping',
                'names': [
                    'City Market', 'The Corner Store', 'Fashion Forward', 'Tech Hub',
                    'Book Nook', 'Artisan Crafts', 'Home & Garden', 'Sports Central',
                    'Music World', 'Vintage Finds', 'The Gift Shop', 'Electronics Plus'
                ],
                'price_range': (1, 3),
                'rating_range': (3.0, 4.7)
            },
            'service': {
                'category': 'Services',
                'names': [
                    'Quick Fix Auto', 'City Bank', 'Hair Studio', 'Fitness First',
                    'Medical Center', 'Legal Associates', 'Dental Care', 'Pet Clinic',
                    'Dry Cleaners', 'Insurance Agency', 'Real Estate Office', 'Tax Services'
                ],
                'price_range': (2, 4),
                'rating_range': (3.2, 4.6)
            },
            'landmark': {
                'category': 'Landmarks & Attractions',
                'names': [
                    'City Hall', 'Central Park', 'Historic Museum', 'Art Gallery',
                    'Memorial Plaza', 'Observation Tower', 'Cultural Center', 'Public Library',
                    'Botanical Garden', 'Heritage Site', 'Monument Square', 'Civic Center'
                ],
                'price_range': (1, 2),
                'rating_range': (3.5, 4.9)
            },
            'entertainment': {
                'category': 'Entertainment',
                'names': [
                    'Cinema Complex', 'Live Music Venue', 'Comedy Club', 'Sports Bar',
                    'Bowling Alley', 'Arcade Center', 'Theater District', 'Night Club',
                    'Pool Hall', 'Karaoke Lounge', 'Dance Studio', 'Gaming Center'
                ],
                'price_range': (2, 4),
                'rating_range': (3.0, 4.5)
            },
            'hotel': {
                'category': 'Accommodation',
                'names': [
                    'Grand Hotel', 'City Inn', 'Comfort Lodge', 'Business Suites',
                    'Boutique Hotel', 'Extended Stay', 'Resort & Spa', 'Budget Motel',
                    'Historic Inn', 'Luxury Suites', 'Traveler\'s Rest', 'Downtown Hotel'
                ],
                'price_range': (2, 4),
                'rating_range': (3.0, 4.8)
            },
            'gas_station': {
                'category': 'Automotive',
                'names': [
                    'Shell', 'BP', 'Exxon', 'Chevron', 'Mobil', 'Texaco',
                    'Sunoco', 'Marathon', 'Citgo', 'Valero', 'Phillips 66', 'Speedway'
                ],
                'price_range': (1, 2),
                'rating_range': (3.0, 4.2)
            },
            'hospital': {
                'category': 'Healthcare',
                'names': [
                    'General Hospital', 'Medical Center', 'Emergency Care', 'Specialty Clinic',
                    'Children\'s Hospital', 'Urgent Care', 'Rehabilitation Center', 'Surgery Center',
                    'Cancer Center', 'Heart Institute', 'Women\'s Health', 'Mental Health Center'
                ],
                'price_range': (3, 4),
                'rating_range': (3.5, 4.7)
            }
        }
    
    def generate_pois(self, 
                     city: str, 
                     country: str, 
                     count: int = 10, 
                     poi_types: List[str] = None,
                     **kwargs) -> List[Dict[str, Any]]:
        """
        Generate points of interest for a city
        
        Args:
            city: Target city name
            country: Country containing the city
            count: Number of POIs to generate
            poi_types: Types of POIs to generate (restaurants, shops, etc.)
            **kwargs: Additional parameters (center_lat, center_lon, radius_km)
            
        Returns:
            List[Dict[str, Any]]: List of POI dictionaries
        """
        if poi_types is None:
            poi_types = ['restaurant', 'shop', 'service', 'landmark']
        
        # Validate POI types
        valid_types = set(self.poi_definitions.keys())
        poi_types = [t for t in poi_types if t in valid_types]
        if not poi_types:
            poi_types = ['restaurant', 'shop']
        
        pois = []
        addr_gen = AddressGenerator(self.seeder, country)
        
        for i in range(count):
            poi_type = self._select_poi_type(poi_types, i)
            poi_data = self._generate_single_poi(poi_type, city, country, addr_gen, **kwargs)
            pois.append(poi_data)
        
        return pois
    
    def _select_poi_type(self, poi_types: List[str], index: int) -> str:
        """Select POI type with some distribution logic"""
        # Use seeded random for reproducibility
        type_seed = self.seeder.get_contextual_seed(f'poi_type_{index}')
        random.seed(type_seed)
        
        # Weight certain types more heavily for realism
        weights = {
            'restaurant': 3,
            'shop': 3,
            'service': 2,
            'landmark': 1,
            'entertainment': 2,
            'hotel': 1,
            'gas_station': 2,
            'hospital': 1
        }
        
        weighted_types = []
        for poi_type in poi_types:
            weight = weights.get(poi_type, 1)
            weighted_types.extend([poi_type] * weight)
        
        return random.choice(weighted_types)
    
    def _generate_single_poi(self, 
                           poi_type: str, 
                           city: str, 
                           country: str, 
                           addr_gen: AddressGenerator,
                           **kwargs) -> Dict[str, Any]:
        """Generate a single POI"""
        poi_def = self.poi_definitions[poi_type]
        
        # Generate name
        name = self._generate_poi_name(poi_type, poi_def)
        
        # Generate address and coordinates
        if 'center_lat' in kwargs and 'center_lon' in kwargs:
            radius_km = kwargs.get('radius_km', 5.0)
            address = addr_gen.generate_address_near(
                kwargs['center_lat'], kwargs['center_lon'], radius_km
            )
        else:
            address = addr_gen._generate_single_address(city=city)
        
        # Generate rating and price level
        rating = self._generate_rating(poi_def['rating_range'])
        price_level = random.randint(*poi_def['price_range'])
        
        # Generate hours
        hours = self._generate_hours(poi_type)
        
        # Generate contact info
        phone = self._generate_phone_number(country)
        website = self._generate_website(name)
        
        return {
            'name': name,
            'type': poi_type,
            'category': poi_def['category'],
            'address': address.__dict__,
            'coordinates': address.coordinates,
            'rating': rating,
            'price_level': price_level,
            'hours': hours,
            'phone': phone,
            'website': website,
            'description': self._generate_description(poi_type, name)
        }
    
    def _generate_poi_name(self, poi_type: str, poi_def: Dict) -> str:
        """Generate realistic POI name"""
        names = poi_def['names']
        
        # For some types, add variety with prefixes/suffixes
        if poi_type == 'restaurant':
            prefixes = ['', 'The ', 'Chez ', 'Casa ', 'Le ', 'La ']
            suffixes = ['', ' Restaurant', ' Bistro', ' Cafe', ' Grill', ' Kitchen']
            base_name = random.choice(names)
            prefix = random.choice(prefixes)
            suffix = random.choice(suffixes)
            return f"{prefix}{base_name}{suffix}".strip()
        
        elif poi_type == 'shop':
            suffixes = ['', ' Store', ' Shop', ' Outlet', ' Market', ' Emporium']
            base_name = random.choice(names)
            suffix = random.choice(suffixes)
            return f"{base_name}{suffix}".strip()
        
        else:
            return random.choice(names)
    
    def _generate_rating(self, rating_range: Tuple[float, float]) -> float:
        """Generate realistic rating"""
        min_rating, max_rating = rating_range
        rating = random.uniform(min_rating, max_rating)
        return round(rating, 1)
    
    def _generate_hours(self, poi_type: str) -> Dict[str, str]:
        """Generate realistic operating hours"""
        hour_patterns = {
            'restaurant': {
                'monday': '11:00 AM - 10:00 PM',
                'tuesday': '11:00 AM - 10:00 PM',
                'wednesday': '11:00 AM - 10:00 PM',
                'thursday': '11:00 AM - 10:00 PM',
                'friday': '11:00 AM - 11:00 PM',
                'saturday': '10:00 AM - 11:00 PM',
                'sunday': '10:00 AM - 9:00 PM'
            },
            'shop': {
                'monday': '9:00 AM - 8:00 PM',
                'tuesday': '9:00 AM - 8:00 PM',
                'wednesday': '9:00 AM - 8:00 PM',
                'thursday': '9:00 AM - 8:00 PM',
                'friday': '9:00 AM - 9:00 PM',
                'saturday': '9:00 AM - 9:00 PM',
                'sunday': '11:00 AM - 6:00 PM'
            },
            'service': {
                'monday': '8:00 AM - 5:00 PM',
                'tuesday': '8:00 AM - 5:00 PM',
                'wednesday': '8:00 AM - 5:00 PM',
                'thursday': '8:00 AM - 5:00 PM',
                'friday': '8:00 AM - 5:00 PM',
                'saturday': '9:00 AM - 2:00 PM',
                'sunday': 'Closed'
            },
            'gas_station': {
                'monday': '24 hours',
                'tuesday': '24 hours',
                'wednesday': '24 hours',
                'thursday': '24 hours',
                'friday': '24 hours',
                'saturday': '24 hours',
                'sunday': '24 hours'
            },
            'hospital': {
                'monday': '24 hours',
                'tuesday': '24 hours',
                'wednesday': '24 hours',
                'thursday': '24 hours',
                'friday': '24 hours',
                'saturday': '24 hours',
                'sunday': '24 hours'
            }
        }
        
        return hour_patterns.get(poi_type, hour_patterns['shop'])
    
    def _generate_phone_number(self, country: str) -> str:
        """Generate phone number using localization"""
        from ..core.localization import LocalizationEngine
        localization = LocalizationEngine()
        return localization.format_phone(country)
    
    def _generate_website(self, name: str) -> str:
        """Generate realistic website URL"""
        # Clean name for URL
        clean_name = name.lower().replace(' ', '').replace('\'', '').replace('&', 'and')
        domains = ['.com', '.net', '.org', '.biz']
        domain = random.choice(domains)
        return f"www.{clean_name}{domain}"
    
    def _generate_description(self, poi_type: str, name: str) -> str:
        """Generate realistic description"""
        descriptions = {
            'restaurant': f"{name} offers delicious cuisine in a welcoming atmosphere. Perfect for family dining or special occasions.",
            'shop': f"{name} provides quality products and excellent customer service. Visit us for all your shopping needs.",
            'service': f"{name} delivers professional services with experienced staff. We're committed to customer satisfaction.",
            'landmark': f"{name} is a notable landmark and popular destination. A must-see attraction for visitors and locals alike.",
            'entertainment': f"{name} offers great entertainment and fun activities. Perfect for a night out with friends or family.",
            'hotel': f"{name} provides comfortable accommodations and excellent amenities. Your home away from home.",
            'gas_station': f"{name} offers fuel, convenience items, and automotive services. Quick and convenient service.",
            'hospital': f"{name} provides comprehensive medical care with state-of-the-art facilities and experienced medical professionals."
        }
        
        return descriptions.get(poi_type, f"{name} is a local business serving the community.")
    
    def generate_pois_near_route(self, 
                                route_coordinates: List[Tuple[float, float]], 
                                poi_types: List[str] = None,
                                radius_km: float = 2.0,
                                pois_per_point: int = 3) -> List[Dict[str, Any]]:
        """
        Generate POIs near a route
        
        Args:
            route_coordinates: List of (lat, lon) coordinates along route
            poi_types: Types of POIs to generate
            radius_km: Search radius around each point
            pois_per_point: Number of POIs per route point
            
        Returns:
            List[Dict[str, Any]]: List of POIs near the route
        """
        if poi_types is None:
            poi_types = ['gas_station', 'restaurant', 'shop']
        
        all_pois = []
        
        for i, (lat, lon) in enumerate(route_coordinates):
            pois = self.generate_pois(
                city=f"RoutePoint{i}",
                country='united_states',  # Default country
                count=pois_per_point,
                poi_types=poi_types,
                center_lat=lat,
                center_lon=lon,
                radius_km=radius_km
            )
            
            # Add route point reference
            for poi in pois:
                poi['route_point_index'] = i
                poi['distance_from_route'] = random.uniform(0.1, radius_km)
            
            all_pois.extend(pois)
        
        return all_pois
    
    def get_poi_recommendations(self, 
                               poi_type: str, 
                               user_preferences: Dict[str, Any] = None) -> List[str]:
        """
        Get POI recommendations based on type and preferences
        
        Args:
            poi_type: Type of POI
            user_preferences: User preferences (rating_min, price_max, etc.)
            
        Returns:
            List[str]: List of recommended POI features
        """
        if user_preferences is None:
            user_preferences = {}
        
        recommendations = {
            'restaurant': [
                'Outdoor seating available',
                'Family-friendly atmosphere',
                'Live music on weekends',
                'Happy hour specials',
                'Vegetarian options',
                'Local ingredients',
                'Wine selection',
                'Takeout available'
            ],
            'hotel': [
                'Free WiFi',
                'Fitness center',
                'Pool and spa',
                'Business center',
                'Pet-friendly',
                'Complimentary breakfast',
                'Airport shuttle',
                'Room service'
            ],
            'shop': [
                'Extended hours',
                'Online ordering',
                'Loyalty program',
                'Expert staff',
                'Price matching',
                'Gift wrapping',
                'Returns accepted',
                'Local products'
            ]
        }
        
        base_features = recommendations.get(poi_type, ['Quality service', 'Convenient location'])
        
        # Filter based on preferences
        rating_min = user_preferences.get('rating_min', 0)
        if rating_min >= 4.0:
            base_features.insert(0, 'Highly rated by customers')  # Insert at beginning to ensure it's included
        
        price_max = user_preferences.get('price_max', 4)
        if price_max <= 2:
            base_features.insert(0, 'Budget-friendly prices')  # Insert at beginning to ensure it's included
        
        # Return up to 4 features, ensuring preference-based features are included
        return base_features[:4]