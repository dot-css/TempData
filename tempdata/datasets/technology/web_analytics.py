"""
Web analytics data generator

Generates realistic web analytics data with user session patterns, page view
distributions, device types, bounce rates, and conversion funnels.
"""

import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from ...core.base_generator import BaseGenerator


class WebAnalyticsGenerator(BaseGenerator):
    """
    Generator for realistic web analytics data
    
    Creates web analytics datasets with user session patterns, page view
    distributions, device types, realistic bounce rates, and conversion funnels.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_device_distributions()
        self._setup_page_categories()
        self._setup_user_behavior_patterns()
        self._setup_conversion_funnels()
        self._setup_traffic_sources()
    
    def _setup_device_distributions(self):
        """Setup realistic device type distributions and characteristics"""
        self.device_data = {
            'desktop': {
                'probability': 0.45,
                'browsers': ['Chrome', 'Firefox', 'Safari', 'Edge', 'Opera'],
                'browser_weights': [0.65, 0.15, 0.10, 0.08, 0.02],
                'screen_resolutions': ['1920x1080', '1366x768', '1440x900', '1536x864', '1280x720'],
                'avg_session_duration': 180,  # seconds
                'bounce_rate': 0.35,
                'pages_per_session': 4.2
            },
            'mobile': {
                'probability': 0.48,
                'browsers': ['Chrome Mobile', 'Safari Mobile', 'Samsung Internet', 'Firefox Mobile', 'Opera Mobile'],
                'browser_weights': [0.60, 0.25, 0.08, 0.05, 0.02],
                'screen_resolutions': ['375x667', '414x896', '360x640', '375x812', '412x915'],
                'avg_session_duration': 95,  # seconds
                'bounce_rate': 0.55,
                'pages_per_session': 2.8
            },
            'tablet': {
                'probability': 0.07,
                'browsers': ['Safari', 'Chrome', 'Samsung Internet', 'Firefox', 'Edge'],
                'browser_weights': [0.50, 0.35, 0.08, 0.05, 0.02],
                'screen_resolutions': ['768x1024', '834x1112', '810x1080', '800x1280', '1024x768'],
                'avg_session_duration': 145,  # seconds
                'bounce_rate': 0.42,
                'pages_per_session': 3.5
            }
        }
    
    def _setup_page_categories(self):
        """Setup page categories with realistic traffic patterns"""
        self.page_categories = {
            'home': {
                'probability': 0.25,
                'paths': ['/', '/home', '/index'],
                'avg_time_on_page': 45,
                'exit_rate': 0.15,
                'conversion_rate': 0.02
            },
            'product': {
                'probability': 0.30,
                'paths': ['/product/', '/item/', '/p/'],
                'avg_time_on_page': 120,
                'exit_rate': 0.35,
                'conversion_rate': 0.04
            },
            'category': {
                'probability': 0.15,
                'paths': ['/category/', '/browse/', '/shop/'],
                'avg_time_on_page': 85,
                'exit_rate': 0.25,
                'conversion_rate': 0.015
            },
            'checkout': {
                'probability': 0.08,
                'paths': ['/checkout', '/cart', '/payment'],
                'avg_time_on_page': 180,
                'exit_rate': 0.45,
                'conversion_rate': 0.25
            },
            'blog': {
                'probability': 0.12,
                'paths': ['/blog/', '/article/', '/news/'],
                'avg_time_on_page': 200,
                'exit_rate': 0.60,
                'conversion_rate': 0.005
            },
            'support': {
                'probability': 0.06,
                'paths': ['/help', '/support', '/faq', '/contact'],
                'avg_time_on_page': 150,
                'exit_rate': 0.70,
                'conversion_rate': 0.003
            },
            'account': {
                'probability': 0.04,
                'paths': ['/account', '/profile', '/dashboard', '/settings'],
                'avg_time_on_page': 90,
                'exit_rate': 0.20,
                'conversion_rate': 0.05
            }
        }
    
    def _setup_user_behavior_patterns(self):
        """Setup user behavior patterns and segments"""
        self.user_segments = {
            'new_visitor': {
                'probability': 0.60,
                'session_duration_multiplier': 0.8,
                'bounce_rate_multiplier': 1.3,
                'pages_per_session_multiplier': 0.7,
                'conversion_rate_multiplier': 0.5
            },
            'returning_visitor': {
                'probability': 0.35,
                'session_duration_multiplier': 1.2,
                'bounce_rate_multiplier': 0.7,
                'pages_per_session_multiplier': 1.4,
                'conversion_rate_multiplier': 1.8
            },
            'loyal_customer': {
                'probability': 0.05,
                'session_duration_multiplier': 1.5,
                'bounce_rate_multiplier': 0.4,
                'pages_per_session_multiplier': 2.0,
                'conversion_rate_multiplier': 3.0
            }
        }
    
    def _setup_conversion_funnels(self):
        """Setup conversion funnel stages and drop-off rates"""
        self.conversion_funnel = {
            'awareness': {'stage': 1, 'drop_off_rate': 0.20},
            'interest': {'stage': 2, 'drop_off_rate': 0.35},
            'consideration': {'stage': 3, 'drop_off_rate': 0.45},
            'intent': {'stage': 4, 'drop_off_rate': 0.25},
            'purchase': {'stage': 5, 'drop_off_rate': 0.15}
        }
    
    def _setup_traffic_sources(self):
        """Setup traffic source distributions"""
        self.traffic_sources = {
            'organic_search': {
                'probability': 0.35,
                'bounce_rate_multiplier': 0.9,
                'conversion_rate_multiplier': 1.2
            },
            'direct': {
                'probability': 0.25,
                'bounce_rate_multiplier': 0.7,
                'conversion_rate_multiplier': 1.5
            },
            'social_media': {
                'probability': 0.15,
                'bounce_rate_multiplier': 1.4,
                'conversion_rate_multiplier': 0.6
            },
            'paid_search': {
                'probability': 0.12,
                'bounce_rate_multiplier': 1.1,
                'conversion_rate_multiplier': 1.8
            },
            'email': {
                'probability': 0.08,
                'bounce_rate_multiplier': 0.6,
                'conversion_rate_multiplier': 2.2
            },
            'referral': {
                'probability': 0.05,
                'bounce_rate_multiplier': 1.0,
                'conversion_rate_multiplier': 1.0
            }
        }
    
    def generate(self, rows: int, **kwargs) -> pd.DataFrame:
        """
        Generate web analytics dataset
        
        Args:
            rows: Number of analytics records to generate
            **kwargs: Additional parameters (time_series, date_range, etc.)
            
        Returns:
            pd.DataFrame: Generated web analytics data with realistic patterns
        """
        # Create time series configuration if requested
        ts_config = self._create_time_series_config(**kwargs)
        
        if ts_config:
            return self._generate_time_series_analytics(rows, ts_config, **kwargs)
        else:
            return self._generate_snapshot_analytics(rows, **kwargs)
    
    def _generate_snapshot_analytics(self, rows: int, **kwargs) -> pd.DataFrame:
        """Generate snapshot web analytics data (random timestamps)"""
        date_range = kwargs.get('date_range', None)
        
        data = []
        session_counter = 1
        
        # Generate page views to reach exact row count
        while len(data) < rows:
            session_data = self._generate_session(session_counter, date_range, False)
            
            # Calculate how many pages this session should have
            remaining_rows = rows - len(data)
            pages_in_session = min(session_data['pages_per_session'], remaining_rows)
            
            for page_idx in range(pages_in_session):
                if len(data) >= rows:
                    break
                
                page_view = self._generate_page_view(
                    session_data, page_idx, pages_in_session
                )
                data.append(page_view)
            
            session_counter += 1
        
        df = pd.DataFrame(data)
        return self._apply_realistic_patterns(df)
    
    def _generate_time_series_analytics(self, rows: int, ts_config, **kwargs) -> pd.DataFrame:
        """Generate time series web analytics data using integrated time series system"""
        # Generate timestamps from time series config
        timestamps = self._generate_time_series_timestamps(ts_config, rows)
        
        data = []
        session_counter = 1
        
        # Generate base time series for key metrics
        base_sessions = self.time_series_generator.generate_time_series_base(
            ts_config, base_value=100.0, value_range=(10.0, 1000.0)
        )
        
        # Create session distribution based on time series patterns
        for i, timestamp in enumerate(timestamps):
            if i >= rows:
                break
                
            # Use time series value to influence session characteristics
            ts_multiplier = base_sessions.iloc[i % len(base_sessions)]['value'] / 100.0
            
            # Generate session with time-aware characteristics
            session_data = self._generate_time_aware_session(
                session_counter, timestamp, ts_multiplier
            )
            
            # Generate page view for this timestamp
            page_view = self._generate_page_view(session_data, 0, 1)
            page_view['timestamp'] = timestamp
            
            data.append(page_view)
            session_counter += 1
        
        df = pd.DataFrame(data)
        
        # Apply time series correlations to key metrics
        df = self._apply_time_series_correlation(df, ts_config, 'time_on_page_seconds')
        df = self._apply_time_series_correlation(df, ts_config, 'scroll_depth_percent')
        
        # Add temporal relationships
        df = self._add_temporal_relationships(df, ts_config)
        
        return self._apply_realistic_patterns(df)
    
    def _generate_time_aware_session(self, session_id: int, timestamp: datetime, 
                                   ts_multiplier: float) -> Dict[str, Any]:
        """Generate session data with time-aware characteristics"""
        # Apply time-of-day effects
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Business hours effect (9 AM - 5 PM weekdays)
        if 9 <= hour <= 17 and day_of_week < 5:
            device_bias = {'desktop': 1.5, 'mobile': 0.7, 'tablet': 0.8}
            traffic_bias = {'organic_search': 1.2, 'direct': 1.3, 'social_media': 0.8}
        else:
            device_bias = {'desktop': 0.8, 'mobile': 1.3, 'tablet': 1.1}
            traffic_bias = {'organic_search': 0.9, 'direct': 1.1, 'social_media': 1.4}
        
        # Weekend effect
        if day_of_week >= 5:
            device_bias['mobile'] *= 1.2
            traffic_bias['social_media'] *= 1.3
        
        # Select device type with time bias
        device_probs = []
        for device in self.device_data.keys():
            base_prob = self.device_data[device]['probability']
            adjusted_prob = base_prob * device_bias.get(device, 1.0) * ts_multiplier
            device_probs.append(adjusted_prob)
        
        # Normalize probabilities
        total_prob = sum(device_probs)
        device_probs = [p / total_prob for p in device_probs]
        
        device_type = self._select_weighted_choice(
            list(self.device_data.keys()), device_probs
        )
        device_info = self.device_data[device_type]
        
        # Select traffic source with time bias
        traffic_probs = []
        for source in self.traffic_sources.keys():
            base_prob = self.traffic_sources[source]['probability']
            adjusted_prob = base_prob * traffic_bias.get(source, 1.0)
            traffic_probs.append(adjusted_prob)
        
        # Normalize probabilities
        total_prob = sum(traffic_probs)
        traffic_probs = [p / total_prob for p in traffic_probs]
        
        traffic_source = self._select_weighted_choice(
            list(self.traffic_sources.keys()), traffic_probs
        )
        source_info = self.traffic_sources[traffic_source]
        
        # Select user segment (less time-dependent)
        user_segment = self._select_weighted_choice(
            list(self.user_segments.keys()),
            [self.user_segments[s]['probability'] for s in self.user_segments.keys()]
        )
        segment_info = self.user_segments[user_segment]
        
        # Calculate session characteristics with time series influence
        base_duration = device_info['avg_session_duration']
        session_duration = max(10, int(base_duration * segment_info['session_duration_multiplier'] * 
                                     ts_multiplier * self.faker.random.gauss(1.0, 0.3)))
        
        # Generate browser and other details
        browser = self._select_weighted_choice(
            device_info['browsers'], device_info['browser_weights']
        )
        
        screen_resolution = self.faker.random_element(device_info['screen_resolutions'])
        
        return {
            'session_id': f'SES_{session_id:08d}',
            'user_id': f'USR_{self.faker.random_int(1, 100000):06d}',
            'device_type': device_type,
            'browser': browser,
            'screen_resolution': screen_resolution,
            'user_segment': user_segment,
            'traffic_source': traffic_source,
            'session_start': timestamp,
            'session_duration': session_duration,
            'pages_per_session': 1,  # Single page for time series
            'device_info': device_info,
            'segment_info': segment_info,
            'source_info': source_info
        }
    
    def _generate_session(self, session_id: int, date_range: Tuple = None, 
                         time_series: bool = False) -> Dict[str, Any]:
        """Generate session-level data"""
        # Select device type
        device_type = self._select_weighted_choice(
            list(self.device_data.keys()),
            [self.device_data[d]['probability'] for d in self.device_data.keys()]
        )
        device_info = self.device_data[device_type]
        
        # Select user segment
        user_segment = self._select_weighted_choice(
            list(self.user_segments.keys()),
            [self.user_segments[s]['probability'] for s in self.user_segments.keys()]
        )
        segment_info = self.user_segments[user_segment]
        
        # Select traffic source
        traffic_source = self._select_weighted_choice(
            list(self.traffic_sources.keys()),
            [self.traffic_sources[s]['probability'] for s in self.traffic_sources.keys()]
        )
        source_info = self.traffic_sources[traffic_source]
        
        # Generate session timestamp
        if time_series and date_range:
            start_date, end_date = date_range
            session_start = self.faker.date_time_between(start_date=start_date, end_date=end_date)
        else:
            session_start = self.faker.date_time_this_year()
        
        # Calculate session characteristics
        base_duration = device_info['avg_session_duration']
        session_duration = max(10, int(base_duration * segment_info['session_duration_multiplier'] * 
                                     self.faker.random.gauss(1.0, 0.3)))
        
        base_pages = device_info['pages_per_session']
        pages_multiplier = segment_info['pages_per_session_multiplier']
        
        # Apply bounce rate probability to determine single vs multi-page sessions
        base_bounce_rate = device_info['bounce_rate']
        adjusted_bounce_rate = (base_bounce_rate * 
                              segment_info['bounce_rate_multiplier'] *
                              source_info['bounce_rate_multiplier'])
        
        # If this should be a bounce session, set to 1 page
        if self.faker.random.random() < adjusted_bounce_rate:
            pages_per_session = 1
        else:
            # Multi-page session
            pages_per_session = max(2, int(base_pages * pages_multiplier * 
                                         self.faker.random.gauss(1.0, 0.4)))
        
        # Generate browser and other details
        browser = self._select_weighted_choice(
            device_info['browsers'], device_info['browser_weights']
        )
        
        screen_resolution = self.faker.random_element(device_info['screen_resolutions'])
        
        return {
            'session_id': f'SES_{session_id:08d}',
            'user_id': f'USR_{self.faker.random_int(1, 100000):06d}',
            'device_type': device_type,
            'browser': browser,
            'screen_resolution': screen_resolution,
            'user_segment': user_segment,
            'traffic_source': traffic_source,
            'session_start': session_start,
            'session_duration': session_duration,
            'pages_per_session': pages_per_session,
            'device_info': device_info,
            'segment_info': segment_info,
            'source_info': source_info
        }
    
    def _generate_page_view(self, session_data: Dict, page_index: int, 
                           total_pages: int) -> Dict[str, Any]:
        """Generate individual page view within a session"""
        # Select page category based on user journey
        if page_index == 0:
            # First page - more likely to be home or landing page
            page_category = self._select_first_page_category(session_data['traffic_source'])
        else:
            # Subsequent pages - follow user journey patterns
            page_category = self._select_subsequent_page_category(page_index, total_pages)
        
        category_info = self.page_categories[page_category]
        
        # Generate page URL
        base_path = self.faker.random_element(category_info['paths'])
        if base_path.endswith('/'):
            page_url = f"{base_path}{self.faker.slug()}"
        else:
            page_url = base_path
        
        # Calculate time on page
        base_time = category_info['avg_time_on_page']
        time_on_page = max(5, int(base_time * self.faker.random.gauss(1.0, 0.5)))
        
        # Calculate timestamp within session
        time_offset = sum([30 + self.faker.random_int(0, 60) for _ in range(page_index)])
        page_timestamp = session_data['session_start'] + timedelta(seconds=time_offset)
        
        # Determine if this is an exit page
        is_exit = (page_index == total_pages - 1) or (
            self.faker.random.random() < category_info['exit_rate']
        )
        
        # Determine if conversion occurred
        base_conversion_rate = category_info['conversion_rate']
        adjusted_conversion_rate = (base_conversion_rate * 
                                  session_data['segment_info']['conversion_rate_multiplier'] *
                                  session_data['source_info']['conversion_rate_multiplier'])
        
        converted = self.faker.random.random() < adjusted_conversion_rate
        
        # Calculate bounce (single page session) - apply device and segment multipliers
        base_bounce_rate = session_data['device_info']['bounce_rate']
        adjusted_bounce_rate = (base_bounce_rate * 
                              session_data['segment_info']['bounce_rate_multiplier'] *
                              session_data['source_info']['bounce_rate_multiplier'])
        
        # For single page sessions, use adjusted bounce rate probability
        # For multi-page sessions, bounce is False
        if total_pages == 1:
            is_bounce = self.faker.random.random() < adjusted_bounce_rate
        else:
            is_bounce = False
        
        return {
            'session_id': session_data['session_id'],
            'user_id': session_data['user_id'],
            'page_view_id': f'PV_{hash(f"{session_data["session_id"]}_{page_index}") % 100000000:08d}',
            'timestamp': page_timestamp,
            'page_url': page_url,
            'page_category': page_category,
            'page_title': self._generate_page_title(page_category),
            'device_type': session_data['device_type'],
            'browser': session_data['browser'],
            'screen_resolution': session_data['screen_resolution'],
            'user_segment': session_data['user_segment'],
            'traffic_source': session_data['traffic_source'],
            'time_on_page_seconds': time_on_page,
            'is_bounce': is_bounce,
            'is_exit': is_exit,
            'converted': converted,
            'page_load_time_ms': self.faker.random_int(200, 3000),
            'scroll_depth_percent': min(100, max(10, int(self.faker.random.gauss(65, 25)))),
            'clicks_on_page': self.faker.random_int(0, 8),
            'referrer_url': self._generate_referrer_url(session_data['traffic_source'], page_index),
            'user_agent': self._generate_user_agent(session_data['device_type'], session_data['browser']),
            'ip_address': self.faker.ipv4(),
            'country': self.faker.country_code(),
            'city': self.faker.city()
        }
    
    def _select_weighted_choice(self, choices: List, weights: List) -> Any:
        """Select item from choices based on weights using seeded random"""
        total_weight = sum(weights)
        rand_val = self.faker.random.uniform(0, total_weight)
        
        cumulative_weight = 0
        for choice, weight in zip(choices, weights):
            cumulative_weight += weight
            if rand_val <= cumulative_weight:
                return choice
        
        return choices[-1]  # Fallback
    
    def _select_first_page_category(self, traffic_source: str) -> str:
        """Select first page category based on traffic source"""
        source_landing_preferences = {
            'organic_search': ['product', 'blog', 'category', 'home'],
            'direct': ['home', 'account', 'product', 'category'],
            'social_media': ['blog', 'product', 'home', 'category'],
            'paid_search': ['product', 'category', 'home', 'checkout'],
            'email': ['product', 'home', 'blog', 'account'],
            'referral': ['home', 'product', 'blog', 'category']
        }
        
        preferred_categories = source_landing_preferences.get(
            traffic_source, ['home', 'product', 'category']
        )
        
        # Weight first choice higher
        weights = [0.5, 0.3, 0.15, 0.05][:len(preferred_categories)]
        return self._select_weighted_choice(preferred_categories, weights)
    
    def _select_subsequent_page_category(self, page_index: int, total_pages: int) -> str:
        """Select subsequent page category based on user journey"""
        # User journey patterns
        if page_index == total_pages - 1:
            # Last page - more likely to be checkout or exit pages
            return self._select_weighted_choice(
                ['checkout', 'product', 'support', 'account'],
                [0.4, 0.3, 0.2, 0.1]
            )
        else:
            # Middle pages - follow typical browsing patterns
            return self._select_weighted_choice(
                list(self.page_categories.keys()),
                [self.page_categories[cat]['probability'] for cat in self.page_categories.keys()]
            )
    
    def _generate_page_title(self, category: str) -> str:
        """Generate realistic page title based on category"""
        title_templates = {
            'home': ['Welcome to {}', '{} - Home', 'Shop {} Online'],
            'product': ['{} - Product Details', 'Buy {} Online', '{} Reviews & Specs'],
            'category': ['{} Category', 'Browse {} Products', 'Shop {} Collection'],
            'checkout': ['Checkout', 'Complete Your Order', 'Payment & Shipping'],
            'blog': ['{} Blog Post', 'Latest News - {}', '{} Article'],
            'support': ['Help & Support', 'FAQ - {}', 'Contact Support'],
            'account': ['My Account', 'User Dashboard', 'Account Settings']
        }
        
        templates = title_templates.get(category, ['{} Page'])
        template = self.faker.random_element(templates)
        
        if '{}' in template:
            return template.format(self.faker.company())
        return template
    
    def _generate_referrer_url(self, traffic_source: str, page_index: int) -> str:
        """Generate referrer URL based on traffic source"""
        if page_index == 0:
            # First page referrers
            referrer_patterns = {
                'organic_search': ['https://www.google.com/search', 'https://www.bing.com/search'],
                'social_media': ['https://www.facebook.com/', 'https://twitter.com/', 'https://www.instagram.com/'],
                'paid_search': ['https://www.google.com/ads', 'https://www.bing.com/ads'],
                'email': ['https://mail.google.com/', 'https://outlook.com/'],
                'referral': [self.faker.url() for _ in range(3)],
                'direct': ['']
            }
            
            patterns = referrer_patterns.get(traffic_source, [''])
            return self.faker.random_element(patterns)
        else:
            # Internal referrers for subsequent pages
            return f"https://example.com{self.faker.uri_path()}"
    
    def _generate_user_agent(self, device_type: str, browser: str) -> str:
        """Generate realistic user agent string"""
        user_agents = {
            'desktop': {
                'Chrome': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Firefox': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
                'Safari': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
                'Edge': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59'
            },
            'mobile': {
                'Chrome Mobile': 'Mozilla/5.0 (Linux; Android 10; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36',
                'Safari Mobile': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1'
            },
            'tablet': {
                'Safari': 'Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1',
                'Chrome': 'Mozilla/5.0 (Linux; Android 10; SM-T870) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Safari/537.36'
            }
        }
        
        device_agents = user_agents.get(device_type, user_agents['desktop'])
        return device_agents.get(browser, list(device_agents.values())[0])
    
    def _apply_realistic_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply additional realistic patterns to the generated data"""
        # Calculate session-level metrics
        session_metrics = data.groupby('session_id').agg({
            'time_on_page_seconds': 'sum',
            'page_view_id': 'count',
            'converted': 'any',
            'is_bounce': 'first'
        }).rename(columns={
            'time_on_page_seconds': 'total_session_duration',
            'page_view_id': 'total_page_views'
        })
        
        # Merge session metrics back to main data
        data = data.merge(session_metrics, on='session_id', suffixes=('', '_session'))
        
        # Add derived metrics
        data['bounce_rate'] = data['is_bounce'].astype(float)
        data['conversion_rate'] = data['converted'].astype(float)
        
        # Add time-based patterns (hour of day effects)
        data['hour_of_day'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        
        # Add engagement score based on multiple factors
        data['engagement_score'] = (
            (data['time_on_page_seconds'] / 60) * 0.3 +  # Time weight
            (data['scroll_depth_percent'] / 100) * 0.2 +  # Scroll weight
            (data['clicks_on_page'] / 5) * 0.2 +  # Clicks weight
            (1 - data['bounce_rate']) * 0.3  # Non-bounce weight
        ).round(2)
        
        # Sort by timestamp for realistic chronological order
        data = data.sort_values(['timestamp', 'session_id']).reset_index(drop=True)
        
        return data