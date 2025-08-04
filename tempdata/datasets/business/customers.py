"""
Customer database generator

Generates realistic customer data with demographic distributions.
"""

import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, List, Any
from ...core.base_generator import BaseGenerator


class CustomerGenerator(BaseGenerator):
    """
    Generator for realistic customer database
    
    Creates customer datasets with demographic distributions, registration patterns,
    customer segmentation, and lifecycle patterns.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_demographic_distributions()
        self._setup_registration_patterns()
        self._setup_customer_segments()
        self._setup_lifecycle_patterns()
    
    def _setup_demographic_distributions(self):
        """Setup realistic demographic distributions"""
        # Age distribution based on typical customer demographics
        self.age_distribution = {
            'gen_z': {'range': (18, 26), 'weight': 0.20},      # Gen Z
            'millennial': {'range': (27, 42), 'weight': 0.35}, # Millennials
            'gen_x': {'range': (43, 58), 'weight': 0.25},      # Gen X
            'boomer': {'range': (59, 77), 'weight': 0.20}      # Boomers
        }
        
        # Gender distribution
        self.gender_distribution = {
            'male': 0.48,
            'female': 0.50,
            'other': 0.02
        }
        
        # Income brackets by age group
        self.income_by_age = {
            'gen_z': {'low': 0.4, 'medium': 0.5, 'high': 0.1},
            'millennial': {'low': 0.25, 'medium': 0.55, 'high': 0.2},
            'gen_x': {'low': 0.2, 'medium': 0.5, 'high': 0.3},
            'boomer': {'low': 0.3, 'medium': 0.4, 'high': 0.3}
        }
        
        # Income ranges
        self.income_ranges = {
            'low': (25000, 45000),
            'medium': (45000, 85000),
            'high': (85000, 200000)
        }
    
    def _setup_registration_patterns(self):
        """Setup registration patterns over time"""
        # Monthly registration multipliers (seasonal patterns)
        self.monthly_registration_patterns = {
            1: 0.9,   # January - New Year resolutions
            2: 0.8,   # February - low activity
            3: 1.1,   # March - spring activity
            4: 1.0,   # April - normal
            5: 1.2,   # May - spring peak
            6: 1.1,   # June - summer start
            7: 0.9,   # July - vacation time
            8: 0.8,   # August - vacation continues
            9: 1.3,   # September - back to business
            10: 1.2,  # October - fall activity
            11: 1.4,  # November - holiday shopping prep
            12: 1.1   # December - holiday season
        }
        
        # Registration channels and their distributions
        self.registration_channels = {
            'organic_search': 0.35,
            'social_media': 0.25,
            'referral': 0.15,
            'email_marketing': 0.10,
            'paid_ads': 0.10,
            'direct': 0.05
        }
    
    def _setup_customer_segments(self):
        """Setup customer segmentation logic"""
        self.segment_criteria = {
            'VIP': {
                'min_lifetime_value': 5000,
                'min_orders': 20,
                'probability': 0.05
            },
            'Premium': {
                'min_lifetime_value': 1500,
                'min_orders': 8,
                'probability': 0.15
            },
            'Standard': {
                'min_lifetime_value': 300,
                'min_orders': 3,
                'probability': 0.60
            },
            'Basic': {
                'min_lifetime_value': 0,
                'min_orders': 1,
                'probability': 0.20
            }
        }
    
    def _setup_lifecycle_patterns(self):
        """Setup customer lifecycle patterns"""
        self.lifecycle_stages = {
            'new': {'days_since_registration': (0, 30), 'weight': 0.15},
            'active': {'days_since_registration': (31, 365), 'weight': 0.40},
            'returning': {'days_since_registration': (366, 1095), 'weight': 0.25},
            'dormant': {'days_since_registration': (1096, 1825), 'weight': 0.15},
            'churned': {'days_since_registration': (1826, 3650), 'weight': 0.05}
        }
        
        # Activity patterns by lifecycle stage
        self.activity_by_stage = {
            'new': {'avg_orders': 1.5, 'avg_value': 150},
            'active': {'avg_orders': 8.0, 'avg_value': 800},
            'returning': {'avg_orders': 15.0, 'avg_value': 1200},
            'dormant': {'avg_orders': 3.0, 'avg_value': 200},
            'churned': {'avg_orders': 1.0, 'avg_value': 100}
        }
    
    def generate(self, rows: int, **kwargs) -> pd.DataFrame:
        """
        Generate customer database
        
        Args:
            rows: Number of customers to generate
            **kwargs: Additional parameters (country, date_range, time_series, etc.)
            
        Returns:
            pd.DataFrame: Generated customer data with realistic patterns
        """
        # Check if time series generation is requested
        ts_config = self._create_time_series_config(**kwargs)
        
        if ts_config:
            return self._generate_time_series_customers(rows, ts_config, **kwargs)
        else:
            return self._generate_snapshot_customers(rows, **kwargs)
    
    def _generate_snapshot_customers(self, rows: int, **kwargs) -> pd.DataFrame:
        """Generate snapshot customer data (random registration dates)"""
        country = kwargs.get('country', 'global')
        date_range = kwargs.get('date_range', None)
        
        data = []
        
        for i in range(rows):
            # Generate demographic information
            age_group = self._select_age_group()
            age = self._generate_age_for_group(age_group)
            gender = self._select_gender()
            income_bracket = self._select_income_bracket(age_group)
            annual_income = self._generate_income(income_bracket)
            
            # Generate lifecycle stage first, then registration date to match
            lifecycle_stage = self._select_lifecycle_stage()
            
            # Generate registration date based on desired lifecycle stage
            if date_range:
                start_date, end_date = date_range
                registration_date = self._generate_registration_date_for_stage(lifecycle_stage, start_date, end_date)
            else:
                registration_date = self._generate_registration_date_for_stage(lifecycle_stage)
            
            registration_channel = self._select_registration_channel()
            
            # Generate activity metrics based on lifecycle
            activity_data = self._generate_activity_metrics(lifecycle_stage, annual_income)
            
            # Determine customer segment
            segment = self._determine_segment(activity_data['lifetime_value'], activity_data['total_orders'])
            
            # Generate personal information
            if gender == 'male':
                first_name = self.faker.first_name_male()
            elif gender == 'female':
                first_name = self.faker.first_name_female()
            else:
                first_name = self.faker.first_name()
            
            last_name = self.faker.last_name()
            email = self._generate_email(first_name, last_name)
            phone = self.faker.phone_number()
            
            # Generate address information
            address = self._generate_address(country)
            
            # Generate preferences
            preferences = self._generate_preferences(age_group, gender)
            
            customer = {
                'customer_id': f'CUST_{i+1:06d}',
                'first_name': first_name,
                'last_name': last_name,
                'email': email,
                'phone': phone,
                'gender': gender,
                'age': age,
                'age_group': age_group,
                'annual_income': annual_income,
                'income_bracket': income_bracket,
                'registration_date': registration_date,
                'registration_channel': registration_channel,
                'lifecycle_stage': lifecycle_stage,
                'segment': segment,
                'total_orders': activity_data['total_orders'],
                'lifetime_value': activity_data['lifetime_value'],
                'avg_order_value': activity_data['avg_order_value'],
                'last_order_date': activity_data['last_order_date'],
                'days_since_last_order': activity_data['days_since_last_order'],
                'address_line1': address['street'],
                'city': address['city'],
                'state_province': address['state'],
                'postal_code': address['postal_code'],
                'country': address['country'],
                'preferred_contact': preferences['contact_method'],
                'marketing_opt_in': preferences['marketing_opt_in'],
                'preferred_category': preferences['product_category']
            }
            
            data.append(customer)
        
        df = pd.DataFrame(data)
        return self._apply_realistic_patterns(df)
    
    def _generate_time_series_customers(self, rows: int, ts_config, **kwargs) -> pd.DataFrame:
        """Generate time series customer data using integrated time series system"""
        country = kwargs.get('country', 'global')
        
        # Generate timestamps from time series config
        timestamps = self._generate_time_series_timestamps(ts_config, rows)
        
        # Generate base registration volume time series
        base_registration_volume = 10  # Base daily registrations
        
        # Create base time series for registration volume
        volume_series = self.time_series_generator.generate_time_series_base(
            ts_config,
            base_value=base_registration_volume,
            value_range=(base_registration_volume * 0.1, base_registration_volume * 5.0)
        )
        
        data = []
        
        # Track registration patterns for realistic correlations
        registration_history = {}
        
        for i, timestamp in enumerate(timestamps):
            if i >= rows or i >= len(volume_series):
                break
            
            # Get time series volume value (represents registration intensity)
            registration_intensity = volume_series.iloc[i]['value'] / base_registration_volume
            
            # Apply time-based registration patterns
            registration_date = timestamp.date()
            
            # Generate demographic information with time-based trends
            age_group = self._select_time_series_age_group(timestamp, registration_intensity)
            age = self._generate_age_for_group(age_group)
            gender = self._select_gender()
            income_bracket = self._select_income_bracket(age_group)
            annual_income = self._generate_income(income_bracket)
            
            # Select registration channel with time-based preferences
            registration_channel = self._select_time_series_registration_channel(timestamp, age_group)
            
            # Generate lifecycle stage based on registration date
            days_since_registration = (datetime.now().date() - registration_date).days
            lifecycle_stage = self._determine_lifecycle_stage(days_since_registration)
            
            # Generate activity metrics based on lifecycle and time patterns
            activity_data = self._generate_time_series_activity_metrics(
                lifecycle_stage, annual_income, timestamp, registration_intensity
            )
            
            # Determine customer segment
            segment = self._determine_segment(activity_data['lifetime_value'], activity_data['total_orders'])
            
            # Generate personal information
            if gender == 'male':
                first_name = self.faker.first_name_male()
            elif gender == 'female':
                first_name = self.faker.first_name_female()
            else:
                first_name = self.faker.first_name()
            
            last_name = self.faker.last_name()
            email = self._generate_email(first_name, last_name)
            phone = self.faker.phone_number()
            
            # Generate address information
            address = self._generate_address(country)
            
            # Generate preferences with time-based trends
            preferences = self._generate_time_series_preferences(age_group, gender, timestamp)
            
            customer = {
                'customer_id': f'CUST_{i+1:06d}',
                'first_name': first_name,
                'last_name': last_name,
                'email': email,
                'phone': phone,
                'gender': gender,
                'age': age,
                'age_group': age_group,
                'annual_income': annual_income,
                'income_bracket': income_bracket,
                'registration_date': registration_date,
                'registration_datetime': timestamp,
                'registration_channel': registration_channel,
                'lifecycle_stage': lifecycle_stage,
                'segment': segment,
                'total_orders': activity_data['total_orders'],
                'lifetime_value': activity_data['lifetime_value'],
                'avg_order_value': activity_data['avg_order_value'],
                'last_order_date': activity_data['last_order_date'],
                'days_since_last_order': activity_data['days_since_last_order'],
                'address_line1': address['street'],
                'city': address['city'],
                'state_province': address['state'],
                'postal_code': address['postal_code'],
                'country': address['country'],
                'preferred_contact': preferences['contact_method'],
                'marketing_opt_in': preferences['marketing_opt_in'],
                'preferred_category': preferences['product_category'],
                'registration_intensity': round(registration_intensity, 3)
            }
            
            # Track registration patterns for correlation
            month_key = f"{timestamp.year}-{timestamp.month:02d}"
            if month_key not in registration_history:
                registration_history[month_key] = []
            registration_history[month_key].append({
                'timestamp': timestamp,
                'age_group': age_group,
                'channel': registration_channel,
                'intensity': registration_intensity
            })
            
            data.append(customer)
        
        df = pd.DataFrame(data)
        
        # Add temporal relationships using base generator functionality
        df = self._add_temporal_relationships(df, ts_config)
        
        # Apply customer-specific time series correlations
        df = self._apply_customer_time_series_correlations(df, ts_config)
        
        return self._apply_realistic_patterns(df)
    
    def _select_time_series_age_group(self, timestamp: datetime, registration_intensity: float) -> str:
        """Select age group with time-based trends"""
        # Base age group selection
        base_age_group = self._select_age_group()
        
        # Apply time-based trends
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        month = timestamp.month
        
        # Time-based age group preferences
        if 9 <= hour <= 17:  # Business hours - more Gen X and Boomers
            if base_age_group in ['gen_z', 'millennial'] and self.faker.random.random() < 0.3:
                base_age_group = self.faker.random_element(['gen_x', 'boomer'])
        elif 18 <= hour <= 23:  # Evening - more Gen Z and Millennials
            if base_age_group in ['gen_x', 'boomer'] and self.faker.random.random() < 0.4:
                base_age_group = self.faker.random_element(['gen_z', 'millennial'])
        
        # Weekend patterns - more leisure-oriented demographics
        if day_of_week >= 5:  # Weekend
            if self.faker.random.random() < 0.2:
                base_age_group = self.faker.random_element(['millennial', 'gen_x'])
        
        # Seasonal patterns
        if month in [11, 12]:  # Holiday season - all age groups more active
            pass  # Keep base selection
        elif month in [6, 7, 8]:  # Summer - younger demographics more active
            if base_age_group in ['gen_x', 'boomer'] and self.faker.random.random() < 0.3:
                base_age_group = self.faker.random_element(['gen_z', 'millennial'])
        
        return base_age_group
    
    def _select_time_series_registration_channel(self, timestamp: datetime, age_group: str) -> str:
        """Select registration channel with time-based patterns"""
        # Base channel selection
        base_channel = self._select_registration_channel()
        
        # Apply time-based channel preferences
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Time-based channel adjustments
        if 9 <= hour <= 17:  # Business hours
            # More organic search and direct traffic
            if base_channel == 'social_media' and self.faker.random.random() < 0.3:
                base_channel = 'organic_search'
        elif 18 <= hour <= 23:  # Evening
            # More social media and email marketing
            if base_channel == 'organic_search' and self.faker.random.random() < 0.4:
                base_channel = 'social_media'
        
        # Weekend patterns
        if day_of_week >= 5:  # Weekend
            # More social media and referrals
            if base_channel in ['organic_search', 'paid_ads'] and self.faker.random.random() < 0.3:
                base_channel = self.faker.random_element(['social_media', 'referral'])
        
        # Age group preferences
        if age_group in ['gen_z', 'millennial']:
            if base_channel == 'direct' and self.faker.random.random() < 0.5:
                base_channel = 'social_media'
        elif age_group in ['gen_x', 'boomer']:
            if base_channel == 'social_media' and self.faker.random.random() < 0.4:
                base_channel = self.faker.random_element(['organic_search', 'email_marketing'])
        
        return base_channel
    
    def _generate_time_series_activity_metrics(self, lifecycle_stage: str, annual_income: int,
                                             timestamp: datetime, registration_intensity: float) -> Dict[str, Any]:
        """Generate activity metrics with time series patterns"""
        # Start with base activity metrics
        base_activity = self._generate_activity_metrics(lifecycle_stage, annual_income)
        
        # Apply time-based adjustments
        month = timestamp.month
        
        # Seasonal activity adjustments
        seasonal_multiplier = self.monthly_registration_patterns.get(month, 1.0)
        
        # Apply registration intensity (higher intensity periods = more engaged customers)
        intensity_multiplier = 0.8 + (registration_intensity * 0.4)  # Scale to 0.8-1.2 range
        
        # Adjust metrics
        adjusted_orders = max(1, int(base_activity['total_orders'] * seasonal_multiplier * intensity_multiplier))
        adjusted_value = max(50, int(base_activity['lifetime_value'] * seasonal_multiplier * intensity_multiplier))
        
        return {
            'total_orders': adjusted_orders,
            'lifetime_value': adjusted_value,
            'avg_order_value': round(adjusted_value / adjusted_orders, 2),
            'last_order_date': base_activity['last_order_date'],
            'days_since_last_order': base_activity['days_since_last_order']
        }
    
    def _generate_time_series_preferences(self, age_group: str, gender: str, timestamp: datetime) -> Dict[str, Any]:
        """Generate preferences with time-based trends"""
        # Start with base preferences
        base_preferences = self._generate_preferences(age_group, gender)
        
        # Apply time-based trends
        year = timestamp.year
        month = timestamp.month
        
        # Technology adoption trends over time
        if year >= 2020:  # Recent years - higher app notification preference
            if base_preferences['contact_method'] == 'mail' and self.faker.random.random() < 0.6:
                base_preferences['contact_method'] = 'app_notification'
        
        # Seasonal marketing opt-in patterns
        if month in [11, 12]:  # Holiday season - higher opt-in rates
            if not base_preferences['marketing_opt_in'] and self.faker.random.random() < 0.3:
                base_preferences['marketing_opt_in'] = True
        
        return base_preferences
    
    def _apply_customer_time_series_correlations(self, data: pd.DataFrame, ts_config) -> pd.DataFrame:
        """Apply customer-specific time series correlations"""
        if len(data) < 2:
            return data
        
        # Sort by registration datetime to ensure proper time series order
        data = data.sort_values('registration_datetime').reset_index(drop=True)
        
        # Apply registration intensity persistence (viral/word-of-mouth effects)
        if 'registration_intensity' in data.columns:
            for i in range(1, len(data)):
                prev_intensity = data.loc[i-1, 'registration_intensity']
                curr_intensity = data.loc[i, 'registration_intensity']
                
                # Apply intensity correlation (viral registration effects)
                intensity_persistence = 0.3
                correlated_intensity = (prev_intensity * intensity_persistence + 
                                      curr_intensity * (1 - intensity_persistence))
                
                data.loc[i, 'registration_intensity'] = round(correlated_intensity, 3)
        
        # Apply channel correlation (similar channels cluster in time)
        for i in range(1, len(data)):
            prev_channel = data.loc[i-1, 'registration_channel']
            curr_channel = data.loc[i, 'registration_channel']
            
            # Check if registrations are close in time (within 1 day)
            time_diff = (data.loc[i, 'registration_datetime'] - 
                        data.loc[i-1, 'registration_datetime']).total_seconds()
            
            if time_diff <= 86400:  # Within 1 day
                # Apply channel correlation (campaigns tend to cluster)
                if prev_channel in ['paid_ads', 'email_marketing'] and self.faker.random.random() < 0.4:
                    data.loc[i, 'registration_channel'] = prev_channel
        
        # Apply demographic clustering (similar demographics register together)
        for i in range(1, len(data)):
            prev_age_group = data.loc[i-1, 'age_group']
            curr_age_group = data.loc[i, 'age_group']
            
            # Check if registrations are close in time (within 2 hours)
            time_diff = (data.loc[i, 'registration_datetime'] - 
                        data.loc[i-1, 'registration_datetime']).total_seconds()
            
            if time_diff <= 7200:  # Within 2 hours
                # Apply demographic correlation (social influence)
                if self.faker.random.random() < 0.25:
                    data.loc[i, 'age_group'] = prev_age_group
                    # Regenerate age within the new group
                    data.loc[i, 'age'] = self._generate_age_for_group(prev_age_group)
        
        return data
    
    def _select_age_group(self) -> str:
        """Select age group based on distribution"""
        choices = list(self.age_distribution.keys())
        weights = [self.age_distribution[group]['weight'] for group in choices]
        
        # Use weighted selection with faker's random
        cumulative_weights = []
        total = 0
        for weight in weights:
            total += weight
            cumulative_weights.append(total)
        
        rand_val = self.faker.random.uniform(0, total)
        for i, cum_weight in enumerate(cumulative_weights):
            if rand_val <= cum_weight:
                return choices[i]
        
        return choices[-1]
    
    def _generate_age_for_group(self, age_group: str) -> int:
        """Generate age within the specified group range"""
        min_age, max_age = self.age_distribution[age_group]['range']
        return self.faker.random_int(min_age, max_age)
    
    def _select_gender(self) -> str:
        """Select gender based on distribution"""
        choices = list(self.gender_distribution.keys())
        weights = list(self.gender_distribution.values())
        
        cumulative_weights = []
        total = 0
        for weight in weights:
            total += weight
            cumulative_weights.append(total)
        
        rand_val = self.faker.random.uniform(0, total)
        for i, cum_weight in enumerate(cumulative_weights):
            if rand_val <= cum_weight:
                return choices[i]
        
        return choices[-1]
    
    def _select_income_bracket(self, age_group: str) -> str:
        """Select income bracket based on age group"""
        distribution = self.income_by_age[age_group]
        choices = list(distribution.keys())
        weights = list(distribution.values())
        
        cumulative_weights = []
        total = 0
        for weight in weights:
            total += weight
            cumulative_weights.append(total)
        
        rand_val = self.faker.random.uniform(0, total)
        for i, cum_weight in enumerate(cumulative_weights):
            if rand_val <= cum_weight:
                return choices[i]
        
        return choices[-1]
    
    def _generate_income(self, income_bracket: str) -> int:
        """Generate income within the specified bracket"""
        min_income, max_income = self.income_ranges[income_bracket]
        return self.faker.random_int(min_income, max_income)
    
    def _generate_registration_date(self, start_date: date, end_date: date) -> date:
        """Generate registration date with seasonal patterns"""
        reg_date = self.faker.date_between(start_date=start_date, end_date=end_date)
        
        # Apply seasonal adjustment (small probability to adjust based on month)
        month_multiplier = self.monthly_registration_patterns.get(reg_date.month, 1.0)
        if self.faker.random.random() < (month_multiplier - 1.0) * 0.1:
            # Slightly adjust date within the month for seasonal effect
            days_in_month = 28 if reg_date.month == 2 else 30
            adjustment = self.faker.random_int(-5, 5)
            try:
                reg_date = reg_date.replace(day=min(max(1, reg_date.day + adjustment), days_in_month))
            except ValueError:
                pass  # Keep original date if adjustment fails
        
        return reg_date
    
    def _select_registration_channel(self) -> str:
        """Select registration channel based on distribution"""
        choices = list(self.registration_channels.keys())
        weights = list(self.registration_channels.values())
        
        cumulative_weights = []
        total = 0
        for weight in weights:
            total += weight
            cumulative_weights.append(total)
        
        rand_val = self.faker.random.uniform(0, total)
        for i, cum_weight in enumerate(cumulative_weights):
            if rand_val <= cum_weight:
                return choices[i]
        
        return choices[-1]
    
    def _select_lifecycle_stage(self) -> str:
        """Select lifecycle stage based on distribution"""
        choices = list(self.lifecycle_stages.keys())
        weights = [self.lifecycle_stages[stage]['weight'] for stage in choices]
        
        cumulative_weights = []
        total = 0
        for weight in weights:
            total += weight
            cumulative_weights.append(total)
        
        rand_val = self.faker.random.uniform(0, total)
        for i, cum_weight in enumerate(cumulative_weights):
            if rand_val <= cum_weight:
                return choices[i]
        
        return choices[-1]
    
    def _generate_registration_date_for_stage(self, lifecycle_stage: str, start_date: date = None, end_date: date = None) -> date:
        """Generate registration date that matches the desired lifecycle stage"""
        if not end_date:
            end_date = datetime.now().date()
        
        # Calculate the date range that would result in the desired lifecycle stage
        stage_criteria = self.lifecycle_stages[lifecycle_stage]['days_since_registration']
        min_days_ago, max_days_ago = stage_criteria
        
        # Calculate the registration date range
        latest_reg_date = end_date - timedelta(days=min_days_ago)
        earliest_reg_date = end_date - timedelta(days=max_days_ago)
        
        # If start_date is provided, respect it
        if start_date:
            earliest_reg_date = max(earliest_reg_date, start_date)
            latest_reg_date = min(latest_reg_date, end_date)
        
        # Ensure we have a valid date range
        if earliest_reg_date > latest_reg_date:
            # Fallback to a reasonable date
            return self.faker.date_between(start_date=start_date or earliest_reg_date, end_date=end_date)
        
        # Generate registration date within the calculated range
        reg_date = self.faker.date_between(start_date=earliest_reg_date, end_date=latest_reg_date)
        
        # Apply seasonal adjustment
        month_multiplier = self.monthly_registration_patterns.get(reg_date.month, 1.0)
        if self.faker.random.random() < (month_multiplier - 1.0) * 0.1:
            # Slightly adjust date within the month for seasonal effect
            days_in_month = 28 if reg_date.month == 2 else 30
            adjustment = self.faker.random_int(-5, 5)
            try:
                reg_date = reg_date.replace(day=min(max(1, reg_date.day + adjustment), days_in_month))
            except ValueError:
                pass  # Keep original date if adjustment fails
        
        return reg_date
    
    def _determine_lifecycle_stage(self, days_since_registration: int) -> str:
        """Determine lifecycle stage based on days since registration"""
        for stage, criteria in self.lifecycle_stages.items():
            min_days, max_days = criteria['days_since_registration']
            if min_days <= days_since_registration <= max_days:
                return stage
        
        return 'churned'  # Default for very old customers
    
    def _generate_activity_metrics(self, lifecycle_stage: str, annual_income: int) -> Dict[str, Any]:
        """Generate activity metrics based on lifecycle stage and income"""
        base_activity = self.activity_by_stage[lifecycle_stage]
        
        # Stronger correlation with income (higher income = much more activity)
        income_multiplier = 1.0
        if annual_income > 120000:
            income_multiplier = 2.5  # High earners spend much more
        elif annual_income > 85000:
            income_multiplier = 1.8
        elif annual_income > 60000:
            income_multiplier = 1.3
        elif annual_income > 45000:
            income_multiplier = 1.0
        else:
            income_multiplier = 0.6
        
        # Generate metrics with controlled randomness to ensure some VIP customers
        variance = self.faker.random.uniform(0.7, 1.4)
        
        # Occasionally create high-value customers regardless of other factors
        if self.faker.random.random() < 0.08:  # 8% chance for high-value customer
            income_multiplier *= self.faker.random.uniform(2.0, 4.0)
        
        total_orders = max(1, int(base_activity['avg_orders'] * income_multiplier * variance))
        lifetime_value = max(50, int(base_activity['avg_value'] * income_multiplier * variance))
        
        # Ensure some customers meet VIP criteria
        if total_orders >= 15 and lifetime_value >= 3000:
            if self.faker.random.random() < 0.3:  # 30% chance to boost to VIP level
                total_orders = max(total_orders, self.faker.random_int(20, 35))
                lifetime_value = max(lifetime_value, self.faker.random_int(5000, 15000))
        
        avg_order_value = round(lifetime_value / total_orders, 2)
        
        # Generate last order date based on lifecycle stage
        if lifecycle_stage == 'new':
            days_ago = self.faker.random_int(1, 30)
        elif lifecycle_stage == 'active':
            days_ago = self.faker.random_int(1, 60)
        elif lifecycle_stage == 'returning':
            days_ago = self.faker.random_int(30, 180)
        elif lifecycle_stage == 'dormant':
            days_ago = self.faker.random_int(180, 730)
        else:  # churned
            days_ago = self.faker.random_int(730, 1095)
        
        last_order_date = datetime.now().date() - timedelta(days=days_ago)
        
        return {
            'total_orders': total_orders,
            'lifetime_value': lifetime_value,
            'avg_order_value': avg_order_value,
            'last_order_date': last_order_date,
            'days_since_last_order': days_ago
        }
    
    def _determine_segment(self, lifetime_value: float, total_orders: int) -> str:
        """Determine customer segment based on activity"""
        # Check segments in order of priority
        for segment, criteria in self.segment_criteria.items():
            if (lifetime_value >= criteria['min_lifetime_value'] and 
                total_orders >= criteria['min_orders']):
                return segment
        
        return 'Basic'  # Default segment
    
    def _generate_email(self, first_name: str, last_name: str) -> str:
        """Generate realistic email address"""
        # Common email patterns
        patterns = [
            f"{first_name.lower()}.{last_name.lower()}",
            f"{first_name.lower()}{last_name.lower()}",
            f"{first_name.lower()}_{last_name.lower()}",
            f"{first_name[0].lower()}{last_name.lower()}",
            f"{first_name.lower()}{self.faker.random_int(1, 999)}"
        ]
        
        pattern = self.faker.random_element(patterns)
        domain = self.faker.random_element([
            'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 
            'icloud.com', 'aol.com', 'protonmail.com'
        ])
        
        return f"{pattern}@{domain}"
    
    def _generate_address(self, country: str) -> Dict[str, str]:
        """Generate address information"""
        return {
            'street': self.faker.street_address(),
            'city': self.faker.city(),
            'state': self.faker.state(),
            'postal_code': self.faker.postcode(),
            'country': country if country != 'global' else self.faker.country()
        }
    
    def _generate_preferences(self, age_group: str, gender: str) -> Dict[str, Any]:
        """Generate customer preferences based on demographics"""
        # Contact method preferences by age group
        contact_prefs = {
            'gen_z': ['sms', 'app_notification', 'email'],
            'millennial': ['email', 'sms', 'app_notification'],
            'gen_x': ['email', 'phone', 'sms'],
            'boomer': ['email', 'phone', 'mail']
        }
        
        # Product category preferences by gender and age
        if gender == 'male':
            if age_group in ['gen_z', 'millennial']:
                categories = ['technology', 'gaming', 'sports', 'automotive']
            else:
                categories = ['automotive', 'tools', 'sports', 'technology']
        elif gender == 'female':
            if age_group in ['gen_z', 'millennial']:
                categories = ['fashion', 'beauty', 'technology', 'home_decor']
            else:
                categories = ['home_decor', 'fashion', 'health', 'books']
        else:
            categories = ['technology', 'books', 'art', 'travel']
        
        return {
            'contact_method': self.faker.random_element(contact_prefs[age_group]),
            'marketing_opt_in': self.faker.random.random() < 0.65,  # 65% opt-in rate
            'product_category': self.faker.random_element(categories)
        }
    
    def _apply_realistic_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply additional realistic patterns to the generated data"""
        # Add customer satisfaction score based on segment and activity
        data['satisfaction_score'] = data.apply(
            lambda row: self._calculate_satisfaction_score(row), axis=1
        ).round(1)
        
        # Add churn risk score
        data['churn_risk_score'] = data.apply(
            lambda row: self._calculate_churn_risk(row), axis=1
        ).round(2)
        
        # Add preferred communication time
        data['preferred_contact_time'] = data['age_group'].apply(
            lambda age_group: self._get_preferred_contact_time(age_group)
        )
        
        # Sort by registration date for realistic chronological order
        data = data.sort_values('registration_date').reset_index(drop=True)
        
        return data
    
    def _calculate_satisfaction_score(self, row) -> float:
        """Calculate customer satisfaction score"""
        base_score = 3.5
        
        # Segment adjustments
        segment_adjustments = {
            'VIP': 0.8,
            'Premium': 0.4,
            'Standard': 0.0,
            'Basic': -0.3
        }
        
        # Lifecycle adjustments
        lifecycle_adjustments = {
            'new': 0.2,
            'active': 0.3,
            'returning': 0.4,
            'dormant': -0.5,
            'churned': -1.0
        }
        
        score = (base_score + 
                segment_adjustments.get(row['segment'], 0) +
                lifecycle_adjustments.get(row['lifecycle_stage'], 0) +
                self.faker.random.gauss(0, 0.3))
        
        return max(1.0, min(5.0, score))
    
    def _calculate_churn_risk(self, row) -> float:
        """Calculate churn risk score (0-1, higher = more risk)"""
        base_risk = 0.3
        
        # Days since last order impact
        if row['days_since_last_order'] > 365:
            base_risk += 0.4
        elif row['days_since_last_order'] > 180:
            base_risk += 0.2
        elif row['days_since_last_order'] > 90:
            base_risk += 0.1
        
        # Lifecycle stage impact
        lifecycle_risk = {
            'new': 0.4,      # New customers are risky
            'active': 0.1,   # Active customers are low risk
            'returning': 0.2, # Returning customers medium risk
            'dormant': 0.7,  # Dormant customers high risk
            'churned': 0.9   # Churned customers very high risk
        }
        
        base_risk += lifecycle_risk.get(row['lifecycle_stage'], 0.3)
        
        # Segment impact
        if row['segment'] == 'VIP':
            base_risk *= 0.5
        elif row['segment'] == 'Premium':
            base_risk *= 0.7
        elif row['segment'] == 'Basic':
            base_risk *= 1.2
        
        # Add some randomness
        base_risk += self.faker.random.gauss(0, 0.1)
        
        return max(0.0, min(1.0, base_risk))
    
    def _get_preferred_contact_time(self, age_group: str) -> str:
        """Get preferred contact time based on age group"""
        time_preferences = {
            'gen_z': ['evening', 'afternoon', 'night'],
            'millennial': ['morning', 'afternoon', 'evening'],
            'gen_x': ['morning', 'afternoon', 'early_evening'],
            'boomer': ['morning', 'early_afternoon', 'early_evening']
        }
        
        return self.faker.random_element(time_preferences[age_group])