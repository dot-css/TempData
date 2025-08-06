"""
Marketing campaign dataset generator

Generates realistic marketing campaign data with lead conversion patterns,
seasonal variations, and channel performance metrics.
"""

import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
from ...core.base_generator import BaseGenerator


class MarketingGenerator(BaseGenerator):
    """
    Generator for realistic marketing campaign data
    
    Creates marketing datasets with campaign performance, lead conversion patterns,
    channel effectiveness, seasonal variations, and realistic budget allocations.
    """
    
    def __init__(self, seeder, locale: str = 'en_US'):
        super().__init__(seeder, locale)
        self._setup_campaign_data()
        self._setup_channel_data()
        self._setup_seasonal_patterns()
        self._setup_conversion_patterns()
        self._setup_lead_scoring()
    
    def _setup_campaign_data(self):
        """Setup campaign types and characteristics"""
        self.campaign_types = {
            'brand_awareness': {
                'typical_budget_range': (5000, 50000),
                'conversion_rate_range': (0.5, 2.0),
                'ctr_range': (0.8, 2.5),
                'lead_quality_score': (3, 6)
            },
            'lead_generation': {
                'typical_budget_range': (10000, 100000),
                'conversion_rate_range': (2.0, 8.0),
                'ctr_range': (1.5, 4.0),
                'lead_quality_score': (6, 9)
            },
            'product_launch': {
                'typical_budget_range': (20000, 200000),
                'conversion_rate_range': (1.0, 5.0),
                'ctr_range': (2.0, 6.0),
                'lead_quality_score': (4, 7)
            },
            'retargeting': {
                'typical_budget_range': (3000, 30000),
                'conversion_rate_range': (3.0, 12.0),
                'ctr_range': (2.5, 8.0),
                'lead_quality_score': (7, 10)
            },
            'seasonal_promotion': {
                'typical_budget_range': (8000, 80000),
                'conversion_rate_range': (2.5, 10.0),
                'ctr_range': (2.0, 5.5),
                'lead_quality_score': (5, 8)
            },
            'content_marketing': {
                'typical_budget_range': (2000, 25000),
                'conversion_rate_range': (1.5, 6.0),
                'ctr_range': (1.0, 3.5),
                'lead_quality_score': (4, 8)
            }
        }
        
        self.campaign_objectives = [
            'increase_brand_awareness', 'generate_leads', 'drive_sales',
            'customer_acquisition', 'customer_retention', 'product_education',
            'market_expansion', 'competitive_positioning'
        ]
    
    def _setup_channel_data(self):
        """Setup marketing channel characteristics"""
        self.marketing_channels = {
            'google_ads': {
                'avg_cpc': (1.50, 8.00),
                'avg_ctr': (2.0, 4.5),
                'conversion_rate': (2.5, 6.0),
                'impression_volume': (10000, 500000),
                'lead_quality_multiplier': 1.0
            },
            'facebook_ads': {
                'avg_cpc': (0.80, 3.50),
                'avg_ctr': (1.5, 3.8),
                'conversion_rate': (1.8, 4.5),
                'impression_volume': (15000, 800000),
                'lead_quality_multiplier': 0.9
            },
            'linkedin_ads': {
                'avg_cpc': (3.00, 12.00),
                'avg_ctr': (0.8, 2.2),
                'conversion_rate': (3.0, 8.0),
                'impression_volume': (2000, 100000),
                'lead_quality_multiplier': 1.3
            },
            'email_marketing': {
                'avg_cpc': (0.10, 0.50),
                'avg_ctr': (2.5, 8.0),
                'conversion_rate': (4.0, 15.0),
                'impression_volume': (5000, 200000),
                'lead_quality_multiplier': 1.1
            },
            'content_marketing': {
                'avg_cpc': (0.50, 2.00),
                'avg_ctr': (1.0, 3.0),
                'conversion_rate': (2.0, 7.0),
                'impression_volume': (3000, 150000),
                'lead_quality_multiplier': 1.2
            },
            'display_advertising': {
                'avg_cpc': (0.30, 2.50),
                'avg_ctr': (0.5, 1.8),
                'conversion_rate': (0.8, 3.0),
                'impression_volume': (50000, 2000000),
                'lead_quality_multiplier': 0.7
            },
            'video_advertising': {
                'avg_cpc': (2.00, 6.00),
                'avg_ctr': (1.8, 4.2),
                'conversion_rate': (2.2, 5.5),
                'impression_volume': (8000, 300000),
                'lead_quality_multiplier': 1.0
            },
            'influencer_marketing': {
                'avg_cpc': (1.00, 4.00),
                'avg_ctr': (2.0, 6.0),
                'conversion_rate': (1.5, 4.0),
                'impression_volume': (5000, 250000),
                'lead_quality_multiplier': 0.8
            }
        }
    
    def _setup_seasonal_patterns(self):
        """Setup seasonal marketing performance patterns"""
        # Monthly performance multipliers (1.0 = average)
        self.seasonal_multipliers = {
            1: 0.7,   # January - post-holiday low
            2: 0.8,   # February - Valentine's boost for some sectors
            3: 1.0,   # March - normal activity
            4: 1.1,   # April - spring campaigns
            5: 1.2,   # May - increased activity
            6: 1.0,   # June - summer start
            7: 0.9,   # July - vacation impact
            8: 0.8,   # August - continued vacation impact
            9: 1.2,   # September - back-to-business surge
            10: 1.3,  # October - Q4 push begins
            11: 1.5,  # November - Black Friday/holiday prep
            12: 1.4   # December - holiday campaigns
        }
        
        # Day of week patterns (0=Monday, 6=Sunday)
        self.weekly_patterns = {
            0: 1.1,  # Monday - strong start
            1: 1.2,  # Tuesday - peak performance
            2: 1.2,  # Wednesday - peak performance
            3: 1.1,  # Thursday - good performance
            4: 1.0,  # Friday - normal
            5: 0.8,  # Saturday - lower B2B activity
            6: 0.7   # Sunday - lowest activity
        }
        
        # Hour of day patterns for campaign performance
        self.daily_patterns = {
            0: 0.2, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.2,
            6: 0.4, 7: 0.6, 8: 0.8, 9: 1.0, 10: 1.2, 11: 1.1,
            12: 1.0, 13: 1.1, 14: 1.2, 15: 1.1, 16: 1.0, 17: 0.9,
            18: 0.8, 19: 0.7, 20: 0.6, 21: 0.5, 22: 0.4, 23: 0.3
        }
    
    def _setup_conversion_patterns(self):
        """Setup realistic conversion funnel patterns"""
        self.conversion_funnel_stages = {
            'impressions': 1.0,
            'clicks': 0.025,      # 2.5% CTR average
            'landing_page_views': 0.85,  # 85% of clicks reach landing page
            'form_submissions': 0.15,     # 15% of landing page views submit form
            'qualified_leads': 0.60,      # 60% of submissions are qualified
            'sales_opportunities': 0.40,  # 40% of qualified leads become opportunities
            'conversions': 0.25           # 25% of opportunities convert
        }
        
        # Channel-specific funnel performance modifiers
        self.channel_funnel_modifiers = {
            'google_ads': {'clicks': 1.2, 'qualified_leads': 1.1, 'conversions': 1.0},
            'facebook_ads': {'clicks': 1.0, 'qualified_leads': 0.9, 'conversions': 0.9},
            'linkedin_ads': {'clicks': 0.8, 'qualified_leads': 1.3, 'conversions': 1.2},
            'email_marketing': {'clicks': 1.5, 'qualified_leads': 1.2, 'conversions': 1.3},
            'content_marketing': {'clicks': 0.9, 'qualified_leads': 1.1, 'conversions': 1.1},
            'display_advertising': {'clicks': 0.7, 'qualified_leads': 0.8, 'conversions': 0.7},
            'video_advertising': {'clicks': 1.1, 'qualified_leads': 1.0, 'conversions': 1.0},
            'influencer_marketing': {'clicks': 1.3, 'qualified_leads': 0.8, 'conversions': 0.8}
        }
    
    def _setup_lead_scoring(self):
        """Setup lead scoring patterns"""
        self.lead_score_factors = {
            'demographic_score': (1, 10),
            'behavioral_score': (1, 10),
            'engagement_score': (1, 10),
            'firmographic_score': (1, 10)  # For B2B leads
        }
        
        self.lead_quality_tiers = {
            'cold': (1, 30),
            'warm': (31, 60),
            'hot': (61, 80),
            'qualified': (81, 100)
        }
    
    def generate(self, rows: int, **kwargs) -> pd.DataFrame:
        """
        Generate marketing campaign dataset
        
        Args:
            rows: Number of campaign records to generate
            **kwargs: Additional parameters (time_series, date_range, etc.)
            
        Returns:
            pd.DataFrame: Generated marketing campaign data with realistic patterns
        """
        # Check if time series generation is requested
        ts_config = self._create_time_series_config(**kwargs)
        
        if ts_config:
            return self._generate_time_series_marketing(rows, ts_config, **kwargs)
        else:
            return self._generate_snapshot_marketing(rows, **kwargs)
    
    def _generate_snapshot_marketing(self, rows: int, **kwargs) -> pd.DataFrame:
        """Generate snapshot marketing data (random timestamps)"""
        date_range = kwargs.get('date_range', None)
        
        data = []
        
        for i in range(rows):
            # Generate campaign date
            if date_range:
                start_date, end_date = date_range
                campaign_date = self.faker.date_between(start_date=start_date, end_date=end_date)
            else:
                campaign_date = self.faker.date_this_year()
            
            # Generate campaign record
            campaign = self._generate_marketing_campaign(i, campaign_date)
            data.append(campaign)
        
        df = pd.DataFrame(data)
        return self._apply_realistic_patterns(df)
    
    def _generate_time_series_marketing(self, rows: int, ts_config, **kwargs) -> pd.DataFrame:
        """Generate time series marketing data using integrated time series system"""
        
        # Generate timestamps from time series config
        timestamps = self._generate_time_series_timestamps(ts_config, rows)
        
        # Create base time series for campaign performance
        base_performance = 100.0  # Base performance score
        
        performance_series = self.time_series_generator.generate_time_series_base(
            ts_config,
            base_value=base_performance,
            value_range=(base_performance * 0.3, base_performance * 2.0)
        )
        
        data = []
        
        for i, timestamp in enumerate(timestamps):
            if i >= rows or i >= len(performance_series):
                break
            
            # Get time series performance value
            performance_multiplier = performance_series.iloc[i]['value'] / base_performance
            
            # Generate marketing campaign with temporal patterns
            campaign = self._generate_time_series_marketing_campaign(
                i, timestamp, performance_multiplier
            )
            
            data.append(campaign)
        
        df = pd.DataFrame(data)
        
        # Add temporal relationships
        df = self._add_temporal_relationships(df, ts_config)
        
        # Apply marketing-specific time series correlations
        df = self._apply_marketing_time_series_correlations(df, ts_config)
        
        return self._apply_realistic_patterns(df)
    
    def _generate_marketing_campaign(self, campaign_index: int, campaign_date: datetime.date) -> Dict:
        """Generate a single marketing campaign record"""
        
        # Select campaign type and channel
        campaign_type = self.faker.random_element(list(self.campaign_types.keys()))
        channel = self.faker.random_element(list(self.marketing_channels.keys()))
        
        # Get campaign and channel characteristics
        campaign_config = self.campaign_types[campaign_type]
        channel_config = self.marketing_channels[channel]
        
        # Generate budget
        budget_min, budget_max = campaign_config['typical_budget_range']
        budget = round(self.faker.random.uniform(budget_min, budget_max), 2)
        
        # Generate impressions based on channel and budget
        impression_min, impression_max = channel_config['impression_volume']
        # Scale impressions based on budget (higher budget = more impressions)
        budget_multiplier = min(budget / budget_min, 3.0)  # Cap at 3x
        scaled_impression_max = int(impression_max * budget_multiplier)
        impressions = self.faker.random_int(impression_min, scaled_impression_max)
        
        # Generate clicks based on CTR
        ctr_min, ctr_max = channel_config['avg_ctr']
        ctr = self.faker.random.uniform(ctr_min, ctr_max)
        clicks = int(impressions * (ctr / 100))
        
        # Generate conversions based on conversion rate
        conv_rate_min, conv_rate_max = campaign_config['conversion_rate_range']
        conversion_rate = self.faker.random.uniform(conv_rate_min, conv_rate_max)
        conversions = int(clicks * (conversion_rate / 100))
        
        # Calculate cost metrics
        cpc_min, cpc_max = channel_config['avg_cpc']
        cost_per_click = self.faker.random.uniform(cpc_min, cpc_max)
        total_cost = min(clicks * cost_per_click, budget)  # Don't exceed budget
        cost_per_conversion = total_cost / max(conversions, 1)
        
        # Generate lead scoring
        lead_score = self._generate_lead_score(campaign_type, channel)
        
        # Apply seasonal adjustments
        seasonal_mult = self.seasonal_multipliers.get(campaign_date.month, 1.0)
        weekly_mult = self.weekly_patterns.get(campaign_date.weekday(), 1.0)
        
        # Adjust metrics based on seasonal patterns
        impressions = int(impressions * seasonal_mult * weekly_mult)
        clicks = int(clicks * seasonal_mult * weekly_mult)
        conversions = int(conversions * seasonal_mult * weekly_mult)
        
        return {
            'campaign_id': f'CAMP_{campaign_index+1:06d}',
            'campaign_name': self._generate_campaign_name(campaign_type),
            'campaign_type': campaign_type,
            'channel': channel,
            'start_date': campaign_date,
            'end_date': campaign_date + timedelta(days=self.faker.random_int(7, 90)),
            'budget': budget,
            'total_cost': round(total_cost, 2),
            'impressions': impressions,
            'clicks': clicks,
            'conversions': conversions,
            'ctr': round((clicks / max(impressions, 1)) * 100, 2),
            'conversion_rate': round((conversions / max(clicks, 1)) * 100, 2),
            'cost_per_click': round(cost_per_click, 2),
            'cost_per_conversion': round(cost_per_conversion, 2),
            'lead_score_avg': round(lead_score, 1),
            'campaign_objective': self.faker.random_element(self.campaign_objectives),
            'target_audience': self._generate_target_audience(),
            'campaign_manager': self.faker.name(),
            'status': self.faker.random_element(['active', 'paused', 'completed', 'draft']),
            'roi': round(((conversions * 100) - total_cost) / max(total_cost, 1) * 100, 2)  # Assume $100 value per conversion
        }
    
    def _generate_time_series_marketing_campaign(self, campaign_index: int, 
                                               timestamp: datetime, 
                                               performance_multiplier: float) -> Dict:
        """Generate time series marketing campaign with temporal patterns"""
        
        # Select campaign type and channel with time-aware preferences
        campaign_type = self._select_time_aware_campaign_type(timestamp)
        channel = self._select_time_aware_channel(timestamp, campaign_type)
        
        # Get campaign and channel characteristics
        campaign_config = self.campaign_types[campaign_type]
        channel_config = self.marketing_channels[channel]
        
        # Generate budget with performance adjustment
        budget_min, budget_max = campaign_config['typical_budget_range']
        base_budget = self.faker.random.uniform(budget_min, budget_max)
        budget = round(base_budget * performance_multiplier, 2)
        
        # Apply temporal patterns to performance metrics
        temporal_performance = self._apply_marketing_temporal_patterns(
            performance_multiplier, timestamp
        )
        
        # Generate impressions with temporal adjustment
        impression_min, impression_max = channel_config['impression_volume']
        budget_multiplier = min(budget / budget_min, 3.0)
        scaled_impression_max = int(impression_max * budget_multiplier * temporal_performance)
        impressions = self.faker.random_int(
            int(impression_min * temporal_performance), 
            scaled_impression_max
        )
        
        # Generate clicks and conversions with temporal patterns
        ctr_min, ctr_max = channel_config['avg_ctr']
        ctr = self.faker.random.uniform(ctr_min, ctr_max) * temporal_performance
        clicks = int(impressions * (ctr / 100))
        
        conv_rate_min, conv_rate_max = campaign_config['conversion_rate_range']
        conversion_rate = self.faker.random.uniform(conv_rate_min, conv_rate_max) * temporal_performance
        conversions = int(clicks * (conversion_rate / 100))
        
        # Calculate cost metrics
        cpc_min, cpc_max = channel_config['avg_cpc']
        cost_per_click = self.faker.random.uniform(cpc_min, cpc_max)
        total_cost = min(clicks * cost_per_click, budget)
        cost_per_conversion = total_cost / max(conversions, 1)
        
        # Generate lead scoring with temporal adjustment
        lead_score = self._generate_lead_score(campaign_type, channel) * temporal_performance
        
        return {
            'campaign_id': f'CAMP_{campaign_index+1:06d}',
            'campaign_name': self._generate_campaign_name(campaign_type),
            'campaign_type': campaign_type,
            'channel': channel,
            'timestamp': timestamp,
            'date': timestamp.date(),
            'start_date': timestamp.date(),
            'end_date': timestamp.date() + timedelta(days=self.faker.random_int(7, 90)),
            'budget': budget,
            'total_cost': round(total_cost, 2),
            'impressions': impressions,
            'clicks': clicks,
            'conversions': conversions,
            'ctr': round((clicks / max(impressions, 1)) * 100, 2),
            'conversion_rate': round((conversions / max(clicks, 1)) * 100, 2),
            'cost_per_click': round(cost_per_click, 2),
            'cost_per_conversion': round(cost_per_conversion, 2),
            'lead_score_avg': round(lead_score, 1),
            'campaign_objective': self.faker.random_element(self.campaign_objectives),
            'target_audience': self._generate_target_audience(),
            'campaign_manager': self.faker.name(),
            'status': self.faker.random_element(['active', 'paused', 'completed', 'draft']),
            'roi': round(((conversions * 100) - total_cost) / max(total_cost, 1) * 100, 2),
            'performance_score': round(temporal_performance * 100, 1)
        }
    
    def _generate_campaign_name(self, campaign_type: str) -> str:
        """Generate realistic campaign names"""
        prefixes = {
            'brand_awareness': ['Brand Boost', 'Awareness Drive', 'Brand Focus', 'Recognition'],
            'lead_generation': ['Lead Gen', 'Prospect Hunt', 'Lead Drive', 'Generation'],
            'product_launch': ['Launch Pad', 'New Product', 'Product Intro', 'Launch'],
            'retargeting': ['Retarget Pro', 'Return Focus', 'Re-engage', 'Comeback'],
            'seasonal_promotion': ['Seasonal Sale', 'Holiday Special', 'Season Push', 'Promo'],
            'content_marketing': ['Content Hub', 'Thought Leader', 'Content Drive', 'Education']
        }
        
        suffixes = ['2024', 'Q1', 'Q2', 'Q3', 'Q4', 'Spring', 'Summer', 'Fall', 'Winter',
                   'Campaign', 'Initiative', 'Push', 'Blitz', 'Focus', 'Drive']
        
        prefix = self.faker.random_element(prefixes.get(campaign_type, ['Marketing']))
        suffix = self.faker.random_element(suffixes)
        
        return f"{prefix} {suffix}"
    
    def _generate_target_audience(self) -> str:
        """Generate target audience descriptions"""
        demographics = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        interests = ['technology', 'business', 'lifestyle', 'health', 'finance', 'education',
                    'entertainment', 'sports', 'travel', 'food', 'fashion', 'automotive']
        behaviors = ['online shoppers', 'mobile users', 'social media active', 'email subscribers',
                    'frequent buyers', 'price conscious', 'brand loyal', 'early adopters']
        
        demo = self.faker.random_element(demographics)
        interest = self.faker.random_element(interests)
        behavior = self.faker.random_element(behaviors)
        
        return f"{demo} {interest} enthusiasts, {behavior}"
    
    def _generate_lead_score(self, campaign_type: str, channel: str) -> float:
        """Generate realistic lead scores based on campaign type and channel"""
        # Base score from campaign type
        score_min, score_max = self.campaign_types[campaign_type]['lead_quality_score']
        base_score = self.faker.random.uniform(score_min, score_max)
        
        # Apply channel multiplier
        channel_multiplier = self.marketing_channels[channel]['lead_quality_multiplier']
        
        # Generate individual scoring components
        demographic_score = self.faker.random.uniform(*self.lead_score_factors['demographic_score'])
        behavioral_score = self.faker.random.uniform(*self.lead_score_factors['behavioral_score'])
        engagement_score = self.faker.random.uniform(*self.lead_score_factors['engagement_score'])
        firmographic_score = self.faker.random.uniform(*self.lead_score_factors['firmographic_score'])
        
        # Calculate weighted average
        total_score = (base_score * 0.4 + 
                      demographic_score * 0.2 + 
                      behavioral_score * 0.2 + 
                      engagement_score * 0.1 + 
                      firmographic_score * 0.1) * channel_multiplier
        
        return min(max(total_score, 1.0), 10.0)  # Clamp between 1-10
    
    def _select_time_aware_campaign_type(self, timestamp: datetime) -> str:
        """Select campaign type based on time patterns"""
        month = timestamp.month
        hour = timestamp.hour
        
        # Seasonal campaign preferences
        if month in [11, 12]:  # Holiday season
            preferred_types = ['seasonal_promotion', 'product_launch', 'brand_awareness']
        elif month in [1, 2]:  # New year period
            preferred_types = ['lead_generation', 'content_marketing', 'retargeting']
        elif month in [9, 10]:  # Back-to-business season
            preferred_types = ['lead_generation', 'product_launch', 'brand_awareness']
        else:
            preferred_types = list(self.campaign_types.keys())
        
        # Business hours preference for B2B campaigns
        if 9 <= hour <= 17:
            if 'lead_generation' not in preferred_types:
                preferred_types.append('lead_generation')
        
        return self.faker.random_element(preferred_types)
    
    def _select_time_aware_channel(self, timestamp: datetime, campaign_type: str) -> str:
        """Select marketing channel based on time and campaign type"""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Business hours - favor B2B channels
        if 9 <= hour <= 17 and day_of_week < 5:
            preferred_channels = ['linkedin_ads', 'email_marketing', 'content_marketing', 'google_ads']
        # Evening hours - favor B2C channels
        elif 18 <= hour <= 22:
            preferred_channels = ['facebook_ads', 'video_advertising', 'influencer_marketing', 'display_advertising']
        # Weekend - favor social channels
        elif day_of_week >= 5:
            preferred_channels = ['facebook_ads', 'influencer_marketing', 'video_advertising']
        else:
            preferred_channels = list(self.marketing_channels.keys())
        
        # Campaign type preferences
        if campaign_type == 'lead_generation':
            preferred_channels = [ch for ch in preferred_channels if ch in ['linkedin_ads', 'google_ads', 'email_marketing']]
        elif campaign_type == 'brand_awareness':
            preferred_channels = [ch for ch in preferred_channels if ch in ['display_advertising', 'video_advertising', 'facebook_ads']]
        
        if not preferred_channels:
            preferred_channels = list(self.marketing_channels.keys())
        
        return self.faker.random_element(preferred_channels)
    
    def _apply_marketing_temporal_patterns(self, base_performance: float, timestamp: datetime) -> float:
        """Apply marketing-specific temporal patterns"""
        
        # Apply seasonal multiplier
        seasonal_mult = self.seasonal_multipliers.get(timestamp.month, 1.0)
        
        # Apply weekly pattern
        weekly_mult = self.weekly_patterns.get(timestamp.weekday(), 1.0)
        
        # Apply daily pattern
        daily_mult = self.daily_patterns.get(timestamp.hour, 1.0)
        
        # Calculate adjusted performance
        adjusted_performance = base_performance * seasonal_mult * weekly_mult * daily_mult
        
        # Add some random variation
        adjusted_performance *= self.faker.random.uniform(0.9, 1.1)
        
        return max(adjusted_performance, 0.1)  # Minimum performance threshold
    
    def _apply_marketing_time_series_correlations(self, data: pd.DataFrame, ts_config) -> pd.DataFrame:
        """Apply realistic time-based correlations for marketing data"""
        if len(data) < 2:
            return data
        
        # Sort by timestamp to ensure proper time series order
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Apply temporal correlations for budget and performance
        for i in range(1, len(data)):
            prev_budget = data.iloc[i-1]['budget']
            prev_conversions = data.iloc[i-1]['conversions']
            
            # Budget persistence (similar budgets tend to cluster)
            if prev_budget > 0:
                correlation_strength = 0.3
                budget_adjustment = 1 + (correlation_strength * self.faker.random.uniform(-0.2, 0.2))
                new_budget = data.iloc[i]['budget'] * budget_adjustment
                data.loc[i, 'budget'] = max(1000, min(new_budget, data.iloc[i]['budget'] * 1.5))
            
            # Performance momentum (good performance tends to continue)
            if prev_conversions > 0:
                momentum_strength = 0.2
                if prev_conversions > data.iloc[i-1]['clicks'] * 0.05:  # Good conversion rate
                    performance_boost = 1 + (momentum_strength * self.faker.random.uniform(0, 0.3))
                    data.loc[i, 'conversions'] = int(data.iloc[i]['conversions'] * performance_boost)
        
        return data
    
    def _apply_realistic_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply additional realistic patterns to marketing data"""
        
        # Add derived fields
        if 'timestamp' in data.columns:
            data['day_of_week'] = pd.to_datetime(data['timestamp']).dt.day_name()
            data['month'] = pd.to_datetime(data['timestamp']).dt.month
            data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
            data['is_weekend'] = pd.to_datetime(data['timestamp']).dt.weekday >= 5
        else:
            data['day_of_week'] = pd.to_datetime(data['start_date']).dt.day_name()
            data['month'] = pd.to_datetime(data['start_date']).dt.month
            data['is_weekend'] = pd.to_datetime(data['start_date']).dt.weekday >= 5
        
        # Calculate additional metrics
        data['click_through_rate'] = data['ctr']  # Already calculated
        data['cost_per_impression'] = round(data['total_cost'] / data['impressions'].replace(0, 1), 4)
        data['conversion_value'] = data['conversions'] * 100  # Assume $100 per conversion
        data['profit'] = data['conversion_value'] - data['total_cost']
        data['profit_margin'] = round((data['profit'] / data['conversion_value'].replace(0, 1)) * 100, 2)
        
        # Add campaign performance tiers
        data['performance_tier'] = pd.cut(
            data['roi'],
            bins=[-float('inf'), 0, 50, 150, float('inf')],
            labels=['poor', 'average', 'good', 'excellent']
        )
        
        # Add lead quality categories
        data['lead_quality_tier'] = pd.cut(
            data['lead_score_avg'],
            bins=[0, 3, 5, 7, 10],
            labels=['low', 'medium', 'high', 'premium']
        )
        
        # Add budget categories
        data['budget_tier'] = pd.cut(
            data['budget'],
            bins=[0, 10000, 50000, 100000, float('inf')],
            labels=['small', 'medium', 'large', 'enterprise']
        )
        
        # Sort by date for realistic chronological order
        sort_column = 'timestamp' if 'timestamp' in data.columns else 'start_date'
        data = data.sort_values(sort_column).reset_index(drop=True)
        
        return data