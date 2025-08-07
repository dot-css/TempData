"""
CRM (Customer Relationship Management) dataset generator

Generates realistic CRM data with contact records, interaction histories,
opportunity pipelines, and deal values with realistic sales progression patterns.
"""

import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional
from ...core.base_generator import BaseGenerator


class CRMGenerator(BaseGenerator):
    """
    Generator for realistic CRM (Customer Relationship Management) data
    
    Creates CRM datasets with contact records, interaction histories, opportunity pipelines,
    deal values, and realistic sales pipeline progression and win rate patterns.
    """
    
    def __init__(self, seeder, locale: str = 'en_US'):
        super().__init__(seeder, locale)
        self._setup_interaction_types()
        self._setup_opportunity_stages()
        self._setup_deal_characteristics()
        self._setup_communication_patterns()
        self._setup_win_rate_patterns()
    
    def _setup_interaction_types(self):
        """Setup interaction types and their characteristics"""
        self.interaction_types = {
            'phone_call': {
                'duration_range': (5, 45),  # minutes
                'success_rate': 0.65,
                'follow_up_probability': 0.8,
                'lead_quality_impact': 1.2
            },
            'email': {
                'duration_range': (1, 3),  # minutes to compose/read
                'success_rate': 0.45,
                'follow_up_probability': 0.6,
                'lead_quality_impact': 0.9
            },
            'meeting': {
                'duration_range': (30, 120),  # minutes
                'success_rate': 0.85,
                'follow_up_probability': 0.95,
                'lead_quality_impact': 1.8
            },
            'demo': {
                'duration_range': (45, 90),  # minutes
                'success_rate': 0.75,
                'follow_up_probability': 0.9,
                'lead_quality_impact': 2.0
            },
            'proposal_sent': {
                'duration_range': (2, 5),  # minutes to send
                'success_rate': 0.55,
                'follow_up_probability': 0.85,
                'lead_quality_impact': 1.5
            },
            'follow_up': {
                'duration_range': (10, 25),  # minutes
                'success_rate': 0.5,
                'follow_up_probability': 0.7,
                'lead_quality_impact': 1.0
            },
            'social_media': {
                'duration_range': (1, 5),  # minutes
                'success_rate': 0.25,
                'follow_up_probability': 0.4,
                'lead_quality_impact': 0.7
            }
        }
    
    def _setup_opportunity_stages(self):
        """Setup sales pipeline stages and progression patterns"""
        self.opportunity_stages = {
            'lead': {
                'probability': 0.10,
                'avg_duration_days': 7,
                'next_stages': ['qualified', 'lost'],
                'next_stage_probabilities': [0.6, 0.4]
            },
            'qualified': {
                'probability': 0.25,
                'avg_duration_days': 14,
                'next_stages': ['proposal', 'lost'],
                'next_stage_probabilities': [0.7, 0.3]
            },
            'proposal': {
                'probability': 0.50,
                'avg_duration_days': 21,
                'next_stages': ['negotiation', 'lost'],
                'next_stage_probabilities': [0.65, 0.35]
            },
            'negotiation': {
                'probability': 0.70,
                'avg_duration_days': 14,
                'next_stages': ['closed_won', 'lost'],
                'next_stage_probabilities': [0.75, 0.25]
            },
            'closed_won': {
                'probability': 1.0,
                'avg_duration_days': 0,
                'next_stages': [],
                'next_stage_probabilities': []
            },
            'lost': {
                'probability': 0.0,
                'avg_duration_days': 0,
                'next_stages': [],
                'next_stage_probabilities': []
            }
        }
    
    def _setup_deal_characteristics(self):
        """Setup deal value and characteristics patterns"""
        self.deal_sizes = {
            'small': {'range': (1000, 10000), 'probability': 0.5, 'close_rate': 0.4},
            'medium': {'range': (10000, 50000), 'probability': 0.3, 'close_rate': 0.25},
            'large': {'range': (50000, 200000), 'probability': 0.15, 'close_rate': 0.15},
            'enterprise': {'range': (200000, 1000000), 'probability': 0.05, 'close_rate': 0.1}
        }
        
        self.industries = {
            'technology': {'deal_multiplier': 1.3, 'sales_cycle_days': 45},
            'healthcare': {'deal_multiplier': 1.1, 'sales_cycle_days': 60},
            'finance': {'deal_multiplier': 1.4, 'sales_cycle_days': 75},
            'manufacturing': {'deal_multiplier': 1.2, 'sales_cycle_days': 90},
            'retail': {'deal_multiplier': 0.9, 'sales_cycle_days': 30},
            'education': {'deal_multiplier': 0.8, 'sales_cycle_days': 120},
            'government': {'deal_multiplier': 1.0, 'sales_cycle_days': 150}
        }
        
        self.lead_sources = {
            'website': 0.25,
            'referral': 0.20,
            'cold_outreach': 0.15,
            'trade_show': 0.12,
            'social_media': 0.10,
            'partner': 0.08,
            'advertising': 0.06,
            'other': 0.04
        }
    
    def _setup_communication_patterns(self):
        """Setup realistic communication patterns"""
        self.communication_frequency = {
            'lead': {'interactions_per_week': 2, 'response_rate': 0.3},
            'qualified': {'interactions_per_week': 3, 'response_rate': 0.6},
            'proposal': {'interactions_per_week': 4, 'response_rate': 0.8},
            'negotiation': {'interactions_per_week': 5, 'response_rate': 0.9},
            'closed_won': {'interactions_per_week': 1, 'response_rate': 0.95},
            'lost': {'interactions_per_week': 0, 'response_rate': 0.1}
        }
    
    def _setup_win_rate_patterns(self):
        """Setup win rate patterns by various factors"""
        self.win_rates_by_source = {
            'referral': 0.35,
            'partner': 0.30,
            'website': 0.25,
            'trade_show': 0.28,
            'cold_outreach': 0.15,
            'social_media': 0.18,
            'advertising': 0.20,
            'other': 0.22
        }
        
        self.win_rates_by_rep_experience = {
            'junior': 0.18,      # 0-2 years
            'mid_level': 0.25,   # 2-5 years
            'senior': 0.32,      # 5+ years
            'expert': 0.40       # 10+ years
        }
    
    def generate(self, rows: int, **kwargs) -> pd.DataFrame:
        """
        Generate CRM dataset
        
        Args:
            rows: Number of CRM records to generate
            **kwargs: Additional parameters (time_series, date_range, etc.)
            
        Returns:
            pd.DataFrame: Generated CRM data with realistic patterns
        """
        # Check if time series generation is requested
        ts_config = self._create_time_series_config(**kwargs)
        
        if ts_config:
            return self._generate_time_series_crm(rows, ts_config, **kwargs)
        else:
            return self._generate_snapshot_crm(rows, **kwargs)    

    def _generate_snapshot_crm(self, rows: int, **kwargs) -> pd.DataFrame:
        """Generate snapshot CRM data"""
        date_range = kwargs.get('date_range', None)
        
        data = []
        
        for i in range(rows):
            # Generate contact information
            contact_data = self._generate_contact_record(i)
            
            # Generate account information
            account_data = self._generate_account_record(contact_data)
            
            # Generate opportunity information
            opportunity_data = self._generate_opportunity_record(contact_data, account_data, date_range)
            
            # Generate interaction history
            interaction_data = self._generate_interaction_history(
                contact_data, opportunity_data, date_range
            )
            
            # Combine all data
            crm_record = {
                **contact_data,
                **account_data,
                **opportunity_data,
                **interaction_data
            }
            
            data.append(crm_record)
        
        df = pd.DataFrame(data)
        return self._apply_realistic_patterns(df)
    
    def _generate_contact_record(self, index: int) -> Dict[str, Any]:
        """Generate contact record information"""
        first_name = self.faker.first_name()
        last_name = self.faker.last_name()
        
        return {
            'contact_id': f'CONT_{index+1:06d}',
            'first_name': first_name,
            'last_name': last_name,
            'full_name': f'{first_name} {last_name}',
            'email': f'{first_name.lower()}.{last_name.lower()}@{self.faker.domain_name()}',
            'phone': self.faker.phone_number(),
            'job_title': self.faker.job(),
            'department': self.faker.random_element([
                'Sales', 'Marketing', 'IT', 'Operations', 'Finance', 
                'HR', 'Executive', 'Procurement', 'R&D'
            ]),
            'seniority_level': self.faker.random_element([
                'Individual Contributor', 'Manager', 'Director', 
                'VP', 'C-Level', 'Owner'
            ])
        }
    
    def _generate_account_record(self, contact_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate account record information"""
        company_name = self.faker.company()
        industry = self.faker.random_element(list(self.industries.keys()))
        
        # Company size affects deal potential
        company_sizes = {
            'startup': {'employees': (1, 50), 'revenue_range': (0, 1000000)},
            'small': {'employees': (51, 200), 'revenue_range': (1000000, 10000000)},
            'medium': {'employees': (201, 1000), 'revenue_range': (10000000, 100000000)},
            'large': {'employees': (1001, 5000), 'revenue_range': (100000000, 1000000000)},
            'enterprise': {'employees': (5001, 50000), 'revenue_range': (1000000000, 10000000000)}
        }
        
        company_size = self.faker.random_element(list(company_sizes.keys()))
        size_data = company_sizes[company_size]
        
        return {
            'account_id': f'ACC_{hash(company_name) % 100000:05d}',
            'company_name': company_name,
            'industry': industry,
            'company_size': company_size,
            'employee_count': self.faker.random_int(*size_data['employees']),
            'annual_revenue': self.faker.random_int(*size_data['revenue_range']),
            'website': f'https://www.{company_name.lower().replace(" ", "").replace(",", "")}.com',
            'address': self.faker.address().replace('\n', ', '),
            'city': self.faker.city(),
            'state': self.faker.state(),
            'country': self.faker.country()
        }    

    def _generate_opportunity_record(self, contact_data: Dict[str, Any], 
                                   account_data: Dict[str, Any], 
                                   date_range: Optional[tuple] = None) -> Dict[str, Any]:
        """Generate opportunity record information"""
        # Select deal size category
        deal_categories = list(self.deal_sizes.keys())
        deal_weights = [self.deal_sizes[cat]['probability'] for cat in deal_categories]
        
        # Weighted selection
        total_weight = sum(deal_weights)
        rand_val = self.faker.random.uniform(0, total_weight)
        
        cumulative_weight = 0
        deal_category = deal_categories[-1]  # fallback
        for category, weight in zip(deal_categories, deal_weights):
            cumulative_weight += weight
            if rand_val <= cumulative_weight:
                deal_category = category
                break
        
        # Generate deal value
        deal_range = self.deal_sizes[deal_category]['range']
        base_deal_value = self.faker.random_int(*deal_range)
        
        # Apply industry multiplier
        industry_multiplier = self.industries[account_data['industry']]['deal_multiplier']
        deal_value = int(base_deal_value * industry_multiplier)
        
        # Select current stage
        stage = self.faker.random_element(list(self.opportunity_stages.keys()))
        probability = self.opportunity_stages[stage]['probability']
        
        # Generate dates
        if date_range:
            start_date, end_date = date_range
            created_date = self.faker.date_between(start_date=start_date, end_date=end_date)
        else:
            created_date = self.faker.date_between(start_date='-1y', end_date='today')
        
        # Calculate expected close date based on stage and industry
        sales_cycle_days = self.industries[account_data['industry']]['sales_cycle_days']
        stage_duration = self.opportunity_stages[stage]['avg_duration_days']
        
        # Ensure minimum duration and add some variance
        min_duration = max(stage_duration, sales_cycle_days // 3)
        max_duration = sales_cycle_days + (sales_cycle_days // 2)
        
        expected_close_date = created_date + timedelta(
            days=self.faker.random_int(min_duration, max_duration)
        )
        
        # Generate lead source
        lead_sources = list(self.lead_sources.keys())
        lead_weights = list(self.lead_sources.values())
        
        total_weight = sum(lead_weights)
        rand_val = self.faker.random.uniform(0, total_weight)
        
        cumulative_weight = 0
        lead_source = lead_sources[-1]  # fallback
        for source, weight in zip(lead_sources, lead_weights):
            cumulative_weight += weight
            if rand_val <= cumulative_weight:
                lead_source = source
                break
        
        # Generate sales rep information
        rep_experience = self.faker.random_element(list(self.win_rates_by_rep_experience.keys()))
        
        return {
            'opportunity_id': f'OPP_{hash(f"{contact_data["contact_id"]}{created_date}") % 1000000:06d}',
            'opportunity_name': f'{account_data["company_name"]} - {self.faker.random_element(["Software License", "Consulting Services", "Implementation", "Support Contract", "Upgrade"])}',
            'stage': stage,
            'probability': probability,
            'deal_value': deal_value,
            'deal_category': deal_category,
            'created_date': created_date,
            'expected_close_date': expected_close_date,
            'lead_source': lead_source,
            'sales_rep': self.faker.name(),
            'sales_rep_experience': rep_experience,
            'product_interest': self.faker.random_element([
                'CRM Software', 'Marketing Automation', 'Analytics Platform',
                'Integration Services', 'Training & Support', 'Custom Development'
            ]),
            'competitor': self.faker.random_element([
                'Salesforce', 'HubSpot', 'Microsoft', 'Oracle', 'SAP', 'None', 'Unknown'
            ]),
            'budget_confirmed': self.faker.boolean(chance_of_getting_true=60),
            'decision_maker_identified': self.faker.boolean(chance_of_getting_true=70)
        }   
 
    def _generate_interaction_history(self, contact_data: Dict[str, Any],
                                    opportunity_data: Dict[str, Any],
                                    date_range: Optional[tuple] = None) -> Dict[str, Any]:
        """Generate interaction history summary"""
        stage = opportunity_data['stage']
        created_date = opportunity_data['created_date']
        
        # Calculate interaction frequency based on stage
        stage_config = self.communication_frequency[stage]
        interactions_per_week = stage_config['interactions_per_week']
        response_rate = stage_config['response_rate']
        
        # Calculate total interactions based on opportunity age
        if date_range:
            end_date = min(date_range[1], date.today())
        else:
            end_date = date.today()
        
        days_active = (end_date - created_date).days
        weeks_active = max(1, days_active / 7)
        
        total_interactions = int(interactions_per_week * weeks_active)
        successful_interactions = int(total_interactions * response_rate)
        
        # Generate interaction type distribution
        interaction_counts = {}
        for interaction_type in self.interaction_types.keys():
            # Weight interactions based on stage appropriateness
            if stage in ['lead', 'qualified']:
                weights = {'email': 0.4, 'phone_call': 0.3, 'social_media': 0.2, 'follow_up': 0.1}
            elif stage in ['proposal', 'negotiation']:
                weights = {'meeting': 0.3, 'demo': 0.25, 'proposal_sent': 0.2, 'phone_call': 0.15, 'email': 0.1}
            else:
                weights = {'phone_call': 0.4, 'email': 0.3, 'meeting': 0.2, 'follow_up': 0.1}
            
            weight = weights.get(interaction_type, 0.1)
            interaction_counts[interaction_type] = int(total_interactions * weight)
        
        # Calculate last interaction date
        last_interaction_date = created_date + timedelta(
            days=self.faker.random_int(0, min(days_active, 30))
        )
        
        return {
            'total_interactions': total_interactions,
            'successful_interactions': successful_interactions,
            'response_rate': round(response_rate, 2),
            'last_interaction_date': last_interaction_date,
            'last_interaction_type': self.faker.random_element(list(self.interaction_types.keys())),
            'days_since_last_interaction': (end_date - last_interaction_date).days,
            'email_interactions': interaction_counts.get('email', 0),
            'phone_interactions': interaction_counts.get('phone_call', 0),
            'meeting_interactions': interaction_counts.get('meeting', 0),
            'demo_interactions': interaction_counts.get('demo', 0),
            'next_follow_up_date': last_interaction_date + timedelta(
                days=self.faker.random_int(1, 14)
            )
        } 
   
    def _generate_time_series_crm(self, rows: int, ts_config, **kwargs) -> pd.DataFrame:
        """Generate time series CRM data using integrated time series system"""
        # Generate timestamps from time series config
        timestamps = self._generate_time_series_timestamps(ts_config, rows)
        
        # Generate base opportunity value time series
        base_opportunity_value = 25000  # Base deal value
        
        # Create base time series for opportunity values
        value_series = self.time_series_generator.generate_time_series_base(
            ts_config,
            base_value=base_opportunity_value,
            value_range=(1000, base_opportunity_value * 4.0)
        )
        
        data = []
        
        for i, timestamp in enumerate(timestamps):
            if i >= rows or i >= len(value_series):
                break
            
            # Get time series opportunity value
            base_deal_value = value_series.iloc[i]['value']
            
            # Generate contact information
            contact_data = self._generate_contact_record(i)
            
            # Generate account information
            account_data = self._generate_account_record(contact_data)
            
            # Generate opportunity with time series value
            opportunity_data = self._generate_time_series_opportunity_record(
                contact_data, account_data, timestamp, base_deal_value
            )
            
            # Generate interaction history with temporal patterns
            interaction_data = self._generate_time_series_interaction_history(
                contact_data, opportunity_data, timestamp
            )
            
            # Combine all data
            crm_record = {
                **contact_data,
                **account_data,
                **opportunity_data,
                **interaction_data,
                'timestamp': timestamp
            }
            
            data.append(crm_record)
        
        df = pd.DataFrame(data)
        
        # Add temporal relationships using base generator functionality
        df = self._add_temporal_relationships(df, ts_config)
        
        # Apply CRM-specific time series correlations
        df = self._apply_crm_time_series_correlations(df, ts_config)
        
        return self._apply_realistic_patterns(df)
    
    def _generate_time_series_opportunity_record(self, contact_data: Dict[str, Any],
                                               account_data: Dict[str, Any],
                                               timestamp: datetime,
                                               base_deal_value: float) -> Dict[str, Any]:
        """Generate opportunity record with time series patterns"""
        # Use time series base value but adjust for industry and company size
        industry_multiplier = self.industries[account_data['industry']]['deal_multiplier']
        
        # Company size multiplier
        size_multipliers = {
            'startup': 0.5, 'small': 0.8, 'medium': 1.0, 'large': 1.5, 'enterprise': 2.0
        }
        size_multiplier = size_multipliers[account_data['company_size']]
        
        deal_value = int(base_deal_value * industry_multiplier * size_multiplier)
        
        # Determine deal category based on value
        deal_category = 'small'
        for category, data in self.deal_sizes.items():
            if data['range'][0] <= deal_value <= data['range'][1]:
                deal_category = category
                break
        
        # Select stage with time-based progression
        stage = self._select_time_aware_stage(timestamp)
        probability = self.opportunity_stages[stage]['probability']
        
        # Generate dates relative to timestamp
        created_date = timestamp.date() - timedelta(
            days=self.faker.random_int(0, 90)
        )
        
        # Calculate expected close date
        sales_cycle_days = self.industries[account_data['industry']]['sales_cycle_days']
        expected_close_date = created_date + timedelta(days=sales_cycle_days)
        
        # Generate other fields
        lead_source = self.faker.random_element(list(self.lead_sources.keys()))
        rep_experience = self.faker.random_element(list(self.win_rates_by_rep_experience.keys()))
        
        return {
            'opportunity_id': f'OPP_{hash(f"{contact_data["contact_id"]}{timestamp}") % 1000000:06d}',
            'opportunity_name': f'{account_data["company_name"]} - {self.faker.random_element(["Software License", "Consulting Services", "Implementation", "Support Contract", "Upgrade"])}',
            'stage': stage,
            'probability': probability,
            'deal_value': deal_value,
            'deal_category': deal_category,
            'created_date': created_date,
            'expected_close_date': expected_close_date,
            'lead_source': lead_source,
            'sales_rep': self.faker.name(),
            'sales_rep_experience': rep_experience,
            'product_interest': self.faker.random_element([
                'CRM Software', 'Marketing Automation', 'Analytics Platform',
                'Integration Services', 'Training & Support', 'Custom Development'
            ]),
            'competitor': self.faker.random_element([
                'Salesforce', 'HubSpot', 'Microsoft', 'Oracle', 'SAP', 'None', 'Unknown'
            ]),
            'budget_confirmed': self.faker.boolean(chance_of_getting_true=60),
            'decision_maker_identified': self.faker.boolean(chance_of_getting_true=70)
        }    

    def _generate_time_series_interaction_history(self, contact_data: Dict[str, Any],
                                                opportunity_data: Dict[str, Any],
                                                timestamp: datetime) -> Dict[str, Any]:
        """Generate interaction history with temporal patterns"""
        stage = opportunity_data['stage']
        created_date = opportunity_data['created_date']
        
        # Calculate interactions based on time patterns
        stage_config = self.communication_frequency[stage]
        base_interactions = stage_config['interactions_per_week']
        response_rate = stage_config['response_rate']
        
        # Apply time-based multipliers
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Business hours have more interactions
        if 9 <= hour <= 17:
            time_multiplier = 1.2
        else:
            time_multiplier = 0.8
        
        # Weekdays have more interactions
        if day_of_week < 5:
            day_multiplier = 1.0
        else:
            day_multiplier = 0.6
        
        total_interactions = max(1, int(base_interactions * time_multiplier * day_multiplier))
        successful_interactions = int(total_interactions * response_rate)
        
        # Generate interaction type distribution with time awareness
        interaction_counts = self._generate_time_aware_interaction_distribution(
            total_interactions, stage, timestamp
        )
        
        # Calculate last interaction date
        days_since_created = (timestamp.date() - created_date).days
        last_interaction_date = created_date + timedelta(
            days=self.faker.random_int(0, min(days_since_created, 7))
        )
        
        return {
            'total_interactions': total_interactions,
            'successful_interactions': successful_interactions,
            'response_rate': round(response_rate, 2),
            'last_interaction_date': last_interaction_date,
            'last_interaction_type': self.faker.random_element(list(self.interaction_types.keys())),
            'days_since_last_interaction': (timestamp.date() - last_interaction_date).days,
            'email_interactions': interaction_counts.get('email', 0),
            'phone_interactions': interaction_counts.get('phone_call', 0),
            'meeting_interactions': interaction_counts.get('meeting', 0),
            'demo_interactions': interaction_counts.get('demo', 0),
            'next_follow_up_date': timestamp.date() + timedelta(
                days=self.faker.random_int(1, 7)
            )
        }   
 
    def _select_time_aware_stage(self, timestamp: datetime) -> str:
        """Select opportunity stage with time-based patterns"""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        month = timestamp.month
        
        # Business hours tend to have more advanced stages
        if 9 <= hour <= 17 and day_of_week < 5:
            advanced_stages = ['proposal', 'negotiation', 'closed_won']
            if self.faker.random.random() < 0.4:
                return self.faker.random_element(advanced_stages)
        
        # End of quarter (March, June, September, December) - more closed deals
        if month in [3, 6, 9, 12]:
            if self.faker.random.random() < 0.3:
                return self.faker.random_element(['closed_won', 'negotiation'])
        
        # Default stage selection
        stages = list(self.opportunity_stages.keys())
        stage_weights = [1.0 if stage != 'closed_won' else 0.2 for stage in stages]
        
        # Weighted selection
        total_weight = sum(stage_weights)
        rand_val = self.faker.random.uniform(0, total_weight)
        
        cumulative_weight = 0
        for stage, weight in zip(stages, stage_weights):
            cumulative_weight += weight
            if rand_val <= cumulative_weight:
                return stage
        
        return 'lead'  # Fallback
    
    def _generate_time_aware_interaction_distribution(self, total_interactions: int,
                                                    stage: str, timestamp: datetime) -> Dict[str, int]:
        """Generate interaction type distribution with time awareness"""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Base weights by stage
        if stage in ['lead', 'qualified']:
            base_weights = {'email': 0.4, 'phone_call': 0.3, 'social_media': 0.2, 'follow_up': 0.1}
        elif stage in ['proposal', 'negotiation']:
            base_weights = {'meeting': 0.3, 'demo': 0.25, 'proposal_sent': 0.2, 'phone_call': 0.15, 'email': 0.1}
        else:
            base_weights = {'phone_call': 0.4, 'email': 0.3, 'meeting': 0.2, 'follow_up': 0.1}
        
        # Time-based adjustments
        if 9 <= hour <= 17 and day_of_week < 5:  # Business hours
            # More phone calls and meetings
            base_weights['phone_call'] = base_weights.get('phone_call', 0) * 1.3
            base_weights['meeting'] = base_weights.get('meeting', 0) * 1.2
            base_weights['email'] = base_weights.get('email', 0) * 0.9
        else:  # After hours
            # More emails and social media
            base_weights['email'] = base_weights.get('email', 0) * 1.4
            base_weights['social_media'] = base_weights.get('social_media', 0) * 1.3
            base_weights['phone_call'] = base_weights.get('phone_call', 0) * 0.7
        
        # Normalize weights
        total_weight = sum(base_weights.values())
        normalized_weights = {k: v / total_weight for k, v in base_weights.items()}
        
        # Distribute interactions
        interaction_counts = {}
        for interaction_type, weight in normalized_weights.items():
            interaction_counts[interaction_type] = int(total_interactions * weight)
        
        return interaction_counts 
   
    def _apply_crm_time_series_correlations(self, data: pd.DataFrame, ts_config) -> pd.DataFrame:
        """Apply CRM-specific time series correlations"""
        if len(data) < 2:
            return data
        
        # Sort by timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Apply deal value correlation (similar deals cluster in time)
        for i in range(1, len(data)):
            prev_value = data.loc[i-1, 'deal_value']
            curr_value = data.loc[i, 'deal_value']
            
            # Time difference in hours
            time_diff = (data.loc[i, 'timestamp'] - data.loc[i-1, 'timestamp']).total_seconds() / 3600
            
            # Stronger correlation for closer time periods
            if time_diff < 24:  # Same day
                correlation_strength = 0.3
            elif time_diff < 168:  # Same week
                correlation_strength = 0.2
            else:
                correlation_strength = 0.1
            
            # Apply correlation
            if prev_value > 0:
                adjustment_factor = 1 + (correlation_strength * self.faker.random.uniform(-0.2, 0.2))
                new_value = curr_value * adjustment_factor
                
                # Keep within reasonable bounds
                new_value = max(1000, min(new_value, curr_value * 2.0))
                data.loc[i, 'deal_value'] = int(new_value)
        
        # Apply stage progression correlation
        for i in range(1, len(data)):
            prev_stage = data.loc[i-1, 'stage']
            curr_stage = data.loc[i, 'stage']
            
            # If same sales rep, apply stage correlation
            if data.loc[i-1, 'sales_rep'] == data.loc[i, 'sales_rep']:
                if prev_stage in ['proposal', 'negotiation'] and self.faker.random.random() < 0.3:
                    # Successful reps tend to have more advanced deals
                    advanced_stages = ['proposal', 'negotiation', 'closed_won']
                    if curr_stage not in advanced_stages:
                        data.loc[i, 'stage'] = self.faker.random_element(advanced_stages)
                        data.loc[i, 'probability'] = self.opportunity_stages[data.loc[i, 'stage']]['probability']
        
        return data    

    def _apply_realistic_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply additional realistic patterns to CRM data"""
        # Add derived fields
        data['days_in_pipeline'] = (pd.to_datetime('today') - pd.to_datetime(data['created_date'])).dt.days
        
        # Calculate weighted deal value (probability * deal_value)
        data['weighted_deal_value'] = data['deal_value'] * data['probability']
        
        # Add lead quality score based on multiple factors
        data['lead_quality_score'] = self._calculate_lead_quality_score(data)
        
        # Add win probability based on historical patterns
        data['predicted_win_probability'] = self._calculate_win_probability(data)
        
        # Add sales velocity (deal_value / days_in_pipeline)
        data['sales_velocity'] = data['weighted_deal_value'] / (data['days_in_pipeline'] + 1)
        
        # Add engagement score based on interactions
        data['engagement_score'] = self._calculate_engagement_score(data)
        
        # Sort by created_date for realistic chronological order
        data = data.sort_values('created_date').reset_index(drop=True)
        
        return data
    
    def _calculate_lead_quality_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate lead quality score based on multiple factors"""
        scores = []
        
        for _, row in data.iterrows():
            score = 5.0  # Base score
            
            # Lead source impact
            source_multiplier = {
                'referral': 1.4, 'partner': 1.3, 'website': 1.1, 'trade_show': 1.2,
                'cold_outreach': 0.8, 'social_media': 0.9, 'advertising': 1.0, 'other': 0.9
            }
            score *= source_multiplier.get(row['lead_source'], 1.0)
            
            # Company size impact
            size_multiplier = {
                'startup': 0.8, 'small': 0.9, 'medium': 1.0, 'large': 1.2, 'enterprise': 1.4
            }
            score *= size_multiplier.get(row['company_size'], 1.0)
            
            # Seniority level impact
            seniority_multiplier = {
                'Individual Contributor': 0.7, 'Manager': 1.0, 'Director': 1.3,
                'VP': 1.5, 'C-Level': 1.8, 'Owner': 1.6
            }
            score *= seniority_multiplier.get(row['seniority_level'], 1.0)
            
            # Budget and decision maker confirmation
            if row['budget_confirmed']:
                score *= 1.2
            if row['decision_maker_identified']:
                score *= 1.3
            
            # Response rate impact
            if row['response_rate'] > 0.7:
                score *= 1.2
            elif row['response_rate'] < 0.3:
                score *= 0.8
            
            # Cap the score between 1 and 10
            score = max(1.0, min(10.0, score))
            scores.append(round(score, 1))
        
        return pd.Series(scores)   
 
    def _calculate_win_probability(self, data: pd.DataFrame) -> pd.Series:
        """Calculate predicted win probability based on historical patterns"""
        probabilities = []
        
        for _, row in data.iterrows():
            # Start with stage probability
            base_prob = row['probability']
            
            # Special handling for closed_won deals
            if row['stage'] == 'closed_won':
                probabilities.append(1.0)
                continue
            
            # Adjust based on lead source
            source_win_rate = self.win_rates_by_source.get(row['lead_source'], 0.25)
            
            # Adjust based on sales rep experience
            rep_win_rate = self.win_rates_by_rep_experience.get(row['sales_rep_experience'], 0.25)
            
            # Adjust based on deal size (smaller deals close more often)
            size_multiplier = self.deal_sizes[row['deal_category']]['close_rate'] / 0.25  # Normalize to average
            
            # Combine factors
            predicted_prob = (base_prob * 0.4 + source_win_rate * 0.3 + rep_win_rate * 0.3) * size_multiplier
            
            # Adjust based on engagement
            if row['response_rate'] > 0.8:
                predicted_prob *= 1.2
            elif row['response_rate'] < 0.3:
                predicted_prob *= 0.7
            
            # Cap between 0 and 1
            predicted_prob = max(0.0, min(1.0, predicted_prob))
            probabilities.append(round(predicted_prob, 2))
        
        return pd.Series(probabilities)
    
    def _calculate_engagement_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate engagement score based on interaction patterns"""
        scores = []
        
        for _, row in data.iterrows():
            score = 0.0
            
            # Base score from total interactions
            score += min(row['total_interactions'] * 0.5, 5.0)
            
            # Bonus for successful interactions
            score += min(row['successful_interactions'] * 0.8, 5.0)
            
            # Response rate impact
            score += row['response_rate'] * 3.0
            
            # Recency impact (more recent = higher score)
            days_since_last = row['days_since_last_interaction']
            if days_since_last <= 7:
                score += 2.0
            elif days_since_last <= 14:
                score += 1.0
            elif days_since_last > 30:
                score -= 1.0
            
            # High-value interaction types
            if row['meeting_interactions'] > 0:
                score += 1.5
            if row['demo_interactions'] > 0:
                score += 2.0
            
            # Cap between 0 and 10
            score = max(0.0, min(10.0, score))
            scores.append(round(score, 1))
        
        return pd.Series(scores)