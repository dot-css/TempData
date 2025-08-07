"""
Suppliers dataset generator

Generates realistic supplier and vendor data with contract terms, performance metrics,
and product categories for procurement and vendor management systems.
"""

import pandas as pd
import json
import os
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional
from ...core.base_generator import BaseGenerator


class SuppliersGenerator(BaseGenerator):
    """
    Generator for realistic supplier and vendor data
    
    Creates supplier datasets with company information, contract terms, performance metrics,
    product categories, and realistic renewal cycles for procurement systems.
    """
    
    def __init__(self, seeder, locale: str = 'en_US'):
        super().__init__(seeder, locale)
        self._setup_supplier_categories()
        self._setup_contract_data()
        self._setup_performance_patterns()
        self._setup_geographic_data()
        self._setup_product_categories()
        self._setup_service_levels()
    
    def _setup_supplier_categories(self):
        """Setup supplier categories and characteristics"""
        self.supplier_categories = {
            'manufacturing': {
                'weight': 0.25,
                'avg_contract_value': (50000, 500000),
                'typical_contract_length': (12, 36),  # months
                'performance_variance': 0.15,
                'quality_focus': True
            },
            'technology': {
                'weight': 0.20,
                'avg_contract_value': (25000, 250000),
                'typical_contract_length': (6, 24),
                'performance_variance': 0.20,
                'quality_focus': True
            },
            'logistics': {
                'weight': 0.15,
                'avg_contract_value': (30000, 200000),
                'typical_contract_length': (12, 24),
                'performance_variance': 0.25,
                'quality_focus': False
            },
            'professional_services': {
                'weight': 0.12,
                'avg_contract_value': (15000, 150000),
                'typical_contract_length': (3, 18),
                'performance_variance': 0.18,
                'quality_focus': True
            },
            'raw_materials': {
                'weight': 0.10,
                'avg_contract_value': (40000, 300000),
                'typical_contract_length': (6, 12),
                'performance_variance': 0.30,
                'quality_focus': False
            },
            'maintenance': {
                'weight': 0.08,
                'avg_contract_value': (10000, 100000),
                'typical_contract_length': (12, 36),
                'performance_variance': 0.22,
                'quality_focus': False
            },
            'office_supplies': {
                'weight': 0.06,
                'avg_contract_value': (5000, 50000),
                'typical_contract_length': (6, 12),
                'performance_variance': 0.12,
                'quality_focus': False
            },
            'consulting': {
                'weight': 0.04,
                'avg_contract_value': (20000, 200000),
                'typical_contract_length': (3, 12),
                'performance_variance': 0.25,
                'quality_focus': True
            }
        }
    
    def _setup_contract_data(self):
        """Setup contract terms and patterns"""
        self.contract_types = {
            'fixed_price': {
                'weight': 0.40,
                'price_variance': 0.05,
                'renewal_probability': 0.75
            },
            'time_and_materials': {
                'weight': 0.25,
                'price_variance': 0.15,
                'renewal_probability': 0.65
            },
            'cost_plus': {
                'weight': 0.15,
                'price_variance': 0.20,
                'renewal_probability': 0.70
            },
            'unit_price': {
                'weight': 0.12,
                'price_variance': 0.10,
                'renewal_probability': 0.80
            },
            'retainer': {
                'weight': 0.08,
                'price_variance': 0.08,
                'renewal_probability': 0.85
            }
        }
        
        self.payment_terms = {
            'net_30': 0.45,
            'net_60': 0.25,
            'net_15': 0.15,
            'net_90': 0.10,
            'immediate': 0.05
        }
        
        self.contract_statuses = {
            'active': 0.70,
            'pending_renewal': 0.15,
            'expired': 0.08,
            'terminated': 0.04,
            'draft': 0.03
        }
    
    def _setup_performance_patterns(self):
        """Setup supplier performance metrics and patterns"""
        self.performance_metrics = {
            'delivery_performance': {
                'excellent': (95, 100),
                'good': (85, 95),
                'average': (75, 85),
                'poor': (60, 75),
                'unacceptable': (0, 60)
            },
            'quality_score': {
                'excellent': (95, 100),
                'good': (85, 95),
                'average': (75, 85),
                'poor': (60, 75),
                'unacceptable': (0, 60)
            },
            'compliance_score': {
                'excellent': (98, 100),
                'good': (90, 98),
                'average': (80, 90),
                'poor': (70, 80),
                'unacceptable': (0, 70)
            }
        }
        
        self.performance_distribution = {
            'excellent': 0.20,
            'good': 0.45,
            'average': 0.25,
            'poor': 0.08,
            'unacceptable': 0.02
        }
        
        # Performance tends to correlate with contract value and length
        self.performance_correlations = {
            'high_value_contracts': 1.15,  # Better performance for high-value contracts
            'long_term_contracts': 1.10,   # Better performance for long-term relationships
            'quality_focused_categories': 1.12  # Better performance for quality-focused categories
        }
    
    def _setup_geographic_data(self):
        """Setup geographic distribution and regional characteristics"""
        self.regions = {
            'north_america': {
                'weight': 0.45,
                'countries': ['United States', 'Canada', 'Mexico'],
                'cost_multiplier': 1.0,
                'quality_multiplier': 1.0
            },
            'europe': {
                'weight': 0.25,
                'countries': ['Germany', 'United Kingdom', 'France', 'Netherlands', 'Italy'],
                'cost_multiplier': 1.15,
                'quality_multiplier': 1.10
            },
            'asia_pacific': {
                'weight': 0.20,
                'countries': ['China', 'Japan', 'South Korea', 'Singapore', 'Australia'],
                'cost_multiplier': 0.75,
                'quality_multiplier': 0.95
            },
            'latin_america': {
                'weight': 0.06,
                'countries': ['Brazil', 'Argentina', 'Chile', 'Colombia'],
                'cost_multiplier': 0.70,
                'quality_multiplier': 0.85
            },
            'other': {
                'weight': 0.04,
                'countries': ['India', 'South Africa', 'UAE', 'Israel'],
                'cost_multiplier': 0.65,
                'quality_multiplier': 0.80
            }
        }
    
    def _setup_product_categories(self):
        """Setup product categories and subcategories"""
        self.product_categories = {
            'electronics': ['semiconductors', 'circuit_boards', 'sensors', 'displays', 'cables'],
            'mechanical': ['fasteners', 'bearings', 'gears', 'springs', 'gaskets'],
            'software': ['licenses', 'development_tools', 'security_software', 'databases'],
            'services': ['consulting', 'maintenance', 'training', 'support', 'installation'],
            'materials': ['metals', 'plastics', 'chemicals', 'textiles', 'composites'],
            'packaging': ['boxes', 'labels', 'protective_materials', 'containers'],
            'office': ['furniture', 'supplies', 'equipment', 'catering', 'cleaning'],
            'logistics': ['shipping', 'warehousing', 'customs', 'insurance', 'tracking']
        }
    
    def _setup_service_levels(self):
        """Setup service level agreements and characteristics"""
        self.service_levels = {
            'premium': {
                'weight': 0.15,
                'cost_multiplier': 1.25,
                'performance_boost': 1.15,
                'response_time_hours': (1, 4)
            },
            'standard': {
                'weight': 0.60,
                'cost_multiplier': 1.0,
                'performance_boost': 1.0,
                'response_time_hours': (4, 24)
            },
            'basic': {
                'weight': 0.20,
                'cost_multiplier': 0.85,
                'performance_boost': 0.90,
                'response_time_hours': (24, 72)
            },
            'economy': {
                'weight': 0.05,
                'cost_multiplier': 0.70,
                'performance_boost': 0.80,
                'response_time_hours': (72, 168)
            }
        }
    
    def generate(self, rows: int, **kwargs) -> pd.DataFrame:
        """
        Generate suppliers dataset
        
        Args:
            rows: Number of supplier records to generate
            **kwargs: Additional parameters (time_series, date_range, etc.)
            
        Returns:
            pd.DataFrame: Generated supplier data with realistic patterns
        """
        # Check if time series generation is requested
        ts_config = self._create_time_series_config(**kwargs)
        
        if ts_config:
            return self._generate_time_series_suppliers(rows, ts_config, **kwargs)
        else:
            return self._generate_snapshot_suppliers(rows, **kwargs)
    
    def _generate_snapshot_suppliers(self, rows: int, **kwargs) -> pd.DataFrame:
        """Generate snapshot supplier data (random timestamps)"""
        date_range = kwargs.get('date_range', None)
        
        data = []
        
        for i in range(rows):
            # Generate contract start date
            if date_range:
                start_date, end_date = date_range
                contract_date = self.faker.date_between(start_date=start_date, end_date=end_date)
            else:
                contract_date = self.faker.date_between(start_date='-2y', end_date='today')
            
            # Generate supplier record
            supplier = self._generate_supplier_record(i, contract_date)
            data.append(supplier)
        
        df = pd.DataFrame(data)
        return self._apply_realistic_patterns(df)
    
    def _generate_time_series_suppliers(self, rows: int, ts_config, **kwargs) -> pd.DataFrame:
        """Generate time series supplier data using integrated time series system"""
        
        # Generate timestamps from time series config
        timestamps = self._generate_time_series_timestamps(ts_config, rows)
        
        # Create base time series for supplier onboarding activity
        base_onboarding_rate = 2.0  # Base daily supplier onboarding
        
        onboarding_series = self.time_series_generator.generate_time_series_base(
            ts_config,
            base_value=base_onboarding_rate,
            value_range=(base_onboarding_rate * 0.3, base_onboarding_rate * 2.5)
        )
        
        data = []
        
        for i, timestamp in enumerate(timestamps):
            if i >= rows or i >= len(onboarding_series):
                break
            
            # Get time series onboarding intensity
            onboarding_intensity = onboarding_series.iloc[i]['value'] / base_onboarding_rate
            
            # Generate supplier with temporal patterns
            supplier = self._generate_time_series_supplier_record(
                i, timestamp, onboarding_intensity
            )
            
            data.append(supplier)
        
        df = pd.DataFrame(data)
        
        # Add temporal relationships
        df = self._add_temporal_relationships(df, ts_config)
        
        # Apply supplier-specific time series correlations
        df = self._apply_supplier_time_series_correlations(df, ts_config)
        
        return self._apply_realistic_patterns(df)
    
    def _generate_supplier_record(self, supplier_index: int, contract_date: date) -> Dict:
        """Generate a single supplier record"""
        
        # Select supplier category and region
        category = self._select_supplier_category()
        region = self._select_region()
        
        # Get category and region configurations
        category_config = self.supplier_categories[category]
        region_config = self.regions[region]
        
        # Generate company information
        company_name = self._generate_company_name(category)
        country = self.faker.random_element(region_config['countries'])
        
        # Generate contract information
        contract_type = self._select_contract_type()
        contract_value_min, contract_value_max = category_config['avg_contract_value']
        contract_value = round(self.faker.random.uniform(contract_value_min, contract_value_max), 2)
        
        # Apply regional cost adjustments
        contract_value *= region_config['cost_multiplier']
        
        # Ensure minimum contract value
        contract_value = max(contract_value, 5000)
        
        # Generate contract length and dates
        length_min, length_max = category_config['typical_contract_length']
        contract_length_months = self.faker.random_int(length_min, length_max)
        contract_end_date = contract_date + timedelta(days=contract_length_months * 30)
        
        # Generate performance metrics
        performance_tier = self._select_performance_tier(category, contract_value, contract_length_months)
        delivery_performance = self._generate_performance_score('delivery_performance', performance_tier, region_config)
        quality_score = self._generate_performance_score('quality_score', performance_tier, region_config)
        compliance_score = self._generate_performance_score('compliance_score', performance_tier, region_config)
        
        # Ensure performance tier consistency after regional adjustments
        overall_performance = (delivery_performance + quality_score + compliance_score) / 3
        performance_tier = self._determine_performance_tier_from_score(overall_performance)
        
        # Generate service level and contact information
        service_level = self._select_service_level()
        service_config = self.service_levels[service_level]
        
        # Generate product categories
        primary_category = self.faker.random_element(list(self.product_categories.keys()))
        product_subcategories = self.faker.random_elements(
            self.product_categories[primary_category],
            length=self.faker.random_int(1, 3),
            unique=True
        )
        
        # Generate payment terms and status
        payment_terms = self._select_payment_terms()
        contract_status = self._select_contract_status(contract_date, contract_end_date)
        
        # Calculate renewal probability and risk factors
        renewal_probability = self._calculate_renewal_probability(
            performance_tier, contract_type, contract_length_months, delivery_performance
        )
        
        # Generate response time based on service level
        response_time_min, response_time_max = service_config['response_time_hours']
        avg_response_time = self.faker.random.uniform(response_time_min, response_time_max)
        
        return {
            'supplier_id': f'SUP_{supplier_index+1:06d}',
            'company_name': company_name,
            'category': category,
            'primary_product_category': primary_category,
            'product_subcategories': ', '.join(product_subcategories),
            'country': country,
            'region': region,
            'contact_name': self.faker.name(),
            'contact_email': self._generate_business_email(company_name),
            'contact_phone': self.faker.phone_number(),
            'address': self.faker.address().replace('\n', ', '),
            'contract_type': contract_type,
            'contract_value': contract_value,
            'contract_start_date': contract_date,
            'contract_end_date': contract_end_date,
            'contract_length_months': contract_length_months,
            'contract_status': contract_status,
            'service_level': service_level,
            'payment_terms': payment_terms,
            'delivery_performance': round(delivery_performance, 1),
            'quality_score': round(quality_score, 1),
            'compliance_score': round(compliance_score, 1),
            'overall_performance': round((delivery_performance + quality_score + compliance_score) / 3, 1),
            'performance_tier': performance_tier,
            'avg_response_time_hours': round(avg_response_time, 1),
            'renewal_probability': round(renewal_probability, 3),
            'last_audit_date': self._generate_audit_date(contract_date),
            'certification_status': self._generate_certification_status(),
            'risk_level': self._calculate_risk_level(delivery_performance, quality_score, compliance_score),
            'preferred_supplier': self._is_preferred_supplier(performance_tier, contract_value),
            'onboarding_date': contract_date,
            'last_performance_review': self._generate_review_date(contract_date)
        }
    
    def _generate_time_series_supplier_record(self, supplier_index: int, 
                                            timestamp: datetime, 
                                            onboarding_intensity: float) -> Dict:
        """Generate time series supplier record with temporal patterns"""
        
        # Generate base supplier record
        supplier = self._generate_supplier_record(supplier_index, timestamp.date())
        
        # Apply time-based onboarding patterns
        contract_date = timestamp.date()
        
        # Adjust contract value based on market conditions (time-based)
        market_adjustment = self._apply_market_conditions(timestamp, onboarding_intensity)
        supplier['contract_value'] *= market_adjustment
        supplier['contract_value'] = round(supplier['contract_value'], 2)
        
        # Apply temporal performance patterns
        temporal_performance = self._apply_temporal_performance_patterns(
            supplier['performance_tier'], timestamp
        )
        
        supplier['delivery_performance'] *= temporal_performance
        supplier['quality_score'] *= temporal_performance
        supplier['compliance_score'] *= temporal_performance
        
        # Recalculate overall performance
        supplier['overall_performance'] = round(
            (supplier['delivery_performance'] + supplier['quality_score'] + supplier['compliance_score']) / 3, 1
        )
        
        # Update time-sensitive fields
        supplier.update({
            'onboarding_datetime': timestamp,
            'contract_start_date': contract_date,
            'onboarding_intensity': round(onboarding_intensity, 3),
            'market_adjustment_factor': round(market_adjustment, 3)
        })
        
        return supplier
    
    def _select_supplier_category(self) -> str:
        """Select supplier category based on distribution"""
        choices = list(self.supplier_categories.keys())
        weights = [self.supplier_categories[cat]['weight'] for cat in choices]
        
        return self._weighted_choice(choices, weights)
    
    def _select_region(self) -> str:
        """Select region based on distribution"""
        choices = list(self.regions.keys())
        weights = [self.regions[region]['weight'] for region in choices]
        
        return self._weighted_choice(choices, weights)
    
    def _select_contract_type(self) -> str:
        """Select contract type based on distribution"""
        choices = list(self.contract_types.keys())
        weights = [self.contract_types[ct]['weight'] for ct in choices]
        
        return self._weighted_choice(choices, weights)
    
    def _select_performance_tier(self, category: str, contract_value: float, contract_length: int) -> str:
        """Select performance tier with correlations"""
        choices = list(self.performance_distribution.keys())
        weights = list(self.performance_distribution.values())
        
        # Apply correlations
        category_config = self.supplier_categories[category]
        
        # High-value contracts tend to have better performance
        if contract_value > 100000:
            weights = self._adjust_weights_for_better_performance(weights)
        
        # Long-term contracts tend to have better performance
        if contract_length > 24:
            weights = self._adjust_weights_for_better_performance(weights)
        
        # Quality-focused categories have better performance
        if category_config['quality_focus']:
            weights = self._adjust_weights_for_better_performance(weights)
        
        return self._weighted_choice(choices, weights)
    
    def _determine_performance_tier_from_score(self, overall_score: float) -> str:
        """Determine performance tier based on overall score"""
        if overall_score >= 95:
            return 'excellent'
        elif overall_score >= 85:
            return 'good'
        elif overall_score >= 75:
            return 'average'
        elif overall_score >= 60:
            return 'poor'
        else:
            return 'unacceptable'
    
    def _generate_performance_score(self, metric: str, tier: str, region_config: Dict) -> float:
        """Generate performance score for a specific metric"""
        score_min, score_max = self.performance_metrics[metric][tier]
        base_score = self.faker.random.uniform(score_min, score_max)
        
        # Apply regional quality multiplier
        adjusted_score = base_score * region_config['quality_multiplier']
        
        return min(100.0, max(0.0, adjusted_score))
    
    def _select_service_level(self) -> str:
        """Select service level based on distribution"""
        choices = list(self.service_levels.keys())
        weights = [self.service_levels[sl]['weight'] for sl in choices]
        
        return self._weighted_choice(choices, weights)
    
    def _select_payment_terms(self) -> str:
        """Select payment terms based on distribution"""
        choices = list(self.payment_terms.keys())
        weights = list(self.payment_terms.values())
        
        return self._weighted_choice(choices, weights)
    
    def _select_contract_status(self, start_date: date, end_date: date) -> str:
        """Select contract status based on dates"""
        today = datetime.now().date()
        
        if start_date > today:
            return 'draft'
        elif end_date < today:
            # Contract has expired
            if self.faker.random.random() < 0.7:
                return 'expired'
            else:
                return 'terminated'
        elif (end_date - today).days < 90:
            # Contract expires soon
            return 'pending_renewal'
        else:
            return 'active'
    
    def _generate_company_name(self, category: str) -> str:
        """Generate realistic company names based on category"""
        prefixes = {
            'manufacturing': ['Industrial', 'Manufacturing', 'Production', 'Assembly', 'Precision'],
            'technology': ['Tech', 'Digital', 'Systems', 'Solutions', 'Innovation'],
            'logistics': ['Logistics', 'Transport', 'Shipping', 'Freight', 'Supply'],
            'professional_services': ['Consulting', 'Advisory', 'Professional', 'Strategic', 'Expert'],
            'raw_materials': ['Materials', 'Resources', 'Supply', 'Industrial', 'Global'],
            'maintenance': ['Maintenance', 'Service', 'Repair', 'Technical', 'Support'],
            'office_supplies': ['Office', 'Business', 'Corporate', 'Workplace', 'Commercial'],
            'consulting': ['Consulting', 'Advisory', 'Strategic', 'Management', 'Business']
        }
        
        suffixes = ['Corp', 'Inc', 'LLC', 'Ltd', 'Group', 'Solutions', 'Services', 'Systems', 'Partners', 'Co']
        
        category_prefixes = prefixes.get(category, ['Global', 'International', 'Premier'])
        prefix = self.faker.random_element(category_prefixes)
        suffix = self.faker.random_element(suffixes)
        
        # Sometimes add a descriptive word
        if self.faker.random.random() < 0.3:
            descriptors = ['Advanced', 'Premier', 'Elite', 'Superior', 'Quality', 'Reliable', 'Trusted']
            descriptor = self.faker.random_element(descriptors)
            return f"{descriptor} {prefix} {suffix}"
        else:
            return f"{prefix} {suffix}"
    
    def _generate_business_email(self, company_name: str) -> str:
        """Generate business email based on company name"""
        # Extract first word from company name for domain
        domain_base = company_name.split()[0].lower().replace(',', '')
        
        # Common business email patterns
        patterns = [
            f"contact@{domain_base}.com",
            f"info@{domain_base}.com",
            f"sales@{domain_base}.com",
            f"procurement@{domain_base}.com"
        ]
        
        return self.faker.random_element(patterns)
    
    def _calculate_renewal_probability(self, performance_tier: str, contract_type: str, 
                                     contract_length: int, delivery_performance: float) -> float:
        """Calculate contract renewal probability"""
        # Base probability from contract type
        base_prob = self.contract_types[contract_type]['renewal_probability']
        
        # Performance tier adjustments
        performance_adjustments = {
            'excellent': 1.15,
            'good': 1.05,
            'average': 1.0,
            'poor': 0.85,
            'unacceptable': 0.60
        }
        
        performance_mult = performance_adjustments.get(performance_tier, 1.0)
        
        # Contract length adjustment (longer contracts have higher renewal rates)
        length_mult = 1.0 + (contract_length - 12) * 0.01
        
        # Delivery performance adjustment
        delivery_mult = 0.8 + (delivery_performance / 100) * 0.4
        
        final_prob = base_prob * performance_mult * length_mult * delivery_mult
        
        return min(1.0, max(0.0, final_prob))
    
    def _generate_audit_date(self, contract_date: date) -> date:
        """Generate last audit date"""
        # Audits typically happen 3-12 months after contract start
        days_after_start = self.faker.random_int(90, 365)
        audit_date = contract_date + timedelta(days=days_after_start)
        
        # Don't audit in the future
        today = datetime.now().date()
        if audit_date > today:
            # If calculated audit date is in future, set it to a random date between contract start and today
            days_since_contract = (today - contract_date).days
            if days_since_contract > 30:  # Only if contract has been active for at least 30 days
                random_days = self.faker.random_int(30, days_since_contract)
                audit_date = contract_date + timedelta(days=random_days)
            else:
                # For very recent contracts, set audit date to contract start date
                audit_date = contract_date
        
        return audit_date
    
    def _generate_certification_status(self) -> str:
        """Generate certification status"""
        statuses = {
            'iso_9001': 0.35,
            'iso_14001': 0.20,
            'iso_45001': 0.15,
            'multiple_certifications': 0.15,
            'pending_certification': 0.08,
            'no_certification': 0.07
        }
        
        choices = list(statuses.keys())
        weights = list(statuses.values())
        
        return self._weighted_choice(choices, weights)
    
    def _calculate_risk_level(self, delivery: float, quality: float, compliance: float) -> str:
        """Calculate risk level based on performance scores"""
        avg_score = (delivery + quality + compliance) / 3
        
        if avg_score >= 90:
            return 'low'
        elif avg_score >= 75:
            return 'medium'
        elif avg_score >= 60:
            return 'high'
        else:
            return 'critical'
    
    def _is_preferred_supplier(self, performance_tier: str, contract_value: float) -> bool:
        """Determine if supplier is preferred based on performance and value"""
        if performance_tier in ['excellent', 'good'] and contract_value > 50000:
            return self.faker.random.random() < 0.8
        elif performance_tier == 'excellent':
            return self.faker.random.random() < 0.6
        elif performance_tier == 'good' and contract_value > 25000:
            return self.faker.random.random() < 0.4
        else:
            return self.faker.random.random() < 0.1
    
    def _generate_review_date(self, contract_date: date) -> date:
        """Generate last performance review date"""
        # Reviews typically happen quarterly or semi-annually
        months_after_start = self.faker.random_element([3, 6, 9, 12])
        review_date = contract_date + timedelta(days=months_after_start * 30)
        
        # Don't review in the future
        today = datetime.now().date()
        if review_date > today:
            # If calculated review date is in future, set it to a random date between contract start and today
            days_since_contract = (today - contract_date).days
            if days_since_contract > 30:  # Only if contract has been active for at least 30 days
                random_days = self.faker.random_int(30, days_since_contract)
                review_date = contract_date + timedelta(days=random_days)
            else:
                # For very recent contracts, set review date to contract start date
                review_date = contract_date
        
        return review_date
    
    def _weighted_choice(self, choices: List[str], weights: List[float]) -> str:
        """Make a weighted random choice"""
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
    
    def _adjust_weights_for_better_performance(self, weights: List[float]) -> List[float]:
        """Adjust weights to favor better performance tiers"""
        # Increase weights for excellent and good, decrease for poor and unacceptable
        adjusted = weights.copy()
        adjusted[0] *= 1.5  # excellent
        adjusted[1] *= 1.2  # good
        adjusted[2] *= 1.0  # average
        adjusted[3] *= 0.7  # poor
        adjusted[4] *= 0.4  # unacceptable
        
        return adjusted
    
    def _apply_market_conditions(self, timestamp: datetime, intensity: float) -> float:
        """Apply market condition adjustments based on time"""
        # Seasonal patterns in procurement
        month = timestamp.month
        
        # Q4 and Q1 tend to have higher contract values (budget cycles)
        if month in [10, 11, 12, 1]:
            seasonal_mult = 1.1
        elif month in [6, 7, 8]:  # Summer slowdown
            seasonal_mult = 0.95
        else:
            seasonal_mult = 1.0
        
        # Apply intensity factor
        intensity_mult = 0.9 + (intensity * 0.2)  # Range: 0.9 to 1.1
        
        return seasonal_mult * intensity_mult
    
    def _apply_temporal_performance_patterns(self, performance_tier: str, timestamp: datetime) -> float:
        """Apply temporal patterns to performance metrics"""
        # Performance tends to improve over time (learning curve)
        days_since_epoch = (timestamp - datetime(2020, 1, 1)).days
        improvement_factor = 1.0 + (days_since_epoch / 365) * 0.02  # 2% improvement per year
        
        # Seasonal performance variations
        month = timestamp.month
        if month in [12, 1]:  # Holiday season - slight performance dip
            seasonal_mult = 0.98
        elif month in [6, 7]:  # Summer - vacation impact
            seasonal_mult = 0.97
        else:
            seasonal_mult = 1.0
        
        # Performance tier affects temporal stability
        tier_stability = {
            'excellent': 0.98,  # More stable
            'good': 0.95,
            'average': 0.92,
            'poor': 0.88,
            'unacceptable': 0.85  # Less stable
        }
        
        stability = tier_stability.get(performance_tier, 0.95)
        
        return improvement_factor * seasonal_mult * stability
    
    def _apply_supplier_time_series_correlations(self, data: pd.DataFrame, ts_config) -> pd.DataFrame:
        """Apply realistic time-based correlations for supplier data"""
        if len(data) < 2:
            return data
        
        # Sort by timestamp to ensure proper time series order
        data = data.sort_values('onboarding_datetime').reset_index(drop=True)
        
        # Apply temporal correlations for contract values and performance
        for i in range(1, len(data)):
            prev_performance = data.iloc[i-1]['overall_performance']
            prev_contract_value = data.iloc[i-1]['contract_value']
            
            # Performance momentum (good suppliers tend to maintain performance)
            if prev_performance > 85:
                correlation_strength = 0.2
                performance_boost = 1 + (correlation_strength * self.faker.random.uniform(0, 0.1))
                data.loc[i, 'overall_performance'] *= performance_boost
                data.loc[i, 'delivery_performance'] *= performance_boost
                data.loc[i, 'quality_score'] *= performance_boost
                data.loc[i, 'compliance_score'] *= performance_boost
            
            # Contract value clustering (similar value contracts tend to cluster)
            if prev_contract_value > 0:
                correlation_strength = 0.15
                value_adjustment = 1 + (correlation_strength * self.faker.random.uniform(-0.1, 0.1))
                new_value = data.iloc[i]['contract_value'] * value_adjustment
                data.loc[i, 'contract_value'] = max(5000, min(new_value, data.iloc[i]['contract_value'] * 1.3))
        
        return data
    
    def _apply_realistic_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply additional realistic patterns to supplier data"""
        
        # Add derived fields
        if 'onboarding_datetime' in data.columns:
            data['onboarding_day_of_week'] = pd.to_datetime(data['onboarding_datetime']).dt.day_name()
            data['onboarding_month'] = pd.to_datetime(data['onboarding_datetime']).dt.month
            data['onboarding_quarter'] = pd.to_datetime(data['onboarding_datetime']).dt.quarter
        else:
            data['contract_day_of_week'] = pd.to_datetime(data['contract_start_date']).dt.day_name()
            data['contract_month'] = pd.to_datetime(data['contract_start_date']).dt.month
            data['contract_quarter'] = pd.to_datetime(data['contract_start_date']).dt.quarter
        
        # Calculate additional metrics
        data['contract_value_per_month'] = round(data['contract_value'] / data['contract_length_months'], 2)
        data['days_until_expiry'] = (pd.to_datetime(data['contract_end_date']) - datetime.now()).dt.days
        data['contract_active_days'] = (datetime.now() - pd.to_datetime(data['contract_start_date'])).dt.days
        
        # Add performance categories
        data['performance_category'] = pd.cut(
            data['overall_performance'],
            bins=[0, 60, 75, 85, 95, 100],
            labels=['unacceptable', 'poor', 'average', 'good', 'excellent']
        )
        
        # Add contract value tiers
        data['contract_value_tier'] = pd.cut(
            data['contract_value'],
            bins=[0, 25000, 100000, 250000, float('inf')],
            labels=['small', 'medium', 'large', 'enterprise']
        )
        
        # Add risk indicators
        data['high_risk'] = (
            (data['overall_performance'] < 75) |
            (data['risk_level'].isin(['high', 'critical'])) |
            (data['days_until_expiry'] < 30)
        )
        
        # Add renewal urgency
        data['renewal_urgent'] = (
            (data['contract_status'] == 'pending_renewal') &
            (data['days_until_expiry'] < 60)
        )
        
        return data