"""
Employee dataset generator

Generates realistic employee data with hierarchical reporting structures,
departments, roles, salaries, and performance ratings.
"""

import pandas as pd
import json
import os
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional
from ...core.base_generator import BaseGenerator


class EmployeesGenerator(BaseGenerator):
    """
    Generator for realistic employee database
    
    Creates employee datasets with hierarchical reporting structures, 
    department organization, realistic salary distributions, performance ratings,
    and tenure patterns.
    """
    
    def __init__(self, seeder, locale: str = 'en_US'):
        super().__init__(seeder, locale)
        self._setup_department_data()
        self._setup_role_hierarchy()
        self._setup_salary_data()
        self._setup_performance_patterns()
        self._setup_tenure_patterns()
        self._setup_location_data()
    
    def _setup_department_data(self):
        """Setup department structure and characteristics"""
        self.departments = {
            'engineering': {
                'size_weight': 0.25,
                'avg_salary_multiplier': 1.3,
                'roles': ['software_engineer', 'senior_engineer', 'lead_engineer', 'engineering_manager', 'cto'],
                'performance_variance': 0.15
            },
            'sales': {
                'size_weight': 0.20,
                'avg_salary_multiplier': 1.1,
                'roles': ['sales_rep', 'senior_sales_rep', 'sales_manager', 'sales_director', 'vp_sales'],
                'performance_variance': 0.25
            },
            'marketing': {
                'size_weight': 0.15,
                'avg_salary_multiplier': 1.0,
                'roles': ['marketing_specialist', 'marketing_manager', 'senior_marketing_manager', 'marketing_director', 'cmo'],
                'performance_variance': 0.18
            },
            'operations': {
                'size_weight': 0.12,
                'avg_salary_multiplier': 0.95,
                'roles': ['operations_specialist', 'operations_manager', 'senior_operations_manager', 'operations_director', 'coo'],
                'performance_variance': 0.12
            },
            'finance': {
                'size_weight': 0.10,
                'avg_salary_multiplier': 1.15,
                'roles': ['financial_analyst', 'senior_analyst', 'finance_manager', 'finance_director', 'cfo'],
                'performance_variance': 0.10
            },
            'human_resources': {
                'size_weight': 0.08,
                'avg_salary_multiplier': 0.9,
                'roles': ['hr_specialist', 'hr_manager', 'senior_hr_manager', 'hr_director', 'chro'],
                'performance_variance': 0.14
            },
            'customer_support': {
                'size_weight': 0.10,
                'avg_salary_multiplier': 0.8,
                'roles': ['support_rep', 'senior_support_rep', 'support_manager', 'support_director'],
                'performance_variance': 0.20
            }
        }
    
    def _setup_role_hierarchy(self):
        """Setup role hierarchy and characteristics"""
        self.role_levels = {
            # Individual Contributors (Level 1-3)
            'software_engineer': {'level': 1, 'base_salary': 75000, 'reports_to': 'lead_engineer'},
            'sales_rep': {'level': 1, 'base_salary': 55000, 'reports_to': 'sales_manager'},
            'marketing_specialist': {'level': 1, 'base_salary': 50000, 'reports_to': 'marketing_manager'},
            'operations_specialist': {'level': 1, 'base_salary': 48000, 'reports_to': 'operations_manager'},
            'financial_analyst': {'level': 1, 'base_salary': 60000, 'reports_to': 'finance_manager'},
            'hr_specialist': {'level': 1, 'base_salary': 45000, 'reports_to': 'hr_manager'},
            'support_rep': {'level': 1, 'base_salary': 40000, 'reports_to': 'support_manager'},
            
            # Senior Individual Contributors (Level 2-3)
            'senior_engineer': {'level': 2, 'base_salary': 95000, 'reports_to': 'lead_engineer'},
            'senior_sales_rep': {'level': 2, 'base_salary': 70000, 'reports_to': 'sales_manager'},
            'senior_analyst': {'level': 2, 'base_salary': 75000, 'reports_to': 'finance_manager'},
            'senior_support_rep': {'level': 2, 'base_salary': 50000, 'reports_to': 'support_manager'},
            'senior_marketing_manager': {'level': 3, 'base_salary': 85000, 'reports_to': 'marketing_director'},
            'senior_operations_manager': {'level': 3, 'base_salary': 80000, 'reports_to': 'operations_director'},
            'senior_hr_manager': {'level': 3, 'base_salary': 75000, 'reports_to': 'hr_director'},
            
            # Team Leads and Managers (Level 3-4)
            'lead_engineer': {'level': 3, 'base_salary': 115000, 'reports_to': 'engineering_manager'},
            'sales_manager': {'level': 3, 'base_salary': 85000, 'reports_to': 'sales_director'},
            'marketing_manager': {'level': 3, 'base_salary': 70000, 'reports_to': 'marketing_director'},
            'operations_manager': {'level': 3, 'base_salary': 68000, 'reports_to': 'operations_director'},
            'finance_manager': {'level': 3, 'base_salary': 80000, 'reports_to': 'finance_director'},
            'hr_manager': {'level': 3, 'base_salary': 65000, 'reports_to': 'hr_director'},
            'support_manager': {'level': 3, 'base_salary': 60000, 'reports_to': 'support_director'},
            'engineering_manager': {'level': 4, 'base_salary': 140000, 'reports_to': 'cto'},
            
            # Directors (Level 4-5)
            'sales_director': {'level': 4, 'base_salary': 120000, 'reports_to': 'vp_sales'},
            'marketing_director': {'level': 4, 'base_salary': 110000, 'reports_to': 'cmo'},
            'operations_director': {'level': 4, 'base_salary': 105000, 'reports_to': 'coo'},
            'finance_director': {'level': 4, 'base_salary': 115000, 'reports_to': 'cfo'},
            'hr_director': {'level': 4, 'base_salary': 100000, 'reports_to': 'chro'},
            'support_director': {'level': 4, 'base_salary': 95000, 'reports_to': 'coo'},
            
            # VPs and C-Suite (Level 5-6)
            'vp_sales': {'level': 5, 'base_salary': 180000, 'reports_to': 'ceo'},
            'cto': {'level': 5, 'base_salary': 200000, 'reports_to': 'ceo'},
            'cmo': {'level': 5, 'base_salary': 170000, 'reports_to': 'ceo'},
            'coo': {'level': 5, 'base_salary': 190000, 'reports_to': 'ceo'},
            'cfo': {'level': 5, 'base_salary': 185000, 'reports_to': 'ceo'},
            'chro': {'level': 5, 'base_salary': 160000, 'reports_to': 'ceo'},
            'ceo': {'level': 6, 'base_salary': 250000, 'reports_to': None}
        }
    
    def _setup_salary_data(self):
        """Setup salary distribution patterns"""
        # Location-based salary multipliers
        self.location_salary_multipliers = {
            'san_francisco': 1.4,
            'new_york': 1.3,
            'seattle': 1.25,
            'boston': 1.2,
            'los_angeles': 1.15,
            'chicago': 1.1,
            'austin': 1.05,
            'denver': 1.0,
            'atlanta': 0.95,
            'phoenix': 0.9,
            'remote': 1.1
        }
        
        # Experience-based salary adjustments
        self.experience_multipliers = {
            (0, 1): 0.8,    # 0-1 years
            (1, 3): 0.9,    # 1-3 years
            (3, 5): 1.0,    # 3-5 years
            (5, 8): 1.15,   # 5-8 years
            (8, 12): 1.3,   # 8-12 years
            (12, 20): 1.45, # 12-20 years
            (20, 30): 1.5   # 20+ years
        }
        
        # Performance-based salary adjustments
        self.performance_salary_multipliers = {
            'exceeds_expectations': 1.1,
            'meets_expectations': 1.0,
            'below_expectations': 0.95,
            'needs_improvement': 0.9
        }
    
    def _setup_performance_patterns(self):
        """Setup performance rating patterns"""
        self.performance_ratings = {
            'exceeds_expectations': 0.15,
            'meets_expectations': 0.70,
            'below_expectations': 0.12,
            'needs_improvement': 0.03
        }
        
        # Performance tends to correlate with tenure (up to a point)
        self.tenure_performance_correlation = {
            (0, 0.5): {'exceeds': 0.05, 'meets': 0.60, 'below': 0.25, 'needs': 0.10},  # New hires
            (0.5, 2): {'exceeds': 0.10, 'meets': 0.70, 'below': 0.15, 'needs': 0.05},  # 6mo-2yr
            (2, 5): {'exceeds': 0.20, 'meets': 0.75, 'below': 0.04, 'needs': 0.01},    # 2-5yr
            (5, 10): {'exceeds': 0.25, 'meets': 0.70, 'below': 0.04, 'needs': 0.01},   # 5-10yr
            (10, 20): {'exceeds': 0.15, 'meets': 0.75, 'below': 0.08, 'needs': 0.02},  # 10-20yr
            (20, 40): {'exceeds': 0.10, 'meets': 0.80, 'below': 0.08, 'needs': 0.02}   # 20+yr
        }
    
    def _setup_tenure_patterns(self):
        """Setup employee tenure patterns"""
        # Tenure distribution by role level
        self.tenure_by_level = {
            1: {'mean_years': 2.5, 'std_years': 1.8},  # Individual contributors
            2: {'mean_years': 4.0, 'std_years': 2.5},  # Senior ICs
            3: {'mean_years': 6.0, 'std_years': 3.0},  # Managers
            4: {'mean_years': 8.0, 'std_years': 3.5},  # Directors
            5: {'mean_years': 10.0, 'std_years': 4.0}, # VPs
            6: {'mean_years': 12.0, 'std_years': 5.0}  # C-Suite
        }
        
        # Turnover risk factors
        self.turnover_risk_factors = {
            'low_performance': 0.3,
            'below_market_salary': 0.25,
            'long_tenure_no_promotion': 0.2,
            'new_hire_adjustment': 0.15
        }
    
    def _setup_location_data(self):
        """Setup office location data"""
        self.office_locations = {
            'san_francisco': {'weight': 0.20, 'timezone': 'PST'},
            'new_york': {'weight': 0.18, 'timezone': 'EST'},
            'seattle': {'weight': 0.12, 'timezone': 'PST'},
            'chicago': {'weight': 0.10, 'timezone': 'CST'},
            'austin': {'weight': 0.08, 'timezone': 'CST'},
            'boston': {'weight': 0.07, 'timezone': 'EST'},
            'los_angeles': {'weight': 0.06, 'timezone': 'PST'},
            'denver': {'weight': 0.05, 'timezone': 'MST'},
            'atlanta': {'weight': 0.04, 'timezone': 'EST'},
            'phoenix': {'weight': 0.03, 'timezone': 'MST'},
            'remote': {'weight': 0.07, 'timezone': 'Various'}
        }
    
    def generate(self, rows: int, **kwargs) -> pd.DataFrame:
        """
        Generate employee dataset
        
        Args:
            rows: Number of employees to generate
            **kwargs: Additional parameters (time_series, date_range, etc.)
            
        Returns:
            pd.DataFrame: Generated employee data with realistic patterns
        """
        # Check if time series generation is requested
        ts_config = self._create_time_series_config(**kwargs)
        
        if ts_config:
            return self._generate_time_series_employees(rows, ts_config, **kwargs)
        else:
            return self._generate_snapshot_employees(rows, **kwargs)
    
    def _generate_snapshot_employees(self, rows: int, **kwargs) -> pd.DataFrame:
        """Generate snapshot employee data"""
        data = []
        
        # First, determine organizational structure
        org_structure = self._create_organizational_structure(rows)
        
        for i, employee_info in enumerate(org_structure):
            employee = self._generate_employee_record(i, employee_info, **kwargs)
            data.append(employee)
        
        df = pd.DataFrame(data)
        return self._apply_realistic_patterns(df)
    
    def _generate_time_series_employees(self, rows: int, ts_config, **kwargs) -> pd.DataFrame:
        """Generate time series employee data using integrated time series system"""
        
        # Generate timestamps from time series config
        timestamps = self._generate_time_series_timestamps(ts_config, rows)
        
        # Create base time series for hiring activity
        base_hiring_rate = 5.0  # Base daily hires
        
        hiring_series = self.time_series_generator.generate_time_series_base(
            ts_config,
            base_value=base_hiring_rate,
            value_range=(base_hiring_rate * 0.2, base_hiring_rate * 3.0)
        )
        
        data = []
        org_structure = self._create_organizational_structure(rows)
        
        for i, timestamp in enumerate(timestamps):
            if i >= rows or i >= len(hiring_series) or i >= len(org_structure):
                break
            
            # Get time series hiring intensity
            hiring_intensity = hiring_series.iloc[i]['value'] / base_hiring_rate
            
            # Generate employee with temporal patterns
            employee = self._generate_time_series_employee_record(
                i, org_structure[i], timestamp, hiring_intensity, **kwargs
            )
            
            data.append(employee)
        
        df = pd.DataFrame(data)
        
        # Add temporal relationships
        df = self._add_temporal_relationships(df, ts_config)
        
        # Apply employee-specific time series correlations
        df = self._apply_employee_time_series_correlations(df, ts_config)
        
        return self._apply_realistic_patterns(df)    

    def _create_organizational_structure(self, total_employees: int) -> List[Dict]:
        """Create realistic organizational structure with proper hierarchy"""
        org_structure = []
        
        # Calculate department sizes
        dept_sizes = {}
        remaining_employees = total_employees
        
        for dept, config in self.departments.items():
            size = max(1, int(total_employees * config['size_weight']))
            dept_sizes[dept] = min(size, remaining_employees)
            remaining_employees -= dept_sizes[dept]
        
        # Distribute any remaining employees
        if remaining_employees > 0:
            largest_dept = max(dept_sizes.keys(), key=lambda k: dept_sizes[k])
            dept_sizes[largest_dept] += remaining_employees
        
        # Create hierarchy for each department
        employee_id = 1
        for dept, size in dept_sizes.items():
            dept_structure = self._create_department_hierarchy(dept, size, employee_id)
            org_structure.extend(dept_structure)
            employee_id += len(dept_structure)
        
        return org_structure
    
    def _create_department_hierarchy(self, department: str, size: int, start_id: int) -> List[Dict]:
        """Create hierarchy for a specific department"""
        dept_config = self.departments[department]
        roles = dept_config['roles']
        
        # Determine role distribution (pyramid structure)
        role_distribution = self._calculate_role_distribution(roles, size)
        
        dept_employees = []
        current_id = start_id
        
        # Create employees by role level (top-down)
        role_hierarchy = sorted(roles, key=lambda r: self.role_levels[r]['level'], reverse=True)
        
        for role in role_hierarchy:
            count = role_distribution.get(role, 0)
            if count == 0:
                continue
                
            for _ in range(count):
                employee_info = {
                    'employee_id': current_id,
                    'department': department,
                    'role': role,
                    'level': self.role_levels[role]['level'],
                    'reports_to_role': self.role_levels[role]['reports_to']
                }
                dept_employees.append(employee_info)
                current_id += 1
        
        # Assign manager relationships
        dept_employees = self._assign_manager_relationships(dept_employees)
        
        return dept_employees
    
    def _calculate_role_distribution(self, roles: List[str], total_size: int) -> Dict[str, int]:
        """Calculate how many employees should be in each role"""
        distribution = {}
        
        # Sort roles by level (highest first)
        sorted_roles = sorted(roles, key=lambda r: self.role_levels[r]['level'], reverse=True)
        
        remaining = total_size
        
        # Calculate distribution based on pyramid structure
        for i, role in enumerate(sorted_roles):
            level = self.role_levels[role]['level']
            
            if remaining <= 0:
                distribution[role] = 0
                continue
            
            if i == len(sorted_roles) - 1:
                # Last role (lowest level) gets all remaining
                distribution[role] = remaining
            else:
                # Higher levels get fewer people (pyramid structure)
                if level >= 5:  # C-suite/VP level
                    count = min(1, remaining)
                elif level == 4:  # Director level
                    count = min(max(1, total_size // 25), remaining)
                elif level == 3:  # Manager level
                    count = min(max(1, total_size // 10), remaining)
                elif level == 2:  # Senior IC level
                    count = min(max(1, total_size // 5), remaining)
                else:  # IC level (level 1)
                    count = remaining
                
                distribution[role] = count
                remaining -= count
        
        # Ensure we don't exceed total size
        total_assigned = sum(distribution.values())
        if total_assigned > total_size:
            # Reduce from highest levels first
            excess = total_assigned - total_size
            for role in sorted_roles:
                if excess <= 0:
                    break
                reduction = min(distribution[role], excess)
                distribution[role] -= reduction
                excess -= reduction
        
        return distribution
    
    def _assign_manager_relationships(self, dept_employees: List[Dict]) -> List[Dict]:
        """Assign manager-employee relationships within department"""
        # Create lookup by role
        employees_by_role = {}
        for emp in dept_employees:
            role = emp['role']
            if role not in employees_by_role:
                employees_by_role[role] = []
            employees_by_role[role].append(emp)
        
        # Assign managers (avoid circular references)
        for emp in dept_employees:
            reports_to_role = emp['reports_to_role']
            
            if reports_to_role and reports_to_role in employees_by_role:
                # Find a manager that is not the employee themselves
                managers = employees_by_role[reports_to_role]
                suitable_manager = None
                
                for manager in managers:
                    if manager['employee_id'] != emp['employee_id']:
                        suitable_manager = manager
                        break
                
                if suitable_manager:
                    emp['manager_id'] = suitable_manager['employee_id']
                else:
                    emp['manager_id'] = None
            else:
                emp['manager_id'] = None
        
        return dept_employees
    
    def _generate_employee_record(self, index: int, employee_info: Dict, **kwargs) -> Dict:
        """Generate a single employee record"""
        
        # Basic info from org structure
        employee_id = employee_info['employee_id']
        department = employee_info['department']
        role = employee_info['role']
        level = employee_info['level']
        manager_id = employee_info.get('manager_id')
        
        # Generate personal information
        gender = self._select_gender()
        if gender == 'male':
            first_name = self.faker.first_name_male()
        elif gender == 'female':
            first_name = self.faker.first_name_female()
        else:
            first_name = self.faker.first_name()
        
        last_name = self.faker.last_name()
        email = self._generate_work_email(first_name, last_name)
        
        # Generate location
        location = self._select_location()
        
        # Generate tenure and hire date
        tenure_years = self._generate_tenure(level)
        hire_date = self._calculate_hire_date(tenure_years)
        
        # Generate salary
        salary = self._calculate_salary(role, location, tenure_years)
        
        # Generate performance rating
        performance_rating = self._generate_performance_rating(tenure_years)
        
        # Generate additional attributes
        employment_status = self._select_employment_status()
        work_schedule = self._select_work_schedule()
        
        return {
            'employee_id': f'EMP_{employee_id:06d}',
            'first_name': first_name,
            'last_name': last_name,
            'email': email,
            'department': department,
            'role': role,
            'level': level,
            'manager_id': f'EMP_{manager_id:06d}' if manager_id else None,
            'hire_date': hire_date,
            'tenure_years': round(tenure_years, 2),
            'location': location,
            'salary': salary,
            'performance_rating': performance_rating,
            'employment_status': employment_status,
            'work_schedule': work_schedule,
            'gender': gender,
            'phone': self.faker.phone_number(),
            'birth_date': self._generate_birth_date(tenure_years),
            'emergency_contact': self.faker.name(),
            'emergency_phone': self.faker.phone_number()
        }
    
    def _generate_time_series_employee_record(self, index: int, employee_info: Dict, 
                                            timestamp: datetime, hiring_intensity: float, 
                                            **kwargs) -> Dict:
        """Generate time series employee record with temporal patterns"""
        
        # Generate base employee record
        employee = self._generate_employee_record(index, employee_info, **kwargs)
        
        # Apply time-based hiring patterns
        hire_date = timestamp.date()
        
        # Adjust tenure based on hiring date
        days_since_hire = (datetime.now().date() - hire_date).days
        tenure_years = max(0, days_since_hire / 365.25)
        
        # Recalculate salary and performance based on actual tenure
        salary = self._calculate_salary(employee_info['role'], employee['location'], tenure_years)
        performance_rating = self._generate_performance_rating(tenure_years)
        
        # Update time-sensitive fields
        employee.update({
            'hire_date': hire_date,
            'hire_datetime': timestamp,
            'tenure_years': round(tenure_years, 2),
            'salary': salary,
            'performance_rating': performance_rating,
            'hiring_intensity': round(hiring_intensity, 3)
        })
        
        return employee
    
    def _select_gender(self) -> str:
        """Select gender with realistic distribution"""
        choices = ['male', 'female', 'other']
        weights = [0.48, 0.50, 0.02]
        
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
    
    def _generate_work_email(self, first_name: str, last_name: str) -> str:
        """Generate work email address"""
        # Common corporate email patterns
        patterns = [
            f"{first_name.lower()}.{last_name.lower()}@company.com",
            f"{first_name[0].lower()}{last_name.lower()}@company.com",
            f"{first_name.lower()}{last_name[0].lower()}@company.com",
            f"{first_name.lower()}_{last_name.lower()}@company.com"
        ]
        
        return self.faker.random_element(patterns)
    
    def _select_location(self) -> str:
        """Select office location based on distribution"""
        choices = list(self.office_locations.keys())
        weights = [self.office_locations[loc]['weight'] for loc in choices]
        
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
    
    def _generate_tenure(self, level: int) -> float:
        """Generate tenure based on role level"""
        tenure_config = self.tenure_by_level[level]
        mean_years = tenure_config['mean_years']
        std_years = tenure_config['std_years']
        
        # Use normal distribution but ensure positive values
        tenure = max(0.1, self.faker.random.normalvariate(mean_years, std_years))
        
        # Cap at reasonable maximum (30 years)
        return min(tenure, 30.0)
    
    def _calculate_hire_date(self, tenure_years: float) -> date:
        """Calculate hire date based on tenure"""
        days_ago = int(tenure_years * 365.25)
        hire_date = datetime.now().date() - timedelta(days=days_ago)
        
        # Adjust to avoid weekends for hire dates
        while hire_date.weekday() >= 5:  # Saturday or Sunday
            hire_date += timedelta(days=1)
        
        return hire_date
    
    def _calculate_salary(self, role: str, location: str, tenure_years: float) -> int:
        """Calculate salary based on role, location, and tenure"""
        # Base salary from role
        base_salary = self.role_levels[role]['base_salary']
        
        # Apply location multiplier
        location_multiplier = self.location_salary_multipliers.get(location, 1.0)
        
        # Apply experience multiplier
        experience_multiplier = self._get_experience_multiplier(tenure_years)
        
        # Calculate final salary
        salary = base_salary * location_multiplier * experience_multiplier
        
        # Add some random variation (±10%)
        variation = self.faker.random.uniform(0.9, 1.1)
        salary *= variation
        
        # Round to nearest $1000
        return int(round(salary / 1000) * 1000)
    
    def _get_experience_multiplier(self, tenure_years: float) -> float:
        """Get experience multiplier based on tenure"""
        for (min_years, max_years), multiplier in self.experience_multipliers.items():
            if min_years <= tenure_years < max_years:
                return multiplier
        
        # Default to highest multiplier for very experienced employees
        return 1.5
    
    def _generate_performance_rating(self, tenure_years: float) -> str:
        """Generate performance rating based on tenure patterns"""
        # Find appropriate tenure bracket
        for (min_years, max_years), distribution in self.tenure_performance_correlation.items():
            if min_years <= tenure_years < max_years:
                choices = ['exceeds_expectations', 'meets_expectations', 'below_expectations', 'needs_improvement']
                weights = [distribution['exceeds'], distribution['meets'], distribution['below'], distribution['needs']]
                
                cumulative_weights = []
                total = 0
                for weight in weights:
                    total += weight
                    cumulative_weights.append(total)
                
                rand_val = self.faker.random.uniform(0, total)
                for i, cum_weight in enumerate(cumulative_weights):
                    if rand_val <= cum_weight:
                        return choices[i]
        
        # Default to meets expectations
        return 'meets_expectations'
    
    def _select_employment_status(self) -> str:
        """Select employment status"""
        choices = ['full_time', 'part_time', 'contract', 'intern']
        weights = [0.85, 0.08, 0.05, 0.02]
        
        cumulative_weights = []
        total = 0
        for weight in weights:
            total += weight
            cumulative_weights.append(total)
        
        rand_val = self.faker.random.uniform(0, total)
        for i, cum_weight in enumerate(cumulative_weights):
            if rand_val <= cum_weight:
                return choices[i]
        
        return 'full_time'
    
    def _select_work_schedule(self) -> str:
        """Select work schedule"""
        choices = ['standard', 'flexible', 'remote', 'hybrid']
        weights = [0.40, 0.25, 0.15, 0.20]
        
        cumulative_weights = []
        total = 0
        for weight in weights:
            total += weight
            cumulative_weights.append(total)
        
        rand_val = self.faker.random.uniform(0, total)
        for i, cum_weight in enumerate(cumulative_weights):
            if rand_val <= cum_weight:
                return choices[i]
        
        return 'standard'
    
    def _generate_birth_date(self, tenure_years: float) -> date:
        """Generate realistic birth date based on tenure"""
        # Assume minimum working age of 18, typical graduation age 22-26
        min_age = max(18, int(tenure_years) + 22)
        max_age = min(65, int(tenure_years) + 45)
        
        age = self.faker.random_int(min_age, max_age)
        birth_date = datetime.now().date() - timedelta(days=age * 365.25)
        
        return birth_date
    
    def _apply_employee_time_series_correlations(self, data: pd.DataFrame, ts_config) -> pd.DataFrame:
        """Apply employee-specific time series correlations"""
        if len(data) < 2:
            return data
        
        # Sort by hire datetime to ensure proper time series order
        data = data.sort_values('hire_datetime').reset_index(drop=True)
        
        # Apply hiring wave correlations (similar roles hired together)
        for i in range(1, len(data)):
            prev_role = data.iloc[i-1]['role']
            prev_dept = data.iloc[i-1]['department']
            
            # Check if hires are close in time (within 30 days)
            time_diff = (data.iloc[i]['hire_datetime'] - data.iloc[i-1]['hire_datetime']).total_seconds()
            
            if time_diff <= 2592000:  # Within 30 days
                # Apply role clustering (hiring waves for same roles)
                if prev_role == data.iloc[i]['role'] and self.faker.random.random() < 0.3:
                    # Similar salary for same role hired around same time
                    prev_salary = data.iloc[i-1]['salary']
                    salary_adjustment = self.faker.random.uniform(0.95, 1.05)
                    data.loc[i, 'salary'] = int(prev_salary * salary_adjustment)
                
                # Apply department clustering
                if prev_dept == data.iloc[i]['department'] and self.faker.random.random() < 0.4:
                    # Similar location preferences within department
                    if self.faker.random.random() < 0.6:
                        data.loc[i, 'location'] = data.iloc[i-1]['location']
        
        return data
    
    def _apply_realistic_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply additional realistic patterns to employee data"""
        
        # Add derived fields
        if 'hire_datetime' in data.columns:
            data['hire_year'] = pd.to_datetime(data['hire_datetime']).dt.year
            data['hire_month'] = pd.to_datetime(data['hire_datetime']).dt.month
            data['hire_quarter'] = pd.to_datetime(data['hire_datetime']).dt.quarter
        else:
            data['hire_year'] = pd.to_datetime(data['hire_date']).dt.year
            data['hire_month'] = pd.to_datetime(data['hire_date']).dt.month
            data['hire_quarter'] = pd.to_datetime(data['hire_date']).dt.quarter
        
        # Calculate age
        data['age'] = ((datetime.now().date() - pd.to_datetime(data['birth_date']).dt.date) / timedelta(days=365.25)).astype(int)
        
        # Add salary bands
        data['salary_band'] = pd.cut(
            data['salary'],
            bins=[0, 50000, 75000, 100000, 150000, 200000, float('inf')],
            labels=['entry', 'junior', 'mid', 'senior', 'principal', 'executive']
        )
        
        # Add tenure bands
        data['tenure_band'] = pd.cut(
            data['tenure_years'],
            bins=[0, 1, 3, 5, 10, 20, float('inf')],
            labels=['new', 'junior', 'experienced', 'senior', 'veteran', 'legacy']
        )
        
        # Add performance score (numeric version of rating)
        performance_scores = {
            'exceeds_expectations': 4,
            'meets_expectations': 3,
            'below_expectations': 2,
            'needs_improvement': 1
        }
        data['performance_score'] = data['performance_rating'].map(performance_scores)
        
        # Add promotion eligibility flag
        data['promotion_eligible'] = (
            (data['tenure_years'] >= 2) & 
            (data['performance_rating'].isin(['meets_expectations', 'exceeds_expectations']))
        )
        
        # Add retention risk score
        data['retention_risk'] = self._calculate_retention_risk(data)
        
        # Add full name
        data['full_name'] = data['first_name'] + ' ' + data['last_name']
        
        # Add years to retirement (assuming retirement at 65)
        data['years_to_retirement'] = (65 - data['age']).clip(lower=0)
        
        return data
    
    def _calculate_retention_risk(self, data: pd.DataFrame) -> pd.Series:
        """Calculate retention risk score for each employee"""
        risk_scores = pd.Series(0.0, index=data.index)
        
        # Low performance increases risk
        risk_scores += (data['performance_rating'] == 'below_expectations') * 0.3
        risk_scores += (data['performance_rating'] == 'needs_improvement') * 0.5
        
        # Long tenure without promotion increases risk
        risk_scores += ((data['tenure_years'] > 5) & (data['level'] <= 2)) * 0.2
        
        # New hires have adjustment risk
        risk_scores += (data['tenure_years'] < 0.5) * 0.15
        
        # Below market salary increases risk (simplified)
        median_salary_by_role = data.groupby('role')['salary'].median()
        below_median = data.apply(lambda row: row['salary'] < median_salary_by_role[row['role']] * 0.9, axis=1)
        risk_scores += below_median * 0.25
        
        # Cap at 1.0
        return risk_scores.clip(upper=1.0).round(3)