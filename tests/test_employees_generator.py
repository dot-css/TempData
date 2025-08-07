"""
Unit tests for EmployeesGenerator

Tests hierarchical reporting structures, department organization, salary distributions,
performance ratings, and tenure patterns.
"""

import pytest
import pandas as pd
from datetime import datetime, date, timedelta
from tempdata.core.seeding import MillisecondSeeder
from tempdata.datasets.business.employees import EmployeesGenerator


class TestEmployeesGenerator:
    """Test suite for EmployeesGenerator functionality"""
    
    @pytest.fixture
    def seeder(self):
        """Create a fixed seeder for reproducible tests"""
        return MillisecondSeeder(fixed_seed=123456789)
    
    @pytest.fixture
    def generator(self, seeder):
        """Create EmployeesGenerator instance"""
        return EmployeesGenerator(seeder)
    
    def test_basic_generation(self, generator):
        """Test basic data generation functionality"""
        rows = 100
        data = generator.generate(rows)
        
        # Check basic structure
        assert isinstance(data, pd.DataFrame)
        assert len(data) == rows
        assert not data.empty
    
    def test_required_columns(self, generator):
        """Test that all required columns are present"""
        data = generator.generate(50)
        
        required_columns = [
            'employee_id', 'first_name', 'last_name', 'email', 'department',
            'role', 'level', 'manager_id', 'hire_date', 'tenure_years',
            'location', 'salary', 'performance_rating', 'employment_status',
            'work_schedule', 'gender', 'phone', 'birth_date'
        ]
        
        for column in required_columns:
            assert column in data.columns, f"Missing required column: {column}"
    
    def test_employee_id_uniqueness(self, generator):
        """Test that employee IDs are unique"""
        data = generator.generate(500)
        assert data['employee_id'].nunique() == len(data)
        
        # Check format
        assert all(data['employee_id'].str.startswith('EMP_'))
        assert all(data['employee_id'].str.len() == 10)  # EMP_ + 6 digits
    
    def test_department_distribution(self, generator):
        """Test that departments are distributed according to weights"""
        data = generator.generate(1000)
        
        expected_departments = {
            'engineering', 'sales', 'marketing', 'operations',
            'finance', 'human_resources', 'customer_support'
        }
        
        actual_departments = set(data['department'].unique())
        assert actual_departments.issubset(expected_departments)
        
        # Engineering should be the largest department
        dept_counts = data['department'].value_counts()
        assert dept_counts.index[0] in ['engineering', 'sales']  # Top 2 departments
    
    def test_hierarchical_structure(self, generator):
        """Test hierarchical reporting structure"""
        data = generator.generate(200)
        
        # Check that levels are consistent with roles
        role_levels = data.groupby('role')['level'].first().to_dict()
        
        # Higher level roles should have higher level numbers
        if 'ceo' in role_levels and 'software_engineer' in role_levels:
            assert role_levels['ceo'] > role_levels['software_engineer']
        
        # Managers should have higher levels than their reports
        for _, employee in data.iterrows():
            if pd.notna(employee['manager_id']):
                manager = data[data['employee_id'] == employee['manager_id']]
                if not manager.empty:
                    assert manager.iloc[0]['level'] >= employee['level']
    
    def test_manager_relationships(self, generator):
        """Test manager-employee relationships are valid"""
        data = generator.generate(300)
        
        # All manager IDs should exist as employee IDs (except for CEO)
        manager_ids = data['manager_id'].dropna().unique()
        employee_ids = data['employee_id'].unique()
        
        for manager_id in manager_ids:
            assert manager_id in employee_ids, f"Manager ID {manager_id} not found in employee list"
        
        # CEO should not have a manager
        ceo_data = data[data['role'] == 'ceo']
        if not ceo_data.empty:
            assert pd.isna(ceo_data.iloc[0]['manager_id'])
        
        # No circular reporting relationships
        for _, employee in data.iterrows():
            visited = set()
            current_id = employee['employee_id']
            
            while current_id and current_id not in visited:
                visited.add(current_id)
                manager_row = data[data['employee_id'] == current_id]
                if manager_row.empty:
                    break
                current_id = manager_row.iloc[0]['manager_id']
            
            # If we found a cycle, current_id should be in visited
            # But the original employee should only appear once (at the start)
            if current_id and current_id in visited:
                # This indicates a circular reference
                assert False, f"Circular reference detected: {current_id} reports to someone who eventually reports back to them"
    
    def test_salary_ranges_realistic(self, generator):
        """Test that salary ranges are realistic"""
        data = generator.generate(500)
        
        # Salaries should be positive
        assert (data['salary'] > 0).all()
        
        # Salaries should be reasonable (between $30K and $500K)
        assert (data['salary'] >= 30000).all()
        assert (data['salary'] <= 500000).all()
        
        # Higher level roles should generally have higher salaries
        level_salaries = data.groupby('level')['salary'].median()
        
        # Check that median salary generally increases with level
        for level in range(1, 6):
            if level in level_salaries.index and level + 1 in level_salaries.index:
                assert level_salaries[level + 1] >= level_salaries[level] * 0.8  # Allow some variance
    
    def test_salary_location_adjustment(self, generator):
        """Test salary adjustments by location"""
        data = generator.generate(1000)
        
        # San Francisco should have higher average salaries than other locations
        location_salaries = data.groupby('location')['salary'].mean()
        
        if 'san_francisco' in location_salaries.index and 'phoenix' in location_salaries.index:
            assert location_salaries['san_francisco'] > location_salaries['phoenix']
    
    def test_tenure_patterns(self, generator):
        """Test tenure patterns by role level"""
        data = generator.generate(800)
        
        # Higher level roles should generally have longer tenure
        level_tenure = data.groupby('level')['tenure_years'].mean()
        
        # Check general trend (allowing for some variance)
        if len(level_tenure) > 2:
            # At least 60% of level comparisons should follow the pattern
            correct_comparisons = 0
            total_comparisons = 0
            
            for level in range(1, 6):
                if level in level_tenure.index and level + 1 in level_tenure.index:
                    if level_tenure[level + 1] >= level_tenure[level]:
                        correct_comparisons += 1
                    total_comparisons += 1
            
            if total_comparisons > 0:
                assert correct_comparisons / total_comparisons >= 0.6
    
    def test_performance_rating_distribution(self, generator):
        """Test performance rating distribution"""
        data = generator.generate(1000)
        
        expected_ratings = {'exceeds_expectations', 'meets_expectations', 'below_expectations', 'needs_improvement'}
        actual_ratings = set(data['performance_rating'].unique())
        
        assert actual_ratings.issubset(expected_ratings)
        
        # Most employees should meet expectations
        rating_dist = data['performance_rating'].value_counts(normalize=True)
        assert rating_dist['meets_expectations'] > 0.5  # At least 50%
        
        # Very few should need improvement
        if 'needs_improvement' in rating_dist.index:
            assert rating_dist['needs_improvement'] < 0.1  # Less than 10%
    
    def test_performance_tenure_correlation(self, generator):
        """Test correlation between performance and tenure"""
        data = generator.generate(1000)
        
        # New employees (< 1 year) should have more variable performance
        new_employees = data[data['tenure_years'] < 1.0]
        experienced_employees = data[data['tenure_years'] >= 3.0]
        
        if len(new_employees) > 10 and len(experienced_employees) > 10:
            new_exceeds_rate = (new_employees['performance_rating'] == 'exceeds_expectations').mean()
            exp_exceeds_rate = (experienced_employees['performance_rating'] == 'exceeds_expectations').mean()
            
            # Experienced employees should have higher rate of exceeding expectations
            assert exp_exceeds_rate >= new_exceeds_rate * 0.8  # Allow some variance
    
    def test_employment_status_distribution(self, generator):
        """Test employment status distribution"""
        data = generator.generate(500)
        
        expected_statuses = {'full_time', 'part_time', 'contract', 'intern'}
        actual_statuses = set(data['employment_status'].unique())
        
        assert actual_statuses.issubset(expected_statuses)
        
        # Most employees should be full-time
        status_dist = data['employment_status'].value_counts(normalize=True)
        assert status_dist['full_time'] > 0.7  # At least 70%
    
    def test_work_schedule_distribution(self, generator):
        """Test work schedule distribution"""
        data = generator.generate(400)
        
        expected_schedules = {'standard', 'flexible', 'remote', 'hybrid'}
        actual_schedules = set(data['work_schedule'].unique())
        
        assert actual_schedules.issubset(expected_schedules)
        
        # Should have reasonable distribution
        schedule_dist = data['work_schedule'].value_counts(normalize=True)
        assert all(schedule_dist <= 0.6)  # No single schedule > 60%
    
    def test_location_distribution(self, generator):
        """Test office location distribution"""
        data = generator.generate(600)
        
        expected_locations = {
            'san_francisco', 'new_york', 'seattle', 'chicago', 'austin',
            'boston', 'los_angeles', 'denver', 'atlanta', 'phoenix', 'remote'
        }
        
        actual_locations = set(data['location'].unique())
        assert actual_locations.issubset(expected_locations)
        
        # San Francisco and New York should be most common
        location_counts = data['location'].value_counts()
        top_locations = set(location_counts.head(3).index)
        assert 'san_francisco' in top_locations or 'new_york' in top_locations
    
    def test_gender_distribution(self, generator):
        """Test gender distribution"""
        data = generator.generate(1000)
        
        expected_genders = {'male', 'female', 'other'}
        actual_genders = set(data['gender'].unique())
        
        assert actual_genders.issubset(expected_genders)
        
        # Should be roughly balanced between male and female
        gender_dist = data['gender'].value_counts(normalize=True)
        assert gender_dist['male'] > 0.4 and gender_dist['male'] < 0.6
        assert gender_dist['female'] > 0.4 and gender_dist['female'] < 0.6
        
        if 'other' in gender_dist.index:
            assert gender_dist['other'] < 0.05  # Small percentage
    
    def test_email_format(self, generator):
        """Test work email format"""
        data = generator.generate(100)
        
        # All emails should contain @company.com
        assert all(data['email'].str.contains('@company.com'))
        
        # Emails should be unique
        assert data['email'].nunique() == len(data)
        
        # Email format should be reasonable
        assert all(data['email'].str.len() > 5)  # Minimum reasonable length
    
    def test_age_consistency(self, generator):
        """Test age consistency with tenure and birth date"""
        data = generator.generate(200)
        
        # Calculate age from birth date
        calculated_age = ((datetime.now().date() - pd.to_datetime(data['birth_date']).dt.date) / timedelta(days=365.25)).astype(int)
        
        # Should match the age column
        age_diff = abs(data['age'] - calculated_age)
        assert (age_diff <= 1).all()  # Allow 1 year difference due to rounding
        
        # Age should be reasonable for working population
        assert (data['age'] >= 18).all()
        assert (data['age'] <= 70).all()
        
        # Age should be consistent with tenure (age >= tenure + 18)
        min_expected_age = data['tenure_years'] + 18
        assert (data['age'] >= min_expected_age - 2).all()  # Allow 2 years variance
    
    def test_hire_date_consistency(self, generator):
        """Test hire date consistency with tenure"""
        data = generator.generate(150)
        
        # Calculate tenure from hire date
        calculated_tenure = ((datetime.now().date() - pd.to_datetime(data['hire_date']).dt.date) / timedelta(days=365.25))
        
        # Should match tenure_years column (within reasonable tolerance)
        tenure_diff = abs(data['tenure_years'] - calculated_tenure)
        assert (tenure_diff <= 0.1).all()  # Within 0.1 years
        
        # Hire dates should not be in the future
        assert (pd.to_datetime(data['hire_date']) <= datetime.now()).all()
        
        # Hire dates should not be on weekends (business rule)
        hire_weekdays = pd.to_datetime(data['hire_date']).dt.dayofweek
        assert (hire_weekdays < 5).all()  # Monday=0, Friday=4
    
    def test_derived_fields(self, generator):
        """Test derived fields are calculated correctly"""
        data = generator.generate(100)
        
        # Should have derived fields
        derived_fields = [
            'hire_year', 'hire_month', 'hire_quarter', 'age', 'salary_band',
            'tenure_band', 'performance_score', 'promotion_eligible',
            'retention_risk', 'full_name', 'years_to_retirement'
        ]
        
        for field in derived_fields:
            assert field in data.columns
        
        # Full name should be first + last name
        expected_full_name = data['first_name'] + ' ' + data['last_name']
        assert (data['full_name'] == expected_full_name).all()
        
        # Performance score should match rating
        score_mapping = {
            'exceeds_expectations': 4,
            'meets_expectations': 3,
            'below_expectations': 2,
            'needs_improvement': 1
        }
        expected_scores = data['performance_rating'].map(score_mapping)
        assert (data['performance_score'] == expected_scores).all()
        
        # Years to retirement should be 65 - age (minimum 0)
        expected_retirement = (65 - data['age']).clip(lower=0)
        assert (data['years_to_retirement'] == expected_retirement).all()
    
    def test_salary_bands(self, generator):
        """Test salary band categorization"""
        data = generator.generate(300)
        
        expected_bands = {'entry', 'junior', 'mid', 'senior', 'principal', 'executive'}
        actual_bands = set(data['salary_band'].dropna().unique())
        
        assert actual_bands.issubset(expected_bands)
        
        # Verify band logic
        entry_employees = data[data['salary_band'] == 'entry']
        executive_employees = data[data['salary_band'] == 'executive']
        
        if len(entry_employees) > 0 and len(executive_employees) > 0:
            assert entry_employees['salary'].max() <= executive_employees['salary'].min()
    
    def test_tenure_bands(self, generator):
        """Test tenure band categorization"""
        data = generator.generate(300)
        
        expected_bands = {'new', 'junior', 'experienced', 'senior', 'veteran', 'legacy'}
        actual_bands = set(data['tenure_band'].dropna().unique())
        
        assert actual_bands.issubset(expected_bands)
        
        # Verify band logic
        new_employees = data[data['tenure_band'] == 'new']
        veteran_employees = data[data['tenure_band'] == 'veteran']
        
        if len(new_employees) > 0:
            assert (new_employees['tenure_years'] < 1.0).all()
        
        if len(veteran_employees) > 0:
            assert (veteran_employees['tenure_years'] >= 10.0).all()
    
    def test_promotion_eligibility(self, generator):
        """Test promotion eligibility logic"""
        data = generator.generate(400)
        
        # Promotion eligible should require tenure >= 2 years and good performance
        eligible = data[data['promotion_eligible'] == True]
        
        if len(eligible) > 0:
            assert (eligible['tenure_years'] >= 2.0).all()
            assert eligible['performance_rating'].isin(['meets_expectations', 'exceeds_expectations']).all()
        
        # Non-eligible should fail at least one criterion
        non_eligible = data[data['promotion_eligible'] == False]
        
        if len(non_eligible) > 0:
            condition1 = non_eligible['tenure_years'] < 2.0
            condition2 = non_eligible['performance_rating'].isin(['below_expectations', 'needs_improvement'])
            assert (condition1 | condition2).all()
    
    def test_retention_risk_calculation(self, generator):
        """Test retention risk score calculation"""
        data = generator.generate(500)
        
        # Retention risk should be between 0 and 1
        assert (data['retention_risk'] >= 0.0).all()
        assert (data['retention_risk'] <= 1.0).all()
        
        # Poor performers should have higher risk
        poor_performers = data[data['performance_rating'].isin(['below_expectations', 'needs_improvement'])]
        good_performers = data[data['performance_rating'] == 'exceeds_expectations']
        
        if len(poor_performers) > 0 and len(good_performers) > 0:
            assert poor_performers['retention_risk'].mean() > good_performers['retention_risk'].mean()
    
    def test_time_series_generation(self, generator):
        """Test time series data generation"""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        data = generator.generate(
            100,
            time_series=True,
            start_date=start_date,
            end_date=end_date,
            interval='1day'
        )
        
        # Should have timestamp column
        assert 'hire_datetime' in data.columns
        
        # Timestamps should be within range
        timestamps = pd.to_datetime(data['hire_datetime'])
        assert (timestamps >= start_date).all()
        assert (timestamps <= end_date).all()
        
        # Should be chronologically ordered
        timestamps_list = timestamps.tolist()
        assert timestamps_list == sorted(timestamps_list)
        
        # Should have hiring_intensity column for time series
        assert 'hiring_intensity' in data.columns
        assert (data['hiring_intensity'] > 0).all()
    
    def test_organizational_pyramid(self, generator):
        """Test organizational pyramid structure"""
        data = generator.generate(500)
        
        # Count employees by level
        level_counts = data['level'].value_counts().sort_index()
        
        # Should have pyramid structure (more employees at lower levels)
        if len(level_counts) > 2:
            # Level 1 should have more employees than level 2
            if 1 in level_counts.index and 2 in level_counts.index:
                assert level_counts[1] >= level_counts[2] * 0.8  # Allow some variance
            
            # Higher levels should generally have fewer employees (allow some variance)
            pyramid_violations = 0
            total_comparisons = 0
            
            for level in range(2, 6):
                if level in level_counts.index and level + 1 in level_counts.index:
                    total_comparisons += 1
                    if level_counts[level] < level_counts[level + 1]:
                        pyramid_violations += 1
            
            # Allow up to 40% of comparisons to violate pyramid structure
            if total_comparisons > 0:
                violation_rate = pyramid_violations / total_comparisons
                assert violation_rate <= 0.4, f"Too many pyramid violations: {violation_rate:.2%}"
    
    def test_department_role_consistency(self, generator):
        """Test that roles are consistent with departments"""
        data = generator.generate(300)
        
        # Check that roles belong to appropriate departments
        dept_roles = data.groupby('department')['role'].unique()
        
        for dept, roles in dept_roles.items():
            if dept == 'engineering':
                eng_roles = {'software_engineer', 'senior_engineer', 'lead_engineer', 'engineering_manager', 'cto'}
                assert set(roles).issubset(eng_roles)
            elif dept == 'sales':
                sales_roles = {'sales_rep', 'senior_sales_rep', 'sales_manager', 'sales_director', 'vp_sales'}
                assert set(roles).issubset(sales_roles)
    
    def test_reproducibility(self, generator):
        """Test that generation is reproducible with same seed"""
        data1 = generator.generate(50)
        
        # Create new generator with same seed
        seeder2 = MillisecondSeeder(fixed_seed=123456789)
        generator2 = EmployeesGenerator(seeder2)
        data2 = generator2.generate(50)
        
        # Should be identical
        pd.testing.assert_frame_equal(data1, data2)
    
    def test_data_quality_validation(self, generator):
        """Test overall data quality"""
        data = generator.generate(500)
        
        # No null values in required fields
        critical_fields = ['employee_id', 'first_name', 'last_name', 'department', 'role', 'salary']
        for field in critical_fields:
            assert data[field].notna().all()
        
        # Reasonable data distributions
        assert data['salary'].std() > 0  # Should have variance
        assert data['tenure_years'].std() > 0
        assert data['age'].std() > 0
        
        # No negative values where inappropriate
        non_negative_fields = ['salary', 'tenure_years', 'age', 'level', 'performance_score']
        for field in non_negative_fields:
            assert (data[field] >= 0).all()
    
    def test_phone_number_format(self, generator):
        """Test phone number format"""
        data = generator.generate(100)
        
        # Phone numbers should be present
        assert data['phone'].notna().all()
        assert data['emergency_phone'].notna().all()
        
        # Should have reasonable length
        assert (data['phone'].str.len() >= 10).all()
        assert (data['emergency_phone'].str.len() >= 10).all()
    
    def test_emergency_contact_data(self, generator):
        """Test emergency contact data"""
        data = generator.generate(100)
        
        # Emergency contacts should be present
        assert data['emergency_contact'].notna().all()
        
        # Should be different from employee name
        same_name_count = (data['emergency_contact'] == data['full_name']).sum()
        assert same_name_count < len(data) * 0.1  # Less than 10% should be same (very unlikely)
        
        # Should have reasonable length
        assert (data['emergency_contact'].str.len() >= 3).all()