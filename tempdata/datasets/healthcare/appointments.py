"""
Medical appointment generator

Generates realistic appointment data with scheduling patterns, doctor availability,
seasonal trends, and appointment types.
"""

import pandas as pd
from datetime import datetime, date, timedelta, time
from typing import Dict, List, Any, Tuple
from ...core.base_generator import BaseGenerator


class AppointmentGenerator(BaseGenerator):
    """
    Generator for realistic medical appointment data
    
    Creates appointment datasets with scheduling patterns, doctor availability,
    seasonal trends, appointment types, and duration patterns.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_appointment_types()
        self._setup_scheduling_patterns()
        self._setup_doctor_availability()
        self._setup_seasonal_patterns()
        self._setup_duration_patterns()
    
    def _setup_appointment_types(self):
        """Setup appointment types with realistic distributions"""
        self.appointment_types = {
            # Primary care appointments
            'Annual Physical': {'weight': 0.15, 'duration': 45, 'specialty': 'Primary Care'},
            'Routine Checkup': {'weight': 0.20, 'duration': 30, 'specialty': 'Primary Care'},
            'Follow-up Visit': {'weight': 0.18, 'duration': 20, 'specialty': 'Primary Care'},
            'Sick Visit': {'weight': 0.12, 'duration': 25, 'specialty': 'Primary Care'},
            
            # Specialist appointments
            'Cardiology Consultation': {'weight': 0.08, 'duration': 60, 'specialty': 'Cardiology'},
            'Dermatology Screening': {'weight': 0.06, 'duration': 30, 'specialty': 'Dermatology'},
            'Orthopedic Consultation': {'weight': 0.05, 'duration': 45, 'specialty': 'Orthopedics'},
            'Endocrinology Visit': {'weight': 0.04, 'duration': 45, 'specialty': 'Endocrinology'},
            'Psychiatry Session': {'weight': 0.03, 'duration': 50, 'specialty': 'Psychiatry'},
            'Ophthalmology Exam': {'weight': 0.04, 'duration': 40, 'specialty': 'Ophthalmology'},
            
            # Preventive care
            'Mammography': {'weight': 0.02, 'duration': 30, 'specialty': 'Radiology'},
            'Colonoscopy': {'weight': 0.01, 'duration': 60, 'specialty': 'Gastroenterology'},
            'Vaccination': {'weight': 0.02, 'duration': 15, 'specialty': 'Primary Care'}
        }
        
        # Appointment urgency levels
        self.urgency_levels = {
            'Routine': 0.70,
            'Urgent': 0.25,
            'Emergency': 0.05
        }
        
        # Appointment status distribution
        self.status_distribution = {
            'Scheduled': 0.40,
            'Completed': 0.50,
            'Cancelled': 0.08,
            'No-Show': 0.02
        }
    
    def _setup_scheduling_patterns(self):
        """Setup realistic scheduling patterns"""
        # Time slot preferences (8 AM to 6 PM)
        self.time_slots = {
            '08:00': 0.08,  # Early morning
            '08:30': 0.10,
            '09:00': 0.12,  # Popular morning slot
            '09:30': 0.11,
            '10:00': 0.10,
            '10:30': 0.09,
            '11:00': 0.08,
            '11:30': 0.07,
            '13:00': 0.06,  # After lunch
            '13:30': 0.07,
            '14:00': 0.08,
            '14:30': 0.09,
            '15:00': 0.10,  # Popular afternoon slot
            '15:30': 0.09,
            '16:00': 0.08,
            '16:30': 0.07,
            '17:00': 0.05,
            '17:30': 0.03   # Late afternoon
        }
        
        # Day of week preferences
        self.day_preferences = {
            'Monday': 0.22,
            'Tuesday': 0.20,
            'Wednesday': 0.18,
            'Thursday': 0.20,
            'Friday': 0.15,
            'Saturday': 0.05,  # Limited weekend hours
            'Sunday': 0.00     # Most practices closed
        }
        
        # Lead time patterns (days in advance appointments are scheduled)
        self.lead_time_patterns = {
            'Routine': {'mean': 21, 'std': 14, 'min': 7, 'max': 90},
            'Urgent': {'mean': 3, 'std': 2, 'min': 0, 'max': 7},
            'Emergency': {'mean': 0, 'std': 0, 'min': 0, 'max': 1}
        }
    
    def _setup_doctor_availability(self):
        """Setup doctor availability patterns"""
        # Doctor specialties and their typical schedules
        self.doctor_specialties = {
            'Primary Care': {
                'count': 40,
                'days_per_week': 5,
                'hours_per_day': 8,
                'patients_per_hour': 2.5
            },
            'Cardiology': {
                'count': 8,
                'days_per_week': 4,
                'hours_per_day': 7,
                'patients_per_hour': 1.5
            },
            'Dermatology': {
                'count': 6,
                'days_per_week': 4,
                'hours_per_day': 6,
                'patients_per_hour': 2.0
            },
            'Orthopedics': {
                'count': 5,
                'days_per_week': 4,
                'hours_per_day': 7,
                'patients_per_hour': 1.8
            },
            'Endocrinology': {
                'count': 3,
                'days_per_week': 3,
                'hours_per_day': 6,
                'patients_per_hour': 1.5
            },
            'Psychiatry': {
                'count': 4,
                'days_per_week': 5,
                'hours_per_day': 8,
                'patients_per_hour': 1.2
            },
            'Ophthalmology': {
                'count': 4,
                'days_per_week': 4,
                'hours_per_day': 6,
                'patients_per_hour': 1.8
            },
            'Radiology': {
                'count': 3,
                'days_per_week': 5,
                'hours_per_day': 8,
                'patients_per_hour': 4.0
            },
            'Gastroenterology': {
                'count': 2,
                'days_per_week': 3,
                'hours_per_day': 6,
                'patients_per_hour': 1.0
            }
        }
        
        # Generate doctor pool
        self.doctors = self._generate_doctor_pool()
    
    def _setup_seasonal_patterns(self):
        """Setup seasonal appointment patterns"""
        # Monthly appointment volume multipliers
        self.seasonal_patterns = {
            1: 1.1,   # January - New Year health resolutions
            2: 0.9,   # February - low activity
            3: 1.0,   # March - normal
            4: 1.1,   # April - spring checkups
            5: 1.0,   # May - normal
            6: 0.9,   # June - vacation season starts
            7: 0.8,   # July - vacation peak
            8: 0.8,   # August - vacation continues
            9: 1.2,   # September - back to routine
            10: 1.1,  # October - flu season prep
            11: 1.0,  # November - normal
            12: 0.9   # December - holidays
        }
        
        # Seasonal appointment type preferences
        self.seasonal_appointment_preferences = {
            'winter': ['Sick Visit', 'Vaccination', 'Routine Checkup'],
            'spring': ['Annual Physical', 'Dermatology Screening', 'Routine Checkup'],
            'summer': ['Dermatology Screening', 'Ophthalmology Exam', 'Follow-up Visit'],
            'fall': ['Annual Physical', 'Vaccination', 'Mammography']
        }
    
    def _setup_duration_patterns(self):
        """Setup appointment duration patterns"""
        # Buffer time between appointments by specialty
        self.buffer_times = {
            'Primary Care': 5,
            'Cardiology': 10,
            'Dermatology': 5,
            'Orthopedics': 10,
            'Endocrinology': 10,
            'Psychiatry': 10,
            'Ophthalmology': 10,
            'Radiology': 5,
            'Gastroenterology': 15
        }
        
        # Overtime probability (appointments running late)
        self.overtime_probability = 0.15
        self.overtime_minutes = {'mean': 10, 'std': 5, 'max': 30}
    
    def _generate_doctor_pool(self) -> List[Dict[str, Any]]:
        """Generate pool of doctors with specialties"""
        doctors = []
        doctor_id = 1
        
        for specialty, config in self.doctor_specialties.items():
            for i in range(config['count']):
                doctor = {
                    'doctor_id': f'DOC_{doctor_id:03d}',
                    'name': f"Dr. {self.faker.first_name()} {self.faker.last_name()}",
                    'specialty': specialty,
                    'days_per_week': config['days_per_week'],
                    'hours_per_day': config['hours_per_day'],
                    'patients_per_hour': config['patients_per_hour']
                }
                doctors.append(doctor)
                doctor_id += 1
        
        return doctors
    
    def generate(self, rows: int, **kwargs) -> pd.DataFrame:
        """
        Generate medical appointment dataset
        
        Args:
            rows: Number of appointments to generate
            **kwargs: Additional parameters (date_range, patient_pool_size, time_series, etc.)
            
        Returns:
            pd.DataFrame: Generated appointment data with realistic patterns
        """
        # Check if time series generation is requested
        ts_config = self._create_time_series_config(**kwargs)
        
        if ts_config:
            return self._generate_time_series_appointments(rows, ts_config, **kwargs)
        else:
            return self._generate_snapshot_appointments(rows, **kwargs)
    
    def _generate_snapshot_appointments(self, rows: int, **kwargs) -> pd.DataFrame:
        """Generate snapshot appointment data (random appointment dates)"""
        date_range = kwargs.get('date_range', None)
        patient_pool_size = kwargs.get('patient_pool_size', 5000)
        include_historical = kwargs.get('include_historical', True)
        
        data = []
        
        for i in range(rows):
            # Select appointment type and get its properties
            appointment_type = self._select_appointment_type()
            type_info = self.appointment_types[appointment_type]
            
            # Select urgency level
            urgency = self._select_urgency_level()
            
            # Generate appointment date based on urgency and seasonal patterns
            appointment_date, appointment_time = self._generate_appointment_datetime(
                urgency, date_range, include_historical
            )
            
            # Select appropriate doctor for the appointment type
            doctor = self._select_doctor_for_appointment(type_info['specialty'])
            
            # Generate patient ID
            patient_id = f'PAT_{self.faker.random_int(1, patient_pool_size):06d}'
            
            # Generate appointment status
            status = self._select_appointment_status(appointment_date)
            
            # Calculate actual duration (may include overtime)
            scheduled_duration = type_info['duration']
            actual_duration = self._calculate_actual_duration(scheduled_duration, status)
            
            # Generate additional appointment details
            appointment_details = self._generate_appointment_details(
                appointment_type, urgency, status, appointment_date
            )
            
            # Generate room assignment
            room_number = self._generate_room_assignment(type_info['specialty'])
            
            # Generate insurance and billing information
            billing_info = self._generate_billing_info(appointment_type, status)
            
            appointment = {
                'appointment_id': f'APT_{i+1:06d}',
                'patient_id': patient_id,
                'doctor_id': doctor['doctor_id'],
                'doctor_name': doctor['name'],
                'specialty': doctor['specialty'],
                'appointment_date': appointment_date,
                'appointment_time': appointment_time,
                'appointment_type': appointment_type,
                'urgency_level': urgency,
                'scheduled_duration_minutes': scheduled_duration,
                'actual_duration_minutes': actual_duration,
                'status': status,
                'room_number': room_number,
                'chief_complaint': appointment_details['chief_complaint'],
                'notes': appointment_details['notes'],
                'follow_up_needed': appointment_details['follow_up_needed'],
                'follow_up_weeks': appointment_details['follow_up_weeks'],
                'billing_code': billing_info['code'],
                'billing_amount': billing_info['amount'],
                'insurance_covered': billing_info['insurance_covered'],
                'copay_amount': billing_info['copay'],
                'created_date': appointment_details['created_date'],
                'last_modified': appointment_details['last_modified']
            }
            
            data.append(appointment)
        
        df = pd.DataFrame(data)
        
        # Convert date and time columns to proper types
        df['appointment_date'] = pd.to_datetime(df['appointment_date']).dt.date
        df['appointment_time'] = pd.to_datetime(df['appointment_time'], format='%H:%M:%S').dt.time
        df['created_date'] = pd.to_datetime(df['created_date']).dt.date
        df['last_modified'] = pd.to_datetime(df['last_modified']).dt.date
        
        return self._apply_realistic_patterns(df)
    
    def _generate_time_series_appointments(self, rows: int, ts_config, **kwargs) -> pd.DataFrame:
        """Generate time series appointment data with temporal scheduling patterns"""
        patient_pool_size = kwargs.get('patient_pool_size', 5000)
        
        # Generate timestamps from time series config
        timestamps = self._generate_time_series_timestamps(ts_config, rows)
        
        data = []
        
        # Track appointment scheduling patterns over time
        daily_appointment_counts = {}
        doctor_schedules = {}
        
        for i, timestamp in enumerate(timestamps):
            if i >= rows:
                break
                
            # Convert timestamp to date for appointment scheduling
            appointment_date = timestamp.date()
            
            # Ensure it's a valid business day
            appointment_date = self._adjust_to_business_day(appointment_date)
            
            # Apply seasonal patterns to appointment type selection
            appointment_type = self._select_seasonal_appointment_type(appointment_date)
            type_info = self.appointment_types[appointment_type]
            
            # Select urgency level (time series may have different patterns)
            urgency = self._select_urgency_level_for_time_series(appointment_date)
            
            # Select appropriate doctor for the appointment type
            doctor = self._select_doctor_for_appointment(type_info['specialty'])
            
            # Generate appointment time with realistic scheduling constraints
            appointment_time = self._select_appointment_time_with_constraints(
                appointment_date, doctor, daily_appointment_counts, doctor_schedules
            )
            
            # Generate patient ID
            patient_id = f'PAT_{self.faker.random_int(1, patient_pool_size):06d}'
            
            # Generate appointment status based on date
            status = self._select_appointment_status(appointment_date)
            
            # Calculate actual duration (may include overtime)
            scheduled_duration = type_info['duration']
            actual_duration = self._calculate_actual_duration(scheduled_duration, status)
            
            # Generate additional appointment details
            appointment_details = self._generate_appointment_details(
                appointment_type, urgency, status, appointment_date
            )
            
            # Generate room assignment
            room_number = self._generate_room_assignment(type_info['specialty'])
            
            # Generate insurance and billing information
            billing_info = self._generate_billing_info(appointment_type, status)
            
            appointment = {
                'appointment_id': f'APT_{i+1:06d}',
                'patient_id': patient_id,
                'doctor_id': doctor['doctor_id'],
                'doctor_name': doctor['name'],
                'specialty': doctor['specialty'],
                'appointment_date': appointment_date,
                'appointment_time': appointment_time,
                'appointment_type': appointment_type,
                'urgency_level': urgency,
                'scheduled_duration_minutes': scheduled_duration,
                'actual_duration_minutes': actual_duration,
                'status': status,
                'room_number': room_number,
                'chief_complaint': appointment_details['chief_complaint'],
                'notes': appointment_details['notes'],
                'follow_up_needed': appointment_details['follow_up_needed'],
                'follow_up_weeks': appointment_details['follow_up_weeks'],
                'billing_code': billing_info['code'],
                'billing_amount': billing_info['amount'],
                'insurance_covered': billing_info['insurance_covered'],
                'copay_amount': billing_info['copay'],
                'created_date': appointment_details['created_date'],
                'last_modified': appointment_details['last_modified']
            }
            
            # Track scheduling for realistic constraints
            date_key = appointment_date
            if date_key not in daily_appointment_counts:
                daily_appointment_counts[date_key] = {}
            if doctor['doctor_id'] not in daily_appointment_counts[date_key]:
                daily_appointment_counts[date_key][doctor['doctor_id']] = 0
            daily_appointment_counts[date_key][doctor['doctor_id']] += 1
            
            # Track doctor schedules
            if doctor['doctor_id'] not in doctor_schedules:
                doctor_schedules[doctor['doctor_id']] = {}
            if date_key not in doctor_schedules[doctor['doctor_id']]:
                doctor_schedules[doctor['doctor_id']][date_key] = []
            doctor_schedules[doctor['doctor_id']][date_key].append(appointment_time)
            
            data.append(appointment)
        
        df = pd.DataFrame(data)
        
        # Convert date and time columns to proper types
        df['appointment_date'] = pd.to_datetime(df['appointment_date']).dt.date
        df['appointment_time'] = pd.to_datetime(df['appointment_time'], format='%H:%M:%S').dt.time
        df['created_date'] = pd.to_datetime(df['created_date']).dt.date
        df['last_modified'] = pd.to_datetime(df['last_modified']).dt.date
        
        # Add temporal relationships using base generator functionality
        df = self._add_temporal_relationships(df, ts_config)
        
        # Apply appointment-specific time series correlations
        df = self._apply_appointment_time_series_correlations(df, ts_config)
        
        return self._apply_realistic_patterns(df)
    
    def _select_seasonal_appointment_type(self, appointment_date: date) -> str:
        """Select appointment type based on seasonal patterns"""
        month = appointment_date.month
        
        # Determine season
        if month in [12, 1, 2]:
            season = 'winter'
        elif month in [3, 4, 5]:
            season = 'spring'
        elif month in [6, 7, 8]:
            season = 'summer'
        else:
            season = 'fall'
        
        # Get seasonal preferences
        seasonal_types = self.seasonal_appointment_preferences.get(season, [])
        
        # 60% chance to use seasonal preference, 40% use normal distribution
        if seasonal_types and self.faker.random.random() < 0.6:
            return self.faker.random_element(seasonal_types)
        else:
            return self._select_appointment_type()
    
    def _select_urgency_level_for_time_series(self, appointment_date: date) -> str:
        """Select urgency level with time series considerations"""
        # During flu season (winter months), more urgent appointments
        month = appointment_date.month
        if month in [12, 1, 2, 3]:
            # Increase urgent appointments during flu season
            adjusted_urgency = {
                'Routine': 0.60,
                'Urgent': 0.35,
                'Emergency': 0.05
            }
        else:
            adjusted_urgency = self.urgency_levels
        
        choices = list(adjusted_urgency.keys())
        weights = list(adjusted_urgency.values())
        
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
    
    def _select_appointment_time_with_constraints(self, appointment_date: date, doctor: Dict[str, Any],
                                                daily_counts: Dict, doctor_schedules: Dict) -> time:
        """Select appointment time considering scheduling constraints"""
        # Check doctor's daily capacity
        doctor_id = doctor['doctor_id']
        date_key = appointment_date
        
        # Get current appointment count for this doctor on this date
        current_count = daily_counts.get(date_key, {}).get(doctor_id, 0)
        max_daily_appointments = int(doctor['hours_per_day'] * doctor['patients_per_hour'])
        
        # If doctor is at capacity, try to find alternative time or adjust
        if current_count >= max_daily_appointments:
            # Try to find a less busy time slot
            available_times = list(self.time_slots.keys())
            used_times = doctor_schedules.get(doctor_id, {}).get(date_key, [])
            
            # Filter out already used times
            available_times = [t for t in available_times if t not in [str(ut) for ut in used_times]]
            
            if available_times:
                selected_time = self.faker.random_element(available_times)
            else:
                # If no available times, use normal selection (overbooking scenario)
                selected_time = self._select_appointment_time()
                return selected_time
        else:
            # Normal time selection
            selected_time = self._select_appointment_time()
            return selected_time
        
        # Parse time string to time object
        hour, minute = map(int, selected_time.split(':'))
        return time(hour, minute)
    
    def _apply_appointment_time_series_correlations(self, data: pd.DataFrame, ts_config) -> pd.DataFrame:
        """Apply realistic time-based correlations for appointment data"""
        if len(data) < 2:
            return data
        
        # Sort by appointment date and time to ensure proper time series order
        data = data.sort_values(['appointment_date', 'appointment_time']).reset_index(drop=True)
        
        # Apply temporal correlations for appointment scheduling patterns
        for i in range(1, len(data)):
            prev_row = data.iloc[i-1]
            current_row = data.iloc[i]
            
            # Follow-up appointment correlation
            # If previous appointment needed follow-up, increase chance of related appointment
            if (prev_row['follow_up_needed'] and 
                prev_row['patient_id'] == current_row['patient_id'] and
                (current_row['appointment_date'] - prev_row['appointment_date']).days <= 90):
                
                # This might be a follow-up appointment
                if self.faker.random.random() < 0.3:  # 30% chance
                    data.loc[i, 'appointment_type'] = 'Follow-up Visit'
                    data.loc[i, 'chief_complaint'] = 'Follow-up from previous visit'
                    data.loc[i, 'notes'] = 'Follow-up appointment as scheduled'
        
        # Apply seasonal volume adjustments
        for i in range(len(data)):
            appointment_date = data.iloc[i]['appointment_date']
            month = appointment_date.month
            
            # Apply seasonal billing adjustments (higher costs during peak seasons)
            seasonal_multiplier = self.seasonal_patterns.get(month, 1.0)
            
            # Adjust billing amount slightly based on seasonal demand
            current_amount = data.loc[i, 'billing_amount']
            if seasonal_multiplier > 1.1:  # High demand months
                adjusted_amount = current_amount * 1.05  # 5% increase
                data.loc[i, 'billing_amount'] = round(adjusted_amount, 2)
                
                # Recalculate insurance and copay
                insurance_rate = data.loc[i, 'insurance_covered'] / current_amount if current_amount > 0 else 0.8
                new_insurance_covered = round(adjusted_amount * insurance_rate, 2)
                new_copay = round(adjusted_amount - new_insurance_covered, 2)
                
                data.loc[i, 'insurance_covered'] = new_insurance_covered
                data.loc[i, 'copay_amount'] = new_copay
        
        # Apply doctor workload correlations
        doctor_workloads = data.groupby(['appointment_date', 'doctor_id']).size().reset_index(name='daily_appointments')
        
        for _, workload_row in doctor_workloads.iterrows():
            if workload_row['daily_appointments'] > 8:  # High workload day
                # Increase wait times and decrease satisfaction for this doctor on this date
                mask = ((data['appointment_date'] == workload_row['appointment_date']) & 
                       (data['doctor_id'] == workload_row['doctor_id']))
                
                # These adjustments will be applied in _apply_realistic_patterns
                # We can add a workload indicator for later processing
                data.loc[mask, 'high_workload_day'] = True
        
        # Fill NaN values for the workload indicator
        data['high_workload_day'] = data.get('high_workload_day', False)
        
        return data

    def _select_appointment_type(self) -> str:
        """Select appointment type based on distribution"""
        choices = list(self.appointment_types.keys())
        weights = [self.appointment_types[apt_type]['weight'] for apt_type in choices]
        
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
    
    def _select_urgency_level(self) -> str:
        """Select urgency level based on distribution"""
        choices = list(self.urgency_levels.keys())
        weights = list(self.urgency_levels.values())
        
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
    
    def _generate_appointment_datetime(self, urgency: str, date_range: Tuple[date, date] = None, 
                                     include_historical: bool = True) -> Tuple[date, time]:
        """Generate appointment date and time based on urgency and patterns"""
        # Determine date range
        if date_range:
            start_date, end_date = date_range
        else:
            if include_historical:
                start_date = datetime.now().date() - timedelta(days=180)
                end_date = datetime.now().date() + timedelta(days=90)
            else:
                start_date = datetime.now().date()
                end_date = datetime.now().date() + timedelta(days=90)
        
        # Generate lead time based on urgency
        lead_pattern = self.lead_time_patterns[urgency]
        lead_days = max(lead_pattern['min'], 
                       min(lead_pattern['max'],
                           int(self.faker.random.gauss(lead_pattern['mean'], lead_pattern['std']))))
        
        # Calculate appointment date
        if include_historical and self.faker.random.random() < 0.6:
            # 60% chance for historical appointments
            historical_end = min(datetime.now().date(), end_date)
            if start_date <= historical_end:
                appointment_date = self.faker.date_between(start_date=start_date, end_date=historical_end)
            else:
                # If no valid historical range, use future
                appointment_date = datetime.now().date() + timedelta(days=lead_days)
        else:
            # Future appointment
            appointment_date = datetime.now().date() + timedelta(days=lead_days)
        
        # Ensure appointment is within valid range
        appointment_date = max(start_date, min(end_date, appointment_date))
        
        # Adjust to business day
        appointment_date = self._adjust_to_business_day(appointment_date)
        
        # Generate appointment time
        appointment_time = self._select_appointment_time()
        
        return appointment_date, appointment_time
    
    def _adjust_to_business_day(self, target_date: date) -> date:
        """Adjust date to nearest business day"""
        # Skip weekends (Saturday=5, Sunday=6)
        while target_date.weekday() >= 5:
            target_date += timedelta(days=1)
        return target_date
    
    def _select_appointment_time(self) -> time:
        """Select appointment time based on preferences"""
        choices = list(self.time_slots.keys())
        weights = list(self.time_slots.values())
        
        cumulative_weights = []
        total = 0
        for weight in weights:
            total += weight
            cumulative_weights.append(total)
        
        rand_val = self.faker.random.uniform(0, total)
        for i, cum_weight in enumerate(cumulative_weights):
            if rand_val <= cum_weight:
                selected_time = choices[i]
                break
        else:
            selected_time = choices[-1]
        
        # Parse time string to time object
        hour, minute = map(int, selected_time.split(':'))
        return time(hour, minute)
    
    def _select_doctor_for_appointment(self, specialty: str) -> Dict[str, Any]:
        """Select appropriate doctor for appointment type"""
        specialty_doctors = [doc for doc in self.doctors if doc['specialty'] == specialty]
        
        if not specialty_doctors:
            # Fallback to primary care if specialty not found
            specialty_doctors = [doc for doc in self.doctors if doc['specialty'] == 'Primary Care']
        
        return self.faker.random_element(specialty_doctors)
    
    def _select_appointment_status(self, appointment_date: date) -> str:
        """Select appointment status based on date"""
        today = datetime.now().date()
        
        if appointment_date > today:
            # Future appointments are scheduled
            return 'Scheduled'
        elif appointment_date == today:
            # Today's appointments could be any status
            return self._weighted_choice(self.status_distribution)
        else:
            # Past appointments are mostly completed
            past_status_dist = {
                'Completed': 0.85,
                'Cancelled': 0.10,
                'No-Show': 0.05
            }
            return self._weighted_choice(past_status_dist)
    
    def _calculate_actual_duration(self, scheduled_duration: int, status: str) -> int:
        """Calculate actual appointment duration"""
        if status in ['Cancelled', 'No-Show']:
            return 0
        
        # Base duration
        actual_duration = scheduled_duration
        
        # Add overtime with some probability
        if self.faker.random.random() < self.overtime_probability:
            overtime = max(0, min(self.overtime_minutes['max'],
                                int(self.faker.random.gauss(
                                    self.overtime_minutes['mean'],
                                    self.overtime_minutes['std']
                                ))))
            actual_duration += overtime
        
        return actual_duration
    
    def _generate_appointment_details(self, appointment_type: str, urgency: str, 
                                    status: str, appointment_date: date) -> Dict[str, Any]:
        """Generate additional appointment details"""
        # Generate chief complaint based on appointment type
        chief_complaints = {
            'Annual Physical': ['Annual checkup', 'Routine physical exam', 'Preventive care visit'],
            'Routine Checkup': ['Follow-up visit', 'Routine check', 'General health assessment'],
            'Sick Visit': ['Cold symptoms', 'Flu-like symptoms', 'Fever', 'Cough', 'Sore throat'],
            'Cardiology Consultation': ['Chest pain', 'Heart palpitations', 'High blood pressure'],
            'Dermatology Screening': ['Skin lesion', 'Mole check', 'Rash', 'Skin cancer screening'],
            'Vaccination': ['Flu shot', 'COVID vaccine', 'Travel vaccines', 'Routine immunization']
        }
        
        complaint_options = chief_complaints.get(appointment_type, ['General consultation'])
        chief_complaint = self.faker.random_element(complaint_options)
        
        # Generate notes based on status
        if status == 'Completed':
            notes = f'Patient seen for {chief_complaint.lower()}. Examination completed.'
        elif status == 'Cancelled':
            notes = 'Appointment cancelled by patient.'
        elif status == 'No-Show':
            notes = 'Patient did not show for scheduled appointment.'
        else:
            notes = f'Scheduled appointment for {chief_complaint.lower()}.'
        
        # Determine follow-up needs
        follow_up_needed = False
        follow_up_weeks = 0
        
        if status == 'Completed' and urgency in ['Urgent', 'Emergency']:
            follow_up_needed = self.faker.random.random() < 0.4  # 40% chance
            if follow_up_needed:
                follow_up_weeks = self.faker.random_int(1, 8)
        elif status == 'Completed' and appointment_type in ['Annual Physical', 'Cardiology Consultation']:
            follow_up_needed = self.faker.random.random() < 0.2  # 20% chance
            if follow_up_needed:
                follow_up_weeks = self.faker.random_int(4, 26)
        
        # Generate creation and modification dates
        created_date = appointment_date - timedelta(days=self.faker.random_int(1, 30))
        last_modified = max(created_date, appointment_date - timedelta(days=self.faker.random_int(0, 7)))
        
        return {
            'chief_complaint': chief_complaint,
            'notes': notes,
            'follow_up_needed': follow_up_needed,
            'follow_up_weeks': follow_up_weeks,
            'created_date': created_date,
            'last_modified': last_modified
        }
    
    def _generate_room_assignment(self, specialty: str) -> str:
        """Generate room assignment based on specialty"""
        room_prefixes = {
            'Primary Care': 'PC',
            'Cardiology': 'CARD',
            'Dermatology': 'DERM',
            'Orthopedics': 'ORTHO',
            'Endocrinology': 'ENDO',
            'Psychiatry': 'PSY',
            'Ophthalmology': 'EYE',
            'Radiology': 'RAD',
            'Gastroenterology': 'GI'
        }
        
        prefix = room_prefixes.get(specialty, 'GEN')
        room_number = self.faker.random_int(1, 20)
        
        return f'{prefix}-{room_number:02d}'
    
    def _generate_billing_info(self, appointment_type: str, status: str) -> Dict[str, Any]:
        """Generate billing information"""
        # Base billing codes and amounts
        billing_codes = {
            'Annual Physical': {'code': '99396', 'amount': 250.00},
            'Routine Checkup': {'code': '99213', 'amount': 180.00},
            'Follow-up Visit': {'code': '99212', 'amount': 120.00},
            'Sick Visit': {'code': '99214', 'amount': 200.00},
            'Cardiology Consultation': {'code': '99243', 'amount': 350.00},
            'Dermatology Screening': {'code': '99203', 'amount': 220.00},
            'Orthopedic Consultation': {'code': '99244', 'amount': 380.00},
            'Vaccination': {'code': '90471', 'amount': 45.00}
        }
        
        billing_info = billing_codes.get(appointment_type, {'code': '99213', 'amount': 180.00})
        
        # Adjust amount for cancelled/no-show appointments
        if status in ['Cancelled', 'No-Show']:
            if status == 'No-Show':
                # Some practices charge a no-show fee
                billing_info['amount'] = 25.00 if self.faker.random.random() < 0.3 else 0.00
            else:
                billing_info['amount'] = 0.00
        
        # Calculate insurance coverage (typically 70-90%)
        if billing_info['amount'] > 0:
            insurance_rate = self.faker.random.uniform(0.70, 0.90)
            insurance_covered = round(billing_info['amount'] * insurance_rate, 2)
            copay = round(billing_info['amount'] - insurance_covered, 2)
        else:
            insurance_covered = 0.00
            copay = 0.00
        
        return {
            'code': billing_info['code'],
            'amount': billing_info['amount'],
            'insurance_covered': insurance_covered,
            'copay': copay
        }
    
    def _weighted_choice(self, choices_dict: Dict[str, float]) -> str:
        """Select item based on weighted probabilities"""
        choices = list(choices_dict.keys())
        weights = list(choices_dict.values())
        
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
    
    def _generate_appointment_date(self, start_date: date, end_date: date, 
                                 appointment_type: str, is_historical: bool) -> date:
        """Generate realistic appointment date"""
        if is_historical:
            # Historical appointments
            appointment_date = self.faker.date_between(start_date=start_date, end_date=end_date)
        else:
            # Future appointments
            base_date = max(datetime.now().date(), start_date)
            if base_date <= end_date:
                appointment_date = base_date + timedelta(days=lead_days)
                if appointment_date > end_date:
                    appointment_date = self.faker.date_between(start_date=base_date, end_date=end_date)
            else:
                # If base_date is after end_date, just use the range as-is
                appointment_date = self.faker.date_between(start_date=start_date, end_date=end_date)
        
        # Apply seasonal adjustment
        month_multiplier = self.seasonal_patterns.get(appointment_date.month, 1.0)
        if self.faker.random.random() > month_multiplier:
            # Adjust date slightly for seasonal effect
            adjustment = self.faker.random_int(-7, 7)
            try:
                appointment_date = appointment_date + timedelta(days=adjustment)
                appointment_date = max(start_date, min(end_date, appointment_date))
            except:
                pass  # Keep original date if adjustment fails
        
        # Ensure it's a valid business day
        appointment_date = self._adjust_to_business_day(appointment_date)
        
        # Generate appointment time
        appointment_time = self._select_appointment_time()
        
        return appointment_date, appointment_time
    
    def _adjust_to_business_day(self, appointment_date: date) -> date:
        """Adjust date to a valid business day"""
        # Most practices are closed on Sundays
        while appointment_date.weekday() == 6:  # Sunday
            appointment_date += timedelta(days=1)
        
        # Limited Saturday hours (only 5% of appointments)
        if appointment_date.weekday() == 5 and self.faker.random.random() > 0.05:
            appointment_date += timedelta(days=2)  # Move to Monday
        
        return appointment_date
    
    def _select_appointment_time(self) -> time:
        """Select appointment time based on preferences"""
        choices = list(self.time_slots.keys())
        weights = list(self.time_slots.values())
        
        cumulative_weights = []
        total = 0
        for weight in weights:
            total += weight
            cumulative_weights.append(total)
        
        rand_val = self.faker.random.uniform(0, total)
        selected_time = choices[-1]
        for i, cum_weight in enumerate(cumulative_weights):
            if rand_val <= cum_weight:
                selected_time = choices[i]
                break
        
        # Parse time string to time object
        hour, minute = map(int, selected_time.split(':'))
        return time(hour, minute)
    
    def _select_doctor_for_appointment(self, required_specialty: str) -> Dict[str, Any]:
        """Select appropriate doctor for appointment type"""
        # Filter doctors by specialty
        specialty_doctors = [doc for doc in self.doctors if doc['specialty'] == required_specialty]
        
        if not specialty_doctors:
            # Fallback to primary care if specialty not available
            specialty_doctors = [doc for doc in self.doctors if doc['specialty'] == 'Primary Care']
        
        return self.faker.random_element(specialty_doctors)
    
    def _select_appointment_status(self, appointment_date: date) -> str:
        """Select appointment status based on date"""
        today = datetime.now().date()
        
        if appointment_date > today:
            # Future appointments are scheduled
            return 'Scheduled'
        elif appointment_date == today:
            # Today's appointments could be any status
            return self.faker.random_element(['Scheduled', 'Completed', 'Cancelled', 'No-Show'])
        else:
            # Past appointments have final status
            choices = list(self.status_distribution.keys())
            weights = list(self.status_distribution.values())
            
            # Adjust weights for historical appointments (no scheduled)
            if 'Scheduled' in choices:
                scheduled_idx = choices.index('Scheduled')
                scheduled_weight = weights[scheduled_idx]
                weights[scheduled_idx] = 0
                
                # Redistribute scheduled weight to completed
                completed_idx = choices.index('Completed')
                weights[completed_idx] += scheduled_weight
            
            cumulative_weights = []
            total = 0
            for weight in weights:
                total += weight
                cumulative_weights.append(total)
            
            rand_val = self.faker.random.uniform(0, total)
            for i, cum_weight in enumerate(cumulative_weights):
                if rand_val <= cum_weight:
                    return choices[i]
            
            return 'Completed'
    
    def _calculate_actual_duration(self, scheduled_duration: int, status: str) -> int:
        """Calculate actual appointment duration"""
        if status in ['Cancelled', 'No-Show']:
            return 0
        
        actual_duration = scheduled_duration
        
        # Add overtime with some probability
        if self.faker.random.random() < self.overtime_probability:
            overtime = max(0, min(self.overtime_minutes['max'],
                                int(self.faker.random.gauss(self.overtime_minutes['mean'],
                                                           self.overtime_minutes['std']))))
            actual_duration += overtime
        
        return actual_duration
    
    def _generate_appointment_details(self, appointment_type: str, urgency: str, 
                                    status: str, appointment_date: date) -> Dict[str, Any]:
        """Generate additional appointment details"""
        # Generate chief complaint based on appointment type
        chief_complaints = {
            'Annual Physical': ['Routine checkup', 'Preventive care', 'Health maintenance'],
            'Routine Checkup': ['Follow-up', 'Medication review', 'General health'],
            'Follow-up Visit': ['Progress check', 'Medication adjustment', 'Test results review'],
            'Sick Visit': ['Cold symptoms', 'Flu-like symptoms', 'Stomach upset', 'Headache'],
            'Cardiology Consultation': ['Chest pain', 'Heart palpitations', 'High blood pressure'],
            'Dermatology Screening': ['Skin lesion', 'Rash', 'Mole check', 'Acne'],
            'Orthopedic Consultation': ['Joint pain', 'Back pain', 'Sports injury'],
            'Endocrinology Visit': ['Diabetes management', 'Thyroid issues', 'Hormone imbalance'],
            'Psychiatry Session': ['Depression', 'Anxiety', 'Medication management'],
            'Ophthalmology Exam': ['Vision changes', 'Eye exam', 'Glaucoma screening'],
            'Mammography': ['Breast screening', 'Routine mammogram'],
            'Colonoscopy': ['Colon screening', 'Preventive care'],
            'Vaccination': ['Flu shot', 'COVID vaccine', 'Travel vaccines']
        }
        
        chief_complaint = self.faker.random_element(
            chief_complaints.get(appointment_type, ['General consultation'])
        )
        
        # Generate notes based on status
        if status == 'Completed':
            notes = self.faker.random_element([
                'Patient seen and evaluated',
                'Treatment plan discussed',
                'Medications reviewed and adjusted',
                'Follow-up scheduled as needed',
                'Patient education provided'
            ])
        elif status == 'Cancelled':
            notes = self.faker.random_element([
                'Patient cancelled due to illness',
                'Scheduling conflict',
                'Transportation issues',
                'Cancelled by patient request'
            ])
        elif status == 'No-Show':
            notes = 'Patient did not show for appointment'
        else:
            notes = 'Appointment scheduled'
        
        # Determine follow-up needs
        follow_up_needed = False
        follow_up_weeks = None
        
        if status == 'Completed' and appointment_type not in ['Vaccination', 'Mammography']:
            if self.faker.random.random() < 0.4:  # 40% need follow-up
                follow_up_needed = True
                if appointment_type in ['Annual Physical', 'Routine Checkup']:
                    follow_up_weeks = 52  # Annual
                elif 'Consultation' in appointment_type:
                    follow_up_weeks = self.faker.random_int(2, 12)
                else:
                    follow_up_weeks = self.faker.random_int(4, 26)
        
        # Generate creation and modification dates
        created_date = appointment_date - timedelta(days=self.faker.random_int(1, 30))
        
        if status in ['Cancelled', 'No-Show']:
            last_modified = appointment_date
        else:
            last_modified = created_date + timedelta(days=self.faker.random_int(0, 7))
        
        return {
            'chief_complaint': chief_complaint,
            'notes': notes,
            'follow_up_needed': follow_up_needed,
            'follow_up_weeks': follow_up_weeks,
            'created_date': created_date,
            'last_modified': last_modified
        }
    
    def _generate_room_assignment(self, specialty: str) -> str:
        """Generate room assignment based on specialty"""
        room_prefixes = {
            'Primary Care': 'PC',
            'Cardiology': 'CARD',
            'Dermatology': 'DERM',
            'Orthopedics': 'ORTH',
            'Endocrinology': 'ENDO',
            'Psychiatry': 'PSY',
            'Ophthalmology': 'EYE',
            'Radiology': 'RAD',
            'Gastroenterology': 'GI'
        }
        
        prefix = room_prefixes.get(specialty, 'GEN')
        room_number = self.faker.random_int(1, 20)
        
        return f'{prefix}-{room_number:02d}'
    
    def _generate_billing_info(self, appointment_type: str, status: str) -> Dict[str, Any]:
        """Generate billing information"""
        # CPT codes and amounts by appointment type
        billing_codes = {
            'Annual Physical': {'code': '99396', 'amount': 250},
            'Routine Checkup': {'code': '99213', 'amount': 180},
            'Follow-up Visit': {'code': '99212', 'amount': 120},
            'Sick Visit': {'code': '99213', 'amount': 180},
            'Cardiology Consultation': {'code': '99243', 'amount': 350},
            'Dermatology Screening': {'code': '99213', 'amount': 200},
            'Orthopedic Consultation': {'code': '99243', 'amount': 320},
            'Endocrinology Visit': {'code': '99213', 'amount': 280},
            'Psychiatry Session': {'code': '90834', 'amount': 150},
            'Ophthalmology Exam': {'code': '92014', 'amount': 220},
            'Mammography': {'code': '77067', 'amount': 300},
            'Colonoscopy': {'code': '45378', 'amount': 800},
            'Vaccination': {'code': '90471', 'amount': 50}
        }
        
        billing_info = billing_codes.get(appointment_type, {'code': '99213', 'amount': 180})
        
        # No billing for cancelled or no-show appointments
        if status in ['Cancelled', 'No-Show']:
            return {
                'code': billing_info['code'],
                'amount': 0,
                'insurance_covered': 0,
                'copay': 0
            }
        
        # Calculate insurance coverage and copay
        base_amount = billing_info['amount']
        insurance_coverage_rate = self.faker.random.uniform(0.7, 0.9)  # 70-90% coverage
        insurance_covered = round(base_amount * insurance_coverage_rate, 2)
        copay = round(base_amount - insurance_covered, 2)
        
        return {
            'code': billing_info['code'],
            'amount': base_amount,
            'insurance_covered': insurance_covered,
            'copay': copay
        }
    
    def _apply_realistic_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply additional realistic patterns to appointment data"""
        # Add wait time for completed appointments
        data['wait_time_minutes'] = data.apply(
            lambda row: self._calculate_wait_time(row), axis=1
        )
        
        # Add patient satisfaction score for completed appointments
        data['patient_satisfaction'] = data.apply(
            lambda row: self._calculate_patient_satisfaction(row), axis=1
        ).round(1)
        
        # Add provider utilization score
        data['provider_utilization'] = data.apply(
            lambda row: self._calculate_provider_utilization(row), axis=1
        ).round(2)
        
        # Sort by appointment date and time
        data = data.sort_values(['appointment_date', 'appointment_time']).reset_index(drop=True)
        
        return data
    
    def _calculate_wait_time(self, row) -> int:
        """Calculate patient wait time"""
        if row['status'] not in ['Completed']:
            return 0
        
        # Base wait time varies by specialty
        base_wait_times = {
            'Primary Care': 15,
            'Cardiology': 25,
            'Dermatology': 20,
            'Orthopedics': 30,
            'Endocrinology': 20,
            'Psychiatry': 10,
            'Ophthalmology': 25,
            'Radiology': 10,
            'Gastroenterology': 20
        }
        
        base_wait = base_wait_times.get(row['specialty'], 15)
        
        # Add randomness and occasional long waits
        if self.faker.random.random() < 0.1:  # 10% chance of long wait
            wait_time = base_wait + self.faker.random_int(30, 90)
        else:
            wait_time = max(0, int(self.faker.random.gauss(base_wait, 8)))
        
        return wait_time
    
    def _calculate_patient_satisfaction(self, row) -> float:
        """Calculate patient satisfaction score (1-5)"""
        if row['status'] != 'Completed':
            return 0.0
        
        base_score = 4.0
        
        # Wait time affects satisfaction
        if row['wait_time_minutes'] > 45:
            base_score -= 1.0
        elif row['wait_time_minutes'] > 30:
            base_score -= 0.5
        elif row['wait_time_minutes'] < 10:
            base_score += 0.2
        
        # Appointment running over time affects satisfaction
        if row['actual_duration_minutes'] > row['scheduled_duration_minutes'] + 15:
            base_score -= 0.3
        
        # Specialty affects satisfaction (specialists generally higher)
        if row['specialty'] != 'Primary Care':
            base_score += 0.1
        
        # Add randomness
        base_score += self.faker.random.gauss(0, 0.4)
        
        return max(1.0, min(5.0, base_score))
    
    def _calculate_provider_utilization(self, row) -> float:
        """Calculate provider utilization score (0-1)"""
        # Base utilization varies by specialty
        base_utilization = {
            'Primary Care': 0.85,
            'Cardiology': 0.75,
            'Dermatology': 0.80,
            'Orthopedics': 0.78,
            'Endocrinology': 0.70,
            'Psychiatry': 0.82,
            'Ophthalmology': 0.76,
            'Radiology': 0.90,
            'Gastroenterology': 0.65
        }
        
        utilization = base_utilization.get(row['specialty'], 0.75)
        
        # Adjust for no-shows and cancellations
        if row['status'] in ['No-Show', 'Cancelled']:
            utilization *= 0.5  # Reduces utilization
        
        # Add some randomness
        utilization += self.faker.random.gauss(0, 0.05)
        
        return max(0.0, min(1.0, utilization))