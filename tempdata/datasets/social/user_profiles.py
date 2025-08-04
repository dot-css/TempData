"""
User profiles generator

Generates realistic user profile data with demographic distributions, interest
correlations, activity patterns, and realistic follower/following relationships.
"""

import pandas as pd
import json
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from collections import OrderedDict
from ...core.base_generator import BaseGenerator


class UserProfilesGenerator(BaseGenerator):
    """
    Generator for realistic user profile data
    
    Creates user profile datasets with demographic distributions, interest
    correlations, activity patterns, and realistic follower/following relationships.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_user_profile_data()
        self._setup_demographic_patterns()
        self._setup_interest_correlations()
        self._setup_relationship_patterns()
        self._setup_activity_patterns()
    
    def _load_user_profile_data(self):
        """Load user profile templates and patterns"""
        try:
            data_path = os.path.join(os.path.dirname(__file__), '../../data/templates/user_profile_data.json')
            with open(data_path, 'r') as f:
                self.profile_data = json.load(f)
            
            self.interests_categories = self.profile_data['interests_categories']
            self.demographic_patterns = self.profile_data['demographic_patterns']
            self.bio_templates = self.profile_data['bio_templates']
            self.username_patterns = self.profile_data['username_patterns']
            self.relationship_patterns = self.profile_data['relationship_patterns']
            self.verification_patterns = self.profile_data['verification_patterns']
            self.activity_patterns = self.profile_data['activity_patterns']
            
        except FileNotFoundError:
            # Fallback data if file not found
            self._setup_fallback_data()
    
    def _setup_fallback_data(self):
        """Setup fallback data if JSON file is not available"""
        self.interests_categories = {
            'technology': {
                'interests': ['programming', 'AI', 'web development'],
                'bio_keywords': ['tech enthusiast', 'developer', 'coder'],
                'hashtags': ['#tech', '#coding', '#programming']
            },
            'fitness': {
                'interests': ['running', 'yoga', 'weightlifting'],
                'bio_keywords': ['fitness lover', 'athlete', 'runner'],
                'hashtags': ['#fitness', '#health', '#workout']
            }
        }
        
        self.demographic_patterns = {
            'age_groups': {
                '25-34': {
                    'probability': 0.4,
                    'preferred_interests': ['technology', 'fitness'],
                    'bio_style': 'casual',
                    'follower_range': [100, 5000],
                    'following_range': [200, 2000]
                }
            },
            'gender_distribution': {'female': 0.5, 'male': 0.5}
        }
        
        self.bio_templates = {
            'casual': ["{interest_description} | {location}"]
        }
        
        self.username_patterns = {
            'styles': ['{first_name}_{last_name}'],
            'adjectives': ['cool', 'awesome'],
            'numbers': ['2024', '123']
        }
        
        self.relationship_patterns = {
            'follower_following_ratios': {
                'average': {'min_ratio': 0.5, 'max_ratio': 2}
            }
        }
        
        self.verification_patterns = {
            'verified_probability': {'regular': 0.001}
        }
        
        self.activity_patterns = {
            'posting_frequency': {
                'medium': {'posts_per_week': 7, 'stories_per_week': 10}
            }
        }
    
    def _setup_demographic_patterns(self):
        """Setup demographic distribution patterns"""
        # Age group weights for selection
        self.age_weights = OrderedDict([
            (age_group, data['probability'])
            for age_group, data in self.demographic_patterns['age_groups'].items()
        ])
        
        # Gender weights for selection
        self.gender_weights = OrderedDict([
            (gender, probability)
            for gender, probability in self.demographic_patterns['gender_distribution'].items()
        ])
        
        # Location type weights
        location_types = self.demographic_patterns.get('location_types', {
            'urban': 0.65, 'suburban': 0.25, 'rural': 0.10
        })
        self.location_weights = OrderedDict(location_types.items())
    
    def _setup_interest_correlations(self):
        """Setup interest correlation patterns"""
        # Create interest correlation matrix
        self.interest_correlations = {
            'technology': ['business', 'education'],
            'fitness': ['lifestyle', 'food'],
            'travel': ['food', 'arts', 'lifestyle'],
            'food': ['lifestyle', 'travel'],
            'arts': ['lifestyle', 'entertainment'],
            'business': ['technology', 'education'],
            'lifestyle': ['fitness', 'food', 'arts'],
            'education': ['technology', 'business', 'social_causes'],
            'entertainment': ['arts', 'lifestyle'],
            'social_causes': ['education', 'lifestyle']
        }
        
        # Interest popularity weights
        self.interest_popularity = OrderedDict([
            ('lifestyle', 0.20),
            ('technology', 0.15),
            ('fitness', 0.12),
            ('food', 0.12),
            ('travel', 0.10),
            ('entertainment', 0.10),
            ('business', 0.08),
            ('arts', 0.06),
            ('education', 0.04),
            ('social_causes', 0.03)
        ])
    
    def _setup_relationship_patterns(self):
        """Setup follower/following relationship patterns"""
        # User types based on follower/following ratios
        self.user_types = {
            'influencer': {'probability': 0.02, 'min_followers': 10000},
            'popular': {'probability': 0.08, 'min_followers': 1000},
            'average': {'probability': 0.80, 'min_followers': 50},
            'new_user': {'probability': 0.10, 'min_followers': 0}
        }
        
        # Engagement patterns by user type
        self.engagement_by_type = {
            'influencer': {'high': 0.6, 'medium': 0.3, 'low': 0.1},
            'popular': {'high': 0.4, 'medium': 0.5, 'low': 0.1},
            'average': {'high': 0.2, 'medium': 0.6, 'low': 0.2},
            'new_user': {'high': 0.1, 'medium': 0.4, 'low': 0.5}
        }
    
    def _setup_activity_patterns(self):
        """Setup user activity patterns"""
        # Activity level distribution
        self.activity_levels = OrderedDict([
            ('high', 0.15),
            ('medium', 0.60),
            ('low', 0.25)
        ])
        
        # Online time preferences
        self.online_preferences = OrderedDict([
            ('morning_person', 0.25),
            ('day_active', 0.35),
            ('evening_person', 0.30),
            ('night_owl', 0.10)
        ])
    
    def generate(self, rows: int, **kwargs) -> pd.DataFrame:
        """
        Generate user profiles dataset
        
        Args:
            rows: Number of user profiles to generate
            **kwargs: Additional parameters (platform, country, etc.)
            
        Returns:
            pd.DataFrame: Generated user profiles data with realistic patterns
        """
        platform = kwargs.get('platform', 'general')
        country = kwargs.get('country', 'global')
        include_relationships = kwargs.get('include_relationships', True)
        
        data = []
        
        for i in range(rows):
            # Generate demographic characteristics
            age_group = self._select_age_group()
            gender = self._select_gender()
            location_type = self._select_location_type()
            
            # Generate basic profile info
            user_id = f'USR_{i+1:06d}'
            first_name, last_name = self._generate_names(gender)
            username = self._generate_username(first_name, last_name, age_group)
            display_name = self._generate_display_name(first_name, last_name, age_group)
            
            # Generate interests and bio
            primary_interests = self._select_interests(age_group)
            bio = self._generate_bio(age_group, primary_interests, location_type)
            
            # Generate follower/following counts and relationships
            user_type = self._determine_user_type()
            followers_count, following_count = self._generate_follower_counts(user_type, age_group)
            
            # Generate activity patterns
            activity_level = self._select_activity_level(age_group)
            online_preference = self._select_online_preference(age_group)
            
            # Generate account details
            join_date = self._generate_join_date(age_group)
            is_verified = self._determine_verification_status(user_type, followers_count)
            is_private = self._determine_privacy_setting(age_group, user_type)
            
            # Generate engagement metrics
            engagement_level = self._select_engagement_level(user_type)
            avg_likes_per_post, avg_comments_per_post = self._calculate_engagement_metrics(
                followers_count, engagement_level
            )
            
            # Generate posting patterns
            posts_per_week, stories_per_week = self._generate_posting_frequency(activity_level)
            
            # Create profile record
            profile = {
                'user_id': user_id,
                'username': username,
                'display_name': display_name,
                'first_name': first_name,
                'last_name': last_name,
                'bio': bio,
                'age_group': age_group,
                'gender': gender,
                'location_type': location_type,
                'location': self._generate_location(location_type, country),
                'primary_interests': ', '.join(primary_interests),
                'followers_count': followers_count,
                'following_count': following_count,
                'user_type': user_type,
                'activity_level': activity_level,
                'online_preference': online_preference,
                'join_date': join_date,
                'account_age_days': (datetime.now().date() - join_date).days,
                'is_verified': is_verified,
                'is_private': is_private,
                'engagement_level': engagement_level,
                'avg_likes_per_post': avg_likes_per_post,
                'avg_comments_per_post': avg_comments_per_post,
                'posts_per_week': posts_per_week,
                'stories_per_week': stories_per_week,
                'follower_following_ratio': round(followers_count / max(following_count, 1), 2),
                'bio_length': len(bio),
                'has_profile_picture': self.faker.random.random() > 0.05,  # 95% have profile pics
                'has_bio': len(bio.strip()) > 0
            }
            
            data.append(profile)
        
        df = pd.DataFrame(data)
        return self._apply_realistic_patterns(df)
    
    def _select_age_group(self) -> str:
        """Select age group based on demographic distributions"""
        return self.faker.random_element(elements=self.age_weights)
    
    def _select_gender(self) -> str:
        """Select gender based on demographic distributions"""
        return self.faker.random_element(elements=self.gender_weights)
    
    def _select_location_type(self) -> str:
        """Select location type (urban, suburban, rural)"""
        return self.faker.random_element(elements=self.location_weights)
    
    def _generate_names(self, gender: str) -> Tuple[str, str]:
        """Generate first and last names based on gender"""
        if gender == 'female':
            first_name = self.faker.first_name_female()
        elif gender == 'male':
            first_name = self.faker.first_name_male()
        else:  # non_binary
            first_name = self.faker.first_name()
        
        last_name = self.faker.last_name()
        return first_name, last_name
    
    def _generate_username(self, first_name: str, last_name: str, age_group: str) -> str:
        """Generate realistic username based on name and age group"""
        # Younger users tend to have more creative usernames
        if age_group in ['18-24', '25-34']:
            creative_probability = 0.6
        else:
            creative_probability = 0.3
        
        if self.faker.random.random() < creative_probability:
            # Creative username
            style = self.faker.random_element(self.username_patterns['styles'])
            
            # Get primary interest for username
            interests = list(self.interests_categories.keys())
            interest = self.faker.random_element(interests)
            
            replacements = {
                'first_name': first_name.lower(),
                'last_name': last_name.lower(),
                'interest': interest,
                'adjective': self.faker.random_element(self.username_patterns['adjectives']),
                'number': self.faker.random_element(self.username_patterns['numbers']),
                'profession': self.faker.job().lower().replace(' ', ''),
                'year': str(self.faker.random_int(2020, 2024)),
                'hobby': self.faker.random_element(['music', 'art', 'travel', 'food']),
                'enthusiast': self.faker.random_element(self.username_patterns['enthusiast_words'])
            }
            
            username = style.format(**replacements)
        else:
            # Simple name-based username
            if self.faker.random.random() < 0.5:
                username = f"{first_name.lower()}_{last_name.lower()}"
            else:
                username = f"{first_name.lower()}{last_name.lower()}"
        
        # Ensure username is unique by adding numbers if needed
        if self.faker.random.random() < 0.2:
            username += str(self.faker.random_int(1, 999))
        
        return username[:30]  # Limit username length
    
    def _generate_display_name(self, first_name: str, last_name: str, age_group: str) -> str:
        """Generate display name based on age group preferences"""
        if age_group in ['18-24']:
            # Younger users often use nicknames or creative names
            if self.faker.random.random() < 0.4:
                return first_name  # Just first name
            elif self.faker.random.random() < 0.3:
                return f"{first_name} {last_name[0]}."  # First name + last initial
        
        # Most users use full name
        return f"{first_name} {last_name}"
    
    def _select_interests(self, age_group: str) -> List[str]:
        """Select primary interests based on age group preferences"""
        age_data = self.demographic_patterns['age_groups'][age_group]
        preferred_interests = age_data.get('preferred_interests', list(self.interests_categories.keys()))
        
        # Select 1-3 primary interests
        num_interests = self.faker.random_int(1, 3)
        
        # 70% chance to select from preferred interests
        if self.faker.random.random() < 0.7 and preferred_interests:
            primary_interest = self.faker.random_element(preferred_interests)
        else:
            primary_interest = self.faker.random_element(elements=self.interest_popularity)
        
        interests = [primary_interest]
        
        # Add correlated interests
        for _ in range(num_interests - 1):
            if primary_interest in self.interest_correlations:
                correlated = self.faker.random_element(self.interest_correlations[primary_interest])
                if correlated not in interests:
                    interests.append(correlated)
            else:
                # Add random interest
                random_interest = self.faker.random_element(elements=self.interest_popularity)
                if random_interest not in interests:
                    interests.append(random_interest)
        
        return interests
    
    def _generate_bio(self, age_group: str, interests: List[str], location_type: str) -> str:
        """Generate realistic bio based on age group and interests"""
        age_data = self.demographic_patterns['age_groups'][age_group]
        bio_style = age_data.get('bio_style', 'casual')
        
        # Select bio template
        templates = self.bio_templates.get(bio_style, self.bio_templates['casual'])
        template = self.faker.random_element(templates)
        
        # Get interest data for bio generation
        primary_interest = interests[0] if interests else 'lifestyle'
        interest_data = self.interests_categories.get(primary_interest, {})
        
        # Generate bio components
        replacements = {
            'interest_description': self._generate_interest_description(interests),
            'location': self._generate_bio_location(location_type),
            'fun_fact': self._generate_fun_fact(),
            'profession': self._generate_profession(age_group, primary_interest),
            'hobby': self.faker.random_element(interest_data.get('interests', ['reading'])),
            'adjective': self.faker.random_element(['passionate', 'creative', 'dedicated', 'enthusiastic']),
            'emoji': self.faker.random_element(['✨', '🌟', '💫', '🚀', '💪', '🎯', '']),
            'age': self._generate_age_from_group(age_group),
            'motivational_quote': self.faker.random_element([
                'Living my best life', 'Dream big', 'Stay positive', 'Keep growing'
            ]),
            'contact_info': self._generate_contact_info(),
            'company_role': self._generate_company_role(age_group),
            'expertise': self._generate_expertise(primary_interest),
            'topic': primary_interest.replace('_', ' '),
            'achievement': self._generate_achievement(primary_interest),
            'senior_title': self._generate_senior_title(age_group),
            'company_type': self.faker.random_element(['startup', 'corporation', 'agency', 'consultancy']),
            'years': str(self.faker.random_int(2, 20)),
            'specialization': self._generate_specialization(primary_interest),
            'title': self._generate_title(age_group),
            'industry': self._generate_industry(primary_interest),
            'team_type': self.faker.random_element(['development', 'marketing', 'sales', 'design']),
            'role': self.faker.random_element(['consultant', 'advisor', 'mentor', 'coach']),
            'executive_title': self.faker.random_element(['CEO', 'CTO', 'VP', 'Director']),
            'professional_goal': self._generate_professional_goal(),
            'formal_title': self._generate_formal_title(age_group),
            'formal_achievement': self._generate_formal_achievement(),
            'formal_goal': self._generate_formal_goal(),
            'credential': self._generate_credential()
        }
        
        # Fill template
        bio = template
        for key, value in replacements.items():
            bio = bio.replace(f'{{{key}}}', str(value))
        
        # Limit bio length (most platforms have limits)
        if len(bio) > 160:
            bio = bio[:157] + '...'
        
        return bio
    
    def _generate_interest_description(self, interests: List[str]) -> str:
        """Generate description based on interests"""
        if not interests:
            return "Exploring life's adventures"
        
        primary_interest = interests[0]
        interest_data = self.interests_categories.get(primary_interest, {})
        keywords = interest_data.get('bio_keywords', [primary_interest])
        
        return self.faker.random_element(keywords)
    
    def _generate_bio_location(self, location_type: str) -> str:
        """Generate location for bio"""
        if location_type == 'urban':
            return self.faker.city()
        elif location_type == 'suburban':
            return f"{self.faker.city()} suburbs"
        else:  # rural
            return f"{self.faker.state()}"
    
    def _generate_fun_fact(self) -> str:
        """Generate fun fact for bio"""
        facts = [
            "Coffee enthusiast ☕", "Dog lover 🐕", "Bookworm 📚", "Foodie 🍕",
            "Adventure seeker 🏔️", "Music lover 🎵", "Art enthusiast 🎨",
            "Fitness fanatic 💪", "Tech geek 💻", "Nature lover 🌿"
        ]
        return self.faker.random_element(facts)
    
    def _generate_profession(self, age_group: str, interest: str) -> str:
        """Generate profession based on age and interests"""
        if interest == 'technology':
            professions = ['Developer', 'Engineer', 'Designer', 'Analyst', 'Consultant']
        elif interest == 'business':
            professions = ['Manager', 'Consultant', 'Analyst', 'Coordinator', 'Specialist']
        elif interest == 'arts':
            professions = ['Designer', 'Artist', 'Writer', 'Photographer', 'Creator']
        else:
            professions = ['Professional', 'Specialist', 'Consultant', 'Manager', 'Expert']
        
        return self.faker.random_element(professions)
    
    def _generate_age_from_group(self, age_group: str) -> int:
        """Generate specific age from age group"""
        ranges = {
            '18-24': (18, 24),
            '25-34': (25, 34),
            '35-44': (35, 44),
            '45-54': (45, 54),
            '55+': (55, 70)
        }
        min_age, max_age = ranges.get(age_group, (25, 35))
        return self.faker.random_int(min_age, max_age)
    
    def _generate_contact_info(self) -> str:
        """Generate contact info for bio"""
        options = [
            f"📧 {self.faker.email()}",
            f"💼 LinkedIn",
            f"📱 DM for collabs",
            f"🌐 {self.faker.domain_name()}"
        ]
        return self.faker.random_element(options)
    
    def _generate_company_role(self, age_group: str) -> str:
        """Generate company role based on age"""
        if age_group in ['18-24']:
            roles = ['Junior Developer', 'Associate', 'Coordinator', 'Analyst']
        elif age_group in ['25-34']:
            roles = ['Developer', 'Manager', 'Consultant', 'Specialist']
        else:
            roles = ['Senior Manager', 'Director', 'VP', 'Principal']
        
        return self.faker.random_element(roles)
    
    def _generate_expertise(self, interest: str) -> str:
        """Generate expertise based on interest"""
        expertise_map = {
            'technology': ['AI/ML', 'Web Development', 'Data Science', 'Cybersecurity'],
            'business': ['Strategy', 'Marketing', 'Operations', 'Finance'],
            'fitness': ['Personal Training', 'Nutrition', 'Wellness', 'Sports'],
            'food': ['Culinary Arts', 'Nutrition', 'Food Photography', 'Recipe Development']
        }
        
        options = expertise_map.get(interest, ['Leadership', 'Innovation', 'Strategy'])
        return self.faker.random_element(options)
    
    def _generate_achievement(self, interest: str) -> str:
        """Generate achievement based on interest"""
        achievements = {
            'technology': ['Built 10+ apps', 'Led dev teams', 'Published researcher'],
            'business': ['Grew revenue 200%', 'Led 50+ projects', 'Award winner'],
            'fitness': ['Marathon finisher', 'Certified trainer', 'Wellness coach']
        }
        
        options = achievements.get(interest, ['Industry expert', 'Thought leader', 'Award winner'])
        return self.faker.random_element(options)
    
    def _generate_senior_title(self, age_group: str) -> str:
        """Generate senior title based on age"""
        if age_group in ['45-54', '55+']:
            return self.faker.random_element(['Senior VP', 'Executive Director', 'Chief Officer'])
        elif age_group in ['35-44']:
            return self.faker.random_element(['Director', 'VP', 'Senior Manager'])
        else:
            return self.faker.random_element(['Manager', 'Lead', 'Principal'])
    
    def _generate_specialization(self, interest: str) -> str:
        """Generate specialization based on interest"""
        return self._generate_expertise(interest)
    
    def _generate_title(self, age_group: str) -> str:
        """Generate professional title"""
        return self._generate_company_role(age_group)
    
    def _generate_industry(self, interest: str) -> str:
        """Generate industry based on interest"""
        industry_map = {
            'technology': 'Tech',
            'business': 'Business',
            'fitness': 'Health & Wellness',
            'food': 'Food & Beverage',
            'arts': 'Creative',
            'education': 'Education'
        }
        
        return industry_map.get(interest, 'Professional Services')
    
    def _generate_professional_goal(self) -> str:
        """Generate professional goal"""
        goals = [
            'Driving innovation', 'Building great teams', 'Creating value',
            'Solving problems', 'Leading change', 'Empowering others'
        ]
        return self.faker.random_element(goals)
    
    def _generate_formal_title(self, age_group: str) -> str:
        """Generate formal title"""
        return self._generate_senior_title(age_group)
    
    def _generate_formal_achievement(self) -> str:
        """Generate formal achievement"""
        achievements = [
            '20+ years experience', 'Industry leader', 'Published author',
            'Keynote speaker', 'Board member', 'Award recipient'
        ]
        return self.faker.random_element(achievements)
    
    def _generate_formal_goal(self) -> str:
        """Generate formal goal"""
        goals = [
            'Advancing industry standards', 'Mentoring next generation',
            'Driving organizational excellence', 'Leading transformation'
        ]
        return self.faker.random_element(goals)
    
    def _generate_credential(self) -> str:
        """Generate professional credential"""
        credentials = ['MBA', 'PhD', 'CPA', 'PMP', 'Certified Expert']
        return self.faker.random_element(credentials)
    
    def _determine_user_type(self) -> str:
        """Determine user type based on probability distribution"""
        user_type_weights = OrderedDict([
            (user_type, data['probability'])
            for user_type, data in self.user_types.items()
        ])
        
        return self.faker.random_element(elements=user_type_weights)
    
    def _generate_follower_counts(self, user_type: str, age_group: str) -> Tuple[int, int]:
        """Generate realistic follower and following counts"""
        age_data = self.demographic_patterns['age_groups'][age_group]
        follower_range = age_data.get('follower_range', [100, 5000])
        following_range = age_data.get('following_range', [200, 2000])
        
        # Adjust ranges based on user type
        if user_type == 'influencer':
            followers = self.faker.random_int(10000, 1000000)
            following = self.faker.random_int(500, 5000)
        elif user_type == 'popular':
            followers = self.faker.random_int(1000, 50000)
            following = self.faker.random_int(300, 3000)
        elif user_type == 'new_user':
            followers = self.faker.random_int(0, 100)
            following = self.faker.random_int(50, 500)
        else:  # average
            followers = self.faker.random_int(follower_range[0], follower_range[1])
            following = self.faker.random_int(following_range[0], following_range[1])
        
        return followers, following
    
    def _select_activity_level(self, age_group: str) -> str:
        """Select activity level based on age group"""
        # Younger users tend to be more active
        if age_group in ['18-24']:
            weights = OrderedDict([('high', 0.3), ('medium', 0.5), ('low', 0.2)])
        elif age_group in ['25-34']:
            weights = OrderedDict([('high', 0.2), ('medium', 0.6), ('low', 0.2)])
        else:
            weights = OrderedDict([('high', 0.1), ('medium', 0.5), ('low', 0.4)])
        
        return self.faker.random_element(elements=weights)
    
    def _select_online_preference(self, age_group: str) -> str:
        """Select online time preference"""
        return self.faker.random_element(elements=self.online_preferences)
    
    def _generate_join_date(self, age_group: str) -> datetime:
        """Generate realistic join date based on age group"""
        # Younger users joined more recently, older users have been around longer
        if age_group in ['18-24']:
            # Joined in last 2-5 years
            start_date = datetime.now() - timedelta(days=5*365)
            end_date = datetime.now() - timedelta(days=2*365)
        elif age_group in ['25-34']:
            # Joined in last 3-8 years
            start_date = datetime.now() - timedelta(days=8*365)
            end_date = datetime.now() - timedelta(days=3*365)
        else:
            # Joined in last 5-12 years
            start_date = datetime.now() - timedelta(days=12*365)
            end_date = datetime.now() - timedelta(days=5*365)
        
        return self.faker.date_between(start_date=start_date, end_date=end_date)
    
    def _determine_verification_status(self, user_type: str, followers_count: int) -> bool:
        """Determine if user should be verified"""
        verification_prob = self.verification_patterns['verified_probability'].get(user_type, 0.001)
        
        # Higher chance if many followers
        if followers_count > 100000:
            verification_prob *= 10
        elif followers_count > 10000:
            verification_prob *= 3
        
        return self.faker.random.random() < verification_prob
    
    def _determine_privacy_setting(self, age_group: str, user_type: str) -> bool:
        """Determine if account should be private"""
        # Younger users and non-influencers more likely to be private
        if age_group in ['18-24']:
            privacy_prob = 0.3
        elif user_type == 'influencer':
            privacy_prob = 0.05  # Influencers rarely private
        else:
            privacy_prob = 0.15
        
        return self.faker.random.random() < privacy_prob
    
    def _select_engagement_level(self, user_type: str) -> str:
        """Select engagement level based on user type"""
        engagement_dist = self.engagement_by_type.get(user_type, {'medium': 1.0})
        engagement_weights = OrderedDict(engagement_dist.items())
        
        return self.faker.random_element(elements=engagement_weights)
    
    def _calculate_engagement_metrics(self, followers_count: int, engagement_level: str) -> Tuple[int, int]:
        """Calculate average engagement metrics"""
        engagement_rates = {
            'high': {'likes': 0.05, 'comments': 0.01},
            'medium': {'likes': 0.02, 'comments': 0.005},
            'low': {'likes': 0.01, 'comments': 0.002}
        }
        
        rates = engagement_rates.get(engagement_level, engagement_rates['medium'])
        
        avg_likes = int(followers_count * rates['likes'])
        avg_comments = int(followers_count * rates['comments'])
        
        return avg_likes, avg_comments
    
    def _generate_posting_frequency(self, activity_level: str) -> Tuple[int, int]:
        """Generate posting frequency based on activity level"""
        frequency_data = self.activity_patterns['posting_frequency'].get(activity_level, {
            'posts_per_week': 7, 'stories_per_week': 10
        })
        
        posts_per_week = frequency_data['posts_per_week']
        stories_per_week = frequency_data['stories_per_week']
        
        # Add some randomness
        posts_per_week = max(1, int(posts_per_week * self.faker.random.uniform(0.7, 1.3)))
        stories_per_week = max(0, int(stories_per_week * self.faker.random.uniform(0.5, 1.5)))
        
        return posts_per_week, stories_per_week
    
    def _generate_location(self, location_type: str, country: str) -> str:
        """Generate location based on type and country"""
        if country != 'global':
            # Use country-specific location
            return self.faker.city()
        
        if location_type == 'urban':
            return self.faker.city()
        elif location_type == 'suburban':
            return f"{self.faker.city()}, {self.faker.state()}"
        else:  # rural
            return f"{self.faker.state()}"
    
    def _apply_realistic_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply additional realistic patterns to the generated data"""
        # Ensure follower/following ratios make sense (keep the rounded values from generation)
        data['follower_following_ratio'] = (data['followers_count'] / data['following_count'].clip(lower=1)).round(2)
        
        # Add derived metrics
        data['engagement_rate'] = (data['avg_likes_per_post'] / data['followers_count'].clip(lower=1) * 100).round(2)
        data['posts_per_day'] = (data['posts_per_week'] / 7).round(1)
        data['total_weekly_content'] = data['posts_per_week'] + data['stories_per_week']
        
        # Add account quality score
        data['account_quality_score'] = self._calculate_account_quality_score(data)
        
        # Add influence score
        data['influence_score'] = self._calculate_influence_score(data)
        
        # Sort by followers count for realistic ordering
        data = data.sort_values('followers_count', ascending=False).reset_index(drop=True)
        
        return data
    
    def _calculate_account_quality_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate account quality score based on various factors"""
        # Factors: has bio, has profile pic, account age, engagement rate, verification
        score = 0
        
        # Bio presence (20 points)
        score += data['has_bio'].astype(int) * 20
        
        # Profile picture (15 points)
        score += data['has_profile_picture'].astype(int) * 15
        
        # Account age (25 points max, scaled)
        max_age = data['account_age_days'].max()
        score += (data['account_age_days'] / max_age * 25).fillna(0)
        
        # Engagement rate (25 points max, scaled)
        max_engagement = data['engagement_rate'].max()
        if max_engagement > 0:
            score += (data['engagement_rate'] / max_engagement * 25).fillna(0)
        
        # Verification bonus (15 points)
        score += data['is_verified'].astype(int) * 15
        
        return score.round(1)
    
    def _calculate_influence_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate influence score based on followers, engagement, and activity"""
        # Normalize followers count (log scale)
        import numpy as np
        
        followers_score = np.log10(data['followers_count'].clip(lower=1)) * 10
        engagement_score = data['engagement_rate'] * 2
        activity_score = data['posts_per_week'] * 1.5
        verification_bonus = data['is_verified'].astype(int) * 10
        
        influence_score = followers_score + engagement_score + activity_score + verification_bonus
        
        # Scale to 0-100
        max_score = influence_score.max()
        if max_score > 0:
            influence_score = (influence_score / max_score * 100).round(1)
        
        return influence_score