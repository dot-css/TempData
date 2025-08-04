"""
Social media posts generator

Generates realistic social media data with posting patterns, engagement distributions,
content types, and realistic hashtag usage with viral content patterns.
"""

import pandas as pd
import json
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from collections import OrderedDict
from ...core.base_generator import BaseGenerator


class SocialMediaGenerator(BaseGenerator):
    """
    Generator for realistic social media posts data
    
    Creates social media datasets with posting patterns, engagement distributions,
    content types, realistic hashtag usage, and viral content patterns.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_social_media_data()
        self._setup_platform_characteristics()
        self._setup_engagement_patterns()
        self._setup_viral_mechanics()
        self._setup_user_behavior_patterns()
    
    def _load_social_media_data(self):
        """Load social media content templates and patterns"""
        try:
            data_path = os.path.join(os.path.dirname(__file__), '../../data/templates/social_media_content.json')
            with open(data_path, 'r') as f:
                self.social_data = json.load(f)
            
            self.hashtag_categories = self.social_data['hashtag_categories']
            self.content_templates = self.social_data['content_templates']
            self.viral_patterns = self.social_data['viral_patterns']
            self.platform_specific = self.social_data['platform_specific']
            self.engagement_patterns = self.social_data['engagement_patterns']
            
        except FileNotFoundError:
            # Fallback data if file not found
            self._setup_fallback_data()
    
    def _setup_fallback_data(self):
        """Setup fallback data if JSON file is not available"""
        self.hashtag_categories = {
            'lifestyle': ['#lifestyle', '#life', '#daily', '#mood'],
            'food': ['#food', '#foodie', '#delicious', '#yummy'],
            'travel': ['#travel', '#wanderlust', '#adventure', '#explore'],
            'fitness': ['#fitness', '#workout', '#gym', '#health'],
            'technology': ['#tech', '#technology', '#innovation', '#digital']
        }
        
        self.content_templates = {
            'personal_update': ["Just finished {activity} and feeling {emotion}! {hashtags}"],
            'question': ["What's your favorite {topic}? {hashtags}"],
            'motivational': ["Remember: {motivational_quote} {hashtags}"],
            'sharing': ["Just discovered {discovery} and had to share! {hashtags}"],
            'event': ["Just attended {event} and it was {adjective}! {hashtags}"]
        }
        
        self.viral_patterns = {
            'trending_topics': ['AI revolution', 'climate change', 'remote work'],
            'viral_formats': ['challenge', 'tutorial', 'reaction'],
            'engagement_boosters': ['tag a friend', 'double tap if', 'comment below']
        }
        
        self.platform_specific = {
            'twitter': {'max_length': 280, 'hashtag_heavy': False},
            'instagram': {'max_length': 2200, 'hashtag_heavy': True},
            'facebook': {'max_length': 63206, 'hashtag_heavy': False}
        }
        
        self.engagement_patterns = {
            'time_based': {'peak_hours': [9, 12, 15, 18, 21]},
            'content_type_multipliers': {'video': 2.5, 'image': 1.8, 'text': 1.0}
        }
    
    def _setup_platform_characteristics(self):
        """Setup platform-specific posting characteristics"""
        self.platforms = ['twitter', 'instagram', 'facebook', 'tiktok', 'linkedin']
        
        # Platform posting frequency (posts per day per active user)
        self.platform_posting_frequency = {
            'twitter': 3.2,
            'instagram': 1.1,
            'facebook': 0.7,
            'tiktok': 2.8,
            'linkedin': 0.3
        }
        
        # Platform user demographics (age distribution)
        self.platform_demographics = {
            'twitter': {'18-24': 0.2, '25-34': 0.35, '35-44': 0.25, '45-54': 0.15, '55+': 0.05},
            'instagram': {'18-24': 0.35, '25-34': 0.32, '35-44': 0.20, '45-54': 0.10, '55+': 0.03},
            'facebook': {'18-24': 0.15, '25-34': 0.25, '35-44': 0.25, '45-54': 0.20, '55+': 0.15},
            'tiktok': {'18-24': 0.50, '25-34': 0.30, '35-44': 0.15, '45-54': 0.04, '55+': 0.01},
            'linkedin': {'18-24': 0.10, '25-34': 0.40, '35-44': 0.30, '45-54': 0.15, '55+': 0.05}
        }
    
    def _setup_engagement_patterns(self):
        """Setup realistic engagement patterns"""
        # Base engagement rates by platform (likes per 1000 followers)
        self.base_engagement_rates = {
            'twitter': {'likes': 24, 'shares': 3, 'comments': 2},
            'instagram': {'likes': 67, 'shares': 8, 'comments': 4},
            'facebook': {'likes': 45, 'shares': 12, 'comments': 6},
            'tiktok': {'likes': 89, 'shares': 15, 'comments': 8},
            'linkedin': {'likes': 32, 'shares': 5, 'comments': 3}
        }
        
        # Content type engagement multipliers
        self.content_multipliers = {
            'video': 2.5,
            'image': 1.8,
            'carousel': 1.6,
            'text': 1.0,
            'poll': 1.4,
            'story': 1.2
        }
        
        # Time-based engagement patterns
        self.hourly_engagement_multipliers = {
            0: 0.3, 1: 0.2, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.2,
            6: 0.4, 7: 0.6, 8: 0.8, 9: 1.2, 10: 1.1, 11: 1.0,
            12: 1.3, 13: 1.1, 14: 0.9, 15: 1.2, 16: 1.1, 17: 1.0,
            18: 1.4, 19: 1.3, 20: 1.2, 21: 1.5, 22: 1.1, 23: 0.7
        }
    
    def _setup_viral_mechanics(self):
        """Setup viral content mechanics"""
        # Viral probability thresholds
        self.viral_thresholds = {
            'micro_viral': {'min_likes': 1000, 'probability': 0.05},
            'viral': {'min_likes': 10000, 'probability': 0.01},
            'mega_viral': {'min_likes': 100000, 'probability': 0.001}
        }
        
        # Viral content characteristics
        self.viral_content_types = [
            'challenge', 'meme', 'controversy', 'heartwarming', 'educational',
            'behind_the_scenes', 'transformation', 'fail', 'success_story', 'trending_topic'
        ]
        
        # Viral boost multipliers
        self.viral_multipliers = {
            'micro_viral': 5.0,
            'viral': 25.0,
            'mega_viral': 100.0
        }
    
    def _setup_user_behavior_patterns(self):
        """Setup realistic user behavior patterns"""
        # User activity levels
        self.user_activity_levels = {
            'inactive': {'post_frequency': 0.1, 'engagement_rate': 0.5, 'probability': 0.3},
            'casual': {'post_frequency': 0.5, 'engagement_rate': 1.0, 'probability': 0.5},
            'active': {'post_frequency': 2.0, 'engagement_rate': 1.5, 'probability': 0.15},
            'influencer': {'post_frequency': 5.0, 'engagement_rate': 3.0, 'probability': 0.05}
        }
        
        # Follower count distributions by activity level
        self.follower_distributions = {
            'inactive': (10, 500),
            'casual': (100, 2000),
            'active': (500, 10000),
            'influencer': (5000, 1000000)
        }
    
    def generate(self, rows: int, **kwargs) -> pd.DataFrame:
        """
        Generate social media posts dataset
        
        Args:
            rows: Number of posts to generate
            **kwargs: Additional parameters (platform, date_range, time_series, etc.)
            
        Returns:
            pd.DataFrame: Generated social media data with realistic patterns
        """
        # Check if time series generation is requested
        ts_config = self._create_time_series_config(**kwargs)
        
        if ts_config:
            return self._generate_time_series_social_media(rows, ts_config, **kwargs)
        else:
            return self._generate_snapshot_social_media(rows, **kwargs)
    
    def _generate_snapshot_social_media(self, rows: int, **kwargs) -> pd.DataFrame:
        """Generate snapshot social media data (random post times)"""
        platform_filter = kwargs.get('platform', None)
        date_range = kwargs.get('date_range', None)
        include_viral = kwargs.get('include_viral', True)
        
        data = []
        
        for i in range(rows):
            # Select platform
            if platform_filter:
                platform = platform_filter
            else:
                platform = self._select_platform()
            
            # Generate post timestamp
            if date_range:
                start_date, end_date = date_range
                post_time = self.faker.date_time_between(start_date=start_date, end_date=end_date)
            else:
                post_time = self.faker.date_time_this_year()
            
            # Determine user activity level and characteristics
            user_activity = self._select_user_activity_level()
            user_id = f'USR_{self.faker.random_int(1, 500000):06d}'
            follower_count = self._generate_follower_count(user_activity)
            
            # Generate content
            content_type = self._select_content_type(platform)
            content_text = self._generate_content_text(platform, content_type)
            hashtags = self._generate_hashtags(platform, content_type)
            
            # Calculate engagement metrics
            engagement_metrics = self._calculate_engagement(
                platform, content_type, follower_count, post_time, user_activity, include_viral
            )
            
            # Determine if post is viral
            is_viral, viral_type = self._determine_viral_status(engagement_metrics, include_viral)
            
            # Create post record
            post = {
                'post_id': f'POST_{i+1:08d}',
                'user_id': user_id,
                'platform': platform,
                'content_type': content_type,
                'content_text': content_text,
                'hashtags': hashtags,
                'timestamp': post_time,
                'likes': engagement_metrics['likes'],
                'shares': engagement_metrics['shares'],
                'comments': engagement_metrics['comments'],
                'views': engagement_metrics.get('views', 0),
                'follower_count': follower_count,
                'user_activity_level': user_activity,
                'is_viral': is_viral,
                'viral_type': viral_type,
                'engagement_rate': round(engagement_metrics['likes'] / max(follower_count, 1) * 100, 2),
                'post_length': len(content_text),
                'hashtag_count': len(hashtags.split()) if hashtags else 0
            }
            
            data.append(post)
        
        df = pd.DataFrame(data)
        return self._apply_realistic_patterns(df)
    
    def _generate_time_series_social_media(self, rows: int, ts_config, **kwargs) -> pd.DataFrame:
        """Generate time series social media data using integrated time series system"""
        platform_filter = kwargs.get('platform', None)
    
    def _select_platform(self) -> str:
        """Select platform based on realistic usage distributions"""
        platform_weights = OrderedDict([
            ('instagram', 0.35),
            ('facebook', 0.25),
            ('twitter', 0.20),
            ('tiktok', 0.15),
            ('linkedin', 0.05)
        ])
        
        return self.faker.random_element(elements=platform_weights)
    
    def _select_user_activity_level(self) -> str:
        """Select user activity level based on realistic distributions"""
        activity_weights = OrderedDict([
            (level, data['probability']) 
            for level, data in self.user_activity_levels.items()
        ])
        
        return self.faker.random_element(elements=activity_weights)
    
    def _generate_follower_count(self, activity_level: str) -> int:
        """Generate realistic follower count based on activity level"""
        min_followers, max_followers = self.follower_distributions[activity_level]
        
        # Use log-normal distribution for realistic follower distribution
        if activity_level == 'influencer':
            # Influencers have more varied follower counts
            return int(self.faker.random.lognormvariate(9, 1.5))
        else:
            return self.faker.random_int(min_followers, max_followers)
    
    def _select_content_type(self, platform: str) -> str:
        """Select content type based on platform characteristics"""
        if platform == 'tiktok':
            return self.faker.random_element(['video', 'challenge', 'duet', 'trend'])
        elif platform == 'instagram':
            return self.faker.random_element(['image', 'video', 'carousel', 'story', 'reel'])
        elif platform == 'twitter':
            return self.faker.random_element(['text', 'image', 'video', 'thread', 'poll'])
        elif platform == 'linkedin':
            return self.faker.random_element(['text', 'article', 'image', 'video', 'poll'])
        else:  # facebook
            return self.faker.random_element(['text', 'image', 'video', 'link', 'poll', 'event'])
    
    def _generate_content_text(self, platform: str, content_type: str) -> str:
        """Generate realistic content text based on platform and type"""
        # Select content template category
        template_category = self.faker.random_element(list(self.content_templates.keys()))
        template = self.faker.random_element(self.content_templates[template_category])
        
        # Fill template with realistic content
        content = self._fill_content_template(template, platform, content_type)
        
        # Adjust length for platform
        max_length = self.platform_specific.get(platform, {}).get('max_length', 280)
        if len(content) > max_length:
            content = content[:max_length-3] + '...'
        
        return content
    
    def _fill_content_template(self, template: str, platform: str, content_type: str) -> str:
        """Fill content template with realistic values"""
        # Define replacement values with more variety
        replacements = {
            'activity': self.faker.random_element([
                'working out', 'cooking dinner', 'reading a book', 'walking the dog',
                'meeting friends', 'watching a movie', 'learning something new', 'traveling',
                'gardening', 'painting', 'writing', 'coding', 'hiking', 'swimming',
                'volunteering', 'shopping', 'cleaning', 'organizing', 'meditating',
                'playing music', 'dancing', 'studying', 'working late', 'relaxing'
            ]),
            'emotion': self.faker.random_element([
                'amazing', 'grateful', 'excited', 'inspired', 'motivated', 'happy', 'blessed',
                'energized', 'peaceful', 'accomplished', 'proud', 'content', 'thrilled',
                'optimistic', 'refreshed', 'fulfilled', 'confident', 'joyful'
            ]),
            'adjective': self.faker.random_element([
                'amazing', 'incredible', 'fantastic', 'wonderful', 'great', 'awesome', 'perfect',
                'outstanding', 'remarkable', 'exceptional', 'brilliant', 'spectacular',
                'magnificent', 'marvelous', 'superb', 'excellent', 'phenomenal', 'terrific'
            ]),
            'achievement': self.faker.random_element([
                'completed my first marathon', 'got promoted', 'learned a new skill',
                'finished a project', 'reached my goal', 'made a new friend',
                'graduated from college', 'started my own business', 'published an article',
                'won a competition', 'overcame a fear', 'helped someone in need',
                'mastered a recipe', 'fixed something broken', 'organized my space'
            ]),
            'time_period': self.faker.random_element([
                'morning', 'afternoon', 'evening', 'weekend', 'day off', 'vacation',
                'lunch break', 'early morning', 'late night', 'holiday', 'birthday',
                'anniversary', 'spring day', 'summer evening', 'winter morning'
            ]),
            'topic': self.faker.random_element([
                'book', 'movie', 'restaurant', 'travel destination', 'hobby', 'app',
                'podcast', 'TV show', 'song', 'recipe', 'workout', 'game',
                'coffee shop', 'park', 'museum', 'beach', 'mountain', 'city'
            ]),
            'answer': self.faker.random_element([
                'this one', 'the classic', 'something unique', 'the popular choice',
                'the hidden gem', 'the trending option', 'my personal favorite',
                'the underrated pick', 'the surprising winner', 'the timeless choice'
            ]),
            'question_text': self.faker.random_element([
                'What motivates you daily', 'How do you stay productive',
                'What\'s your favorite way to relax', 'What inspires you most',
                'How do you handle stress', 'What makes you smile',
                'What\'s your secret to happiness', 'How do you stay creative',
                'What\'s your biggest passion', 'How do you find balance'
            ]),
            'option1': self.faker.random_element([
                'coffee', 'morning workout', 'early start', 'indoor activities',
                'sweet treats', 'city life', 'summer vibes', 'casual style'
            ]),
            'option2': self.faker.random_element([
                'tea', 'evening workout', 'late start', 'outdoor adventures',
                'savory snacks', 'country life', 'winter cozy', 'formal style'
            ]),
            'scenario': self.faker.random_element([
                'you could travel anywhere', 'you had unlimited time',
                'you could learn any skill', 'you could meet anyone',
                'you won the lottery', 'you could change one thing',
                'you had superpowers', 'you could live anywhere',
                'you could master any talent', 'you had no limitations'
            ]),
            'motivational_quote': self.faker.random_element([
                'success is a journey, not a destination',
                'every day is a new opportunity',
                'believe in yourself and anything is possible',
                'the best time to plant a tree was 20 years ago, the second best time is now',
                'you are stronger than you think',
                'progress, not perfection',
                'dream big, work hard, stay focused',
                'your only limit is your mind'
            ]),
            'discovery': self.faker.random_element([
                'an amazing restaurant', 'a great book', 'a useful app',
                'a beautiful place', 'an inspiring podcast', 'a hidden gem',
                'a life-changing course', 'a perfect recipe', 'a stunning view',
                'a helpful tool', 'a fascinating documentary', 'a cozy cafe'
            ]),
            'event': self.faker.random_element([
                'a conference', 'a concert', 'a workshop', 'a meetup', 'a festival',
                'a wedding', 'a graduation', 'a birthday party', 'a reunion',
                'an exhibition', 'a seminar', 'a performance', 'a competition'
            ]),
            'location': self.faker.city(),
            'advice': self.faker.random_element([
                'take time for yourself', 'follow your passion', 'stay curious',
                'be kind to others', 'embrace change', 'trust the process',
                'celebrate small wins', 'learn from failures', 'stay positive'
            ]),
            'inspirational_message': self.faker.random_element([
                'You\'ve got this!', 'Keep pushing forward!', 'Believe in your dreams!',
                'Stay strong and keep going!', 'Your hard work will pay off!',
                'Every step counts!', 'You\'re making progress!', 'Don\'t give up!'
            ]),
            'positive_affirmation': self.faker.random_element([
                'you are capable of amazing things', 'your potential is limitless',
                'you deserve all the good things coming your way',
                'you are exactly where you need to be', 'you have everything it takes',
                'you are worthy of success and happiness'
            ]),
            'uplifting_message': self.faker.random_element([
                'Today is full of possibilities!', 'You are making a difference!',
                'Your journey is unique and valuable!', 'Good things are coming your way!',
                'You have the power to create positive change!'
            ]),
            'content_type': self.faker.random_element([
                'wisdom', 'inspiration', 'motivation', 'advice', 'insight',
                'perspective', 'reflection', 'experience', 'lesson', 'story'
            ]),
            'description': self.faker.sentence(),
            'item': self.faker.random_element([
                'tool', 'resource', 'opportunity', 'experience', 'moment',
                'connection', 'insight', 'discovery', 'adventure', 'journey'
            ]),
            'thing': self.faker.random_element([
                'quote', 'video', 'article', 'photo', 'story', 'moment',
                'experience', 'memory', 'achievement', 'milestone'
            ]),
            'upcoming_event': f"{self.faker.random_element(['conference', 'concert', 'workshop', 'meetup'])} next {self.faker.random_element(['week', 'month'])}",
            'past_event': f"{self.faker.random_element(['conference', 'concert', 'workshop', 'meetup'])} last {self.faker.random_element(['week', 'month', 'year'])}"
        }
        
        # Add some randomness to make content more unique
        if self.faker.random.random() < 0.3:  # 30% chance to add personal touch
            personal_touches = [
                f" (been thinking about this all {self.faker.random_element(['day', 'week', 'morning'])})",
                f" - {self.faker.random_element(['honestly', 'seriously', 'truly', 'really'])}",
                f" {self.faker.random_element(['😊', '✨', '🙌', '💫', '🌟', ''])}",
                f" #{self.faker.random_element(['mood', 'vibes', 'life', 'thoughts', 'feelings'])}"
            ]
            template += self.faker.random_element(personal_touches)
        
        # Replace placeholders
        content = template
        for key, value in replacements.items():
            content = content.replace(f'{{{key}}}', str(value))
        
        # Remove hashtags placeholder for now (will be added separately)
        content = content.replace('{hashtags}', '').strip()
        
        # Add some variety with random additions
        if self.faker.random.random() < 0.2:  # 20% chance to add extra context
            extra_context = [
                f" What do you think?",
                f" Anyone else feel this way?",
                f" Let me know your thoughts!",
                f" Share your experience below!",
                f" Tag someone who needs to see this!"
            ]
            content += self.faker.random_element(extra_context)
        
        return content
    
    def _generate_hashtags(self, platform: str, content_type: str) -> str:
        """Generate realistic hashtags based on platform and content"""
        hashtag_count = self._determine_hashtag_count(platform)
        
        if hashtag_count == 0:
            return ''
        
        # Select hashtag categories
        categories = self.faker.random_elements(
            list(self.hashtag_categories.keys()), 
            length=min(3, hashtag_count)
        )
        
        hashtags = []
        for category in categories:
            category_hashtags = self.hashtag_categories[category]
            selected = self.faker.random_element(category_hashtags)
            hashtags.append(selected)
        
        # Add platform-specific hashtags
        if platform == 'tiktok':
            hashtags.extend(['#fyp', '#viral'])
        elif platform == 'instagram' and content_type == 'reel':
            hashtags.append('#reels')
        elif platform == 'linkedin':
            hashtags.extend(['#professional', '#career'])
        
        # Ensure we don't exceed the desired count
        hashtags = hashtags[:hashtag_count]
        
        return ' '.join(hashtags)
    
    def _determine_hashtag_count(self, platform: str) -> int:
        """Determine number of hashtags based on platform norms"""
        if platform == 'instagram':
            return self.faker.random_int(5, 15)  # Instagram users love hashtags
        elif platform == 'twitter':
            return self.faker.random_int(1, 3)   # Twitter is more conservative
        elif platform == 'tiktok':
            return self.faker.random_int(3, 8)   # TikTok uses moderate hashtags
        elif platform == 'linkedin':
            return self.faker.random_int(2, 5)   # Professional, moderate use
        else:  # facebook
            return self.faker.random_int(0, 3)   # Facebook uses fewer hashtags
    
    def _calculate_engagement(self, platform: str, content_type: str, follower_count: int, 
                            post_time: datetime, user_activity: str, include_viral: bool) -> Dict[str, int]:
        """Calculate realistic engagement metrics"""
        # Base engagement rates
        base_rates = self.base_engagement_rates[platform]
        
        # Calculate base engagement
        base_likes = int(follower_count * base_rates['likes'] / 1000)
        base_shares = int(follower_count * base_rates['shares'] / 1000)
        base_comments = int(follower_count * base_rates['comments'] / 1000)
        
        # Apply content type multiplier
        content_multiplier = self.content_multipliers.get(content_type, 1.0)
        
        # Apply time-based multiplier
        hour = post_time.hour
        time_multiplier = self.hourly_engagement_multipliers.get(hour, 1.0)
        
        # Apply user activity multiplier
        activity_multiplier = self.user_activity_levels[user_activity]['engagement_rate']
        
        # Calculate final engagement
        total_multiplier = content_multiplier * time_multiplier * activity_multiplier
        
        likes = max(0, int(base_likes * total_multiplier * self.faker.random.uniform(0.5, 2.0)))
        shares = max(0, int(base_shares * total_multiplier * self.faker.random.uniform(0.3, 1.8)))
        comments = max(0, int(base_comments * total_multiplier * self.faker.random.uniform(0.4, 1.6)))
        
        # Calculate views (typically 5-10x likes for video content)
        if content_type in ['video', 'reel', 'story']:
            views = int(likes * self.faker.random.uniform(5, 10))
        else:
            views = int(likes * self.faker.random.uniform(2, 4))
        
        return {
            'likes': likes,
            'shares': shares,
            'comments': comments,
            'views': views
        }
    
    def _determine_viral_status(self, engagement_metrics: Dict[str, int], include_viral: bool) -> Tuple[bool, str]:
        """Determine if a post has gone viral"""
        if not include_viral:
            return False, 'none'
        
        likes = engagement_metrics['likes']
        
        # Check viral thresholds
        if likes >= self.viral_thresholds['mega_viral']['min_likes']:
            if self.faker.random.random() < self.viral_thresholds['mega_viral']['probability']:
                return True, 'mega_viral'
        elif likes >= self.viral_thresholds['viral']['min_likes']:
            if self.faker.random.random() < self.viral_thresholds['viral']['probability']:
                return True, 'viral'
        elif likes >= self.viral_thresholds['micro_viral']['min_likes']:
            if self.faker.random.random() < self.viral_thresholds['micro_viral']['probability']:
                return True, 'micro_viral'
        
        return False, 'none'
    
    def _apply_realistic_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply additional realistic patterns to the generated data"""
        # Apply viral boost to engagement metrics
        viral_mask = data['is_viral'] == True
        for viral_type in ['micro_viral', 'viral', 'mega_viral']:
            type_mask = data['viral_type'] == viral_type
            multiplier = self.viral_multipliers[viral_type]
            
            data.loc[type_mask, 'likes'] = (data.loc[type_mask, 'likes'] * multiplier).astype(int)
            data.loc[type_mask, 'shares'] = (data.loc[type_mask, 'shares'] * multiplier).astype(int)
            data.loc[type_mask, 'comments'] = (data.loc[type_mask, 'comments'] * multiplier).astype(int)
            data.loc[type_mask, 'views'] = (data.loc[type_mask, 'views'] * multiplier).astype(int)
        
        # Recalculate engagement rate after viral boost
        data['engagement_rate'] = round(data['likes'] / data['follower_count'].clip(lower=1) * 100, 2)
        
        # Add realistic posting patterns (users don't post randomly)
        data = data.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        
        # Add day of week and hour for analysis
        data['day_of_week'] = data['timestamp'].dt.day_name()
        data['hour_of_day'] = data['timestamp'].dt.hour
        
        # Add sentiment score based on content
        data['sentiment_score'] = data['content_text'].apply(self._calculate_sentiment_score)
        
        # Add reach estimate (views for video, impressions for others)
        data['estimated_reach'] = data.apply(
            lambda row: row['views'] if row['content_type'] in ['video', 'reel', 'story'] 
            else int(row['follower_count'] * self.faker.random.uniform(0.1, 0.3)), axis=1
        )
        
        return data
    
    def _calculate_sentiment_score(self, content: str) -> float:
        """Calculate a simple sentiment score for content"""
        positive_words = ['amazing', 'great', 'awesome', 'fantastic', 'wonderful', 'excited', 
                         'happy', 'love', 'best', 'perfect', 'incredible', 'grateful']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'disappointed', 
                         'angry', 'frustrated', 'sad', 'boring']
        
        content_lower = content.lower()
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        # Score from -1 (very negative) to 1 (very positive)
        if positive_count + negative_count == 0:
            return 0.0  # Neutral
        
        score = (positive_count - negative_count) / (positive_count + negative_count)
        return round(score, 2)