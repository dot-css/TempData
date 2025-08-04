"""
Unit tests for SocialMediaGenerator

Tests realistic social media data generation including posting patterns,
engagement distributions, content types, and viral content patterns.
"""

import unittest
import pandas as pd
from datetime import datetime, timedelta
from tempdata.core.seeding import MillisecondSeeder
from tempdata.datasets.social.social_media import SocialMediaGenerator


class TestSocialMediaGenerator(unittest.TestCase):
    """Test cases for SocialMediaGenerator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.seeder = MillisecondSeeder(fixed_seed=12345)
        self.generator = SocialMediaGenerator(self.seeder)
    
    def test_basic_generation(self):
        """Test basic social media data generation"""
        rows = 100
        data = self.generator.generate(rows)
        
        # Check basic structure
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), rows)
        
        # Check required columns
        required_columns = [
            'post_id', 'user_id', 'platform', 'content_type', 'content_text',
            'hashtags', 'timestamp', 'likes', 'shares', 'comments', 'views',
            'follower_count', 'user_activity_level', 'is_viral', 'viral_type',
            'engagement_rate', 'post_length', 'hashtag_count'
        ]
        
        for column in required_columns:
            self.assertIn(column, data.columns, f"Missing column: {column}")
    
    def test_platform_distribution(self):
        """Test realistic platform distribution"""
        rows = 1000
        data = self.generator.generate(rows)
        
        platform_counts = data['platform'].value_counts()
        
        # Check all expected platforms are present
        expected_platforms = ['instagram', 'facebook', 'twitter', 'tiktok', 'linkedin']
        for platform in expected_platforms:
            self.assertIn(platform, platform_counts.index)
        
        # Instagram should be most popular (around 35%)
        instagram_ratio = platform_counts['instagram'] / rows
        self.assertGreater(instagram_ratio, 0.25, "Instagram should have significant presence")
        
        # LinkedIn should be least popular (around 5%)
        linkedin_ratio = platform_counts['linkedin'] / rows
        self.assertLess(linkedin_ratio, 0.15, "LinkedIn should have lower presence")
    
    def test_content_type_by_platform(self):
        """Test platform-specific content types"""
        rows = 500
        data = self.generator.generate(rows)
        
        # Test TikTok content types
        tiktok_data = data[data['platform'] == 'tiktok']
        if len(tiktok_data) > 0:
            tiktok_content_types = set(tiktok_data['content_type'].unique())
            expected_tiktok_types = {'video', 'challenge', 'duet', 'trend'}
            self.assertTrue(tiktok_content_types.issubset(expected_tiktok_types))
        
        # Test Instagram content types
        instagram_data = data[data['platform'] == 'instagram']
        if len(instagram_data) > 0:
            instagram_content_types = set(instagram_data['content_type'].unique())
            expected_instagram_types = {'image', 'video', 'carousel', 'story', 'reel'}
            self.assertTrue(instagram_content_types.issubset(expected_instagram_types))
    
    def test_user_activity_levels(self):
        """Test realistic user activity level distribution"""
        rows = 1000
        data = self.generator.generate(rows)
        
        activity_counts = data['user_activity_level'].value_counts()
        
        # Check all activity levels are present
        expected_levels = ['inactive', 'casual', 'active', 'influencer']
        for level in expected_levels:
            self.assertIn(level, activity_counts.index)
        
        # Casual users should be most common (around 50%)
        casual_ratio = activity_counts['casual'] / rows
        self.assertGreater(casual_ratio, 0.35, "Casual users should be most common")
        
        # Influencers should be rare (around 5%)
        influencer_ratio = activity_counts['influencer'] / rows
        self.assertLess(influencer_ratio, 0.15, "Influencers should be rare")
    
    def test_follower_count_by_activity(self):
        """Test follower count correlation with activity level"""
        rows = 500
        data = self.generator.generate(rows)
        
        # Group by activity level and check follower ranges
        activity_groups = data.groupby('user_activity_level')['follower_count']
        
        # Influencers should have more followers than casual users
        if 'influencer' in activity_groups.groups and 'casual' in activity_groups.groups:
            influencer_avg = activity_groups.get_group('influencer').mean()
            casual_avg = activity_groups.get_group('casual').mean()
            self.assertGreater(influencer_avg, casual_avg, 
                             "Influencers should have more followers than casual users")
        
        # Inactive users should have fewer followers
        if 'inactive' in activity_groups.groups and 'active' in activity_groups.groups:
            inactive_avg = activity_groups.get_group('inactive').mean()
            active_avg = activity_groups.get_group('active').mean()
            self.assertLess(inactive_avg, active_avg,
                           "Inactive users should have fewer followers than active users")
    
    def test_engagement_metrics(self):
        """Test realistic engagement metrics"""
        rows = 200
        data = self.generator.generate(rows)
        
        # Check that engagement metrics are non-negative
        self.assertTrue((data['likes'] >= 0).all(), "Likes should be non-negative")
        self.assertTrue((data['shares'] >= 0).all(), "Shares should be non-negative")
        self.assertTrue((data['comments'] >= 0).all(), "Comments should be non-negative")
        self.assertTrue((data['views'] >= 0).all(), "Views should be non-negative")
        
        # Check engagement rate calculation
        calculated_rate = (data['likes'] / data['follower_count'].clip(lower=1) * 100).round(2)
        pd.testing.assert_series_equal(data['engagement_rate'], calculated_rate, 
                                     check_names=False)
        
        # Views should generally be higher than likes
        video_content = data[data['content_type'].isin(['video', 'reel', 'story'])]
        if len(video_content) > 0:
            views_vs_likes = (video_content['views'] >= video_content['likes']).mean()
            self.assertGreater(views_vs_likes, 0.8, 
                             "Views should generally be higher than likes for video content")
    
    def test_hashtag_patterns(self):
        """Test realistic hashtag usage patterns"""
        rows = 300
        data = self.generator.generate(rows)
        
        # Instagram should have more hashtags than Twitter
        instagram_data = data[data['platform'] == 'instagram']
        twitter_data = data[data['platform'] == 'twitter']
        
        if len(instagram_data) > 0 and len(twitter_data) > 0:
            instagram_avg_hashtags = instagram_data['hashtag_count'].mean()
            twitter_avg_hashtags = twitter_data['hashtag_count'].mean()
            self.assertGreater(instagram_avg_hashtags, twitter_avg_hashtags,
                             "Instagram should have more hashtags than Twitter")
        
        # Check hashtag format
        hashtag_data = data[data['hashtags'].notna() & (data['hashtags'] != '')]
        if len(hashtag_data) > 0:
            sample_hashtags = hashtag_data['hashtags'].iloc[0]
            hashtag_list = sample_hashtags.split()
            for hashtag in hashtag_list:
                self.assertTrue(hashtag.startswith('#'), f"Hashtag should start with #: {hashtag}")
    
    def test_viral_content_patterns(self):
        """Test viral content identification and boost"""
        rows = 1000
        data = self.generator.generate(rows, include_viral=True)
        
        # Check viral status columns
        self.assertIn('is_viral', data.columns)
        self.assertIn('viral_type', data.columns)
        
        # Check viral types
        viral_data = data[data['is_viral'] == True]
        if len(viral_data) > 0:
            viral_types = set(viral_data['viral_type'].unique())
            expected_viral_types = {'micro_viral', 'viral', 'mega_viral'}
            self.assertTrue(viral_types.issubset(expected_viral_types))
            
            # Viral content should have higher engagement
            non_viral_data = data[data['is_viral'] == False]
            if len(non_viral_data) > 0:
                viral_avg_likes = viral_data['likes'].mean()
                non_viral_avg_likes = non_viral_data['likes'].mean()
                self.assertGreater(viral_avg_likes, non_viral_avg_likes,
                                 "Viral content should have higher engagement")
    
    def test_content_text_generation(self):
        """Test realistic content text generation"""
        rows = 100
        data = self.generator.generate(rows)
        
        # Check that content text is not empty
        self.assertTrue((data['content_text'].str.len() > 0).all(),
                       "Content text should not be empty")
        
        # Check post length calculation
        calculated_length = data['content_text'].str.len()
        pd.testing.assert_series_equal(data['post_length'], calculated_length,
                                     check_names=False)
        
        # Check platform-specific length limits
        twitter_data = data[data['platform'] == 'twitter']
        if len(twitter_data) > 0:
            max_twitter_length = twitter_data['post_length'].max()
            self.assertLessEqual(max_twitter_length, 280,
                               "Twitter posts should respect character limit")
    
    def test_temporal_patterns(self):
        """Test temporal posting patterns"""
        rows = 200
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)
        
        data = self.generator.generate(rows, date_range=(start_date, end_date))
        
        # Check date range
        self.assertTrue((data['timestamp'] >= start_date).all(),
                       "All timestamps should be after start date")
        self.assertTrue((data['timestamp'] <= end_date).all(),
                       "All timestamps should be before end date")
        
        # Check additional temporal columns
        self.assertIn('day_of_week', data.columns)
        self.assertIn('hour_of_day', data.columns)
        
        # Verify temporal column calculations
        expected_days = data['timestamp'].dt.day_name()
        expected_hours = data['timestamp'].dt.hour
        
        pd.testing.assert_series_equal(data['day_of_week'], expected_days,
                                     check_names=False)
        pd.testing.assert_series_equal(data['hour_of_day'], expected_hours,
                                     check_names=False)
    
    def test_sentiment_analysis(self):
        """Test sentiment score calculation"""
        rows = 100
        data = self.generator.generate(rows)
        
        # Check sentiment score range
        self.assertTrue((data['sentiment_score'] >= -1).all(),
                       "Sentiment scores should be >= -1")
        self.assertTrue((data['sentiment_score'] <= 1).all(),
                       "Sentiment scores should be <= 1")
        
        # Test specific sentiment calculation
        test_positive = "This is amazing and wonderful! I love it!"
        test_negative = "This is terrible and awful. I hate it."
        test_neutral = "This is a regular post about daily activities."
        
        positive_score = self.generator._calculate_sentiment_score(test_positive)
        negative_score = self.generator._calculate_sentiment_score(test_negative)
        neutral_score = self.generator._calculate_sentiment_score(test_neutral)
        
        self.assertGreater(positive_score, 0, "Positive content should have positive score")
        self.assertLess(negative_score, 0, "Negative content should have negative score")
        self.assertEqual(neutral_score, 0.0, "Neutral content should have zero score")
    
    def test_platform_specific_generation(self):
        """Test platform-specific generation"""
        rows = 100
        
        # Test Instagram-specific generation
        instagram_data = self.generator.generate(rows, platform='instagram')
        self.assertTrue((instagram_data['platform'] == 'instagram').all(),
                       "All posts should be Instagram posts")
        
        # Instagram should have higher hashtag counts
        avg_hashtags = instagram_data['hashtag_count'].mean()
        self.assertGreater(avg_hashtags, 3, "Instagram should have more hashtags")
    
    def test_data_consistency(self):
        """Test data consistency and relationships"""
        rows = 200
        data = self.generator.generate(rows)
        
        # Check that user_id format is consistent
        user_id_pattern = data['user_id'].str.match(r'^USR_\d{6}$')
        self.assertTrue(user_id_pattern.all(), "User IDs should follow USR_XXXXXX format")
        
        # Check that post_id format is consistent
        post_id_pattern = data['post_id'].str.match(r'^POST_\d{8}$')
        self.assertTrue(post_id_pattern.all(), "Post IDs should follow POST_XXXXXXXX format")
        
        # Check estimated reach calculation
        video_mask = data['content_type'].isin(['video', 'reel', 'story'])
        video_data = data[video_mask]
        non_video_data = data[~video_mask]
        
        if len(video_data) > 0:
            # For video content, estimated reach should equal views
            pd.testing.assert_series_equal(video_data['estimated_reach'], 
                                         video_data['views'],
                                         check_names=False)
        
        if len(non_video_data) > 0:
            # For non-video content, estimated reach should be reasonable
            reach_ratio = (non_video_data['estimated_reach'] / 
                          non_video_data['follower_count']).mean()
            self.assertGreater(reach_ratio, 0.05, "Reach should be reasonable fraction of followers")
            self.assertLess(reach_ratio, 0.5, "Reach should not exceed reasonable limits")
    
    def test_reproducibility(self):
        """Test that generation is reproducible with fixed seed"""
        rows = 50
        
        # Generate data twice with same seed
        seeder1 = MillisecondSeeder(fixed_seed=54321)
        generator1 = SocialMediaGenerator(seeder1)
        data1 = generator1.generate(rows)
        
        seeder2 = MillisecondSeeder(fixed_seed=54321)
        generator2 = SocialMediaGenerator(seeder2)
        data2 = generator2.generate(rows)
        
        # Data should be identical
        pd.testing.assert_frame_equal(data1, data2, 
                                    "Data should be identical with same seed")
    
    def test_data_quality_metrics(self):
        """Test overall data quality metrics"""
        rows = 500
        data = self.generator.generate(rows)
        
        # Check for realistic engagement patterns
        # Engagement rate should be reasonable (0.1% to 20%)
        engagement_rates = data['engagement_rate']
        reasonable_engagement = ((engagement_rates >= 0.1) & 
                               (engagement_rates <= 20.0)).mean()
        self.assertGreater(reasonable_engagement, 0.8,
                         "Most posts should have reasonable engagement rates")
        
        # Check content diversity
        unique_content_ratio = data['content_text'].nunique() / len(data)
        self.assertGreater(unique_content_ratio, 0.6,
                         "Content should be reasonably diverse")
        
        # Check user diversity
        unique_users_ratio = data['user_id'].nunique() / len(data)
        self.assertGreater(unique_users_ratio, 0.3,
                         "Should have reasonable user diversity")


if __name__ == '__main__':
    unittest.main()