"""
Unit tests for UserProfilesGenerator

Tests realistic user profile data generation including demographic distributions,
interest correlations, activity patterns, and follower/following relationships.
"""

import unittest
import pandas as pd
from datetime import datetime, timedelta
from tempdata.core.seeding import MillisecondSeeder
from tempdata.datasets.social.user_profiles import UserProfilesGenerator


class TestUserProfilesGenerator(unittest.TestCase):
    """Test cases for UserProfilesGenerator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.seeder = MillisecondSeeder(fixed_seed=12345)
        self.generator = UserProfilesGenerator(self.seeder)
    
    def test_basic_generation(self):
        """Test basic user profile data generation"""
        rows = 100
        data = self.generator.generate(rows)
        
        # Check basic structure
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), rows)
        
        # Check required columns
        required_columns = [
            'user_id', 'username', 'display_name', 'first_name', 'last_name',
            'bio', 'age_group', 'gender', 'location_type', 'location',
            'primary_interests', 'followers_count', 'following_count', 'user_type',
            'activity_level', 'online_preference', 'join_date', 'account_age_days',
            'is_verified', 'is_private', 'engagement_level', 'avg_likes_per_post',
            'avg_comments_per_post', 'posts_per_week', 'stories_per_week',
            'follower_following_ratio', 'bio_length', 'has_profile_picture', 'has_bio'
        ]
        
        for column in required_columns:
            self.assertIn(column, data.columns, f"Missing column: {column}")
    
    def test_demographic_distributions(self):
        """Test realistic demographic distributions"""
        rows = 1000
        data = self.generator.generate(rows)
        
        # Test age group distribution
        age_counts = data['age_group'].value_counts()
        
        # 25-34 should be most common (35% probability)
        most_common_age = age_counts.index[0]
        self.assertEqual(most_common_age, '25-34', "25-34 should be most common age group")
        
        # Check all expected age groups are present
        expected_ages = ['18-24', '25-34', '35-44', '45-54', '55+']
        for age in expected_ages:
            self.assertIn(age, age_counts.index, f"Missing age group: {age}")
        
        # Test gender distribution
        gender_counts = data['gender'].value_counts()
        expected_genders = ['female', 'male', 'non_binary']
        for gender in expected_genders:
            self.assertIn(gender, gender_counts.index, f"Missing gender: {gender}")
        
        # Female should be slightly more common (52%)
        female_ratio = gender_counts['female'] / rows
        self.assertGreater(female_ratio, 0.45, "Female representation should be significant")
    
    def test_user_type_distribution(self):
        """Test realistic user type distribution"""
        rows = 1000
        data = self.generator.generate(rows)
        
        user_type_counts = data['user_type'].value_counts()
        
        # Check all user types are present
        expected_types = ['influencer', 'popular', 'average', 'new_user']
        for user_type in expected_types:
            self.assertIn(user_type, user_type_counts.index, f"Missing user type: {user_type}")
        
        # Average users should be most common (80%)
        average_ratio = user_type_counts['average'] / rows
        self.assertGreater(average_ratio, 0.6, "Average users should be most common")
        
        # Influencers should be rare (2%)
        influencer_ratio = user_type_counts['influencer'] / rows
        self.assertLess(influencer_ratio, 0.1, "Influencers should be rare")
    
    def test_follower_count_by_user_type(self):
        """Test follower count correlation with user type"""
        rows = 500
        data = self.generator.generate(rows)
        
        # Group by user type and check follower ranges
        user_type_groups = data.groupby('user_type')['followers_count']
        
        # Influencers should have more followers than average users
        if 'influencer' in user_type_groups.groups and 'average' in user_type_groups.groups:
            influencer_avg = user_type_groups.get_group('influencer').mean()
            average_avg = user_type_groups.get_group('average').mean()
            self.assertGreater(influencer_avg, average_avg,
                             "Influencers should have more followers than average users")
        
        # New users should have fewer followers
        if 'new_user' in user_type_groups.groups and 'popular' in user_type_groups.groups:
            new_user_avg = user_type_groups.get_group('new_user').mean()
            popular_avg = user_type_groups.get_group('popular').mean()
            self.assertLess(new_user_avg, popular_avg,
                           "New users should have fewer followers than popular users")
    
    def test_interest_correlations(self):
        """Test interest correlation patterns"""
        rows = 300
        data = self.generator.generate(rows)
        
        # Check that interests are realistic
        all_interests = []
        for interests_str in data['primary_interests']:
            all_interests.extend(interests_str.split(', '))
        
        unique_interests = set(all_interests)
        expected_interests = [
            'technology', 'fitness', 'travel', 'food', 'arts', 'business',
            'lifestyle', 'education', 'entertainment', 'social_causes'
        ]
        
        # Most interests should be from expected categories
        valid_interests = [i for i in unique_interests if i in expected_interests]
        self.assertGreater(len(valid_interests), len(unique_interests) * 0.8,
                         "Most interests should be from expected categories")
    
    def test_username_generation(self):
        """Test realistic username generation"""
        rows = 200
        data = self.generator.generate(rows)
        
        # Check username format
        usernames = data['username']
        
        # All usernames should be non-empty and reasonable length
        self.assertTrue((usernames.str.len() > 0).all(), "All usernames should be non-empty")
        self.assertTrue((usernames.str.len() <= 30).all(), "Usernames should be reasonable length")
        
        # Check for variety in username patterns
        unique_ratio = usernames.nunique() / len(usernames)
        self.assertGreater(unique_ratio, 0.9, "Usernames should be mostly unique")
        
        # Check that usernames don't contain spaces
        has_spaces = usernames.str.contains(' ').any()
        self.assertFalse(has_spaces, "Usernames should not contain spaces")
    
    def test_bio_generation(self):
        """Test realistic bio generation"""
        rows = 150
        data = self.generator.generate(rows)
        
        # Check bio characteristics
        bios = data['bio']
        
        # Most users should have bios
        has_bio_ratio = data['has_bio'].mean()
        self.assertGreater(has_bio_ratio, 0.8, "Most users should have bios")
        
        # Bio lengths should be reasonable
        bio_lengths = data['bio_length']
        self.assertTrue((bio_lengths <= 160).all(), "Bios should respect length limits")
        
        # Check bio diversity
        unique_bio_ratio = bios.nunique() / len(bios)
        self.assertGreater(unique_bio_ratio, 0.7, "Bios should be diverse")
    
    def test_age_group_correlations(self):
        """Test age group correlations with other attributes"""
        rows = 400
        data = self.generator.generate(rows)
        
        # Younger users should be more active
        young_users = data[data['age_group'].isin(['18-24', '25-34'])]
        older_users = data[data['age_group'].isin(['45-54', '55+'])]
        
        if len(young_users) > 0 and len(older_users) > 0:
            young_high_activity = (young_users['activity_level'] == 'high').mean()
            older_high_activity = (older_users['activity_level'] == 'high').mean()
            
            self.assertGreater(young_high_activity, older_high_activity,
                             "Younger users should be more active")
    
    def test_engagement_metrics(self):
        """Test realistic engagement metrics"""
        rows = 200
        data = self.generator.generate(rows)
        
        # Check that engagement metrics are non-negative
        self.assertTrue((data['avg_likes_per_post'] >= 0).all(),
                       "Average likes should be non-negative")
        self.assertTrue((data['avg_comments_per_post'] >= 0).all(),
                       "Average comments should be non-negative")
        
        # Engagement should correlate with follower count
        correlation = data['avg_likes_per_post'].corr(data['followers_count'])
        self.assertGreater(correlation, 0.5,
                         "Engagement should correlate with follower count")
        
        # Check engagement rate calculation
        calculated_rate = (data['avg_likes_per_post'] / data['followers_count'].clip(lower=1) * 100).round(2)
        pd.testing.assert_series_equal(data['engagement_rate'], calculated_rate,
                                     check_names=False)
    
    def test_activity_patterns(self):
        """Test realistic activity patterns"""
        rows = 250
        data = self.generator.generate(rows)
        
        # Check activity level distribution
        activity_counts = data['activity_level'].value_counts()
        expected_levels = ['high', 'medium', 'low']
        
        for level in expected_levels:
            self.assertIn(level, activity_counts.index, f"Missing activity level: {level}")
        
        # Medium activity should be most common
        most_common_activity = activity_counts.index[0]
        self.assertEqual(most_common_activity, 'medium',
                        "Medium activity should be most common")
        
        # Check posting frequency correlation with activity level
        high_activity = data[data['activity_level'] == 'high']
        low_activity = data[data['activity_level'] == 'low']
        
        if len(high_activity) > 0 and len(low_activity) > 0:
            high_avg_posts = high_activity['posts_per_week'].mean()
            low_avg_posts = low_activity['posts_per_week'].mean()
            
            self.assertGreater(high_avg_posts, low_avg_posts,
                             "High activity users should post more frequently")
    
    def test_verification_patterns(self):
        """Test verification status patterns"""
        rows = 500
        data = self.generator.generate(rows)
        
        # Check verification status
        verification_rate = data['is_verified'].mean()
        self.assertLess(verification_rate, 0.1, "Verification should be rare")
        
        # Influencers should be more likely to be verified
        influencers = data[data['user_type'] == 'influencer']
        regular_users = data[data['user_type'] == 'average']
        
        if len(influencers) > 0 and len(regular_users) > 0:
            influencer_verification = influencers['is_verified'].mean()
            regular_verification = regular_users['is_verified'].mean()
            
            # Allow for small sample sizes
            if len(influencers) >= 5:
                self.assertGreaterEqual(influencer_verification, regular_verification,
                                      "Influencers should be more likely to be verified")
    
    def test_privacy_settings(self):
        """Test privacy setting patterns"""
        rows = 300
        data = self.generator.generate(rows)
        
        # Check privacy distribution
        privacy_rate = data['is_private'].mean()
        self.assertLess(privacy_rate, 0.5, "Most accounts should be public")
        
        # Younger users should be more likely to have private accounts
        young_users = data[data['age_group'] == '18-24']
        older_users = data[data['age_group'].isin(['45-54', '55+'])]
        
        if len(young_users) > 10 and len(older_users) > 10:
            young_privacy = young_users['is_private'].mean()
            older_privacy = older_users['is_private'].mean()
            
            # This is a tendency, not a strict rule
            self.assertGreaterEqual(young_privacy, older_privacy * 0.8,
                                  "Younger users tend to be more privacy-conscious")
    
    def test_account_age_patterns(self):
        """Test account age patterns"""
        rows = 200
        data = self.generator.generate(rows)
        
        # Check account age calculation
        current_date = datetime.now().date()
        calculated_age = pd.Series([(current_date - date).days for date in data['join_date']])
        pd.testing.assert_series_equal(data['account_age_days'], calculated_age,
                                     check_names=False)
        
        # Account ages should be reasonable
        self.assertTrue((data['account_age_days'] >= 0).all(),
                       "Account ages should be non-negative")
        self.assertTrue((data['account_age_days'] <= 365 * 15).all(),
                       "Account ages should be reasonable (< 15 years)")
        
        # Older users should have older accounts on average
        young_users = data[data['age_group'] == '18-24']
        older_users = data[data['age_group'].isin(['45-54', '55+'])]
        
        if len(young_users) > 10 and len(older_users) > 10:
            young_avg_age = young_users['account_age_days'].mean()
            older_avg_age = older_users['account_age_days'].mean()
            
            self.assertGreater(older_avg_age, young_avg_age,
                             "Older users should have older accounts on average")
    
    def test_follower_following_ratios(self):
        """Test follower/following ratio patterns"""
        rows = 300
        data = self.generator.generate(rows)
        
        # Check ratio calculation - the generator rounds to 2 decimal places
        calculated_ratio = (data['followers_count'] / data['following_count'].clip(lower=1)).round(2)
        
        # Check that the stored ratio matches the calculated ratio (both rounded to 2 decimal places)
        pd.testing.assert_series_equal(data['follower_following_ratio'], calculated_ratio,
                                     check_names=False)
        
        # Influencers should have higher ratios
        influencers = data[data['user_type'] == 'influencer']
        average_users = data[data['user_type'] == 'average']
        
        if len(influencers) > 0 and len(average_users) > 0:
            influencer_avg_ratio = influencers['follower_following_ratio'].mean()
            average_avg_ratio = average_users['follower_following_ratio'].mean()
            
            self.assertGreater(influencer_avg_ratio, average_avg_ratio,
                             "Influencers should have higher follower/following ratios")
    
    def test_quality_scores(self):
        """Test account quality and influence scores"""
        rows = 200
        data = self.generator.generate(rows)
        
        # Check that quality scores exist and are reasonable
        self.assertIn('account_quality_score', data.columns)
        self.assertIn('influence_score', data.columns)
        
        # Quality scores should be 0-100
        self.assertTrue((data['account_quality_score'] >= 0).all(),
                       "Quality scores should be non-negative")
        self.assertTrue((data['account_quality_score'] <= 100).all(),
                       "Quality scores should not exceed 100")
        
        # Influence scores should be 0-100
        self.assertTrue((data['influence_score'] >= 0).all(),
                       "Influence scores should be non-negative")
        self.assertTrue((data['influence_score'] <= 100).all(),
                       "Influence scores should not exceed 100")
        
        # Verified users should have higher quality scores on average
        verified_users = data[data['is_verified'] == True]
        unverified_users = data[data['is_verified'] == False]
        
        if len(verified_users) > 0 and len(unverified_users) > 0:
            verified_avg_quality = verified_users['account_quality_score'].mean()
            unverified_avg_quality = unverified_users['account_quality_score'].mean()
            
            self.assertGreater(verified_avg_quality, unverified_avg_quality,
                             "Verified users should have higher quality scores")
    
    def test_data_consistency(self):
        """Test data consistency and relationships"""
        rows = 150
        data = self.generator.generate(rows)
        
        # Check that user_id format is consistent
        user_id_pattern = data['user_id'].str.match(r'^USR_\d{6}$')
        self.assertTrue(user_id_pattern.all(), "User IDs should follow USR_XXXXXX format")
        
        # Check that display names contain first names
        name_consistency = data.apply(
            lambda row: row['first_name'] in row['display_name'], axis=1
        )
        consistency_rate = name_consistency.mean()
        self.assertGreater(consistency_rate, 0.8,
                         "Display names should generally contain first names")
        
        # Check that bio length matches actual bio length
        actual_bio_lengths = data['bio'].str.len()
        pd.testing.assert_series_equal(data['bio_length'], actual_bio_lengths,
                                     check_names=False)
        
        # Check that has_bio is consistent with bio content
        has_bio_calculated = (data['bio'].str.strip().str.len() > 0)
        pd.testing.assert_series_equal(data['has_bio'], has_bio_calculated,
                                     check_names=False)
    
    def test_platform_specific_generation(self):
        """Test platform-specific generation"""
        rows = 100
        
        # Test general platform generation
        general_data = self.generator.generate(rows, platform='general')
        self.assertEqual(len(general_data), rows)
        
        # Test with specific country
        country_data = self.generator.generate(rows, country='united_states')
        self.assertEqual(len(country_data), rows)
        
        # All profiles should have required fields
        required_fields = ['username', 'display_name', 'bio', 'followers_count']
        for field in required_fields:
            self.assertIn(field, country_data.columns)
    
    def test_reproducibility(self):
        """Test that generation is reproducible with fixed seed"""
        rows = 50
        
        # Generate data twice with same seed
        seeder1 = MillisecondSeeder(fixed_seed=54321)
        generator1 = UserProfilesGenerator(seeder1)
        data1 = generator1.generate(rows)
        
        seeder2 = MillisecondSeeder(fixed_seed=54321)
        generator2 = UserProfilesGenerator(seeder2)
        data2 = generator2.generate(rows)
        
        # Data should be identical
        pd.testing.assert_frame_equal(data1, data2,
                                    "Data should be identical with same seed")
    
    def test_data_quality_metrics(self):
        """Test overall data quality metrics"""
        rows = 400
        data = self.generator.generate(rows)
        
        # Check for realistic follower distributions
        # Most users should have reasonable follower counts
        reasonable_followers = ((data['followers_count'] >= 0) & 
                              (data['followers_count'] <= 1000000)).mean()
        self.assertGreater(reasonable_followers, 0.95,
                         "Most users should have reasonable follower counts")
        
        # Check username diversity
        unique_username_ratio = data['username'].nunique() / len(data)
        self.assertGreater(unique_username_ratio, 0.95,
                         "Usernames should be highly unique")
        
        # Check interest diversity
        all_interests = []
        for interests_str in data['primary_interests']:
            all_interests.extend(interests_str.split(', '))
        
        unique_interests = len(set(all_interests))
        self.assertGreater(unique_interests, 5,
                         "Should have good variety of interests")
        
        # Check that derived metrics are calculated correctly
        self.assertTrue((data['posts_per_day'] == (data['posts_per_week'] / 7).round(1)).all(),
                       "Posts per day should be calculated correctly")
        
        self.assertTrue((data['total_weekly_content'] == 
                        (data['posts_per_week'] + data['stories_per_week'])).all(),
                       "Total weekly content should be calculated correctly")


if __name__ == '__main__':
    unittest.main()