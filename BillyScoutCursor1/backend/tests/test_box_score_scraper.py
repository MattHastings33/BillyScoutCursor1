import unittest
from scrapers.box_score import BoxScoreScraper
import json

class TestBoxScoreScraper(unittest.TestCase):
    def setUp(self):
        self.scraper = BoxScoreScraper()
        
    def test_url_patterns(self):
        """Test URL pattern recognition for different platforms"""
        test_urls = {
            'northcoastnetwork.com': 'https://www.northcoastnetwork.com/denison/?B=2216031',
            'nsnsports.net': 'https://www.nsnsports.net/colleges/bates/?bfplayvid=2263495',
            'hudl.com': 'https://www.hudl.com/videos/123456',
            'presto.com': 'https://www.prestosports.com/game/123456',
            'stretch.com': 'https://www.stretch.com/game/123456',
            'ne10sports.com': 'https://www.ne10sports.com/game/123456'
        }
        
        for platform, url in test_urls.items():
            result = self.scraper.extract_game_id(url)
            self.assertIsNotNone(result, f"Failed to extract game ID for {platform}")
            self.assertEqual(result['platform'], platform)
            self.assertIsNotNone(result['game_id'])

    def test_box_score_scraping(self):
        """Test box score scraping for each platform"""
        test_cases = [
            {
                'platform': 'northcoastnetwork.com',
                'url': 'https://www.northcoastnetwork.com/denison/?B=2216031',
                'expected_fields': ['pitchers', 'total_innings', 'platform']
            },
            {
                'platform': 'nsnsports.net',
                'url': 'https://www.nsnsports.net/colleges/bates/?bfplayvid=2263495',
                'expected_fields': ['pitchers', 'total_innings', 'platform']
            }
        ]
        
        for case in test_cases:
            result = self.scraper.get_pitcher_info(case['url'])
            
            # Check if result contains expected fields
            for field in case['expected_fields']:
                self.assertIn(field, result, f"Missing field {field} for {case['platform']}")
            
            # Check if pitchers data is properly structured
            if result.get('pitchers'):
                for pitcher in result['pitchers']:
                    self.assertIn('name', pitcher)
                    self.assertIn('innings', pitcher)
                    self.assertIn('start_inning', pitcher)
                    self.assertIn('end_inning', pitcher)
                    
                    # Validate data types
                    self.assertIsInstance(pitcher['innings'], float)
                    self.assertIsInstance(pitcher['start_inning'], int)
                    self.assertIsInstance(pitcher['end_inning'], float)

    def test_error_handling(self):
        """Test error handling for invalid URLs and missing data"""
        invalid_urls = [
            'https://invalid-url.com',
            'https://northcoastnetwork.com/invalid',
            'https://nsnsports.net/invalid',
            'https://hudl.com/invalid'
        ]
        
        for url in invalid_urls:
            result = self.scraper.get_pitcher_info(url)
            self.assertEqual(result, {}, f"Should handle invalid URL: {url}")

    def test_pitcher_data_validation(self):
        """Test validation of pitcher data structure and values"""
        test_url = 'https://www.northcoastnetwork.com/denison/?B=2216031'
        result = self.scraper.get_pitcher_info(test_url)
        
        if result.get('pitchers'):
            for pitcher in result['pitchers']:
                # Check data types
                self.assertIsInstance(pitcher['name'], str)
                self.assertIsInstance(pitcher['innings'], float)
                self.assertIsInstance(pitcher['start_inning'], int)
                self.assertIsInstance(pitcher['end_inning'], float)
                
                # Check value ranges
                self.assertGreaterEqual(pitcher['innings'], 0)
                self.assertGreaterEqual(pitcher['start_inning'], 1)
                self.assertGreaterEqual(pitcher['end_inning'], pitcher['start_inning'])

    def test_platform_specific_scraping(self):
        """Test platform-specific scraping methods"""
        test_cases = [
            {
                'platform': 'northcoastnetwork.com',
                'game_id': '2216031',
                'method': self.scraper._get_ncn_box_score
            },
            {
                'platform': 'nsnsports.net',
                'game_id': '2263495',
                'method': self.scraper._get_nsn_box_score
            }
        ]
        
        for case in test_cases:
            box_score_url = case['method'](case['game_id'])
            self.assertIsNotNone(box_score_url, f"Failed to get box score URL for {case['platform']}")

if __name__ == '__main__':
    unittest.main() 