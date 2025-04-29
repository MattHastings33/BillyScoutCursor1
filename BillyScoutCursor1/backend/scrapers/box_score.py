import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import re
from datetime import datetime
import json
from urllib.parse import urljoin

class BoxScoreScraper:
    def __init__(self):
        # Base URLs for different athletic departments
        self.base_urls = {
            'northcoastnetwork.com': "https://denisonbigred.com",
            'nsnsports.net': "https://www.batesbobcats.com",
            'hudl.com': "https://www.hudl.com",
            'presto.com': "https://www.prestosports.com",
            'stretch.com': "https://www.stretch.com",
            'ne10sports.com': "https://www.ne10sports.com",
            'd3baseball.com': "https://www.d3baseball.com",
            'ncaa.com': "https://www.ncaa.com",
            'hudl.tv': "https://www.hudl.com/api/v2"
        }

        # Platform configurations
        self.supported_platforms = {
            'northcoastnetwork.com': {
                'id_pattern': r'B=(\d+)',
                'get_box_score': self._get_hudl_box_score,
                'table_class': 'sidearm-table'
            },
            'nsnsports.net': {
                'id_pattern': r'bfplayvid=(\d+)',
                'get_box_score': self._get_nsn_box_score,
                'table_class': 'stats-table'
            },
            'hudl.com': {
                'id_pattern': r'videos/(\d+)',
                'get_box_score': self._get_hudl_box_score,
                'table_class': 'hudl-table'
            },
            'presto.com': {
                'id_pattern': r'game/(\d+)',
                'get_box_score': self._get_presto_box_score,
                'table_class': 'presto-table'
            },
            'stretch.com': {
                'id_pattern': r'game/(\d+)',
                'get_box_score': self._get_stretch_box_score,
                'table_class': 'stretch-table'
            },
            'ne10sports.com': {
                'id_pattern': r'game/(\d+)',
                'get_box_score': self._get_ne10_box_score,
                'table_class': 'ne10-table'
            }
        }

        # Headers for Hudl API requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def extract_game_id(self, video_url: str) -> Optional[Dict]:
        """Extract game ID from video URL"""
        try:
            if 'northcoastnetwork.com' in video_url:
                match = re.search(r'B=(\d+)', video_url)
                if match:
                    return {
                        'platform': 'northcoastnetwork.com',
                        'game_id': match.group(1)
                    }
            return None
        except Exception as e:
            print(f"Error extracting game ID: {str(e)}")
            return None

    def _get_box_score_url(self, game_id: str) -> Optional[str]:
        """Get box score URL from game ID"""
        try:
            # For North Coast Network, we can construct the box score URL directly
            box_score_url = f"{self.base_urls['northcoastnetwork.com']}/sports/baseball/stats/2025/kenyon-college/boxscore/{game_id}"
            print(f"Constructed box score URL: {box_score_url}")
            
            # Verify the URL exists
            response = requests.get(box_score_url, headers=self.headers)
            if response.status_code == 200:
                return box_score_url
            
            # If direct URL doesn't work, try searching through schedule
            schedule_url = f"{self.base_urls['northcoastnetwork.com']}/sports/baseball/schedule"
            response = requests.get(schedule_url, headers=self.headers)
            
            if response.status_code != 200:
                print(f"Failed to get schedule page: {response.status_code}")
                return None
            
            # Parse the page to find the game link
            soup = BeautifulSoup(response.text, 'html.parser')
            game_links = soup.find_all('a', href=True)
            
            for link in game_links:
                if game_id in link.get('href', ''):
                    # Found a matching game, check if it has a box score
                    game_url = urljoin(self.base_urls['northcoastnetwork.com'], link['href'])
                    game_response = requests.get(game_url, headers=self.headers)
                    
                    if game_response.status_code == 200:
                        game_soup = BeautifulSoup(game_response.text, 'html.parser')
                        box_score_link = game_soup.find('a', text=re.compile('Box Score', re.I))
                        
                        if box_score_link and box_score_link.get('href'):
                            box_score_url = urljoin(self.base_urls['northcoastnetwork.com'], box_score_link['href'])
                            print(f"Found box score URL: {box_score_url}")
                            return box_score_url
            
            return None
            
        except Exception as e:
            print(f"Error getting box score URL: {str(e)}")
            return None

    def _get_ncn_box_score(self, game_id: str) -> Optional[str]:
        """Get box score URL for North Coast Network games"""
        try:
            search_url = f"{self.base_urls['northcoastnetwork.com']}/sports/baseball/schedule"
            response = requests.get(search_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                if game_id in link['href']:
                    game_url = link['href']
                    box_score_url = game_url.replace('schedule', 'stats/2025') + '/boxscore'
                    return box_score_url
            return None
        except Exception:
            return None

    def _get_nsn_box_score(self, game_id: str) -> Optional[str]:
        """Get box score URL for Northeast Sports Network games"""
        try:
            search_url = f"{self.base_urls['nsnsports.net']}/sports/baseball/schedule"
            response = requests.get(search_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                if game_id in link['href']:
                    game_url = link['href']
                    box_score_url = game_url.replace('schedule', 'stats/2024') + '/boxscore'
                    return box_score_url
            return None
        except Exception:
            return None

    def _get_hudl_box_score(self, game_id: str) -> Optional[Dict]:
        """Get box score data from Hudl TV API"""
        try:
            # First get the game details
            game_url = f"{self.base_urls['hudl.tv']}/games/{game_id}"
            game_response = requests.get(game_url, headers=self.headers)
            
            if game_response.status_code != 200:
                print(f"Failed to get game data: {game_response.status_code}")
                return None
                
            game_data = game_response.json()
            
            # Then get the box score
            box_score_url = f"{self.base_urls['hudl.tv']}/games/{game_id}/boxscore"
            box_score_response = requests.get(box_score_url, headers=self.headers)
            
            if box_score_response.status_code != 200:
                print(f"Failed to get box score: {box_score_response.status_code}")
                return None
                
            box_score_data = box_score_response.json()
            
            # Extract pitcher information
            pitchers = []
            current_inning = 0
            
            if 'pitchers' in box_score_data:
                for pitcher in box_score_data['pitchers']:
                    innings = float(pitcher.get('ip', 0))
                    pitchers.append({
                        'name': pitcher.get('name', 'Unknown'),
                        'innings': innings,
                        'start_inning': current_inning + 1,
                        'end_inning': current_inning + innings
                    })
                    current_inning += innings
            
            return {
                'pitchers': pitchers,
                'total_innings': current_inning,
                'platform': 'northcoastnetwork.com'
            }
            
        except Exception as e:
            print(f"Error getting box score: {str(e)}")
            return None

    def _get_presto_box_score(self, game_id: str) -> Optional[str]:
        """Get box score URL for Presto games"""
        try:
            search_url = f"{self.base_urls['presto.com']}/baseball/schedule"
            response = requests.get(search_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                if game_id in link['href']:
                    game_url = link['href']
                    box_score_url = game_url.replace('schedule', 'stats') + '/boxscore'
                    return box_score_url
            return None
        except Exception:
            return None

    def _get_stretch_box_score(self, game_id: str) -> Optional[str]:
        """Get box score URL for Stretch games"""
        try:
            search_url = f"{self.base_urls['stretch.com']}/baseball/schedule"
            response = requests.get(search_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                if game_id in link['href']:
                    game_url = link['href']
                    box_score_url = game_url.replace('schedule', 'stats') + '/boxscore'
                    return box_score_url
            return None
        except Exception:
            return None

    def _get_ne10_box_score(self, game_id: str) -> Optional[str]:
        """Get box score URL for NE10 games"""
        try:
            search_url = f"{self.base_urls['ne10sports.com']}/baseball/schedule"
            response = requests.get(search_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                if game_id in link['href']:
                    game_url = link['href']
                    box_score_url = game_url.replace('schedule', 'stats') + '/boxscore'
                    return box_score_url
            return None
        except Exception:
            return None

    def scrape_box_score(self, box_score_url: str) -> Dict:
        """Scrape box score data from URL"""
        try:
            print(f"Fetching box score from: {box_score_url}")
            response = requests.get(box_score_url, headers=self.headers)
            
            if response.status_code != 200:
                print(f"Failed to get box score page: {response.status_code}")
                return {}
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the pitching section
            pitchers = []
            current_inning = 0
            
            # First try to find the Kenyon section
            kenyon_section = None
            for section in soup.find_all(['div', 'section']):
                if section.get_text(strip=True).upper().startswith('KENYON'):
                    kenyon_section = section
                    break
            
            if kenyon_section:
                # Look for tables within the Kenyon section
                tables = kenyon_section.find_all('table')
            else:
                # If no Kenyon section found, look through all tables
                tables = soup.find_all('table')
            
            for table in tables:
                # Look for rows that contain pitcher information
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td'])
                    if len(cells) >= 3:
                        # Check if this is a pitcher row
                        pos_cell = cells[0].get_text(strip=True)
                        if pos_cell in ['p', 'p/dh']:
                            try:
                                # Get the pitcher's name
                                name = cells[0].get_text(strip=True)
                                
                                # Find the innings pitched
                                for i, cell in enumerate(cells):
                                    text = cell.get_text(strip=True)
                                    try:
                                        innings = float(text)
                                        if innings > 0:  # Only include pitchers who actually pitched
                                            pitchers.append({
                                                'name': name,
                                                'innings': innings,
                                                'start_inning': current_inning + 1,
                                                'end_inning': current_inning + innings,
                                                'position': pos_cell
                                            })
                                            current_inning += innings
                                            print(f"Found pitcher: {name} ({innings} innings)")
                                            break
                                    except ValueError:
                                        continue
                                
                            except (ValueError, IndexError) as e:
                                print(f"Could not parse data for {pos_cell}: {str(e)}")
                                continue
            
            if not pitchers:
                print("No pitchers found in the box score")
            else:
                print(f"Found {len(pitchers)} pitchers")
            
            # Extract the final score
            score_table = soup.find('table', {'class': 'linescore'})
            if score_table:
                rows = score_table.find_all('tr')
                for row in rows:
                    if 'KENYON' in row.get_text():
                        cells = row.find_all('td')
                        if len(cells) >= 9:  # Should have 8 innings plus R H E
                            runs = cells[-3].get_text(strip=True)
                            hits = cells[-2].get_text(strip=True)
                            errors = cells[-1].get_text(strip=True)
                            print(f"Final line: {runs} runs, {hits} hits, {errors} errors")
            
            return {
                'pitchers': pitchers,
                'total_innings': current_inning,
                'platform': 'northcoastnetwork.com'
            }
            
        except Exception as e:
            print(f"Error scraping box score: {str(e)}")
            return {}

    def get_pitcher_info(self, video_url: str) -> Dict:
        """Get pitcher information for a given video URL"""
        game_info = self.extract_game_id(video_url)
        if not game_info:
            print("Could not extract game ID from URL")
            return {}
            
        game_id = game_info['game_id']
        print(f"Extracted game ID: {game_id}")
        
        # Get box score URL
        box_score_url = self._get_box_score_url(game_id)
        if not box_score_url:
            print("Could not find box score URL")
            return {}
            
        # Scrape box score data
        return self.scrape_box_score(box_score_url) 