from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import json

@dataclass
class GameStats:
    date: datetime
    opponent: str
    pitches_thrown: int = 0
    hits_allowed: int = 0
    strikeouts: int = 0
    walks: int = 0
    pitch_types: Dict[str, int] = field(default_factory=dict)
    avg_pitch_speed: Dict[str, float] = field(default_factory=dict)
    success_rate: Dict[str, float] = field(default_factory=dict)

@dataclass
class SwingStats:
    date: datetime
    opponent: str
    at_bats: int = 0
    hits: int = 0
    strikeouts: int = 0
    walks: int = 0
    avg_swing_speed: float = 0.0
    avg_swing_angle: float = 0.0
    timing_percentage: float = 0.0
    early_swings: int = 0
    late_swings: int = 0
    on_time_swings: int = 0

class Player:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role  # "pitcher" or "batter"
        self.games: List[GameStats] = []
        self.swings: List[SwingStats] = []
        
    def add_game_stats(self, stats: GameStats):
        """Add a game's pitching stats"""
        self.games.append(stats)
        
    def add_swing_stats(self, stats: SwingStats):
        """Add a game's batting stats"""
        self.swings.append(stats)
        
    def get_career_stats(self) -> Dict:
        """Get aggregated career statistics"""
        if self.role == "pitcher":
            return self._get_pitcher_career_stats()
        else:
            return self._get_batter_career_stats()
    
    def _get_pitcher_career_stats(self) -> Dict:
        """Calculate career pitching statistics"""
        if not self.games:
            return {}
            
        total_games = len(self.games)
        total_pitches = sum(game.pitches_thrown for game in self.games)
        total_hits = sum(game.hits_allowed for game in self.games)
        total_strikeouts = sum(game.strikeouts for game in self.games)
        total_walks = sum(game.walks for game in self.games)
        
        # Aggregate pitch types
        pitch_types = {}
        for game in self.games:
            for pitch_type, count in game.pitch_types.items():
                if pitch_type not in pitch_types:
                    pitch_types[pitch_type] = {
                        'count': 0,
                        'total_speed': 0.0,
                        'successes': 0
                    }
                pitch_types[pitch_type]['count'] += count
                pitch_types[pitch_type]['total_speed'] += game.avg_pitch_speed.get(pitch_type, 0) * count
                pitch_types[pitch_type]['successes'] += int(game.success_rate.get(pitch_type, 0) * count / 100)
        
        # Calculate averages
        for pitch_type in pitch_types:
            count = pitch_types[pitch_type]['count']
            pitch_types[pitch_type]['percentage'] = (count / total_pitches) * 100
            pitch_types[pitch_type]['avg_speed'] = pitch_types[pitch_type]['total_speed'] / count
            pitch_types[pitch_type]['success_rate'] = (pitch_types[pitch_type]['successes'] / count) * 100
        
        return {
            'total_games': total_games,
            'total_pitches': total_pitches,
            'hits_allowed': total_hits,
            'strikeouts': total_strikeouts,
            'walks': total_walks,
            'era': (total_hits * 9) / (total_games * 7),  # Simplified ERA calculation
            'k_per_9': (total_strikeouts * 9) / (total_games * 7),
            'bb_per_9': (total_walks * 9) / (total_games * 7),
            'pitch_types': pitch_types
        }
    
    def _get_batter_career_stats(self) -> Dict:
        """Calculate career batting statistics"""
        if not self.swings:
            return {}
            
        total_games = len(self.swings)
        total_at_bats = sum(game.at_bats for game in self.swings)
        total_hits = sum(game.hits for game in self.swings)
        total_strikeouts = sum(game.strikeouts for game in self.swings)
        total_walks = sum(game.walks for game in self.swings)
        
        # Calculate swing metrics
        total_swings = sum(game.early_swings + game.late_swings + game.on_time_swings 
                          for game in self.swings)
        if total_swings > 0:
            avg_speed = sum(game.avg_swing_speed * (game.early_swings + game.late_swings + game.on_time_swings)
                          for game in self.swings) / total_swings
            avg_angle = sum(game.avg_swing_angle * (game.early_swings + game.late_swings + game.on_time_swings)
                          for game in self.swings) / total_swings
            timing_percentage = sum(game.timing_percentage for game in self.swings) / total_games
        else:
            avg_speed = avg_angle = timing_percentage = 0.0
        
        return {
            'total_games': total_games,
            'at_bats': total_at_bats,
            'hits': total_hits,
            'strikeouts': total_strikeouts,
            'walks': total_walks,
            'batting_average': total_hits / total_at_bats if total_at_bats > 0 else 0,
            'on_base_percentage': (total_hits + total_walks) / total_at_bats if total_at_bats > 0 else 0,
            'swing_metrics': {
                'average_speed': avg_speed,
                'average_angle': avg_angle,
                'timing_percentage': timing_percentage
            }
        }
    
    def to_dict(self) -> Dict:
        """Convert player data to dictionary format"""
        return {
            'name': self.name,
            'role': self.role,
            'games': [self._game_to_dict(game) for game in self.games],
            'swings': [self._swing_to_dict(swing) for swing in self.swings],
            'career_stats': self.get_career_stats()
        }
    
    def _game_to_dict(self, game: GameStats) -> Dict:
        """Convert game stats to dictionary"""
        return {
            'date': game.date.isoformat(),
            'opponent': game.opponent,
            'pitches_thrown': game.pitches_thrown,
            'hits_allowed': game.hits_allowed,
            'strikeouts': game.strikeouts,
            'walks': game.walks,
            'pitch_types': game.pitch_types,
            'avg_pitch_speed': game.avg_pitch_speed,
            'success_rate': game.success_rate
        }
    
    def _swing_to_dict(self, swing: SwingStats) -> Dict:
        """Convert swing stats to dictionary"""
        return {
            'date': swing.date.isoformat(),
            'opponent': swing.opponent,
            'at_bats': swing.at_bats,
            'hits': swing.hits,
            'strikeouts': swing.strikeouts,
            'walks': swing.walks,
            'avg_swing_speed': swing.avg_swing_speed,
            'avg_swing_angle': swing.avg_swing_angle,
            'timing_percentage': swing.timing_percentage,
            'early_swings': swing.early_swings,
            'late_swings': swing.late_swings,
            'on_time_swings': swing.on_time_swings
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Player':
        """Create player instance from dictionary data"""
        player = cls(data['name'], data['role'])
        
        for game_data in data['games']:
            game = GameStats(
                date=datetime.fromisoformat(game_data['date']),
                opponent=game_data['opponent'],
                pitches_thrown=game_data['pitches_thrown'],
                hits_allowed=game_data['hits_allowed'],
                strikeouts=game_data['strikeouts'],
                walks=game_data['walks'],
                pitch_types=game_data['pitch_types'],
                avg_pitch_speed=game_data['avg_pitch_speed'],
                success_rate=game_data['success_rate']
            )
            player.add_game_stats(game)
        
        for swing_data in data['swings']:
            swing = SwingStats(
                date=datetime.fromisoformat(swing_data['date']),
                opponent=swing_data['opponent'],
                at_bats=swing_data['at_bats'],
                hits=swing_data['hits'],
                strikeouts=swing_data['strikeouts'],
                walks=swing_data['walks'],
                avg_swing_speed=swing_data['avg_swing_speed'],
                avg_swing_angle=swing_data['avg_swing_angle'],
                timing_percentage=swing_data['timing_percentage'],
                early_swings=swing_data['early_swings'],
                late_swings=swing_data['late_swings'],
                on_time_swings=swing_data['on_time_swings']
            )
            player.add_swing_stats(swing)
        
        return player 