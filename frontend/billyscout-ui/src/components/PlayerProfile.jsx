import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import axios from 'axios';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';
import { Line, Pie } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

function PlayerProfile() {
  const { id } = useParams();
  const [player, setPlayer] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchPlayerData = async () => {
      try {
        const response = await axios.get(`http://localhost:8000/player/${id}`);
        setPlayer(response.data);
      } catch (err) {
        setError(err.response?.data?.detail || 'Error fetching player data');
      } finally {
        setLoading(false);
      }
    };

    fetchPlayerData();
  }, [id]);

  if (loading) {
    return (
      <div className="text-center py-12">
        <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-blue-500 border-t-transparent"></div>
        <p className="mt-2 text-gray-600">Loading player data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-50 text-red-700 rounded-lg">
        {error}
      </div>
    );
  }

  if (!player) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-600">Player not found</p>
      </div>
    );
  }

  const careerStats = player.career_stats;

  // Create pitch type distribution chart data
  const pitchTypeData = {
    labels: Object.keys(careerStats.pitch_types || {}),
    datasets: [
      {
        data: Object.values(careerStats.pitch_types || {}).map(type => type.count),
        backgroundColor: [
          '#FF6384',
          '#36A2EB',
          '#FFCE56',
          '#4BC0C0',
          '#9966FF',
        ],
      },
    ],
  };

  // Create performance trend chart data
  const performanceData = {
    labels: player.games.map(game => new Date(game.date).toLocaleDateString()),
    datasets: [
      {
        label: player.role === 'pitcher' ? 'ERA' : 'Batting Average',
        data: player.role === 'pitcher'
          ? player.games.map(game => (game.hits_allowed * 9) / 7)
          : player.games.map(game => game.hits / game.at_bats),
        borderColor: '#36A2EB',
        tension: 0.1,
      },
    ],
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-white rounded-lg shadow-lg p-6">
        {/* Player Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">
              {player.name}
            </h1>
            <p className="text-gray-600 capitalize">
              {player.role}
            </p>
          </div>
        </div>

        {/* Career Stats Summary */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          {player.role === 'pitcher' ? (
            <>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="text-sm font-medium text-gray-500">ERA</h3>
                <p className="text-2xl font-bold text-gray-900">
                  {careerStats.era.toFixed(2)}
                </p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="text-sm font-medium text-gray-500">K/9</h3>
                <p className="text-2xl font-bold text-gray-900">
                  {careerStats.k_per_9.toFixed(2)}
                </p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="text-sm font-medium text-gray-500">BB/9</h3>
                <p className="text-2xl font-bold text-gray-900">
                  {careerStats.bb_per_9.toFixed(2)}
                </p>
              </div>
            </>
          ) : (
            <>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="text-sm font-medium text-gray-500">Batting Average</h3>
                <p className="text-2xl font-bold text-gray-900">
                  {careerStats.batting_average.toFixed(3)}
                </p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="text-sm font-medium text-gray-500">On-Base %</h3>
                <p className="text-2xl font-bold text-gray-900">
                  {careerStats.on_base_percentage.toFixed(3)}
                </p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="text-sm font-medium text-gray-500">Total Games</h3>
                <p className="text-2xl font-bold text-gray-900">
                  {careerStats.total_games}
                </p>
              </div>
            </>
          )}
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
          {/* Pitch Type Distribution */}
          {player.role === 'pitcher' && (
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Pitch Type Distribution
              </h3>
              <div className="h-64">
                <Pie data={pitchTypeData} />
              </div>
            </div>
          )}

          {/* Performance Trend */}
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Performance Trend
            </h3>
            <div className="h-64">
              <Line data={performanceData} />
            </div>
          </div>
        </div>

        {/* Game History */}
        <div>
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Game History
          </h2>
          <div className="space-y-4">
            {player.games.map((game, index) => (
              <div
                key={index}
                className="bg-gray-50 p-4 rounded-lg"
              >
                <div className="flex justify-between items-center">
                  <div>
                    <p className="font-medium text-gray-900">
                      vs {game.opponent}
                    </p>
                    <p className="text-sm text-gray-500">
                      {new Date(game.date).toLocaleDateString()}
                    </p>
                  </div>
                  <div className="text-right">
                    {player.role === 'pitcher' ? (
                      <>
                        <p className="font-medium text-gray-900">
                          {game.pitches_thrown} pitches
                        </p>
                        <p className="text-sm text-gray-500">
                          {game.strikeouts} K, {game.walks} BB
                        </p>
                      </>
                    ) : (
                      <>
                        <p className="font-medium text-gray-900">
                          {game.hits}/{game.at_bats}
                        </p>
                        <p className="text-sm text-gray-500">
                          {game.strikeouts} K, {game.walks} BB
                        </p>
                      </>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default PlayerProfile; 