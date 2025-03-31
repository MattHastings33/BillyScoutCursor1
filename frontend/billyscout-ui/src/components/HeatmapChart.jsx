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
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function HeatmapChart() {
  const { id } = useParams();
  const [heatmapData, setHeatmapData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedPitchType, setSelectedPitchType] = useState('all');

  useEffect(() => {
    const fetchHeatmapData = async () => {
      try {
        const response = await axios.get(`http://localhost:8000/heatmap/${id}`);
        setHeatmapData(response.data);
      } catch (err) {
        setError(err.response?.data?.detail || 'Error fetching heatmap data');
      } finally {
        setLoading(false);
      }
    };

    fetchHeatmapData();
  }, [id]);

  if (loading) {
    return (
      <div className="text-center py-12">
        <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-blue-500 border-t-transparent"></div>
        <p className="mt-2 text-gray-600">Loading heatmap data...</p>
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

  if (!heatmapData) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-600">No heatmap data available</p>
      </div>
    );
  }

  // Filter points based on selected pitch type
  const filteredPoints = selectedPitchType === 'all'
    ? heatmapData.points
    : heatmapData.points.filter(point => point.pitch_type === selectedPitchType);

  // Create grid data for visualization
  const gridSize = 10;
  const grid = Array(gridSize).fill().map(() => Array(gridSize).fill(0));

  filteredPoints.forEach(point => {
    const x = Math.floor(point.x * gridSize);
    const y = Math.floor(point.y * gridSize);
    if (x >= 0 && x < gridSize && y >= 0 && y < gridSize) {
      grid[y][x] += point.value;
    }
  });

  // Create chart data
  const chartData = {
    labels: Array(gridSize).fill().map((_, i) => i),
    datasets: grid.map((row, i) => ({
      label: `Row ${i}`,
      data: row,
      borderColor: `hsl(${(i / gridSize) * 360}, 70%, 50%)`,
      backgroundColor: `hsla(${(i / gridSize) * 360}, 70%, 50%, 0.2)`,
    })),
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Pitch Location Heatmap',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: Math.max(...grid.flat()),
      },
    },
  };

  // Get unique pitch types
  const pitchTypes = [...new Set(heatmapData.points.map(point => point.pitch_type))];

  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold text-gray-900">
            Pitch Location Heatmap
          </h1>
          
          <select
            value={selectedPitchType}
            onChange={(e) => setSelectedPitchType(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Pitches</option>
            {pitchTypes.map(type => (
              <option key={type} value={type}>
                {type.charAt(0).toUpperCase() + type.slice(1)}
              </option>
            ))}
          </select>
        </div>

        {/* Strike Zone Overlay */}
        <div className="relative mb-8">
          <div className="aspect-w-1 aspect-h-1 bg-gray-100 rounded-lg">
            <div
              className="absolute border-2 border-blue-500"
              style={{
                top: `${heatmapData.strike_zone.top * 100}%`,
                bottom: `${(1 - heatmapData.strike_zone.bottom) * 100}%`,
                left: `${heatmapData.strike_zone.left * 100}%`,
                right: `${(1 - heatmapData.strike_zone.right) * 100}%`,
              }}
            />
          </div>
        </div>

        {/* Heatmap Chart */}
        <div className="mb-8">
          <Line data={chartData} options={options} />
        </div>

        {/* Legend */}
        <div className="bg-gray-50 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            Heatmap Legend
          </h3>
          <div className="flex items-center space-x-4">
            <div className="flex items-center">
              <div className="w-4 h-4 bg-red-500 rounded mr-2"></div>
              <span className="text-sm text-gray-600">High Density</span>
            </div>
            <div className="flex items-center">
              <div className="w-4 h-4 bg-yellow-500 rounded mr-2"></div>
              <span className="text-sm text-gray-600">Medium Density</span>
            </div>
            <div className="flex items-center">
              <div className="w-4 h-4 bg-green-500 rounded mr-2"></div>
              <span className="text-sm text-gray-600">Low Density</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default HeatmapChart; 