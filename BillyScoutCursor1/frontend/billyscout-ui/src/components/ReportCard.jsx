import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import ReactPlayer from 'react-player';
import axios from 'axios';
import { Line } from 'react-chartjs-2';

function ReportCard() {
  const { id } = useParams();
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedPitch, setSelectedPitch] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [selectedPitcher, setSelectedPitcher] = useState(null);

  useEffect(() => {
    const fetchReport = async () => {
      try {
        const response = await axios.get(`http://localhost:8000/report/${id}`);
        setReport(response.data);
        setVideoUrl(response.data.video_url);
      } catch (err) {
        setError(err.response?.data?.detail || 'Error fetching report');
      } finally {
        setLoading(false);
      }
    };

    fetchReport();
  }, [id]);

  const handlePitcherSelect = (pitcherName) => {
    setSelectedPitcher(pitcherName);
  };

  const generateVelocityChartData = (pitcherName) => {
    const velocityStats = report.pitcher_report.pitcher_reports[pitcherName].velocity_stats;
    const trend = velocityStats.trend;
    
    return {
      labels: Array.from({ length: trend.length }, (_, i) => `Pitch ${i + 1}`),
      datasets: [
        {
          label: 'Velocity (mph)',
          data: trend,
          borderColor: '#3B82F6',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          tension: 0.1,
          fill: true,
        },
      ],
    };
  };

  const velocityChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: false,
        title: {
          display: true,
          text: 'Velocity (mph)',
        },
      },
    },
    plugins: {
      title: {
        display: true,
        text: 'Velocity Trend',
      },
    },
  };

  if (loading) {
    return (
      <div className="text-center py-12">
        <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-blue-500 border-t-transparent"></div>
        <p className="mt-2 text-gray-600">Loading report...</p>
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

  if (!report) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-600">Report not found</p>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-6">
          Scouting Report
        </h1>

        {/* Video Player */}
        <div className="mb-8">
          <div className="aspect-w-16 aspect-h-9 bg-gray-100 rounded-lg overflow-hidden">
            {videoUrl && (
              <ReactPlayer
                url={videoUrl}
                controls
                width="100%"
                height="100%"
                playing={false}
                onProgress={({ playedSeconds }) => {
                  // Handle video progress
                }}
              />
            )}
          </div>
        </div>

        {/* Pitcher Selection */}
        <div className="mb-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Pitchers
          </h2>
          <div className="flex space-x-2 overflow-x-auto pb-4">
            {Object.keys(report.pitcher_report.pitcher_reports).map((pitcherName) => (
              <button
                key={pitcherName}
                onClick={() => handlePitcherSelect(pitcherName)}
                className={`px-4 py-2 rounded-lg text-sm font-medium
                  ${selectedPitcher === pitcherName
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}`}
              >
                {pitcherName}
              </button>
            ))}
          </div>
        </div>

        {/* Pitcher Changes Timeline */}
        <div className="mb-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Pitcher Changes
          </h2>
          <div className="space-y-4">
            {report.pitcher_report.pitcher_changes.map((change, index) => (
              <div
                key={index}
                className="bg-gray-50 rounded-lg p-4"
              >
                <div className="flex justify-between items-center">
                  <div>
                    <p className="font-medium text-gray-900">
                      {change.pitcher_name} (#{change.jersey_number})
                    </p>
                    <p className="text-sm text-gray-500">
                      {change.team} - Inning {change.inning}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-gray-500">
                      Score: {change.score}
                    </p>
                    <p className="text-sm text-gray-500">
                      {new Date(change.timestamp * 1000).toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Selected Pitcher Report */}
        {selectedPitcher && (
          <div className="mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">
              {selectedPitcher} Analysis
            </h2>

            {/* Velocity Analysis */}
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Velocity Analysis
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h4 className="text-sm font-medium text-blue-700 mb-1">Max Velocity</h4>
                  <p className="text-2xl font-bold text-blue-900">
                    {report.pitcher_report.pitcher_reports[selectedPitcher].velocity_stats.max.toFixed(1)} mph
                  </p>
                </div>
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h4 className="text-sm font-medium text-blue-700 mb-1">Average Velocity</h4>
                  <p className="text-2xl font-bold text-blue-900">
                    {report.pitcher_report.pitcher_reports[selectedPitcher].velocity_stats.avg.toFixed(1)} mph
                  </p>
                </div>
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h4 className="text-sm font-medium text-blue-700 mb-1">Min Velocity</h4>
                  <p className="text-2xl font-bold text-blue-900">
                    {report.pitcher_report.pitcher_reports[selectedPitcher].velocity_stats.min.toFixed(1)} mph
                  </p>
                </div>
              </div>

              {/* Velocity by Pitch Type */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-4">
                {Object.entries(report.pitcher_report.pitcher_reports[selectedPitcher].velocity_stats.by_pitch_type).map(([type, stats]) => (
                  <div key={type} className="bg-gray-50 p-4 rounded-lg">
                    <h4 className="font-medium text-gray-900 mb-2 capitalize">
                      {type}
                    </h4>
                    <div className="space-y-1">
                      <p className="text-sm text-gray-600">
                        Max: {stats.max.toFixed(1)} mph
                      </p>
                      <p className="text-sm text-gray-600">
                        Avg: {stats.avg.toFixed(1)} mph
                      </p>
                      <p className="text-sm text-gray-600">
                        Min: {stats.min.toFixed(1)} mph
                      </p>
                    </div>
                  </div>
                ))}
              </div>

              {/* Velocity Trend Chart */}
              <div className="bg-white p-4 rounded-lg shadow">
                <div className="h-64">
                  <Line data={generateVelocityChartData(selectedPitcher)} options={velocityChartOptions} />
                </div>
              </div>
            </div>

            {/* Existing pitch type distribution and count-based selection sections */}
            <div className="bg-gray-50 rounded-lg p-4 mb-6">
              <pre className="whitespace-pre-wrap text-gray-700">
                {report.pitcher_report.pitcher_reports[selectedPitcher].summary}
              </pre>
            </div>

            {/* Count-Based Pitch Selection */}
            <div className="mt-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Count-Based Pitch Selection
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {Object.entries(report.pitcher_report.pitcher_reports[selectedPitcher].count_based_selection)
                  .filter(([_, data]) => data.total >= 5) // Only show counts with enough data
                  .map(([count, data]) => (
                    <div key={count} className="bg-white rounded-lg shadow p-4">
                      <h4 className="font-medium text-gray-900 mb-2">
                        {count} Count ({data.total} pitches)
                      </h4>
                      <div className="space-y-2">
                        {Object.entries(data.types).map(([type, percentage]) => (
                          <div key={type} className="flex items-center justify-between">
                            <span className="text-sm text-gray-600 capitalize">
                              {type}
                            </span>
                            <div className="flex items-center">
                              <div className="w-24 h-2 bg-gray-200 rounded-full mr-2">
                                <div
                                  className="h-full bg-blue-500 rounded-full"
                                  style={{ width: `${percentage}%` }}
                                />
                              </div>
                              <span className="text-sm font-medium text-gray-900">
                                {percentage.toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          </div>
        )}

        {/* Batter Report */}
        <div className="mb-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Batter Analysis
          </h2>
          <div className="bg-gray-50 rounded-lg p-4">
            <pre className="whitespace-pre-wrap text-gray-700">
              {report.batter_report.summary}
            </pre>
          </div>
        </div>

        {/* Heatmap */}
        <div>
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Pitch Location Heatmap
          </h2>
          <div className="bg-gray-50 rounded-lg p-4">
            {/* Heatmap visualization would go here */}
            <div className="aspect-w-1 aspect-h-1 bg-gray-200 rounded-lg">
              {/* Placeholder for heatmap */}
            </div>
          </div>
        </div>
      </div>

      {/* Selected Pitch Details Modal */}
      {selectedPitch && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg p-6 max-w-lg w-full">
            <h3 className="text-xl font-semibold text-gray-900 mb-4">
              Pitch Details
            </h3>
            <div className="space-y-2">
              <p>
                <span className="font-medium">Type:</span> {selectedPitch.type}
              </p>
              <p>
                <span className="font-medium">Speed:</span> {selectedPitch.speed} mph
              </p>
              <p>
                <span className="font-medium">Result:</span> {selectedPitch.result}
              </p>
            </div>
            <button
              onClick={() => setSelectedPitch(null)}
              className="mt-4 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default ReportCard; 