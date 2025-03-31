import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import UploadZone from './components/UploadZone';
import ReportCard from './components/ReportCard';
import PlayerProfile from './components/PlayerProfile';
import HeatmapChart from './components/HeatmapChart';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-100">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<UploadZone />} />
            <Route path="/report/:id" element={<ReportCard />} />
            <Route path="/player/:id" element={<PlayerProfile />} />
            <Route path="/heatmap/:id" element={<HeatmapChart />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App; 