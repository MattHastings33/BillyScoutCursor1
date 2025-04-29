import React from 'react';
import { Link } from 'react-router-dom';
import { HomeIcon, ChartBarIcon, UserGroupIcon } from '@heroicons/react/24/outline';

function Navbar() {
  return (
    <nav className="bg-white shadow-lg">
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center">
            <Link to="/" className="flex items-center">
              <span className="text-2xl font-bold text-blue-600">BillyScout</span>
            </Link>
          </div>
          
          <div className="flex items-center space-x-4">
            <Link
              to="/"
              className="flex items-center px-3 py-2 rounded-md text-gray-700 hover:text-blue-600 hover:bg-gray-100"
            >
              <HomeIcon className="h-5 w-5 mr-2" />
              Upload
            </Link>
            
            <Link
              to="/heatmap"
              className="flex items-center px-3 py-2 rounded-md text-gray-700 hover:text-blue-600 hover:bg-gray-100"
            >
              <ChartBarIcon className="h-5 w-5 mr-2" />
              Heatmap
            </Link>
            
            <Link
              to="/players"
              className="flex items-center px-3 py-2 rounded-md text-gray-700 hover:text-blue-600 hover:bg-gray-100"
            >
              <UserGroupIcon className="h-5 w-5 mr-2" />
              Players
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
}

export default Navbar; 