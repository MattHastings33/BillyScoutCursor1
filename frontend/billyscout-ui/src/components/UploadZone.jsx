import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { CloudArrowUpIcon } from '@heroicons/react/24/outline';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

function UploadZone() {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [videoUrl, setVideoUrl] = useState('');
  const [uploadType, setUploadType] = useState('file'); // 'file' or 'url'
  const navigate = useNavigate();

  const onDrop = async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;
    
    const file = acceptedFiles[0];
    if (file.size > 500 * 1024 * 1024) { // 500MB limit
      setError('File size must be less than 500MB');
      return;
    }

    setUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('video', file);

    try {
      const response = await axios.post('http://localhost:8000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      setSuccess(true);
      navigate(`/report/${response.data.report_id}`);
    } catch (err) {
      setError(err.response?.data?.detail || 'Error uploading video');
    } finally {
      setUploading(false);
    }
  };

  const handleUrlSubmit = async (e) => {
    e.preventDefault();
    if (!videoUrl) return;

    setUploading(true);
    setError(null);

    try {
      const response = await axios.post('http://localhost:8000/upload', {
        video_url: videoUrl,
      });
      
      setSuccess(true);
      navigate(`/report/${response.data.report_id}`);
    } catch (err) {
      setError(err.response?.data?.detail || 'Error processing video URL');
    } finally {
      setUploading(false);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/mp4': ['.mp4'],
    },
    maxFiles: 1,
  });

  return (
    <div className="max-w-2xl mx-auto p-6">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Upload Game Video
        </h1>
        <p className="text-gray-600">
          Upload a video file or provide a broadcast URL to analyze
        </p>
      </div>

      {/* Upload Type Selection */}
      <div className="flex justify-center space-x-4 mb-6">
        <button
          onClick={() => setUploadType('file')}
          className={`px-4 py-2 rounded-lg text-sm font-medium
            ${uploadType === 'file'
              ? 'bg-blue-500 text-white'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}`}
        >
          Upload Video File
        </button>
        <button
          onClick={() => setUploadType('url')}
          className={`px-4 py-2 rounded-lg text-sm font-medium
            ${uploadType === 'url'
              ? 'bg-blue-500 text-white'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}`}
        >
          Enter Broadcast URL
        </button>
      </div>

      {/* File Upload Zone */}
      {uploadType === 'file' && (
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
            ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}`}
        >
          <input {...getInputProps()} />
          <CloudArrowUpIcon className="mx-auto h-12 w-12 text-gray-400" />
          <p className="mt-2 text-gray-600">
            {isDragActive
              ? 'Drop the video file here'
              : 'Drag and drop a video file here, or click to select'}
          </p>
          <p className="mt-1 text-sm text-gray-500">
            MP4 files up to 500MB
          </p>
        </div>
      )}

      {/* URL Input Form */}
      {uploadType === 'url' && (
        <form onSubmit={handleUrlSubmit} className="space-y-4">
          <div>
            <label htmlFor="videoUrl" className="block text-sm font-medium text-gray-700 mb-1">
              Broadcast Video URL
            </label>
            <input
              type="url"
              id="videoUrl"
              value={videoUrl}
              onChange={(e) => setVideoUrl(e.target.value)}
              placeholder="https://www.northcoastnetwork.com/denison/?B=2216028"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              required
            />
            <p className="mt-1 text-sm text-gray-500">
              Enter the URL of the broadcast video
            </p>
          </div>
          <button
            type="submit"
            disabled={uploading || !videoUrl}
            className="w-full px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            {uploading ? 'Processing...' : 'Analyze Video'}
          </button>
        </form>
      )}

      {/* Status Messages */}
      {error && (
        <div className="mt-4 p-4 bg-red-50 text-red-700 rounded-lg">
          {error}
        </div>
      )}

      {success && (
        <div className="mt-4 p-4 bg-green-50 text-green-700 rounded-lg">
          Video uploaded successfully! Redirecting to analysis...
        </div>
      )}

      {/* Supported Platforms */}
      <div className="mt-8">
        <h3 className="text-sm font-medium text-gray-500 mb-2">
          Supported Broadcast Platforms
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-3 bg-gray-50 rounded-lg">
            <p className="text-sm font-medium text-gray-700">North Coast Network</p>
          </div>
          <div className="text-center p-3 bg-gray-50 rounded-lg">
            <p className="text-sm font-medium text-gray-700">Hudl</p>
          </div>
          <div className="text-center p-3 bg-gray-50 rounded-lg">
            <p className="text-sm font-medium text-gray-700">YouTube</p>
          </div>
          <div className="text-center p-3 bg-gray-50 rounded-lg">
            <p className="text-sm font-medium text-gray-700">Vimeo</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default UploadZone; 