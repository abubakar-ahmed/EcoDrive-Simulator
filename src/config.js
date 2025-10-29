/**
 * Configuration file for frontend API endpoints
 * Uses environment variables for different deployment environments
 */
const config = {
  // API base URL - defaults to localhost:5000 for development
  API_BASE_URL: process.env.REACT_APP_API_URL || 'http://localhost:5000',
  
  // WebSocket/EventSource base URL
  WS_BASE_URL: process.env.REACT_APP_WS_URL || 'http://localhost:5000',
  
  // Environment
  ENV: process.env.NODE_ENV || 'development',
  
  // API endpoints
  endpoints: {
    simulation: {
      start: '/api/simulation/start',
      status: '/api/web-simulation/status',
      stop: '/api/web-simulation/stop',
      stream: '/api/web-simulation/stream',
      update: (id) => `/api/simulation/${id}/update`,
      complete: (id) => `/api/simulation/${id}/complete`,
    },
    tracks: '/api/tracks',
    models: '/api/models/available',
    comparison: (track) => `/api/comparison/${track}`,
    telemetry: '/api/telemetry/generate',
    admin: {
      systemStats: '/api/admin/system-stats',
      models: '/api/admin/models',
      simulations: '/api/admin/simulations',
    },
  },
  
  // Helper function to get full API URL
  getApiUrl: (endpoint) => {
    // If endpoint already includes http, return as-is
    if (endpoint.startsWith('http://') || endpoint.startsWith('https://')) {
      return endpoint;
    }
    // Remove leading slash if present and combine with base URL
    const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
    return `${config.API_BASE_URL}${cleanEndpoint}`;
  },
};

export default config;

