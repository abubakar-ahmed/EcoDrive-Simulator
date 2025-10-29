import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Download, Code, Map, FileArchive, BookOpen, ExternalLink, CheckCircle } from 'lucide-react';
import './Developer.css';

const Developer = () => {
  const [downloading, setDownloading] = useState({});

  const handleDownload = async (resource, filename) => {
    setDownloading(prev => ({ ...prev, [resource]: true }));
    try {
      const response = await fetch(`/api/developer/download/${resource}`);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Download failed:', error);
    } finally {
      setDownloading(prev => ({ ...prev, [resource]: false }));
    }
  };

  const resources = [
    {
      id: 'models-ppo',
      title: 'PPO Model - Multi Track',
      description: 'Pre-trained PPO model for multi-track racing',
      size: '~50 MB',
      icon: FileArchive,
      type: 'model',
      filename: 'multi_track_ppo_model.zip'
    },
    {
      id: 'models-sac',
      title: 'SAC Model - Multi Track',
      description: 'Pre-trained SAC model for multi-track racing',
      size: '~45 MB',
      icon: FileArchive,
      type: 'model',
      filename: 'multi_track_sac_model.zip'
    },
    {
      id: 'maps',
      title: 'Track Maps',
      description: 'All racing track configurations (F1Tenth format)',
      size: '~15 MB',
      icon: Map,
      type: 'data',
      filename: 'track_maps.zip'
    },
    {
      id: 'environment',
      title: 'Environment Package',
      description: 'F1Tenth gym environment and dependencies',
      size: '~25 MB',
      icon: Code,
      type: 'code',
      filename: 'environment.zip'
    },
    {
      id: 'documentation',
      title: 'API Documentation',
      description: 'Complete API documentation and setup guides',
      size: '~2 MB',
      icon: BookOpen,
      type: 'docs',
      filename: 'api_documentation.pdf'
    }
  ];

  const setupSteps = [
    {
      step: 1,
      title: 'Install Dependencies',
      code: 'pip install -r requirements.txt'
    },
    {
      step: 2,
      title: 'Download Resources',
      description: 'Download models and maps using the buttons above'
    },
    {
      step: 3,
      title: 'Set Up Environment',
      code: 'export PYTHONPATH="${PYTHONPATH}:./f1tenth_rl-main"'
    },
    {
      step: 4,
      title: 'Run Test Script',
      code: 'python scripts/test_sac_silverstone.py'
    }
  ];

  return (
    <div className="developer-page">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="developer-container"
      >
        <div className="developer-header">
          <h1>Developer Resources</h1>
          <p>Download models, maps, and documentation to run EcoDrive Simulator locally</p>
        </div>

        <div className="developer-section">
          <h2>Available Resources</h2>
          <div className="resources-grid">
            {resources.map((resource) => {
              const Icon = resource.icon;
              const isDownloading = downloading[resource.id];
              
              return (
                <motion.div
                  key={resource.id}
                  className="resource-card"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <div className="resource-header">
                    <Icon size={32} className="resource-icon" />
                    <span className={`resource-type ${resource.type}`}>{resource.type}</span>
                  </div>
                  <h3>{resource.title}</h3>
                  <p>{resource.description}</p>
                  <div className="resource-footer">
                    <span className="resource-size">{resource.size}</span>
                    <button
                      className="download-button"
                      onClick={() => handleDownload(resource.id, resource.filename)}
                      disabled={isDownloading}
                    >
                      {isDownloading ? (
                        <>Downloading...</>
                      ) : (
                        <>
                          <Download size={16} />
                          Download
                        </>
                      )}
                    </button>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>

        <div className="developer-section">
          <h2>Quick Start Guide</h2>
          <div className="setup-steps">
            {setupSteps.map((item, index) => (
              <motion.div
                key={index}
                className="setup-step"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <div className="step-number">{item.step}</div>
                <div className="step-content">
                  <h3>{item.title}</h3>
                  {item.code && (
                    <div className="code-block">
                      <code>{item.code}</code>
                    </div>
                  )}
                  {item.description && <p>{item.description}</p>}
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        <div className="developer-section">
          <h2>API Endpoints</h2>
          <div className="api-grid">
            <div className="api-endpoint">
              <span className="method GET">GET</span>
              <code>/api/simulation/start</code>
              <p>Start a new simulation</p>
            </div>
            <div className="api-endpoint">
              <span className="method GET">GET</span>
              <code>/api/simulation/status</code>
              <p>Get current simulation status</p>
            </div>
            <div className="api-endpoint">
              <span className="method GET">GET</span>
              <code>/api/simulation/stop</code>
              <p>Stop the current simulation</p>
            </div>
            <div className="api-endpoint">
              <span className="method POST">POST</span>
              <code>/api/simulation/config</code>
              <p>Update simulation config</p>
            </div>
            <div className="api-endpoint">
              <span className="method GET">GET</span>
              <code>/api/tracks/list</code>
              <p>Get list of available tracks</p>
            </div>
            <div className="api-endpoint">
              <span className="method GET">GET</span>
              <code>/api/models/list</code>
              <p>Get list of available models</p>
            </div>
          </div>
        </div>

        <div className="developer-section">
          <h2>GitHub Repository</h2>
          <a
            href="https://github.com/yourusername/ecodrive-simulator"
            target="_blank"
            rel="noopener noreferrer"
            className="github-link"
          >
            <ExternalLink size={16} />
            View on GitHub
          </a>
        </div>
      </motion.div>
    </div>
  );
};

export default Developer;

