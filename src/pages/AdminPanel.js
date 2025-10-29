import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft, Settings, Users, BarChart3, Server, Database, Cpu, HardDrive, RefreshCw } from 'lucide-react';
import './AdminPanel.css';

const AdminPanel = () => {
  const [systemStats, setSystemStats] = useState({
    activeSimulations: 0,
    totalUsers: 0,
    systemUptime: '99.8%',
    cpuUsage: '0%',
    memoryUsage: '0%',
    diskUsage: '0%'
  });
  
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchSystemStats = async () => {
    try {
      const response = await fetch('/api/admin/system-stats');
      const data = await response.json();
      setSystemStats({
        activeSimulations: data.active_simulations || 0,
        totalUsers: data.total_users || 0,
        systemUptime: data.system_uptime || '99.8%',
        cpuUsage: data.cpu_usage || '0%',
        memoryUsage: data.memory_usage || '0%',
        diskUsage: data.disk_usage || '0%'
      });
    } catch (error) {
      console.error('Failed to fetch system stats:', error);
    }
  };

  const fetchModels = async () => {
    try {
      const response = await fetch('/api/admin/models');
      const data = await response.json();
      setModels(data.models || []);
    } catch (error) {
      console.error('Failed to fetch models:', error);
    }
  };

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await Promise.all([fetchSystemStats(), fetchModels()]);
      setLoading(false);
    };
    
    loadData();
    
    // Set up auto-refresh every 5 seconds
    const interval = setInterval(fetchSystemStats, 5000);
    
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="admin-panel">
      <div className="container">
        <div className="page-header">
          <Link to="/" className="back-button">
            <ArrowLeft size={20} />
            Back to Home
          </Link>
          <div className="header-content">
            <div>
              <h1>Admin Panel</h1>
              <p>System administration, monitoring, and model management</p>
            </div>
            <button className="refresh-button" onClick={() => {
              fetchSystemStats();
              fetchModels();
            }}>
              <RefreshCw size={18} />
              Refresh
            </button>
          </div>
        </div>

        <div className="admin-content">
          <div className="admin-section">
            <h2>System Status</h2>
            <div className="status-grid">
              <div className="status-card">
                <div className="status-icon">
                  <BarChart3 size={20} />
                </div>
                <h3>Active Simulations</h3>
                <div className="status-value">{systemStats.activeSimulations}</div>
              </div>
              <div className="status-card">
                <div className="status-icon">
                  <Users size={20} />
                </div>
                <h3>Total Users</h3>
                <div className="status-value">{systemStats.totalUsers}</div>
              </div>
              <div className="status-card">
                <div className="status-icon">
                  <Server size={20} />
                </div>
                <h3>System Uptime</h3>
                <div className="status-value">{systemStats.systemUptime}</div>
              </div>
              <div className="status-card">
                <div className="status-icon">
                  <Cpu size={20} />
                </div>
                <h3>CPU Usage</h3>
                <div className="status-value">{systemStats.cpuUsage}</div>
              </div>
              <div className="status-card">
                <div className="status-icon">
                  <Database size={20} />
                </div>
                <h3>Memory Usage</h3>
                <div className="status-value">{systemStats.memoryUsage}</div>
              </div>
              <div className="status-card">
                <div className="status-icon">
                  <HardDrive size={20} />
                </div>
                <h3>Disk Usage</h3>
                <div className="status-value">{systemStats.diskUsage}</div>
              </div>
            </div>
          </div>

          <div className="admin-section">
            <h2>Model Management</h2>
            {loading && models.length === 0 ? (
              <div className="loading-state">Loading models...</div>
            ) : models.length === 0 ? (
              <div className="empty-state">No models available</div>
            ) : (
              <div className="model-grid">
                {models.map((model, index) => (
                  <div key={index} className="model-card">
                    <div className="model-header">
                      <h3>{model.name}</h3>
                      <div className="model-version">{model.version}</div>
                    </div>
                    {model.description && (
                      <p className="model-description">{model.description}</p>
                    )}
                    <div className="model-details">
                      <div className="model-detail">
                        <span className="detail-label">Status:</span>
                        <span className="model-status">{model.status}</span>
                      </div>
                      <div className="model-detail">
                        <span className="detail-label">Last Trained:</span>
                        <span className="detail-value">{model.last_trained}</span>
                      </div>
                      <div className="model-detail">
                        <span className="detail-label">Accuracy:</span>
                        <span className="detail-value">{model.accuracy}</span>
                      </div>
                      <div className="model-detail">
                        <span className="detail-label">Size:</span>
                        <span className="detail-value">{model.size}</span>
                      </div>
                      <div className="model-detail">
                        <span className="detail-label">Type:</span>
                        <span className="detail-value" style={{ textTransform: 'uppercase', fontWeight: 600 }}>{model.type}</span>
                      </div>
                    </div>
                    <div className="model-actions">
                      <button className="admin-button primary">Retrain</button>
                      <button className="admin-button secondary">Deploy</button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdminPanel;