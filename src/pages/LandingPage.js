import React from 'react';
import { Link } from 'react-router-dom';
import { 
  Zap, 
  BarChart3, 
  BookOpen, 
  Play, 
  TrendingUp, 
  Leaf, 
  Gauge,
  Award,
  Users,
} from 'lucide-react';
import './LandingPage.css';

const LandingPage = () => {
  const features = [
    {
      icon: Zap,
      title: 'AI-Powered Optimization',
      description: 'Advanced reinforcement learning algorithms optimize driving patterns for maximum efficiency.',
      color: '#00ff88'
    },
    {
      icon: Leaf,
      title: 'Sustainability Focus',
      description: 'Reduce CO₂ emissions and fuel consumption through intelligent driving strategies.',
      color: '#00ff88'
    },
    {
      icon: Gauge,
      title: 'Real-time Analytics',
      description: 'Live telemetry data and performance metrics during simulation runs.',
      color: '#ff6b35'
    },
    {
      icon: Award,
      title: 'Performance Comparison',
      description: 'Compare AI vs Human driving performance across multiple metrics.',
      color: '#ff1744'
    }
  ];

  const stats = [
    { number: '25%', label: 'Fuel Savings', icon: TrendingUp },
    { number: '30%', label: 'CO₂ Reduction', icon: Leaf },
    { number: '15%', label: 'Faster Lap Times', icon: Gauge },
    { number: '1000+', label: 'Simulations Run', icon: Users }
  ];

  return (
    <div className="landing-page">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="container">
          <div className="hero-content">
            <div className="hero-badge">
              <Zap size={16} />
              AI-Powered Eco-Driving
            </div>
            <h1 className="hero-title">EcoDrive Simulator</h1>
            <p className="hero-description">
              Experience the future of sustainable racing with AI-powered eco-driving simulations
            </p>
            <div className="hero-actions">
              <Link to="/simulator" className="btn-primary">
                <Play size={20} />
                Start Simulation
              </Link>
              <Link to="/insights" className="btn-outline">
                <BookOpen size={20} />
                Learn More
              </Link>
            </div>
          </div>
          
          <div className="hero-visual">
            <div className="dashboard-preview">
              <div className="dashboard-header">
                <div className="dashboard-dots">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
                <div className="dashboard-title">Live Simulation</div>
              </div>
              <div className="dashboard-content">
                <div className="metric-card">
                  <div className="metric-value">1:23.4</div>
                  <div className="metric-label">Lap Time</div>
                </div>
                <div className="metric-card">
                  <div className="metric-value">85%</div>
                  <div className="metric-label">Efficiency</div>
                </div>
                <div className="metric-card">
                  <div className="metric-value">-15%</div>
                  <div className="metric-label">CO₂ Saved</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features-section">
        <div className="container">
          <div className="section-header">
            <h2 className="section-title">Why Choose EcoDrive?</h2>
            <p className="section-description">
              Advanced AI technology meets sustainable racing for the ultimate eco-driving experience
            </p>
          </div>
          <div className="features-grid">
            {features.map((feature, index) => (
              <div key={index} className="feature-card">
                <div className="feature-icon" style={{ color: feature.color }}>
                  <feature.icon size={32} />
                </div>
                <h3 className="feature-title">{feature.title}</h3>
                <p className="feature-description">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="stats-section">
        <div className="container">
          <div className="stats-grid">
            {stats.map((stat, index) => (
              <div key={index} className="stat-card">
                <div className="stat-icon">
                  <stat.icon size={24} />
                </div>
                <div className="stat-number">{stat.number}</div>
                <div className="stat-label">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta-section">
        <div className="container">
          <div className="cta-content">
            <h2 className="cta-title">Ready to Experience Eco-Driving?</h2>
            <p className="cta-description">
              Start your simulation and see how AI can optimize your driving for maximum efficiency
            </p>
            <div className="cta-actions">
              <Link to="/simulator" className="btn-primary">
                <Play size={24} />
                Launch Simulator
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default LandingPage;